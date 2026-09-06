"""Immutable prediction snapshots and append-only score revisions for validation."""
from __future__ import annotations

from contextlib import closing
from io import StringIO
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESS_INSTANCE = uuid.uuid4().hex
PROVENANCE_COLUMNS = [
    "snapshot_id", "prediction_generated_at", "model_version", "model_trained_through",
    "model_available_at", "training_cutoff_basis", "probability_semantics",
    "game_start_utc", "odds_recorded_at", "quote_bookmaker", "quote_binding_verified",
    "provider_quotes", "calibrated_probability",
]


def now_utc():
    return pd.Timestamp.now(tz="UTC").isoformat()


def database_path():
    return Path(os.environ.get("PARLAYPICKER_EVIDENCE_DIR", str(ROOT / "data/prediction_evidence"))) / "evidence.sqlite3"


def connect(path=None):
    path = Path(path or database_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS bundles (
            version TEXT PRIMARY KEY, frozen_at TEXT NOT NULL, manifest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_id TEXT PRIMARY KEY, version TEXT NOT NULL REFERENCES bundles(version),
            generated_at TEXT NOT NULL, candidates TEXT NOT NULL, decisions TEXT NOT NULL,
            inputs TEXT NOT NULL, payload_hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS snapshot_runtime (
            snapshot_id TEXT PRIMARY KEY REFERENCES snapshots(snapshot_id), process_instance TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS score_revisions (
            snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
            evidence_hash TEXT NOT NULL, recorded_at TEXT NOT NULL, scores TEXT NOT NULL,
            PRIMARY KEY(snapshot_id, evidence_hash));
    """)
    # Protect against accidental UPDATE/DELETE even from future application code.
    for table in ("bundles", "snapshots", "score_revisions", "snapshot_runtime"):
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{action} BEFORE {action} ON {table} "
                       "BEGIN SELECT RAISE(ABORT, 'prediction evidence is append-only'); END")
    return db


def artifact_manifest(controls, root=ROOT):
    root = Path(root)
    paths = set()
    for pattern in ("core/**/*.py", "app_core/**/*.py", "integrations/*.py", "models/*",
                    "data/calibration/*", "data/backtest_exports/*", "data/tier_results/*.csv"):
        paths.update(p for p in root.glob(pattern) if p.is_file())
    paths.update(p for p in root.glob("*.py") if p.is_file())
    dynamic = root / "data/dynamic_aliases.json"
    if dynamic.exists():
        paths.add(dynamic)
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    # Explicit allowlist excludes uploaded file handles, API keys and session data.
    config = {k: controls[k] for k in ("sports", "use_ml", "use_gemini", "bankroll", "max_legs") if k in controls}
    from app_core import weights_config
    primitive = (str, int, float, bool, type(None))
    runtime_weights = {name: value for name, value in vars(weights_config).items()
                       if name.isupper() and isinstance(value, primitive)}
    return {"schema": 1, "artifacts": hashes, "controls": config, "runtime_weights": runtime_weights,
            "evaluation_policy": "whole Eastern slates after the first bundle freeze day; no tuning during evaluation"}


def begin_run(controls, *, path=None, root=ROOT):
    manifest = artifact_manifest(controls, root)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    version = hashlib.sha256(encoded.encode()).hexdigest()
    frozen = now_utc()
    with closing(connect(path)) as db, db:
        db.execute("INSERT OR IGNORE INTO bundles VALUES (?, ?, ?)", (version, frozen, encoded))
        frozen = db.execute("SELECT frozen_at FROM bundles WHERE version=?", (version,)).fetchone()[0]
    return {"snapshot_id": uuid.uuid4().hex, "model_version": version, "frozen_at": frozen,
            "manifest": manifest, "root": str(root), "controls": controls}


def provider_quotes(game):
    """Keep actual per-book/per-market source times; never stamp a download as a quote."""
    quotes = []
    for book in game.get("bookmakers", []):
        name = str(book.get("key", "")).lower()
        name = "novig" if name.startswith("novig") else name
        for market in book.get("markets", []):
            family = market.get("key")
            for outcome in market.get("outcomes", []):
                side = str(outcome.get("name", "")).strip().casefold()
                if family == "totals" and side in {"over", "under"}:
                    kind = "total_" + side
                elif family in {"spreads", "h2h"}:
                    if side == str(game.get("home_team", "")).strip().casefold():
                        side = "home"
                    elif side == str(game.get("away_team", "")).strip().casefold():
                        side = "away"
                    else:
                        continue
                    kind = ("spread_" if family == "spreads" else "moneyline_") + side
                else:
                    continue
                quotes.append({"book": name, "market_type": kind, "point": outcome.get("point"),
                               "price": outcome.get("price"), "recorded_at": market.get("last_update") or book.get("last_update")})
    return json.dumps(quotes, sort_keys=True)


def bind_quote(row):
    """A timestamp is usable only for the exact offered side, line, price and book."""
    from core.selector_validation import timestamp

    kind = str(row.get("market_type", ""))
    price = pd.to_numeric(row.get("odds_american"), errors="coerce")
    line = pd.to_numeric(row.get("total_line" if kind.startswith("total") else "spread_line"), errors="coerce")
    preferred = str(row.get("opposing_odds_source", ""))
    try:
        quotes = json.loads(row.get("provider_quotes") or "[]")
    except (TypeError, ValueError):
        quotes = []
    matches = []
    for quote in quotes:
        qprice = pd.to_numeric(quote.get("price"), errors="coerce")
        qline = pd.to_numeric(quote.get("point"), errors="coerce")
        same_line = kind.startswith("moneyline") or (pd.notna(line) and pd.notna(qline) and abs(line - qline) < 1e-8)
        if quote.get("market_type") == kind and pd.notna(price) and pd.notna(qprice) and price == qprice and same_line:
            if preferred in {"novig", "fanduel", "draftkings", "betmgm"} and quote.get("book") != preferred:
                continue
            matches.append(quote)
    # Without a known book, multiple matching sources are ambiguous.
    if len(matches) != 1 or pd.isna(timestamp(matches[0].get("recorded_at"))):
        return {"odds_recorded_at": "", "quote_bookmaker": "", "quote_binding_verified": False}
    return {"odds_recorded_at": timestamp(matches[0]["recorded_at"]).isoformat(),
            "quote_bookmaker": matches[0]["book"], "quote_binding_verified": True}


def capture_run(context, audit, final, inputs, *, path=None):
    """Commit inputs, candidates and the final guarded card in one transaction."""
    from core.selector_validation import join_final_selections, text_column, timestamp

    if artifact_manifest(context["controls"], context["root"]) != context["manifest"]:
        raise ValueError("Model/configuration artifacts changed during analysis; run analysis again")
    if audit is None or audit.empty or final is None or final.empty:
        raise ValueError("Cannot capture an empty candidate audit or final card")
    audit, final = audit.copy(), final.copy()
    generated = now_utc()
    run_id = pd.Timestamp(generated).strftime("%Y%m%dT%H%M%S.%fZ")
    audit["export_run_id"], final["export_run_id"] = run_id, run_id
    # Resolve the final card to the audit via existing strict export-identity joins.
    approved = join_final_selections(audit, final)["_approved"]
    audit["wager_approved"] = approved
    if "matchup_id" not in final:
        final["matchup_id"] = ""
    from core.team_mapper import normalize_team_name

    def key(row):
        return tuple(str(row.get(c, "")).strip().casefold() if c not in {"home_team", "away_team"}
                     else normalize_team_name(str(row.get(c, "")))
                     for c in ("league", "home_team", "away_team", "market_type", "best_pick"))

    selected = audit[text_column(audit, "best_available_selected").str.lower().isin(["true", "1"])]
    for idx, row in final.iterrows():
        matches = selected[selected.apply(key, axis=1).map(lambda value: value == key(row))]
        # Include day to avoid joining separate slates of the same matchup.
        day = str(row.get("game_date", ""))[:10]
        matches = matches[text_column(matches, "game_date").str[:10].eq(day)]
        if len(matches) != 1:
            raise ValueError("Final pick does not identify exactly one saved candidate")
        ai = matches.index[0]
        final.at[idx, "matchup_id"] = audit.at[ai, "matchup_id"]
        for column in ("calibrated_probability", "odds_american", "odds_source", "spread_line", "total_line",
                       "Kelly_Bet_Size", "wager_approved", "Pick_Status", "Status_Reason", "qualification_reason",
                       "gemini_approved", "gemini_flags", "production_gate_reason"):
            if column in final:
                audit.at[ai, column] = row[column]

    for frame in (audit, final):
        frame["snapshot_id"] = context["snapshot_id"]
        frame["prediction_generated_at"] = generated
        frame["model_version"] = context["model_version"]
        frame["model_trained_through"] = context["frozen_at"]
        frame["model_available_at"] = context["frozen_at"]
        frame["training_cutoff_basis"] = "frozen_artifact_information_upper_bound"
    for idx, row in audit.iterrows():
        for col, value in bind_quote(row).items():
            audit.at[idx, col] = value
        kind = str(row.get("market_type", ""))
        line = pd.to_numeric(row.get("total_line" if kind.startswith("total") else "spread_line"), errors="coerce")
        no_push = kind.startswith("moneyline") or (pd.notna(line) and abs(line % 1 - .5) < 1e-8)
        audit.at[idx, "probability_semantics"] = "win_conditional_on_decision" if no_push else "push_semantics_unverified"
        if pd.isna(timestamp(row.get("game_start_utc"))):
            try:
                start = pd.to_datetime(str(row.get("game_time_est", "")).replace(" ET", ""), format="%Y-%m-%d %I:%M %p")
                audit.at[idx, "game_start_utc"] = start.tz_localize("America/New_York", ambiguous="raise", nonexistent="raise").tz_convert("UTC").isoformat()
            except (ValueError, TypeError):
                audit.at[idx, "game_start_utc"] = ""
    for idx, row in final.iterrows():
        matched = audit[audit.matchup_id.eq(row.matchup_id) & audit.market_type.eq(row.market_type) & audit.best_pick.eq(row.best_pick)]
        if len(matched) != 1:
            raise ValueError("Ambiguous candidate identity in final card")
        for col in PROVENANCE_COLUMNS:
            if col in matched:
                final.at[idx, col] = matched.iloc[0][col]
    # No outcomes may enter the original prediction record.
    forbidden = [c for c in audit if c.startswith("actual_") or c in {"candidate_outcome", "candidate_graded", "candidate_ledger_key"}]
    audit = audit.drop(columns=forbidden, errors="ignore")
    payload = [audit.to_csv(index=False), final.to_csv(index=False), inputs.to_csv(index=False)]
    digest = hashlib.sha256("\0".join(payload).encode()).hexdigest()
    with closing(connect(path)) as db, db:
        existing = db.execute("SELECT payload_hash FROM snapshots WHERE snapshot_id=?", (context["snapshot_id"],)).fetchone()
        if existing:
            raise ValueError("This run already has an immutable snapshot; start a new run")
        db.execute("INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (context["snapshot_id"], context["model_version"], generated, *payload, digest))
        db.execute("INSERT INTO snapshot_runtime VALUES (?, ?)", (context["snapshot_id"], PROCESS_INSTANCE))
    return audit, final


def load_snapshots(path=None):
    if not Path(path or database_path()).exists():
        return []
    with closing(connect(path)) as db:
        rows = db.execute("SELECT snapshot_id, candidates, decisions, inputs, payload_hash FROM snapshots ORDER BY generated_at").fetchall()
    output = []
    for sid, candidates, decisions, inputs, expected_hash in rows:
        actual_hash = hashlib.sha256("\0".join([candidates, decisions, inputs]).encode()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Saved prediction payload failed its integrity check")
        output.append((sid, pd.read_csv(StringIO(candidates)), pd.read_csv(StringIO(decisions))))
    return output


def record_scores(scored, *, path=None):
    """Join by snapshot id + matchup id, never by a team pair or date alone."""
    required = {"snapshot_id", "matchup_id", "actual_home_score", "actual_away_score"}
    if scored is None or scored.empty or not required.issubset(scored):
        return 0
    if not Path(path or database_path()).exists():
        return 0
    saved = 0
    with closing(connect(path)) as db, db:
        for sid, group in scored.groupby("snapshot_id"):
            original = db.execute("SELECT candidates FROM snapshots WHERE snapshot_id=?", (str(sid),)).fetchone()
            if original is None:
                continue
            ids = set(pd.read_csv(StringIO(original[0])).matchup_id)
            scores = group[["matchup_id", "actual_home_score", "actual_away_score"]].copy()
            for column in ("actual_home_score", "actual_away_score"):
                scores[column] = pd.to_numeric(scores[column], errors="coerce")
            scores = scores.dropna().drop_duplicates()
            import math
            for column in ("actual_home_score", "actual_away_score"):
                scores = scores[scores[column].map(lambda value: math.isfinite(value) and value >= 0 and float(value).is_integer())]
            scores = scores[scores.matchup_id.isin(ids)]
            if scores.duplicated("matchup_id").any():
                raise ValueError("Conflicting final scores in one results batch")
            if scores.empty:
                continue
            raw = scores.sort_values("matchup_id").to_csv(index=False)
            digest = hashlib.sha256(raw.encode()).hexdigest()
            saved += db.execute("INSERT OR IGNORE INTO score_revisions VALUES (?, ?, ?, ?)",
                                (str(sid), digest, now_utc(), raw)).rowcount
    return saved


def materialize(path=None):
    """Return report inputs; score revisions never alter original predictions."""
    from app_core.candidate_recap import _grade_candidate

    audits, finals = [], []
    for sid, audit, final in load_snapshots(path):
        with closing(connect(path)) as db:
            revisions = db.execute("SELECT scores FROM score_revisions WHERE snapshot_id=? ORDER BY rowid", (sid,)).fetchall()
        if revisions:
            scores = pd.concat([pd.read_csv(StringIO(raw)) for (raw,) in revisions], ignore_index=True)
            scores = scores.drop_duplicates("matchup_id", keep="last")
            audit = audit.merge(scores, on="matchup_id", how="left", validate="many_to_one")
        else:
            audit["actual_home_score"] = pd.NA
            audit["actual_away_score"] = pd.NA
        audit["candidate_outcome"] = audit.apply(_grade_candidate, axis=1)
        audits.append(audit)
        finals.append(final)
    return (pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(),
            pd.concat(finals, ignore_index=True) if finals else pd.DataFrame())


def write_validation_reports(path=None):
    """Refresh derived reports after grading, separately for each frozen bundle."""
    from core.selector_validation import build_report, render_markdown

    audits, finals = materialize(path)
    if audits.empty:
        return []
    reports_dir = Path(path or database_path()).parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with closing(connect(path)) as db:
        bundles = db.execute("SELECT version, frozen_at, manifest FROM bundles ORDER BY frozen_at").fetchall()
    outputs = []
    for version, frozen, manifest in bundles:
        candidates = audits[audits.model_version.eq(version)]
        if candidates.empty:
            continue
        decisions = finals[finals.model_version.eq(version)]
        cutoff = pd.Timestamp(frozen).tz_convert("America/New_York").strftime("%Y-%m-%d")
        config = {"report_version": "1", "train_through": cutoff, "probability_column": "calibrated_probability",
                  "market_column": "market_probability", "snapshot_policy": "latest_export_before_start",
                  "slate_timezone": "America/New_York"}
        spec = {"configuration": config, "frozen_at": frozen, "model_version": version}
        frozen_manifest = json.loads(manifest)
        current_evaluator = hashlib.sha256((ROOT / "core/selector_validation.py").read_bytes()).hexdigest()
        evaluator_matches = frozen_manifest["artifacts"].get("core/selector_validation.py") == current_evaluator
        report = build_report(candidates, train_through=cutoff, selections=decisions,
                              specification=spec if evaluator_matches else None)
        if not evaluator_matches:
            report["evidence"]["note"] = "Evaluator code differs from, or is absent in, the frozen manifest; this is descriptive historical evaluation."
        with closing(connect(path)) as db:
            hashes = db.execute("SELECT snapshot_id,payload_hash FROM snapshots WHERE version=? ORDER BY snapshot_id", (version,)).fetchall()
            revisions = db.execute("SELECT r.snapshot_id,r.evidence_hash FROM score_revisions r JOIN snapshots s USING(snapshot_id) WHERE s.version=? ORDER BY r.rowid", (version,)).fetchall()
        report["reproducibility"] = {"frozen_manifest": json.loads(manifest), "snapshot_hashes": hashes, "score_revision_hashes": revisions}
        base = reports_dir / version
        base.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        base.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
        outputs.append(str(base.with_suffix(".md")))
    return outputs
