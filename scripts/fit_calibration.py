#!/usr/bin/env python3
"""
Fit an isotonic (PAV) calibration table mapping effective_win_probability to the
REALIZED win rate, from the graded slates in data/backtest_exports/.

Motivation (Jun 5-10 recaps): effective_win_probability is overconfident in the
0.55-0.65 band — the band that feeds Actionable/HV promotion and parlay legs.
Dozens of hand-tuned shrink/penalty knobs in weights_config.py have been patching
this symptom one slate at a time; this fits the mapping once, from all graded data,
and writes a non-activated candidate artifact for independent review.

Usage
-----
    python3 scripts/fit_calibration.py [exports_dir] [out_json]

Defaults: data/backtest_exports -> data/calibration/candidates/effective_prob_calibration_candidate.json
Re-run after each graded slate is added (scripts/grade_slate.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.probability_calibration import (  # noqa: E402
    CALIBRATION_SCHEMA_VERSION,
    CONDITIONAL_FIT_TARGET,
    CONDITIONAL_PROBABILITY_SEMANTICS,
    DEFAULT_CALIBRATION_PATH,
    FITTING_IMPLEMENTATION_VERSION,
    PER_CANDIDATE_PUSH_CONVERSION,
    apply_calibration,
    fit_isotonic_calibration,
    save_calibration,
    calibration_digest,
)
from core.walk_forward import chronological_split, probability_metrics  # noqa: E402

PROB_COLS = ["effective_win_probability", "WinProbability"]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_PATH = Path(
    "data/calibration/candidates/effective_prob_calibration_candidate.json"
)


def _source_label(path: Path) -> str:
    """Store portable repo-relative provenance instead of a user's absolute path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _to_prob(v: object) -> float:
    """Accept 0.61, '0.61', or '61.0%'."""
    s = str(v).strip().rstrip("%")
    try:
        p = float(s)
    except ValueError:
        return float("nan")
    if "%" in str(v) or p > 1.0:
        p /= 100.0
    return p


def _extract(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.strip(): c for c in df.columns}
    prob_col = next((cols[c] for c in PROB_COLS if c in cols), None)
    wl_col = cols.get("W/L")
    columns = [
        "prob", "outcome_class", "win", "fit_included", "exclusion_reason",
        "slate_date", "outcome_available_at", "sample_weight", "source_record_id",
        "canonical_event_id", "exact_sport", "exact_market_family",
        "source_predictor_version", "probability_semantics",
    ]
    if prob_col is None or wl_col is None:
        return pd.DataFrame(columns=columns)

    def first(names: tuple[str, ...], default=None):
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series(default, index=df.index, dtype="object")

    raw_outcomes = df[wl_col].astype(str).str.strip().str.upper()
    aliases = {
        "W": "WIN", "WIN": "WIN", "L": "LOSS", "LOSS": "LOSS",
        "P": "PUSH", "PUSH": "PUSH", "VOID": "VOID",
    }
    outcome_class = raw_outcomes.map(aliases).fillna("UNKNOWN")
    probabilities = df[prob_col].map(_to_prob)
    weights = pd.to_numeric(first(("sample_weight", "weight"), 1.0), errors="coerce")
    valid_probability = probabilities.notna() & probabilities.between(0.0, 1.0)
    valid_weight = weights.notna() & weights.gt(0.0)
    decided = outcome_class.isin(["WIN", "LOSS"])
    included = valid_probability & valid_weight & decided
    reason = pd.Series("INCLUDED", index=df.index, dtype="object")
    reason.loc[~valid_probability] = "PROBABILITY_INVALID"
    reason.loc[valid_probability & ~valid_weight] = "SAMPLE_WEIGHT_INVALID"
    reason.loc[valid_probability & valid_weight & ~decided] = (
        "PUSH_EXCLUDED_FROM_CONDITIONAL_BINARY_FIT"
    )
    raw_market = first(("exact_market_family", "market_family", "market", "market_type"))

    def market_family(value: object) -> str | None:
        token = str(value or "").strip().upper()
        if token.startswith("TOTAL_") or token in {"OVER", "UNDER", "TOTAL"}:
            return "TOTAL"
        if token.startswith("SPREAD_") or token == "SPREAD":
            return "SPREAD"
        if token in {"RUN_LINE", "PUCK_LINE"}:
            return token
        return token or None

    out = pd.DataFrame({
        "prob": probabilities,
        "outcome_class": outcome_class,
        "win": outcome_class.map({"WIN": 1.0, "LOSS": 0.0}),
        "fit_included": included,
        "exclusion_reason": reason,
        "sample_weight": weights,
        "source_record_id": first(
            ("source_record_id", "recommendation_id", "candidate_id", "quote_id", "id")
        ),
        "canonical_event_id": first(("canonical_event_id", "event_id", "matchup_id")),
        "exact_sport": first(("exact_sport", "sport", "league")),
        "exact_market_family": raw_market.map(market_family),
        "source_predictor_version": first(
            ("source_predictor_version", "predictor_version", "model_version", "ensemble_version")
        ),
        "probability_semantics": first(("probability_semantics",)),
    })
    date_col = next((c for c in ("game_date", "event_date", "date") if c in df.columns), None)
    out["slate_date"] = (
        pd.to_datetime(df[date_col], errors="coerce", utc=True)
        if date_col else pd.NaT
    )
    available_col = next(
        (
            c for c in (
                "outcome_available_at", "result_available_at", "settled_at", "graded_at"
            ) if c in df.columns
        ),
        None,
    )
    out["outcome_available_at"] = (
        pd.to_datetime(df[available_col], errors="coerce", utc=True)
        if available_col else pd.NaT
    )
    return out[columns]


# The May .txt slates are hand-pasted spreadsheet dumps: all rows on ONE line, with
# three column-order variants. Two invariants hold across all of them:
#   * effective_win_probability is the field immediately before the consensus token
#     (effective_expected_value, effective_edge, effective_win_probability, consensus)
#   * the graded W/L cell (" W " / " L ") appears after that consensus token and
#     before the NEXT row's consensus token
# so we pair each consensus match with the first W/L cell that follows it. Summary
# rows ("9-8") and ungraded rows pair nothing and drop out naturally.
_CONSENSUS_RE = re.compile(r"([0-9]*\.?[0-9]+%?),(?:Agrees|Neutral|Disagrees|No Kalshi),")
_WL_CELL_RE = re.compile(r",\s([WL])\s,")


def _extract_txt(text: str) -> pd.DataFrame:
    rows = []
    matches = list(_CONSENSUS_RE.finditer(text))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        wl = _WL_CELL_RE.search(text, m.end(), end)
        if wl:
            rows.append({
                "prob": _to_prob(m.group(1)),
                "outcome_class": "WIN" if wl.group(1) == "W" else "LOSS",
                "win": int(wl.group(1) == "W"),
                "fit_included": True,
                "exclusion_reason": "INCLUDED",
                "sample_weight": 1.0,
            })
    return pd.DataFrame(rows)


def load_graded(exports_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(exports_dir.glob("*.csv")) + sorted(exports_dir.glob("*.txt")):
        try:
            raw = f.read_bytes()
            source_hash = hashlib.sha256(raw).hexdigest()
            if f.suffix == ".txt":
                got = _extract_txt(raw.decode("utf-8"))
            else:
                got = _extract(pd.read_csv(f))
        except Exception as e:  # hand-pasted slates; skip unparseable ones
            print(f"  [skip] {f.name}: {e}", file=sys.stderr)
            continue
        if not got.empty:
            got = got.copy()
            got["source_path"] = _source_label(f)
            got["source_hash"] = source_hash
            if "source_record_id" not in got:
                got["source_record_id"] = None
            generated_ids = [f"{_source_label(f)}#{index}" for index in range(len(got))]
            got["source_record_id"] = got["source_record_id"].where(
                got["source_record_id"].notna() & got["source_record_id"].astype(str).str.strip().ne(""),
                pd.Series(generated_ids, index=got.index),
            )
            for column in (
                "slate_date", "outcome_available_at", "canonical_event_id", "exact_sport",
                "exact_market_family", "source_predictor_version", "probability_semantics",
            ):
                if column not in got:
                    got[column] = pd.NaT if column.endswith("_at") or column == "slate_date" else None
            frames.append(got)
            print(f"  {f.name}: {int(got['fit_included'].sum())} fitted / {len(got)} outcome rows")
    if not frames:
        raise SystemExit(f"no graded picks found under {exports_dir}")
    return pd.concat(frames, ignore_index=True)


def build_fit_manifest(graded: pd.DataFrame) -> dict:
    """Describe every observed row and exactly which rows the fit consumed."""
    rows: list[dict[str, object]] = []
    for index, row in graded.reset_index(drop=True).iterrows():
        def clean(value):
            if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
                return None
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            if hasattr(value, "item"):
                value = value.item()
            return value

        record = {
            "manifest_row": index,
            "source_record_id": clean(row.get("source_record_id")),
            "source_path": clean(row.get("source_path")),
            "source_hash": clean(row.get("source_hash")),
            "canonical_event_id": clean(row.get("canonical_event_id")),
            "exact_sport": str(clean(row.get("exact_sport")) or "").upper() or None,
            "exact_market_family": str(clean(row.get("exact_market_family")) or "").upper() or None,
            "source_predictor_version": clean(row.get("source_predictor_version")),
            "input_probability": clean(row.get("prob")),
            "probability_semantics": clean(row.get("probability_semantics")),
            "outcome_class": clean(row.get("outcome_class")),
            "event_time": clean(row.get("slate_date")),
            "outcome_available_at": clean(row.get("outcome_available_at")),
            "sample_weight": clean(row.get("sample_weight")),
            "fit_included": bool(row.get("fit_included", False)),
            "exclusion_reason": clean(row.get("exclusion_reason")),
        }
        rows.append(record)
    included = [row for row in rows if row["fit_included"]]
    predictors = sorted({str(row["source_predictor_version"]) for row in included if row["source_predictor_version"]})
    scopes = sorted({
        (str(row["exact_sport"]), str(row["exact_market_family"]))
        for row in included if row["exact_sport"] and row["exact_market_family"]
    })
    events = {str(row["canonical_event_id"]) for row in included if row["canonical_event_id"]}
    manifest: dict[str, object] = {
        "schema_version": 1,
        "fit_target": CONDITIONAL_FIT_TARGET,
        "probability_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
        "source_predictor_version": predictors[0] if len(predictors) == 1 else None,
        "training_scope": (
            {"exact_sport": scopes[0][0], "exact_market_family": scopes[0][1]}
            if len(scopes) == 1 else None
        ),
        "observation_count": len(rows),
        "fitted_row_count": len(included),
        "independent_event_count": len(events),
        "rows": rows,
    }
    manifest["manifest_hash"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def report(graded: pd.DataFrame) -> None:
    bins = pd.cut(graded["prob"], [0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0])
    by_bin = graded.groupby(bins, observed=True).agg(n=("win", "size"), predicted=("prob", "mean"), realized=("win", "mean"))
    print("\nPredicted vs realized by bin:")
    print(by_bin.to_string(float_format=lambda v: f"{v:.3f}"))


def validate_calibration_promotion(
    graded: pd.DataFrame,
    *,
    test_fraction: float = 0.20,
    min_train_rows: int = 100,
) -> dict:
    """Evaluate a train-only isotonic map on a strictly future holdout."""
    required = {"prob", "win", "slate_date"}
    if not required.issubset(graded.columns) or graded["slate_date"].isna().any():
        return {"promotable": False, "reason": "dated graded rows are required"}
    if len(graded) <= int(min_train_rows):
        return {
            "promotable": False,
            "reason": (
                f"not enough training rows: need more than {int(min_train_rows)}, "
                f"got {len(graded)}"
            ),
        }
    try:
        train, test = chronological_split(
            graded,
            "slate_date",
            test_fraction=test_fraction,
            min_train_rows=min_train_rows,
        )
    except ValueError as exc:
        return {"promotable": False, "reason": str(exc)}
    knots = fit_isotonic_calibration(train["prob"].tolist(), train["win"].tolist())
    calibrated = apply_calibration(test["prob"], knots)
    base_rate = pd.Series(float(train["win"].mean()), index=test.index)
    metrics = {
        "calibrated": probability_metrics(calibrated, test["win"]),
        "raw": probability_metrics(test["prob"], test["win"]),
        "train_base_rate": probability_metrics(base_rate, test["win"]),
    }
    promotable = all(
        metrics["calibrated"][metric]
        < min(metrics["raw"][metric], metrics["train_base_rate"][metric])
        for metric in ("brier", "log_loss")
    )
    return {
        "promotable": bool(promotable),
        "reason": (
            "beats raw probability and train base rate on the future holdout"
            if promotable
            else "does not beat raw probability and train base rate on the future holdout"
        ),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_end": str(pd.to_datetime(train["slate_date"], utc=True).max()),
        "test_start": str(pd.to_datetime(test["slate_date"], utc=True).min()),
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exports_dir", nargs="?", default="data/backtest_exports")
    parser.add_argument("out_json", nargs="?", default=str(DEFAULT_CANDIDATE_PATH))
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Write a RESEARCH_CANDIDATE_ONLY artifact to a non-production path when "
            "chronological validation fails. This never grants production authority."
        ),
    )
    parser.add_argument(
        "--source-predictor-version",
        help="Immutable source predictor or ensemble version used by every fitted row.",
    )
    parser.add_argument(
        "--training-scope",
        help="Human-readable training scope description (not authority by itself).",
    )
    parser.add_argument("--exact-sport", help="Exact sport expected in every fitted source record.")
    parser.add_argument(
        "--exact-market-family",
        help="Exact market family expected in every fitted source record.",
    )
    args = parser.parse_args(argv)
    exports_dir = Path(args.exports_dir)
    out_json = Path(args.out_json)
    default_live_path = (ROOT / DEFAULT_CALIBRATION_PATH).resolve()
    requested_output = out_json.resolve()
    if requested_output == default_live_path:
        print(
            "ACTIVATION BLOCKED: fitting may only write a candidate artifact; "
            "the live calibration path is immutable in this command",
            file=sys.stderr,
        )
        return 2

    graded = load_graded(exports_dir)
    fitted = graded.loc[graded["fit_included"].fillna(False).astype(bool)].copy()
    if len(fitted) < 2:
        print("PROMOTION BLOCKED: fewer than two valid decided observations", file=sys.stderr)
        return 2
    validation = validate_calibration_promotion(fitted)
    print("\nChronological promotion validation:")
    for name, metrics in validation.get("metrics", {}).items():
        print(
            f"  {name:<16} log_loss={metrics['log_loss']:.4f} "
            f"brier={metrics['brier']:.4f} n={metrics['n']}"
        )
    if not validation["promotable"] and not args.force:
        print(f"\nPROMOTION BLOCKED: {validation['reason']}", file=sys.stderr)
        return 2

    knots = fit_isotonic_calibration(
        fitted["prob"].tolist(),
        fitted["win"].astype(int).tolist(),
        sample_weights=fitted["sample_weight"].tolist(),
    )
    # The final refit consumes every supported endpoint, including 0 and 1. Its
    # cutoff is therefore derived from the same rows passed to the fitter.
    dates = pd.to_datetime(fitted.get("slate_date"), errors="coerce", utc=True)
    cutoff = dates.max().isoformat() if dates is not None and not dates.isna().any() else None
    availability = pd.to_datetime(
        fitted.get("outcome_available_at"), errors="coerce", utc=True
    )
    availability_cutoff = (
        availability.max().isoformat()
        if availability is not None and not availability.isna().any() else None
    )
    manifest = build_fit_manifest(graded)
    expected_scope = (
        {
            "exact_sport": str(args.exact_sport).strip().upper(),
            "exact_market_family": str(args.exact_market_family).strip().upper(),
        }
        if args.exact_sport and args.exact_market_family else None
    )
    candidate_reasons = []
    if not validation["promotable"]:
        candidate_reasons.append("CHRONOLOGICAL_VALIDATION_NOT_PROMOTABLE")
    if not args.source_predictor_version:
        candidate_reasons.append("SOURCE_PREDICTOR_VERSION_NOT_RECORDED")
    elif manifest["source_predictor_version"] != args.source_predictor_version:
        candidate_reasons.append("SOURCE_PREDICTOR_VERSION_MISMATCH")
    if expected_scope is None:
        candidate_reasons.append("EXACT_TRAINING_SCOPE_NOT_RECORDED")
    elif manifest["training_scope"] != expected_scope:
        candidate_reasons.append("EXACT_TRAINING_SCOPE_MISMATCH")
    if cutoff is None:
        candidate_reasons.append("TRAINING_CUTOFF_UNAVAILABLE")
    if availability_cutoff is None:
        candidate_reasons.append("OUTCOME_AVAILABILITY_CUTOFF_UNAVAILABLE")
    if int(manifest["independent_event_count"]) == 0:
        candidate_reasons.append("INDEPENDENT_EVENT_IDENTITIES_UNAVAILABLE")
    declared_semantics = {
        str(value).strip()
        for value in fitted["probability_semantics"].dropna().tolist()
        if str(value).strip()
    }
    if declared_semantics != {CONDITIONAL_PROBABILITY_SEMANTICS}:
        candidate_reasons.append("SOURCE_PROBABILITY_SEMANTICS_UNVERIFIED")
    meta = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_method": "isotonic",
        "fitting_implementation_version": FITTING_IMPLEMENTATION_VERSION,
        "artifact_status": "RESEARCH_CANDIDATE_ONLY" if candidate_reasons else "PRODUCTION_CANDIDATE",
        "candidate_rejection_reasons": candidate_reasons,
        "n_graded": int(len(fitted)),
        "n_outcome_rows": int(len(graded)),
        "source": _source_label(exports_dir),
        "fitted_on": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "prob_col": "effective_win_probability (fallback WinProbability)",
        "probability_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
        "fit_target": CONDITIONAL_FIT_TARGET,
        "output_semantics": CONDITIONAL_PROBABILITY_SEMANTICS,
        "push_conversion": PER_CANDIDATE_PUSH_CONVERSION,
        "source_predictor_version": args.source_predictor_version,
        "training_scope": expected_scope,
        "training_scope_description": args.training_scope,
        "fit_manifest": manifest,
        "validation": validation,
        "calibration_trained_through": cutoff,
        "outcome_available_through": availability_cutoff,
        "calibration_available_at": None,
    }
    # Persist an explicitly non-authorizing draft before recording availability.
    draft_meta = dict(meta, artifact_status="PERSISTING_RESEARCH_DRAFT")
    save_calibration(knots, out_json, meta=draft_meta)
    meta["calibration_available_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    meta["calibration_version"] = calibration_digest({"knots": knots, "meta": meta})
    save_calibration(knots, out_json, meta=meta)
    print(
        f"\nfit on {len(fitted)} decided rows ({len(graded)} total outcomes) -> "
        f"{len(knots)} knots -> {out_json} "
        f"[{meta['artifact_status']}; NOT ACTIVATED]"
    )
    report(fitted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
