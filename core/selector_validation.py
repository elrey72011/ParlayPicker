"""Read-only, provenance-gated evaluation of exported one-pick-per-game cards.

No fitting, live API calls, or production configuration changes occur here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from core.team_mapper import normalize_team_name
from core.probability_semantics import conditional_probabilities


REPORT_VERSION = "1"
KEY = ["export_run_id", "matchup_id", "market_type", "best_pick"]
REQUIRED = KEY + ["league", "home_team", "away_team", "candidate_outcome",
                  "best_available_selected", "odds_american"]


def text_column(frame, name):
    return frame.get(name, pd.Series("", index=frame.index)).astype("string").fillna("").str.strip()


def number_column(frame, name):
    return pd.to_numeric(text_column(frame, name), errors="coerce").astype(float)


def bool_column(frame, name):
    return text_column(frame, name).str.lower().map({"true": True, "1": True, "false": False, "0": False}).astype("boolean")


def timestamp(value):
    """Only explicit, timezone-aware timestamps are admissible evidence."""
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is None:
            return pd.NaT
        return result.tz_convert("UTC")
    except (TypeError, ValueError):
        return pd.NaT


def time_column(frame, name):
    return pd.to_datetime(text_column(frame, name).map(timestamp), utc=True)


def read_inputs(paths):
    """Hash the exact bytes read, so a report identifies its source evidence."""
    from io import BytesIO

    frames, inventory = [], []
    for path in sorted(set(map(str, paths))):
        raw = Path(path).read_bytes()
        frame = pd.read_csv(BytesIO(raw), low_memory=False)
        inventory.append({"file": Path(path).name, "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(frame)})
        frames.append(frame)
    if not frames:
        raise ValueError("No input CSV files matched")
    return pd.concat(frames, ignore_index=True), inventory


def join_final_selections(audits, selections):
    """Audit wager_approved is a backtest placeholder; require the final export.

    Never join by team names alone or silently resolve contradictory downloads.
    """
    out = audits.copy()
    out["_approved"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    if selections is None:
        return out
    selections = selections.copy()
    if "matchup_id" not in selections or text_column(selections, "matchup_id").eq("").any():
        # Compact UI exports use display headers. Recover identity only from an
        # unambiguous exact run/date/league/canonical-team/market/pick match.
        aliases = {"home_team": "Home", "away_team": "Away", "game_date": "Local Date"}
        for canonical, alias in aliases.items():
            selections[canonical] = text_column(selections, canonical).replace("", pd.NA).fillna(text_column(selections, alias))
        identity = ["export_run_id", "league", "home_team", "away_team", "game_date", "market_type", "best_pick"]
        missing_identity = set(identity) - set(audits)
        # A supplied UTC start is also sufficient for the Eastern calendar date.
        source = audits.copy()
        if "game_date" in missing_identity and "game_start_utc" in source:
            source["game_date"] = time_column(source, "game_start_utc").dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            missing_identity.remove("game_date")
        if missing_identity:
            raise ValueError(f"Cannot join compact final exports; audits missing: {sorted(missing_identity)}")

        def join_identity(frame):
            parts = []
            for column in identity:
                values = text_column(frame, column)
                if column in {"home_team", "away_team"}:
                    values = values.map(normalize_team_name)
                elif column == "game_date":
                    values = pd.to_datetime(values, errors="coerce", format="mixed", utc=True).dt.strftime("%Y-%m-%d").fillna("")
                parts.append(values)
            valid = pd.concat(parts, axis=1).ne("").all(axis=1)
            keys = pd.Series(list(zip(*parts)), index=frame.index)
            return keys.where(valid)

        mapping = pd.DataFrame({"_join": join_identity(source), "matchup_id": text_column(source, "matchup_id")}).dropna().drop_duplicates()
        ambiguous = mapping.duplicated("_join", keep=False)
        mapping = mapping[~ambiguous].set_index("_join")["matchup_id"]
        recovered = join_identity(selections).map(mapping)
        selections["matchup_id"] = text_column(selections, "matchup_id").replace("", pd.NA).fillna(recovered)
    missing = set(KEY + ["wager_approved"]) - set(selections)
    if missing:
        raise ValueError(f"Final selections missing columns: {sorted(missing)}")
    final = selections[KEY].copy()
    final["_approved"] = bool_column(selections, "wager_approved")
    final = final.dropna(subset=KEY)
    final = final.drop_duplicates()
    if final.duplicated(KEY).any():
        raise ValueError("Conflicting final wager approvals for the same run/candidate")
    out = out.drop(columns="_approved").merge(final, on=KEY, how="left", validate="many_to_one")
    out["_approved"] = out["_approved"].astype("boolean")
    return out


def _summary(rows, denominator, probability_column):
    outcomes = rows["candidate_outcome"]
    wins, losses, pushes = [int(outcomes.eq(value).sum()) for value in ("WIN", "LOSS", "PUSH")]
    decided = outcomes.isin(["WIN", "LOSS"])
    # Returns are equal-unit counterfactuals at recorded prices, not account P&L.
    decimal = rows["_decimal"]
    profit = pd.Series(np.where(outcomes.eq("WIN"), decimal - 1,
                               np.where(outcomes.eq("LOSS"), -1.0, 0.0)), index=rows.index)
    probabilities = rows[probability_column]
    y = outcomes.loc[decided].eq("WIN").astype(float)
    p = probabilities.loc[decided]
    clipped = p.clip(1e-6, 1 - 1e-6)
    bins = []
    bucket_ids = (p * 10).astype(int).clip(upper=9)
    for bucket in range(10):
        lower = bucket / 10
        mask = bucket_ids.eq(bucket)
        if mask.any():
            bins.append({"lower": round(float(lower), 1), "upper": round(float(lower + .1), 1),
                         "n": int(mask.sum()), "predicted": float(p[mask].mean()), "observed": float(y[mask].mean())})
    # Sum a slate before calculating drawdown; intraday row order is arbitrary.
    cumulative = profit.groupby(rows["_day"]).sum().sort_index().cumsum()
    peak = cumulative.cummax().clip(lower=0)
    return {
        "games": len(rows), "eligible_games": int(denominator),
        "coverage": len(rows) / denominator if denominator else None,
        "slates": int(rows["_day"].nunique()), "wins": wins, "losses": losses, "pushes": pushes,
        "decided": wins + losses, "hit_rate": wins / (wins + losses) if wins + losses else None,
        "flat_stake_units": len(rows), "flat_profit_units": float(profit.sum()) if len(rows) else None,
        "flat_roi": float(profit.mean()) if len(rows) else None,
        "max_slate_drawdown_units": float((peak - cumulative).max()) if len(rows) else None,
        "probability_n": len(p), "brier": float(((p - y) ** 2).mean()) if len(p) else None,
        "log_loss": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()) if len(p) else None,
        "calibration": bins,
    }


def _comparison(pool):
    selected = pool[pool["_selected"]].copy()
    # Fixed deterministic tie break uses candidate identity, never outcomes or model rank.
    baseline = pool.sort_values(["_market", "market_type", "best_pick"], ascending=[False, True, True],
                                kind="stable").drop_duplicates("_event")
    baseline = baseline.set_index("_event", drop=False)
    scopes = {"all_selected": selected,
              "qualified_wagers": selected[selected["_approved"].eq(True).fillna(False)],
              "pass_picks": selected[selected["_approved"].eq(False).fillna(False)],
              "approval_unknown": selected[selected["_approved"].isna()]}
    result = {}
    for name, card in scopes.items():
        paired_market = baseline.loc[card["_event"]].copy()
        # Market comparator is hypothetical, including within the qualified-game subset.
        result[name] = {
            "selector": _summary(card, len(selected), "_probability"),
            "market_only": _summary(paired_market, len(selected), "_market"),
            "by_league_market": [],
        }
        for (league, market), group in card.groupby(["league", "_family"], sort=True):
            same_games = baseline.loc[group["_event"]].copy()
            denominator = pool.loc[pool.league.eq(league) & pool._family.eq(market), "_event"].nunique()
            result[name]["by_league_market"].append({
                "league": league, "market": market,
                "selector": _summary(group, denominator, "_probability"),
                "market_only_on_same_games": _summary(same_games, denominator, "_market"),
            })
    return result


def build_report(audits, *, train_through, probability_column="calibrated_probability",
                 market_column="market_probability", selections=None, specification=None, return_eligible=False):
    """Evaluate only complete, pregame snapshots with declared model provenance.

    Development/evaluation separation is by Eastern calendar slate. A model's
    trained-through timestamp covers all fitted components, including calibration
    and selector tuning. Input declarations are checked, not independently attested.
    """
    missing = set(REQUIRED) - set(audits)
    if missing:
        raise ValueError(f"Candidate audits missing columns: {sorted(missing)}")
    if probability_column in {"selection_probability_used", "best_available_score", "final_family_score", "tier_score"}:
        raise ValueError("Ranking scores cannot be evaluated as calibrated probabilities")
    cutoff = pd.Timestamp(train_through)
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize():
        raise ValueError("train_through must be a calendar date")
    evaluation_start = (cutoff + pd.Timedelta(days=1)).tz_localize("America/New_York").tz_convert("UTC")
    f = audits.copy().drop_duplicates().reset_index(drop=True)
    raw_rows = len(audits)
    # Candidate identity conflicts are excluded at snapshot level, not resolved by file order.
    conflict = f.duplicated(KEY, keep=False)
    conflict_keys = set(map(tuple, f.loc[conflict, ["export_run_id", "matchup_id"]].to_numpy()))
    f = join_final_selections(f, selections)
    f["_start"] = time_column(f, "game_start_utc")
    legacy_start = pd.to_datetime(text_column(f, "game_time_est").str.replace(" ET", "", regex=False),
                                  format="%Y-%m-%d %I:%M %p", errors="coerce")
    legacy_start = legacy_start.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    f["_start"] = f["_start"].fillna(legacy_start)
    f["_day"] = f["_start"].dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    f["_prediction"] = time_column(f, "prediction_generated_at")
    f["_export"] = pd.to_datetime(text_column(f, "export_run_id"), format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)
    f["_export"] = f["_export"].fillna(pd.to_datetime(text_column(f, "export_run_id"), format="%Y%m%dT%H%M%S.%fZ", errors="coerce", utc=True))
    f["_trained"] = time_column(f, "model_trained_through")
    f["_available"] = time_column(f, "model_available_at")
    f["_odds_time"] = time_column(f, "odds_recorded_at")
    f["_probability"] = number_column(f, probability_column)
    f["_market"] = number_column(f, market_column)
    conditional = [conditional_probabilities(row, probability_column, market_column) for _, row in f.iterrows()]
    f["_semantics_verified"] = [value is not None for value in conditional]
    f["_probability"] = [value[0] if value is not None else raw for value, raw in zip(conditional, f._probability)]
    f["_market"] = [value[1] if value is not None else raw for value, raw in zip(conditional, f._market)]
    f["_selected"] = bool_column(f, "best_available_selected").fillna(False).astype(bool)
    f["_family"] = text_column(f, "market_type").str.split("_").str[0].replace({"h2h": "moneyline"})
    f["candidate_outcome"] = text_column(f, "candidate_outcome").str.upper()
    price = number_column(f, "odds_american")
    f["_decimal"] = np.where(price.ge(100), 1 + price / 100, 1 + 100 / price.abs())
    home = text_column(f, "home_team").map(normalize_team_name)
    away = text_column(f, "away_team").map(normalize_team_name)
    # Start time distinguishes doubleheaders; canonical team aliases collapse duplicate feeds.
    f["_event"] = text_column(f, "league").str.upper() + "|" + home + "|" + away + "|" + f["_start"].astype(str)
    f["_snapshot"] = f["_event"] + "|" + text_column(f, "export_run_id")
    f["_reason"] = ""

    def reject(mask, reason):
        f.loc[mask.fillna(True) & f._reason.eq(""), "_reason"] = reason

    reject(pd.Series([any(not str(row[c]).strip() or pd.isna(row[c]) for c in KEY + ["league", "home_team", "away_team"])
                      for _, row in f.iterrows()], index=f.index, dtype=bool), "missing_identity")
    reject(pd.Series([(r, m) in conflict_keys for r, m in zip(f.export_run_id, f.matchup_id)], index=f.index, dtype=bool), "conflicting_candidate_downloads")
    reject(f._start.isna(), "missing_or_ambiguous_game_start")
    reject(f._export.isna() | ~f._export.lt(f._start), "export_not_verified_pregame")
    # Snapshot selection precedes probability/grade/provenance checks. Never cherry-pick
    # an older snapshot because the latest pregame snapshot has missing or losing rows.
    timing_good = f._reason.eq("")
    latest = f.loc[timing_good].groupby("_event")._export.max()
    reject(timing_good & f._export.ne(f._event.map(latest)), "superseded_pregame_snapshot")
    development = f._day.le(cutoff.strftime("%Y-%m-%d")).fillna(False)
    reject(development, "development_slate")
    reject(f._prediction.isna() | ~f._prediction.lt(f._start) | ~f._prediction.le(f._export), "missing_or_invalid_prediction_timestamp")
    reject(text_column(f, "model_version").eq(""), "missing_model_version")
    reject(f._trained.isna() | ~f._trained.lt(evaluation_start) | ~f._trained.lt(f._prediction), "training_cutoff_unverified_or_leaking")
    reject(f._available.isna() | ~f._trained.le(f._available) | ~f._available.le(f._prediction), "model_availability_unverified")
    reject(bool_column(f, "final_line_rejected").fillna(False), "final_line_rejected")
    reject(~f._probability.between(0, 1) | ~f._market.between(0, 1), "missing_or_invalid_probability")
    reject(~f._semantics_verified, "probability_semantics_unverified")
    reject(f._odds_time.isna() | ~f._odds_time.le(f._prediction), "odds_timestamp_unverified")
    source = text_column(f, "odds_source").str.lower()
    reject(~price.abs().ge(100) | ~np.isfinite(price) | source.eq("") |
           source.str.contains("synthetic|inferred|default|unpriced|fallback", regex=True), "price_or_source_unverified")
    reject(~f.candidate_outcome.isin(["WIN", "LOSS", "PUSH"]), "unsettled_or_invalid_outcome")
    # One selected candidate and one source identity per canonical snapshot. Alias
    # collisions are excluded rather than silently counting the same event twice.
    selected_counts = f.groupby("_snapshot")._selected.transform("sum")
    identity_counts = f.groupby("_snapshot").matchup_id.transform("nunique")
    reject(bool_column(f, "best_available_selected").isna(), "invalid_selection_flag")
    reject(selected_counts.ne(1) | identity_counts.ne(1), "ambiguous_snapshot_selection_or_identity")
    expected = number_column(f, "best_available_candidate_count")
    observed = f.groupby("_snapshot")._snapshot.transform("size")
    reject(expected.isna() | expected.ne(observed), "candidate_pool_completeness_unverified")
    failed_snapshots = set(f.loc[f._reason.ne(""), "_snapshot"])
    reject(f._snapshot.isin(failed_snapshots), "incomplete_verified_candidate_pool")
    eligible = f[f._reason.eq("")].copy()
    configuration = {"report_version": REPORT_VERSION, "train_through": cutoff.strftime("%Y-%m-%d"),
                     "probability_column": probability_column, "market_column": market_column,
                     "snapshot_policy": "latest_export_before_start", "slate_timezone": "America/New_York"}
    preregistered = False
    spec_reason = "No frozen specification supplied; historical evaluation only."
    if specification is not None:
        frozen = timestamp(specification.get("frozen_at"))
        config_matches = specification.get("configuration") == configuration
        version = specification.get("model_version", "")
        preregistered = bool(len(eligible) and config_matches and pd.notna(frozen) and
                             frozen < evaluation_start and frozen < eligible._prediction.min() and
                             version and text_column(eligible, "model_version").eq(version).all())
        spec_reason = ("Frozen specification predates evaluation and matches all eligible model versions."
                       if preregistered else "Specification does not establish a matching freeze before evaluation/prediction.")
    exclusions = [{"reason": reason, "rows": len(group), "events": int(group._event.nunique())}
                  for reason, group in f[f._reason.ne("")].groupby("_reason", sort=True)]
    report = {
        "configuration": configuration,
        "status": "evaluated" if len(eligible) else "insufficient_verified_data",
        "evidence": {"preregistered": preregistered, "note": spec_reason,
                     "model_versions": sorted(text_column(eligible, "model_version").unique().tolist()),
                     "out_of_sample_independently_verified": False},
        "inventory": {"raw_rows": raw_rows, "duplicate_rows_removed": raw_rows - len(audits.drop_duplicates()),
                      "development_events": int(f.loc[development, "_event"].nunique()),
                      "evaluation_events_seen": int(f.loc[~development & f._start.notna(), "_event"].nunique()),
                      "eligible_events": int(eligible._event.nunique()), "eligible_candidates": len(eligible),
                      "evaluation_days": sorted(eligible._day.unique().tolist())},
        "missing_evidence_rows": {
            name: int(text_column(f, name).eq("").sum()) for name in
            ["prediction_generated_at", "model_version", "model_trained_through", "model_available_at",
             probability_column, "probability_semantics", "odds_recorded_at", "best_available_candidate_count"]
        },
        "exclusions": exclusions,
        "comparisons": _comparison(eligible),
        "limitations": [
            "Input timestamps and model metadata are declarations, not independent proof of leakage-free training.",
            "Existing ranking scores are never substituted for calibrated probabilities.",
            "Complete verified candidate pools only; exclusions can bias coverage and performance.",
            "Coverage denominators are supplied eligible games, not every scheduled game in the league.",
            "Approval is known only from an exact matching final-pick export; audit approval placeholders are ignored.",
            "Returns are one-unit counterfactuals at recorded prices, not actual stakes, fills, fees, or account profit.",
            "Pushes return zero units and count in ROI turnover; binary scoring uses decided outcomes and conditional probabilities.",
            "Market-only chooses the highest exported market probability on the same games; it is not a qualified betting strategy.",
            "No automatic model promotion or profitability conclusion; a freeze alone cannot prove data were never inspected.",
        ],
    }

    return (report, eligible) if return_eligible else report


def render_markdown(report):
    def fmt(value, percent=False):
        if value is None:
            return "Unavailable"
        return f"{value:.1%}" if percent else f"{value:.4f}"

    inv = report["inventory"]
    lines = ["# Selector Validation Report", "", f"Status: **{report['status']}**", "",
             f"Development cutoff: {report['configuration']['train_through']} (America/New_York slates).",
             f"Verified evaluation: **{inv['eligible_events']} games / {inv['eligible_candidates']} candidates**; "
             f"{inv['evaluation_events_seen']} evaluation events supplied.", "",
             report["evidence"]["note"], "",
             "## Paired comparison", "",
             "Market-only is evaluated on exactly the same games as each selector scope. Returns are flat one-unit simulations.", "",
             "| Scope | Selector | Games | Coverage | W–L–P | Hit rate | ROI | Brier | Log loss |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for scope, comparison in report["comparisons"].items():
        for name in ("selector", "market_only"):
            m = comparison[name]
            lines.append(f"| {scope} | {name} | {m['games']} | {fmt(m['coverage'], True)} | "
                         f"{m['wins']}–{m['losses']}–{m['pushes']} | {fmt(m['hit_rate'], True)} | "
                         f"{fmt(m['flat_roi'], True)} | {fmt(m['brier'])} | {fmt(m['log_loss'])} |")
    lines += ["", "## Sport and market breakdown", "",
              "Market baseline may select a different market on these same games. Segment coverage uses eligible games offering that market family.", "",
              "| Scope | League | Selected market | Games | Slates | Coverage | Selector hit rate | Market hit rate | Selector ROI | Market ROI |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for scope, comparison in report["comparisons"].items():
        for group in comparison["by_league_market"]:
            m, b = group["selector"], group["market_only_on_same_games"]
            lines.append(f"| {scope} | {group['league']} | {group['market']} | {m['games']} | {m['slates']} | "
                         f"{fmt(m['coverage'], True)} | {fmt(m['hit_rate'], True)} | {fmt(b['hit_rate'], True)} | "
                         f"{fmt(m['flat_roi'], True)} | {fmt(b['flat_roi'], True)} |")
    lines += ["", "## Calibration", "", "Probabilities mean win conditional on no push. Full subgroup metrics are in the JSON report.", "",
              "| Scope | Selector | Probability bin | Decisions | Mean prediction | Observed win rate |",
              "|---|---|---|---:|---:|---:|"]
    for scope, comparison in report["comparisons"].items():
        for name in ("selector", "market_only"):
            for bucket in comparison[name]["calibration"]:
                lines.append(f"| {scope} | {name} | {bucket['lower']:.1f}–{bucket['upper']:.1f} | "
                             f"{bucket['n']} | {fmt(bucket['predicted'], True)} | {fmt(bucket['observed'], True)} |")
    lines += ["", "## Exclusions", "", "Each row has one primary reason; event counts across reasons can overlap.", "",
              "| Reason | Rows | Events |", "|---|---:|---:|"]
    lines += [f"| {r['reason']} | {r['rows']} | {r['events']} |" for r in report["exclusions"]]
    lines += ["", "## Missing evidence", "", "Counts cover all unique supplied rows, including development slates.", "",
              "| Field | Rows without evidence |", "|---|---:|"]
    lines += [f"| {field} | {count} |" for field, count in report["missing_evidence_rows"].items() if count]
    lines += ["", "## Interpretation limits", ""] + [f"- {s}" for s in report["limitations"]]
    lines += ["", "## Reproducibility", "", "Input file hashes and the evaluator code hash are recorded in the companion JSON.", ""]
    return "\n".join(lines)
