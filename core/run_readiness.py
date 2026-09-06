"""Read-only diagnostics for supplied prediction runs; never approve a wager."""
from collections import Counter
import math

import pandas as pd

from core.selector_validation import timestamp
from core.probability_semantics import conditional_probabilities


def text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def flag(value):
    return {"true": True, "1": True, "false": False, "0": False}.get(text(value).lower())


def first(row, *columns):
    for column in columns:
        if text(row.get(column)):
            return row[column]
    return None


def probability(value):
    value = number(value)
    return value if value is not None and 0 <= value <= 1 else None


def build_readiness(audit, final=None, *, quote_warning_minutes=15, diagnostics=None):
    """One diagnostic row per supplied snapshot/game, including failed games.

    Quote age is measured at capture, not at report opening. The warning limit
    is diagnostic only, not a new production policy. Outcomes are not consulted.
    """
    if not math.isfinite(quote_warning_minutes) or quote_warning_minutes <= 0:
        raise ValueError("Quote warning minutes must be positive and finite")
    audit = pd.DataFrame() if audit is None else audit.copy()
    audit = audit.drop(columns=[c for c in audit if c.startswith("actual_") or c in {"candidate_outcome", "candidate_graded"}], errors="ignore").drop_duplicates()
    final = pd.DataFrame() if final is None else final.copy().drop_duplicates()
    report = {"version": 1, "quote_warning_minutes": quote_warning_minutes,
              "production_changes": False, "games": [], "candidates": [], "run_warnings": []}
    diagnostics = diagnostics or {}
    for key in ("stale_base_schedule", "prediction_snapshot_error", "run_health_warning"):
        value = diagnostics.get(key)
        if isinstance(value, (str, bool)) and value:
            report["run_warnings"].append(f"{key}: {value}")
    if audit.empty:
        report.update(status="no_candidate_evidence", counts={"games": 0, "ready_for_grading": 0, "approved_wagers": 0})
        return report
    identity = ["snapshot_id", "export_run_id", "matchup_id"]
    for column in identity:
        if column not in audit:
            audit[column] = ""
        audit[column] = audit[column].map(text)
    # Missing IDs must not collapse unrelated rows into a convincing game pool.
    audit["_group"] = [tuple(row[c] for c in identity) if all(row[c] for c in identity)
                       else ("unknown", str(i), "") for i, row in audit.iterrows()]
    for _, pool in audit.groupby("_group", sort=False):
        head = pool.iloc[0]
        sid, run, game = (text(head.get(c)) for c in identity)
        blocks, warnings, details = set(), set(), []
        selectors = pool.get("best_available_selected", pd.Series(None, index=pool.index)).map(flag)
        selected = pool.loc[selectors.eq(True)]
        if selectors.isna().any() or len(selected) != 1:
            blocks.add("ambiguous_candidate_selection")
        counts = pool.get("best_available_candidate_count", pd.Series(None, index=pool.index)).map(number)
        if counts.isna().any() or not counts.eq(len(pool)).all():
            blocks.add("candidate_pool_incomplete")
        candidate_keys = pool.apply(lambda row: (text(row.get("market_type")), text(row.get("best_pick"))), axis=1)
        if candidate_keys.duplicated().any():
            blocks.add("conflicting_candidate_records")
        for column in ("league", "home_team", "away_team", "game_start_utc", "prediction_generated_at", "model_version"):
            if pool.get(column, pd.Series(None, index=pool.index)).map(text).nunique() != 1:
                blocks.add("inconsistent_game_metadata")
        selected_row = selected.iloc[0] if len(selected) == 1 else pd.Series(dtype=object)
        match = final
        for column in identity + ["market_type", "best_pick"]:
            wanted = text(selected_row.get(column))
            if not wanted or column not in match:
                match = match.iloc[0:0]
                break
            match = match.loc[match[column].map(text).eq(wanted)]
        card = match.iloc[0] if len(match) == 1 else pd.Series(dtype=object)
        if len(match) != 1:
            blocks.add("final_decision_missing_or_ambiguous")
        displayed = probability(first(card, "WinProbability", "calibrated_probability"))
        original_probability = probability(selected_row.get("calibrated_probability"))
        if displayed is not None and original_probability is not None and abs(displayed - original_probability) > 1e-9:
            blocks.add("final_probability_mismatch")
        final_price, audit_price = number(card.get("odds_american")), number(selected_row.get("odds_american"))
        if final_price is not None and audit_price is not None and final_price != audit_price:
            blocks.add("final_price_mismatch")
        approved = flag(card.get("wager_approved"))
        bettable = flag(card.get("Bettable"))
        stake = number(first(card, "Play_Stake", "production_bet_amount", "Kelly_Bet_Size"))
        if (approved is not None and bettable is not None and approved != bettable) or (approved is True and stake is not None and stake <= 0):
            approved = None
            blocks.add("conflicting_wager_approval")
        if approved is None:
            warnings.add("wager_approval_unknown")
        for _, row in pool.iterrows():
            issues = []
            generated, start, quote, trained, available = (timestamp(row.get(c)) for c in
                ("prediction_generated_at", "game_start_utc", "odds_recorded_at", "model_trained_through", "model_available_at"))
            if not all(text(row.get(c)) for c in identity + ["league", "home_team", "away_team", "market_type", "best_pick"]):
                issues.append("missing_identity")
            if pd.isna(generated) or pd.isna(start):
                issues.append("prediction_or_start_time_unverified")
            elif generated >= start:
                issues.append("not_pregame_at_capture")
            exported = pd.to_datetime(text(row.get("export_run_id")), format="%Y%m%dT%H%M%S.%fZ", errors="coerce", utc=True)
            if pd.isna(exported):
                exported = pd.to_datetime(text(row.get("export_run_id")), format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True)
            if pd.isna(exported) or pd.isna(generated) or pd.isna(start) or not generated <= exported < start:
                issues.append("export_timing_unverified")
            if not text(row.get("model_version")) or pd.isna(trained) or pd.isna(available) or pd.isna(generated):
                issues.append("model_provenance_missing")
            elif not trained <= available <= generated or not trained < generated:
                issues.append("model_provenance_timing_invalid")
            exact = flag(row.get("quote_binding_verified")) is True
            if not exact or pd.isna(quote):
                issues.append("quote_binding_unverified")
            age = (generated - quote).total_seconds() / 60 if pd.notna(generated) and pd.notna(quote) else None
            if age is not None and age < 0:
                issues.append("quote_after_prediction")
            if age is not None and age > quote_warning_minutes:
                warnings.add("quote_age_above_diagnostic_limit")
            source = text(row.get("odds_source"))
            price = number(row.get("odds_american"))
            if price is None or abs(price) < 100 or not source or any(s in source.lower() for s in ("fallback", "synthetic", "inferred", "default", "unpriced")):
                issues.append("price_or_source_unverified")
            if probability(row.get("calibrated_probability")) is None or probability(row.get("market_probability")) is None:
                issues.append("probability_missing_or_invalid")
            if conditional_probabilities(row) is None:
                issues.append("probability_semantics_unverified")
            rejected_line = (flag(row.get("final_line_rejected")) is True
                             or text(row.get("market_line_source")).startswith("rejected")
                             or "line unresolved" in text(row.get("best_pick")).lower())
            if rejected_line:
                issues.append("final_line_rejected")
            kind = text(row.get("market_type"))
            line = number(row.get("total_line" if kind.startswith("total") else "spread_line"))
            push_possible = line is not None and line.is_integer() and kind.startswith(("spread", "total"))
            detail = {"snapshot_id": sid, "export_run_id": run, "matchup_id": game,
                      "pick": text(row.get("best_pick")), "market": text(row.get("market_type")),
                      "selected": flag(row.get("best_available_selected")), "quote_verified": exact,
                      "line_eligible": not rejected_line, "quoted_line": line,
                      "settlement_rule": "push_on_equal" if push_possible else "no_push" if line is not None else "unresolved",
                      "probability_semantics": text(row.get("probability_semantics")),
                      "quote_age_minutes_at_capture": age, "odds_source": source,
                      "independent_model_probability": probability(row.get("ml_probability")),
                      "theover_probability": probability(row.get("theover_probability")),
                      "issues": sorted(set(issues))}
            blocks.update(issues)
            details.append(detail)
        model_p = probability(selected_row.get("ml_probability"))
        if model_p is None:
            warnings.add("selected_independent_model_probability_unavailable")
        feature_time = timestamp(first(card, "features_generated_at", "stats_updated_at"))
        if pd.isna(feature_time):
            warnings.add("feature_freshness_unavailable")
        warnings.update(text(card.get(c)) for c in ("degraded_feature_subset_reason", "line_provenance_warning") if text(card.get(c)))
        gate_reasons = list(dict.fromkeys(text(card.get(c)) for c in (
            "Production_Gate_Reason", "production_gate_reason", "status_blocker_reason", "Status_Reason", "qualification_reason", "kelly_zero_reason") if text(card.get(c))))
        trained = timestamp(head.get("model_trained_through"))
        start = timestamp(head.get("game_start_utc"))
        first_day = (trained.tz_convert("America/New_York").normalize() + pd.DateOffset(days=1)).date().isoformat() if pd.notna(trained) else None
        eligible_day = start.tz_convert("America/New_York").date().isoformat() >= first_day if pd.notna(start) and first_day else None
        if eligible_day is False:
            warnings.add("capture_on_or_before_model_freeze_day")
        report["games"].append({"snapshot_id": sid, "export_run_id": run, "matchup_id": game,
            "league": text(head.get("league")), "matchup": f"{text(head.get('away_team'))} at {text(head.get('home_team'))}",
            "game_start_utc": text(head.get("game_start_utc")), "prediction_generated_at": text(head.get("prediction_generated_at")),
            "selected_pick": text(selected_row.get("best_pick")), "candidate_count": len(pool),
            "verified_quote_candidates": sum(d["quote_verified"] for d in details),
            "readiness": "ready_for_grading" if not blocks else "blocked",
            "wager_decision": "approved" if approved is True else "pass" if approved is False else "unknown",
            "displayed_probability": probability(first(card, "WinProbability", "calibrated_probability")),
            "production_probability": probability(card.get("production_win_probability")),
            "independent_model_probability": model_p, "market_probability": probability(selected_row.get("market_probability")),
            "feature_timestamp": feature_time.isoformat() if pd.notna(feature_time) else None,
            "earliest_evaluation_slate": first_day, "after_freeze_day": eligible_day,
            "evidence_blockers": sorted(blocks), "data_warnings": sorted(warnings), "wager_reasons": gate_reasons})
        report["candidates"].extend(details)
    report.update(status="reported", counts={"games": len(report["games"]),
        "ready_for_grading": sum(r["readiness"] == "ready_for_grading" for r in report["games"]),
        "approved_wagers": sum(r["wager_decision"] == "approved" for r in report["games"]),
        "blockers_by_game": dict(Counter(c for r in report["games"] for c in r["evidence_blockers"]))})
    return report


def game_table(report):
    rows = []
    for row in report["games"]:
        item = dict(row)
        for key in ("evidence_blockers", "data_warnings", "wager_reasons"):
            item[key] = "; ".join(item[key])
        rows.append(item)
    return pd.DataFrame(rows)


def render_readiness(report):
    def escape(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Run Readiness Report", "", "Read-only diagnostics. Ready for grading does not mean approved to bet or proven accurate.",
             "Formal validation still checks outcomes, snapshot selection, provenance and the evaluation cutoff.",
             "A verified quote may still have a rejected final line. Integer lines can push; scoring requires explicit conditional probabilities or recorded model and market push probabilities.", "",
             f"Quote age warning: {report['quote_warning_minutes']} minutes at capture; diagnostic only.",
             "Feature freshness is unknown unless a source timestamp was exported. Missing model inputs remain unavailable.", "",
             "| Game | Pick | Evidence | Wager | Display probability | Production probability | Blockers / wager reasons |",
             "|---|---|---|---|---|---|---|"]
    for r in report["games"]:
        values = [r["matchup"], r["selected_pick"], r["readiness"], r["wager_decision"],
                  r["displayed_probability"], r["production_probability"], "; ".join(r["evidence_blockers"] + r["wager_reasons"])]
        lines.append("| " + " | ".join(escape(v) if v is not None else "Unavailable" for v in values) + " |")
    lines += ["", "## Run warnings", ""] + ["- " + escape(v) for v in report["run_warnings"]]
    lines += ["", "Candidate-level issues, source coverage and evaluation-day eligibility are included in the JSON download."]
    return "\n".join(lines) + "\n"
