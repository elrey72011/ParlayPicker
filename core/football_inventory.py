"""Read-only football evidence inventory. Never grants training or wager authority."""
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from core.wager_decisions import aware
from core.exposure_ledger import digest


def build(rows, now=None):
    clock = now or datetime.now(timezone.utc)
    sports = {}
    for sport in ("NFL", "NCAAF"):
        source = [r for r in rows if r.get("sport", r.get("league")) == sport]
        exclusions, groups = Counter(), defaultdict(list)
        for r in source:
            if str(r.get("selected_as_best_pick")).lower() not in {"true", "1"}:
                exclusions["not_saved_selection"] += 1
                continue
            pred, saved, start = (aware(r.get(k)) for k in
                                 ("prediction_generated_at", "capture_recorded_at", "game_start_utc"))
            if not pred or not saved or not start or not pred <= saved < start or saved > clock:
                exclusions["unverified_or_late_capture"] += 1
                continue
            event = r.get("matchup_id")
            if not event:
                exclusions["missing_event_identity"] += 1
                continue
            groups[event].append(r)
        records = []
        for event, group in groups.items():
            # Select first capture without consulting outcomes; never select a later winner.
            earliest = min(aware(r["capture_recorded_at"]) for r in group)
            first = {digest(r): r for r in group if aware(r["capture_recorded_at"]) == earliest}
            if len(first) != 1:
                exclusions["conflicting_first_capture"] += 1
                continue
            r = next(iter(first.values()))
            gaps = []
            quote = aware(r.get("odds_recorded_at"))
            if str(r.get("quote_binding_verified")).lower() not in {"true", "1"} or not quote or quote > aware(r["prediction_generated_at"]):
                gaps.append("unverified_original_quote")
            if r.get("market_type") not in {"spread_home", "spread_away", "total_over", "total_under"}:
                gaps.append("unsupported_market")
            outcome_at = aware(r.get("outcome_recorded_at"))
            settled = (r.get("candidate_outcome") in {"WIN", "LOSS", "PUSH"}
                       and r.get("result_source") in {"ESPN", "MLB"}
                       and bool(r.get("result_provider_event_id")) and outcome_at is not None
                       and aware(r["game_start_utc"]) <= outcome_at <= clock)
            home, away = r.get("home_classification"), r.get("away_classification")
            cohort = "NFL" if sport == "NFL" else (home if home == away and home in {"FBS", "FCS"}
                else "FBS/FCS" if {home, away} == {"FBS", "FCS"} else "UNKNOWN")
            missing = [k for k in ("model_version", "model_trained_through", "calibration_version",
                                  "home_team_id", "away_team_id", "features_generated_at") if not r.get(k)]
            records.append({"matchup_id": event, "snapshot_id": r.get("snapshot_id"),
                "capture_recorded_at": r["capture_recorded_at"], "cohort": cohort,
                "settlement_verified": bool(settled), "research_gaps": gaps,
                "missing_model_inputs": missing,
                "settled_research_record": bool(settled and not gaps)})
        if source:
            sports[sport] = {"source_rows": len(source), "unique_pregame_selections": len(records),
                "settled_research_records": sum(r["settled_research_record"] for r in records),
                "cohorts": dict(Counter(r["cohort"] for r in records)),
                "missing_model_inputs": dict(Counter(k for r in records for k in r["missing_model_inputs"])),
                "exclusions": dict(exclusions), "records": records}
    return {"schema": "football-inventory-v1", "sports": sports,
            "training_authorized": False, "wager_authority": False,
            "selection_policy": "First recorded pregame best pick per sport and matchup; conflicting first captures excluded.",
            "limitations": "Research inventory only. Does not validate event mapping, calibration, historical feature availability, split sizes, or model quality."}


def rebuild(database):
    from app_core.prediction_evidence import materialize, connect
    from app_core.candidate_evidence_schema import evidence_value
    frame, _ = materialize(database)
    with closing(connect(database)) as db:
        captured = dict(db.execute("SELECT snapshot_id, generated_at FROM snapshots"))
    rows = []
    for value in frame.to_dict("records"):
        row = evidence_value(value)
        row["capture_recorded_at"] = captured.get(row.get("snapshot_id"))
        rows.append(row)
    return build(rows)
