"""Diagnostic football coverage only; no inferred model or wager authority."""
from collections import Counter


def summarize(report):
    result = {"schema": "football-coverage-v1", "wager_authority": False, "sports": {}}
    for sport in ("NFL", "NCAAF"):
        games = [g for g in report.get("games", []) if g.get("league") == sport]
        if not games:
            continue
        candidates = report.get("candidates", [])
        counts = Counter(g.get("production_probability") for g in games
                         if g.get("production_probability") is not None)
        rows = []
        for game in games:
            pool = [c for c in candidates if c.get("matchup_id") == game.get("matchup_id")
                    and c.get("snapshot_id") == game.get("snapshot_id")]
            selected = [c for c in pool if c.get("selected") is True]
            candidate = selected[0] if len(selected) == 1 else {}
            reasons = []
            if game.get("independent_model_probability") is None:
                reasons.append("independent_model_unavailable")
            if "model_provenance_missing" in game.get("evidence_blockers", []):
                reasons.append("model_provenance_missing")
            if not game.get("verified_quote_candidates"):
                reasons.append("no_verified_candidate_quotes")
            if len(selected) != 1:
                reasons.append("selected_candidate_unresolved")
            home, away = candidate.get("home_classification"), candidate.get("away_classification")
            cohort = "NFL" if sport == "NFL" else (
                home if home == away and home in {"FBS", "FCS"} else
                "FBS/FCS" if {home, away} == {"FBS", "FCS"} else "UNKNOWN")
            rows.append({"matchup": game.get("matchup"), "cohort": cohort,
                         "pick": game.get("selected_pick"), "reasons": reasons,
                         "probability_basis": candidate.get("selection_probability_source") or "NOT_RECORDED",
                         "model_unavailable_reason": candidate.get("ml_unavailable_reason") or None,
                         "status": "LIMITED_EVIDENCE" if reasons else "REQUIRES_VALIDATION"})
        result["sports"][sport] = {"games": len(games), "rows": rows,
            "cohorts": dict(Counter(r["cohort"] for r in rows)),
            "repeated_probabilities": [{"probability": p, "games": n} for p,n in counts.items() if n > 1],
            "note": "Repeated values are diagnostic, not proof of error. TheOver is optional. No training or wagering authorized."}
    return result
