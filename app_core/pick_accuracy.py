"""Read-only paired ranking and source diagnostics on provenance-gated history."""
from __future__ import annotations

import pandas as pd
from core.selector_validation import build_report, number_column, text_column, _summary
from core.walk_forward import probability_metrics
from app_core.gemini_review_comparison import review_comparison

SOURCES = {
    "Independent model": "blend_in_ml",
    "AI Analysis / TheOver": "blend_in_theover",
    "Kalshi": "blend_in_kalshi",
}


def _choose(pool, column):
    # Exact ties use immutable candidate identity, never outcome or existing rank.
    return pool.sort_values([column, "market_type", "best_pick"],
        ascending=[False, True, True], kind="stable").drop_duplicates("_event")


def _paired(current, alternative):
    a = current.set_index("_event")
    b = alternative.set_index("_event").loc[a.index]
    decided = a.candidate_outcome.isin(["WIN", "LOSS"]) & b.candidate_outcome.isin(["WIN", "LOSS"])
    aw, bw = a.candidate_outcome.eq("WIN"), b.candidate_outcome.eq("WIN")
    return {"games": len(a), "changed_picks": int((a.market_type.ne(b.market_type) | a.best_pick.ne(b.best_pick)).sum()),
            "both_decided": int(decided.sum()),
            "current_only_wins": int((decided & aw & ~bw).sum()),
            "alternative_only_wins": int((decided & ~aw & bw).sum()),
            "paired_win_rate_change": float((bw[decided].astype(int)-aw[decided].astype(int)).mean()) if decided.any() else None}


def _source_metrics(rows, probability):
    keep = rows.candidate_outcome.isin(["WIN", "LOSS"]) & probability.notna()
    return probability_metrics(probability.loc[keep], rows.loc[keep, "candidate_outcome"].eq("WIN").astype(int))


def build_accuracy_report(audits, *, evaluation_start):
    """Compare fixed choices; no fitting, ranking mutation, or automatic promotion.

    Reuse selector validation for complete pools, declared training cutoffs,
    exact timestamps/price semantics, duplicate conflicts and pregame snapshots.
    Probability-first uses calibrated_probability, never the composite score.
    Source scoring uses the original selected ticket, with oriented inputs only.
    """
    start = pd.Timestamp(evaluation_start)
    if start.tzinfo is not None or start != start.normalize():
        raise ValueError("Evaluation start must be a calendar date")
    validation, pool = build_report(audits, train_through=(start-pd.Timedelta(days=1)).strftime("%Y-%m-%d"), return_eligible=True)
    current = pool.loc[pool._selected].copy()
    cards = {"Current ranking": (current, "_probability"),
             "Probability first": (_choose(pool, "_probability"), "_probability"),
             "Sportsbook baseline": (_choose(pool, "_market"), "_market")}
    ranking = {name: _summary(card, len(current), probability) for name, (card, probability) in cards.items()}
    paired = {name: _paired(current, card) for name, (card, _) in cards.items() if name != "Current ranking"}
    by_league = []
    for league in sorted(pool.league.unique()):
        events = set(current.loc[current.league.eq(league), "_event"])
        by_league.append({"league": league, "rankings": {name: _summary(card.loc[card._event.isin(events)], len(events), p)
                          for name, (card, p) in cards.items()}})
    selected = current.copy()
    probabilities = {}
    for label, column in SOURCES.items():
        p = number_column(selected, column)
        # The historical pipeline uses 0/1 as source-missing sentinels. Do not
        # invent a 50% source or infer which team/market a raw signal meant.
        p = p.where(p.gt(0) & p.lt(1))
        p = p.where(text_column(selected, "probability_semantics").eq("win_conditional_on_decision"))
        if label == "Independent model":
            target = text_column(selected, "ml_target")
            market = text_column(selected, "market_type")
            valid = (market.str.startswith("spread_") & target.eq("spread_cover")) | (market.str.startswith("total_") & target.eq(market))
            p = p.where(valid & text_column(selected, "ml_probability_source").ne(""))
        probabilities[label] = p
    source_comparisons = []
    segments = [("All", "All", selected)] + [(str(league), str(market), rows)
        for (league, market), rows in selected.groupby(["league", "_family"], sort=True)]
    for league, market, rows in segments:
        for label, values in probabilities.items():
            p = values.loc[rows.index]
            paired_rows = rows.loc[p.notna()]
            source = _source_metrics(paired_rows, p.loc[paired_rows.index])
            baseline = _source_metrics(paired_rows, paired_rows._market)
            source_comparisons.append({"league": league, "market": market, "source": label,
                "selected_games": len(rows), "available_games": len(paired_rows),
                "missing_or_unverified_games": len(rows)-len(paired_rows),
                "source_metrics": source, "sportsbook_on_same_tickets": baseline,
                "brier_improvement": baseline["brier"]-source["brier"] if source["n"] else None,
                "log_loss_improvement": baseline["log_loss"]-source["log_loss"] if source["n"] else None})
    common_mask = pd.Series(True, index=selected.index)
    for values in probabilities.values():
        common_mask &= values.notna()
    common = selected.loc[common_mask]
    common_sources = {label: _source_metrics(common, p.loc[common.index]) for label, p in probabilities.items()}
    common_sources["Sportsbook baseline"] = _source_metrics(common, common._market)
    choices = []
    indexed = {name: card.set_index("_event") for name, (card, _) in cards.items()}
    for _, row in current.iterrows():
        record = {"date": row._day, "league": row.league, "matchup_id": row.matchup_id,
                  "export_run_id": row.export_run_id}
        for name, card in indexed.items():
            chosen = card.loc[row._event]
            record[name] = {"pick": chosen.best_pick, "market": chosen.market_type,
                            "odds": float(chosen.odds_american), "outcome": chosen.candidate_outcome}
        choices.append(record)
    gemini = review_comparison(selected)
    return {"version": 1, "status": "historical_comparison" if len(current) else "insufficient_verified_evidence",
            "evaluation_start": start.strftime("%Y-%m-%d"), "live_changes": False,
            "validation": validation, "rankings": ranking, "paired_ranking_changes": paired,
            "rankings_by_league": by_league, "sources": source_comparisons,
            "all_sources_common_games": len(common), "all_sources_common_metrics": common_sources,
            "gemini_review": gemini.astype(object).where(pd.notna(gemini), None).to_dict("records"),
            "choices": choices,
            "limitations": [
                "Historical diagnostics only; no independently verified out-of-sample improvement or automatic promotion.",
                "Current ranking is the saved selection from each run, not today's code rerun on old games.",
                "Probability-first uses the saved calibrated estimate. Source comparisons score identical original selected tickets.",
                "Paired source cohorts can differ; compare each source with its own baseline, or use the all-source common cohort.",
                "Source-alone scores do not measure incremental value in the live blend. Missing inputs are never imputed.",
                "Gemini is qualitative review, not a probability. Its agreement subset is descriptive, not a causal test.",
                "Return estimates risk one unit at the original odds; exclude fees and fills. Pushes are excluded from win rates, not turnover.",
                "Complete-pool exclusions can bias coverage; all supplied snapshots and rejection counts remain in validation."]}


def render_accuracy_markdown(report):
    def fmt(value, percent=False):
        if value is None: return "Unavailable"
        return f"{value:.1%}" if percent else f"{value:.4f}"
    inv = report["validation"]["inventory"]
    lines = ["# Pick accuracy comparison", "", f"Status: **{report['status']}**", "",
             f"Evaluation starts {report['evaluation_start']} (Eastern). Verified games: {inv['eligible_events']}.", "",
             "Live ranking and weights are unchanged. This is a historical comparison, not a validated improvement.", "",
             "| Ranking | Games | Wins | Losses | Pushes | Win rate | Simulated ROI | Brier | Log loss |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in report["rankings"].items():
        lines.append(f"| {name} | {values['games']} | {values['wins']} | {values['losses']} | {values['pushes']} | {fmt(values['hit_rate'],True)} | {fmt(values['flat_roi'],True)} | {fmt(values['brier'])} | {fmt(values['log_loss'])} |")
    lines += ["", "## Source comparison on identical original selected tickets", "",
              "Lower Brier and log loss are better. Sources have different availability; each baseline uses exactly the same tickets.", "",
              "| Source | Available games | Decisions | Source Brier | Baseline Brier | Source log loss | Baseline log loss |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in report["sources"]:
        if row["league"] != "All": continue
        s, b = row["source_metrics"], row["sportsbook_on_same_tickets"]
        lines.append(f"| {row['source']} | {row['available_games']} | {s['n']} | {fmt(s['brier'])} | {fmt(b['brier'])} | {fmt(s['log_loss'])} | {fmt(b['log_loss'])} |")
    lines += ["", f"Games with all probability sources: {report['all_sources_common_games']}.",
              "Gemini is reported separately as qualitative review in the JSON report.", "", "## Exclusions", "", "```json"]
    import json
    lines += [json.dumps(report["validation"]["exclusions"], indent=2), "```", "", "## Limits", ""]
    lines += ["- "+s for s in report["limitations"]]
    return "\n".join(lines)+"\n"
