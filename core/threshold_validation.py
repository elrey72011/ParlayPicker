"""Development-only threshold comparisons; never choose or promote a live rule."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from core.selector_validation import build_report, _summary

DEFAULT_THRESHOLDS = (.50, .55, .60, .65, .70, .75, .80, .85, .90)


def wilson_interval(wins, decisions):
    if decisions == 0:
        return None
    z = 1.959963984540054
    p = wins / decisions
    denominator = 1 + z * z / decisions
    center = (p + z * z / (2 * decisions)) / denominator
    half = z * math.sqrt(p * (1 - p) / decisions + z * z / (4 * decisions * decisions)) / denominator
    return [max(0., center - half), min(1., center + half)]


def slate_bootstrap(rows, repetitions=1000):
    """Resample whole slates to keep same-day wagers together; deterministic seed."""
    if rows._day.nunique() < 2:
        return {"hit_rate_95": None, "roi_95": None}
    work = pd.DataFrame({"day": rows._day,
                         "wins": rows.candidate_outcome.eq("WIN").astype(int),
                         "decisions": rows.candidate_outcome.isin(["WIN", "LOSS"]).astype(int),
                         "picks": 1,
                         "profit": np.where(rows.candidate_outcome.eq("WIN"), rows._decimal - 1,
                                            np.where(rows.candidate_outcome.eq("LOSS"), -1., 0.))})
    daily = work.groupby("day", sort=True)[["wins", "decisions", "picks", "profit"]].sum().to_numpy()
    rng = np.random.default_rng(20260906)
    # Batch memory remains bounded for long histories.
    totals = np.vstack([daily[rng.integers(0, len(daily), size=len(daily))].sum(axis=0)
                        for _ in range(repetitions)])
    decided = totals[:, 1] > 0
    hit_rate = totals[decided, 0] / totals[decided, 1]
    return {"hit_rate_95": np.quantile(hit_rate, [.025, .975]).tolist() if len(hit_rate) else None,
            "roi_95": np.quantile(totals[:, 3] / totals[:, 2], [.025, .975]).tolist()}


def compare_thresholds(audits, *, train_through, development_through, selections=None,
                       thresholds=DEFAULT_THRESHOLDS, scope="qualified_wagers", bootstrap_repetitions=1000):
    if scope not in {"qualified_wagers", "all_selected"}:
        raise ValueError("Scope must be qualified_wagers or all_selected")
    thresholds = sorted(set(float(t) for t in thresholds))
    if not thresholds or any(not math.isfinite(t) or t < 0 or t > 1 for t in thresholds):
        raise ValueError("Thresholds must be finite probabilities between 0 and 1")
    end = pd.Timestamp(development_through)
    start = pd.Timestamp(train_through)
    if end.tzinfo or start.tzinfo or end != end.normalize() or start != start.normalize() or end <= start:
        raise ValueError("Use calendar dates with development_through later than train_through")
    if bootstrap_repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    # Remove later slates before validation, deduplication or snapshot selection:
    # no future labels/metadata may affect any displayed development result.
    dates = pd.to_datetime(audits.get("game_start_utc", pd.Series(index=audits.index, dtype=str)), errors="coerce", utc=True)
    legacy = pd.to_datetime(audits.get("game_time_est", pd.Series(index=audits.index, dtype=str)).astype(str).str.replace(" ET", "", regex=False),
                            format="%Y-%m-%d %I:%M %p", errors="coerce")
    legacy = legacy.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    days = dates.fillna(legacy).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    supplied = audits.loc[days.le(end.strftime("%Y-%m-%d")).fillna(False)].copy()
    if selections is not None:
        # Exact run/game identity avoids pulling later, contradictory approvals
        # into the development join. Compact exports remain supported by build_report.
        selections = selections[selections.export_run_id.isin(supplied.export_run_id)].copy()
    verification, pool = build_report(supplied, train_through=train_through,
                                      selections=selections, return_eligible=True)
    if pool.empty:
        return {"status": "insufficient_verified_data", "development_through": end.strftime("%Y-%m-%d"),
                "scope": scope, "thresholds": thresholds, "rows": [], "verification": verification,
                "recommended_threshold": None, "production_changes": False}
    selected = pool[pool._selected].copy()
    card = selected[selected._approved.eq(True).fillna(False)] if scope == "qualified_wagers" else selected
    baseline = pool.sort_values(["_market", "market_type", "best_pick"], ascending=[False, True, True],
                                kind="stable").drop_duplicates("_event").set_index("_event", drop=False)
    groups = [("ALL", "ALL", card, len(selected))]
    for league, league_rows in selected.groupby("league", sort=True):
        groups.append((league, "ALL", card[card.league.eq(league)], len(league_rows)))
        for family in sorted(pool.loc[pool.league.eq(league), "_family"].unique()):
            denominator = pool.loc[pool.league.eq(league) & pool._family.eq(family), "_event"].nunique()
            groups.append((league, family, card[card.league.eq(league) & card._family.eq(family)], denominator))
    rows = []
    for league, family, segment, denominator in groups:
        for threshold in thresholds:
            kept = segment[segment._probability.ge(threshold)]
            market = baseline.loc[kept._event].copy()
            metrics = _summary(kept, denominator, "_probability")
            rows.append({"league": league, "market": family, "threshold": threshold,
                         "selector": metrics, "market_only_same_games": _summary(market, denominator, "_market"),
                         "mean_decimal_odds": float(kept._decimal.mean()) if len(kept) else None,
                         "retention_of_scope": len(kept) / len(segment) if len(segment) else None,
                         "hit_rate_wilson_95": wilson_interval(metrics["wins"], metrics["decided"]),
                         "slate_bootstrap": slate_bootstrap(kept, bootstrap_repetitions),
                         "observed_hit_rate_at_least_75": metrics["hit_rate"] is not None and metrics["hit_rate"] >= .75})
    return {"status": "development_comparison" if len(card) else "no_known_approved_wagers",
            "development_through": end.strftime("%Y-%m-%d"), "scope": scope, "thresholds": thresholds,
            "bootstrap_repetitions": bootstrap_repetitions, "rows": rows, "verification": verification,
            "recommended_threshold": None, "production_changes": False}


def render_threshold_report(report):
    def percent(value):
        return "Unavailable" if value is None else f"{value:.1%}"

    def interval(value):
        return "Unavailable" if value is None else "–".join(percent(v) for v in value)

    lines = ["# Selection Threshold Comparison", "", f"Status: **{report['status']}**", "",
             f"Scope: **{report['scope']}**. Development data end: **{report['development_through']}**.", "",
             "Exploratory development results. No threshold is recommended or promoted automatically; later slates are excluded.", "",
             "| League | Market | Minimum probability | Picks | Slates | Coverage | W–L–P | Hit rate | Wilson 95% | Slate bootstrap 95% | ROI | Market ROI | Average decimal odds |",
             "|---|---|---:|---:|---:|---:|---|---:|---|---|---:|---:|---:|"]
    for row in report["rows"]:
        s, b = row["selector"], row["market_only_same_games"]
        odds = "Unavailable" if row["mean_decimal_odds"] is None else f"{row['mean_decimal_odds']:.2f}"
        lines.append(f"| {row['league']} | {row['market']} | {percent(row['threshold'])} | {s['games']} | {s['slates']} | "
                     f"{percent(s['coverage'])} | {s['wins']}–{s['losses']}–{s['pushes']} | {percent(s['hit_rate'])} | "
                     f"{interval(row['hit_rate_wilson_95'])} | {interval(row['slate_bootstrap']['hit_rate_95'])} | "
                     f"{percent(s['flat_roi'])} | {percent(b['flat_roi'])} | {odds} |")
    lines += ["", "## Reading the comparison", "",
              "- Coverage uses all verified supplied games (or games offering that market in each sport/market segment). Unknown approvals never become approved wagers.",
              "- Win rate excludes pushes; one-unit simulated ROI includes pushes in turnover. Market ROI uses the same games, without applying the model threshold to the baseline.",
              "- Wilson intervals assume independent decisions. Slate bootstrap keeps same-day picks together; it needs at least two slates and remains unreliable with few slates or dependence across dates.",
              "- A displayed 75% win rate is descriptive, not proof of a sustainable 75% rate. Intervals are pointwise and do not correct for comparing many thresholds or segments.",
              "- Select any candidate rule using development data, then freeze that exact rule before evaluating untouched future slates. Do not optimize on those future results.",
              "- Proper scoring metrics, calibration bins, retained scope fraction, and bootstrap ROI intervals are included in the companion JSON.", "",
              "## Input verification", "", f"Verified development games: {report['verification']['inventory']['eligible_events']}.", ""]
    lines.extend(f"- {item['reason']}: {item['rows']} rows." for item in report["verification"]["exclusions"])
    return "\n".join(lines) + "\n"
