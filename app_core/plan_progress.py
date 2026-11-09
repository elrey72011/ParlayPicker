"""Is a frozen bound plan on pace to pass? Daily progress against its windows (owner 2026-11-07).

``prospective_evidence.evaluate_validation_plan`` scores a plan's windows as they fill, but
only says pass or fail with blockers. While a window runs, the useful question is whether it
will reach its minimums before it ends. For each window that has started this reports the
share of the window elapsed, the counts so far (games, graded games, still ungraded, bets,
bets with a comparable close) and a straight-line projection to the window's end against:

* ``minimum_independent_sample`` games;
* market-skill plans: ``min_bets`` bets and ``min_bet_close_coverage`` of them with a close.

``on_pace`` is False when a projection falls short, so a slow plan is seen in days, not at
the end of a window. Read-only: it uses the plan's own evaluation and changes nothing. The
bets' ROI and CLV so far are shown as they stand; the plan is frozen, so seeing them cannot
change it.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

WINDOWS = (("validation", "validation_start", "validation_end"), ("holdout", "holdout_start", "holdout_end"))
# Below this share of a window, a straight line is mostly noise: ``games`` already counts games
# predicted but not yet played (about a day ahead), so the first hours projected thousands
# (audit 2026-11-09). Projections and on_pace start once this much of the window has passed.
MIN_ELAPSED_FOR_PACE = 0.1


def _time(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return result if result.tzinfo else result.replace(tzinfo=timezone.utc)


def progress(plan: dict, evaluation: dict, *, now: datetime) -> list[dict]:
    """One row per started window of ``plan`` (a stored validation plan record)."""
    spec = dict(plan)
    payload = plan.get("payload")
    if isinstance(payload, str):
        try:
            spec.update(json.loads(payload))
        except ValueError:
            pass
    method = spec.get("market_skill_methodology") or {}
    rows = []
    for name, start_key, end_key in WINDOWS:
        start, end = _time(spec[start_key]), _time(spec[end_key])
        if now < start:
            continue
        span = (end - start).total_seconds()
        elapsed = min(1.0, max(0.0, (now - start).total_seconds() / span)) if span > 0 else 1.0
        cohort = evaluation.get(name) or {}
        skill = (evaluation.get("market_skill") or {}).get(name) or {}
        outcomes = cohort.get("outcomes") or {}
        games = int(cohort.get("unique_events") or 0)
        bets = int(skill.get("bets") or 0)
        row = {"plan": spec["validation_plan_id"], "sport": spec["sport"], "market_family": spec["market_family"],
               "window": name, "state": "ENDED" if now >= end else "IN_PROGRESS",
               "elapsed": round(elapsed, 3), "days_left": max(0.0, round((end - now).total_seconds() / 86400, 1)),
               "games": games, "graded": int(cohort.get("scored_observations") or 0),
               "ungraded": int(outcomes.get("PENDING", 0)) + int(outcomes.get("NEEDS_REVIEW", 0)),
               "min_games": int(spec.get("minimum_independent_sample") or 0)}

        def projected(count):
            return round(count / elapsed) if elapsed >= MIN_ELAPSED_FOR_PACE else None
        row["projected_games"] = projected(games)
        pace = [row["projected_games"] is not None and row["projected_games"] >= row["min_games"]]
        if method:
            row.update(bets=bets, min_bets=int(method.get("min_bets") or 0), projected_bets=projected(bets),
                       bet_close_coverage=round(float(skill.get("bet_close_coverage") or 0.0), 3),
                       min_close_coverage=float(method.get("min_bet_close_coverage") or 0.0),
                       bet_roi=skill.get("bet_roi"), bet_mean_clv=skill.get("bet_mean_clv"),
                       min_edge=method.get("min_edge"))
            pace.append(row["projected_bets"] is not None and row["projected_bets"] >= row["min_bets"])
            pace.append(bets == 0 or row["bet_close_coverage"] >= row["min_close_coverage"])
        row["on_pace"] = all(pace) if elapsed >= MIN_ELAPSED_FOR_PACE else None  # None: too early to say
        rows.append(row)
    return rows


def from_store(path, *, now=None) -> list[dict]:
    """Progress rows for every frozen bound plan (one with a model) in the store."""
    from pathlib import Path
    from app_core import prospective_evidence as evidence
    if not Path(path).is_file():
        return []
    now = now or datetime.now(timezone.utc)
    rows = []
    for plan in evidence.read_records(path, "prospective_validation_plan"):
        if not plan.get("model_id"):
            continue  # unbound v1 templates have nothing to track
        try:
            rows.extend(progress(plan, evidence.evaluate_validation_plan(path, plan["validation_plan_id"]), now=now))
        except Exception as exc:  # report, never guess
            rows.append({"plan": plan["validation_plan_id"], "sport": plan["sport"],
                         "market_family": plan["market_family"], "error": f"{type(exc).__name__}: {exc}"[:200]})
    return rows


def markdown(rows: list[dict]) -> list[str]:
    if not rows:
        return ["", "### Plan progress", "", "No frozen bound plan has a window running yet."]
    lines = ["", "### Plan progress", "",
             "| plan | window | elapsed | days left | games (graded / ungraded) | projected vs min | bets "
             "| projected vs min | close coverage | ROI | CLV | on pace |", "|" + "---|" * 12]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['plan']} | error: {r['error']} |" + " |" * 10)
            continue
        pct = lambda v: "" if v is None else f"{v:+.1%}"  # noqa: E731
        lines.append(
            f"| {r['plan']} | {r['window']} ({r['state'].lower()}) | {r['elapsed']:.0%} | {r['days_left']} | "
            f"{r['games']} ({r['graded']} / {r['ungraded']}) | {r['projected_games']} / {r['min_games']} | "
            f"{r.get('bets', '')} | {r.get('projected_bets', '')} / {r.get('min_bets', '')} | "
            f"{'' if 'bet_close_coverage' not in r else format(r['bet_close_coverage'], '.0%')} | "
            f"{pct(r.get('bet_roi'))} | {pct(r.get('bet_mean_clv'))} | {'too early' if r['on_pace'] is None else 'yes' if r['on_pace'] else '**NO**'} |")
    lines += ["", "Projections are straight-line to each window's end, shown once "
                  f"{MIN_ELAPSED_FOR_PACE:.0%} of a window has passed. A window passes only with its minimum "
                  "games and bets, ROI and mean CLV >= 0, and closes for enough of its bets."]
    return lines
