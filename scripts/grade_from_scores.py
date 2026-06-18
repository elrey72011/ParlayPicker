#!/usr/bin/env python3
"""
Auto-grade a best-picks export from FINAL SCORES (ESPN), no manual recap needed.

This closes the earned-Actionable validation loop: it grades each stakeable pick
(over/under/spread/moneyline) against the actual final score and writes a
backtest_exports CSV in the same shape scripts/grade_slate.py produces, which
scripts/fit_bucket_stats.py and scripts/fit_calibration.py then consume. As graded
history accumulates, buckets EARN Actionable (>=55% over >=25 graded picks) and the
calibration table re-fits to realized outcomes.

Usage
-----
    python3 scripts/grade_from_scores.py <export.csv> [YYYY-MM-DD] [out.csv]

Defaults: date = the export's game date (or yesterday); out =
data/backtest_exports/<date>.csv. Re-run the fitters afterward (the daily pipeline
already does). No-Play / Missing-Line / unresolved rows and postponed (0-0) games are
excluded — they were never stakeable and would poison the fit.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.espn_results import fetch_espn_results  # noqa: E402
from app_core.feature_processing import robust_normalize_team  # noqa: E402

OUT_COLS = [
    "league", "Home", "Away", "best_pick", "WinProbability",
    "effective_win_probability", "effective_edge", "effective_expected_value",
    "consensus_agreement", "kalshi_probability", "Kelly_Bet_Size", "Win Amount", "W/L",
]


def _norm(s: object) -> str:
    return robust_normalize_team(str(s)) if pd.notna(s) else ""


def _decimal(odds: float) -> float:
    o = float(odds)
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def _line_from_pick(best_pick: str, market_line_used: object) -> float | None:
    """Numeric line for the pick — prefer the explicit market_line_used, else parse
    the trailing number from the pick string (e.g. 'Over 8.5', 'Atlanta -1.5')."""
    ln = pd.to_numeric(market_line_used, errors="coerce")
    if pd.notna(ln):
        return float(ln)
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", str(best_pick))
    return float(m.group(1)) if m else None


def _grade_pick(market_type: str, line: float | None, home_score: int, away_score: int) -> str:
    """WIN / LOSS / PUSH for a pick against the final score. '' if ungradeable."""
    mt = str(market_type or "").strip().lower()
    total = home_score + away_score
    margin = home_score - away_score  # home minus away
    if mt == "total_over":
        if line is None:
            return ""
        return "WIN" if total > line else "LOSS" if total < line else "PUSH"
    if mt == "total_under":
        if line is None:
            return ""
        return "WIN" if total < line else "LOSS" if total > line else "PUSH"
    if mt == "spread_home":
        if line is None:
            return ""
        adj = margin + line  # home covers when (home-away) + line > 0
        return "WIN" if adj > 0 else "LOSS" if adj < 0 else "PUSH"
    if mt == "spread_away":
        if line is None:
            return ""
        adj = (-margin) + line  # away covers when (away-home) + line > 0
        return "WIN" if adj > 0 else "LOSS" if adj < 0 else "PUSH"
    if mt in ("moneyline", "h2h", "spread_home_ml"):
        return "WIN" if margin > 0 else "LOSS" if margin < 0 else "PUSH"
    return ""


def _slate_date(exp: pd.DataFrame) -> datetime.date:
    for col in ("game_date", "Local Date", "Commence (Local)"):
        if col in exp.columns:
            d = pd.to_datetime(exp[col], errors="coerce", utc=True).dropna()
            if not d.empty:
                return d.iloc[0].date()
    return (datetime.utcnow() - timedelta(days=1)).date()


def _gradeable(row) -> bool:
    status = str(row.get("Pick_Status", "")).strip().lower()
    if status in ("no play", "missing line"):
        return False
    if "unresolved" in _norm(row.get("best_pick")):
        return False
    return True


def grade_from_scores(export_csv: Path, target_date: datetime.date | None, out_csv: Path) -> int:
    exp = pd.read_csv(export_csv)
    if target_date is None:
        target_date = _slate_date(exp)

    leagues = sorted({str(l).upper() for l in exp.get("league", pd.Series(dtype=str)).dropna().unique()})
    results = fetch_espn_results(leagues or ["MLB", "NBA", "NHL", "NCAAB"], target_date)
    if results.empty:
        print(f"  [WARN] no ESPN finals for {target_date} ({leagues}); nothing graded.", file=sys.stderr)
        return 1

    # Index finals by the unordered team pair so home/away orientation can't break the match.
    finals: dict[frozenset, dict] = {}
    for _, r in results.iterrows():
        h, a = _norm(r["home_team"]), _norm(r["away_team"])
        hs, as_ = pd.to_numeric(r["home_score"], errors="coerce"), pd.to_numeric(r["away_score"], errors="coerce")
        if not h or not a or pd.isna(hs) or pd.isna(as_):
            continue
        if int(hs) == 0 and int(as_) == 0:  # postponed/void
            continue
        finals[frozenset((h, a))] = {h: int(hs), a: int(as_)}

    rows, unmatched = [], []
    for _, e in exp.iterrows():
        if not _gradeable(e):
            continue
        eh, ea = _norm(e.get("Home")), _norm(e.get("Away"))
        score = finals.get(frozenset((eh, ea)))
        if not score or eh not in score or ea not in score:
            unmatched.append((eh, ea))
            continue
        line = _line_from_pick(e.get("best_pick"), e.get("market_line_used"))
        result = _grade_pick(e.get("market_type"), line, score[eh], score[ea])
        if result == "":
            continue
        kelly = pd.to_numeric(e.get("Kelly_Bet_Size"), errors="coerce")
        odds = pd.to_numeric(e.get("odds_american"), errors="coerce")
        win_amt = ""
        if result == "WIN" and pd.notna(kelly) and pd.notna(odds):
            win_amt = round(float(kelly) * _decimal(odds), 2)
        rows.append({
            "league": e.get("league"), "Home": e.get("Home"), "Away": e.get("Away"),
            "best_pick": e.get("best_pick"), "WinProbability": e.get("WinProbability"),
            "effective_win_probability": e.get("effective_win_probability"),
            "effective_edge": e.get("effective_edge"),
            "effective_expected_value": e.get("effective_expected_value"),
            "consensus_agreement": e.get("consensus_agreement"),
            "kalshi_probability": e.get("kalshi_probability"),
            "Kelly_Bet_Size": kelly, "Win Amount": win_amt, "W/L": result,
        })

    out = pd.DataFrame(rows, columns=OUT_COLS)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    w = sum(1 for r in rows if r["W/L"] == "WIN")
    l = sum(1 for r in rows if r["W/L"] == "LOSS")
    p = sum(1 for r in rows if r["W/L"] == "PUSH")
    print(f"wrote {out_csv}: {len(out)} graded picks, {w}-{l} ({p} push) for {target_date}")
    if unmatched:
        print(f"  [WARN] {len(unmatched)} export rows had no ESPN final: {unmatched[:5]}", file=sys.stderr)
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    export_csv = Path(args[0])
    target_date = None
    out_csv = None
    for a in args[1:]:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a):
            target_date = datetime.strptime(a, "%Y-%m-%d").date()
        else:
            out_csv = Path(a)
    if target_date is None:
        target_date = _slate_date(pd.read_csv(export_csv))
    if out_csv is None:
        out_csv = Path("data/backtest_exports") / f"{target_date}.csv"
    return grade_from_scores(export_csv, target_date, out_csv)


if __name__ == "__main__":
    raise SystemExit(main())
