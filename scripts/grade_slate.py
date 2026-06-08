#!/usr/bin/env python3
"""
Grade a slate: merge a best-picks export with its performance recap into a
backtest_exports CSV that scripts/backtest_*.py can read.

Replaces the error-prone hand-merge. Given:
  * the best-picks export (consensus_agreement, kalshi_probability, Kelly_Bet_Size,
    odds_american, market_type, WinProbability/effective_*, Home/Away/best_pick), and
  * the performance recap (Home, Away, Pick Taken, Outcome),
it matches rows on (Home, Away, best_pick), grades W/L from Outcome, computes
Win Amount = Kelly_Bet_Size * decimal_odds for wins, and writes the graded CSV.

Usage
-----
    python3 scripts/grade_slate.py <export.csv> <recap.csv> data/backtest_exports/2026-06-03.csv

Then re-run:  python3 scripts/backtest_theover_direction.py data/backtest_exports
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _norm(s: object) -> str:
    return " ".join(str(s).strip().lower().split())


def _decimal(odds: float) -> float:
    o = float(odds)
    return 1 + o / 100 if o > 0 else 1 + 100 / abs(o)


def _norm_result(v: object) -> str:
    s = str(v).strip().upper()
    if s.startswith("W"):
        return "WIN"
    if s.startswith("L"):
        return "LOSS"
    if s.startswith("P"):
        return "PUSH"
    return ""


OUT_COLS = [
    "league", "Home", "Away", "best_pick", "WinProbability",
    "effective_win_probability", "effective_edge", "effective_expected_value",
    "consensus_agreement", "kalshi_probability", "Kelly_Bet_Size", "Win Amount", "W/L",
]


def grade(export_csv: Path, recap_csv: Path, out_csv: Path) -> None:
    exp = pd.read_csv(export_csv)
    rec = pd.read_csv(recap_csv)

    # Recap pick column is "Pick Taken"; outcome is "Outcome".
    rec_key = {
        (_norm(r.get("Home")), _norm(r.get("Away")), _norm(r.get("Pick Taken"))): _norm_result(r.get("Outcome"))
        for _, r in rec.iterrows()
    }

    rows, unmatched = [], []
    for _, e in exp.iterrows():
        key = (_norm(e.get("Home")), _norm(e.get("Away")), _norm(e.get("best_pick")))
        result = rec_key.get(key)
        if result is None:
            unmatched.append(key)
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
    print(f"wrote {out_csv}: {len(out)} graded picks, {w}-{l}")
    if unmatched:
        print(f"  [WARN] {len(unmatched)} export rows had no recap match: {unmatched[:5]}", file=sys.stderr)


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    grade(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
