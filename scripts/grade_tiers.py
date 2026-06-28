#!/usr/bin/env python3
"""Track the games card's tier outcomes (No Play vs Below/HV vs Actionable) over time.

Logs every graded pick from a Performance Recap into data/tier_results/tier_results_log.csv
(deduped by date+matchup+pick) and prints each tier's running record, win rate, and a
flat-stake P&L. The whole question this answers: does the DISMISSED "No Play" tier keep
beating the higher-ranked "Below/HV" tier — and does it clear the -110 break-even (52.4%)?

Usage
-----
    python3 scripts/grade_tiers.py <Performance_Recap.csv> [YYYY-MM-DD]

P&L is a flat 1-unit bet at -110 (the standard totals price; the recap carries no odds), so
it is an APPROXIMATION used only to track whether a tier is above/below break-even.

grade_recap / summarize_by_tier are pure and unit-tested.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOG_PATH = Path("data/tier_results/tier_results_log.csv")
FLAT_ODDS = -110.0   # standard totals price for the approximate P&L
_DEC = 1 + 100.0 / 110.0


def tier_of(status) -> str:
    s = str(status).strip()
    if s == "No Play":
        return "No Play"
    if s == "Actionable":
        return "Actionable"
    return "Below/HV"   # Below Threshold + High Variance/Speculative


def _date_from_run_id(run_id) -> str | None:
    s = str(run_id)
    if len(s) >= 8 and s[:8].isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    return None


def grade_recap(recap: pd.DataFrame, date: str | None = None) -> list[dict]:
    """One row per graded (WIN|LOSS) pick: date, matchup, pick, tier, outcome. Pure."""
    rows = []
    for _, r in recap.iterrows():
        outcome = str(r.get("Outcome", "")).strip().upper()
        if outcome not in ("WIN", "LOSS"):
            continue   # skip N/A, PUSH, blanks
        d = date or _date_from_run_id(r.get("export_run_id")) or ""
        home, away = str(r.get("Home", "")).strip(), str(r.get("Away", "")).strip()
        rows.append({
            "date": d,
            "matchup": f"{away} @ {home}",
            "pick": str(r.get("Pick Taken", "")).strip(),
            "tier": tier_of(r.get("Status")),
            "outcome": "WIN" if outcome == "WIN" else "LOSS",
        })
    return rows


def append_log(rows: list[dict], log_path: Path = LOG_PATH) -> pd.DataFrame:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if log_path.exists():
        combined = pd.concat([pd.read_csv(log_path), new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "matchup", "pick"], keep="last")
    else:
        combined = new
    combined.to_csv(log_path, index=False)
    return combined


def summarize_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    """Per-tier record, win rate, and flat-1u-at-(-110) P&L."""
    out = []
    for tier, s in df.groupby("tier")["outcome"]:
        w = int((s == "WIN").sum())
        l = int((s == "LOSS").sum())
        n = w + l
        pnl = w * (_DEC - 1) - l * 1.0
        out.append({
            "tier": tier, "W": w, "L": l, "n": n,
            "win_rate": (w / n) if n else 0.0,
            "units": round(pnl, 2),
            "roi": round(pnl / n, 3) if n else 0.0,
        })
    res = pd.DataFrame(out).sort_values("tier").reset_index(drop=True)
    return res


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    recap_path = sys.argv[1]
    date = sys.argv[2] if len(sys.argv) == 3 else None
    rows = grade_recap(pd.read_csv(recap_path), date)
    slate = summarize_by_tier(pd.DataFrame(rows))
    print(f"=== this recap ({rows[0]['date'] if rows else '?'}) ===")
    print(slate.to_string(index=False))

    full = append_log(rows)
    print(f"\n=== RUNNING TOTAL ({len(full)} graded picks across all slates) ===")
    print(summarize_by_tier(full).to_string(index=False))
    print(f"\n  break-even at -110 = 52.4% win rate.  log: {LOG_PATH}")


if __name__ == "__main__":
    main()
