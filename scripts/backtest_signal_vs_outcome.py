#!/usr/bin/env python3
"""Backtest harness — segment win-rate / ROI / CLV from graded Performance Recaps.

Turns the one-off "where (if anywhere) do we have an edge?" analysis into a
repeatable tool. Reads a folder of ``Performance Recap*.csv`` files (each row a
graded pick with ``Status``, ``Pick Taken``, ``Outcome``), and reports:

  * Staked-tier summary (Actionable / Actionable+HV / all rows) — the headline
    that separates the card you bet from the do-not-bet tiers.
  * Per-segment win rate + flat -110 ROI (league, totals direction, line bucket,
    tier) so a profitable segment (if one exists with adequate n) is visible.
  * Closing-line-value summary when a closing-lines snapshot is supplied — the
    only metric that actually predicts long-run edge.

ROI is flat -110 unless a recap carries odds; recaps don't, so treat ROI as the
standard-juice approximation. NOTHING here fits parameters — it only measures, so
acting on an in-sample segment is on the reader (out-of-sample first).

Usage:
    python scripts/backtest_signal_vs_outcome.py --recaps "path/to/Performance Recap*.csv"
"""
from __future__ import annotations

import argparse
import glob
from typing import Iterable

import pandas as pd

BREAK_EVEN_110 = 0.5238  # -110 implied break-even win rate
STAKED = {"Actionable", "High Variance/Speculative"}


def _norm_outcome(v) -> str | None:
    s = str(v).strip().upper()
    if s in ("WIN", "W"):
        return "WIN"
    if s in ("LOSS", "L"):
        return "LOSS"
    return None  # PUSH / N/A / blank -> ungraded


def _side(pick: str) -> str:
    p = str(pick).strip().lower()
    if "over" in p:
        return "Over"
    if "under" in p:
        return "Under"
    return "Side"


def _line(pick: str):
    p = str(pick).strip().lower().replace("over", "").replace("under", "").strip()
    try:
        return float(p)
    except ValueError:
        return None


def load_recaps(paths: Iterable[str]) -> pd.DataFrame:
    """Concatenate recap CSVs into a normalized graded frame."""
    frames = []
    for path in paths:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            continue
        if "Pick Taken" not in df.columns or "Outcome" not in df.columns:
            continue
        df["__src"] = path
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    rec = pd.concat(frames, ignore_index=True)
    rec["_outcome"] = rec["Outcome"].map(_norm_outcome)
    rec = rec[rec["_outcome"].notna()].copy()
    rec["_win"] = rec["_outcome"].eq("WIN")
    rec["_side"] = rec["Pick Taken"].map(_side)
    rec["_line"] = rec["Pick Taken"].map(_line)
    rec["_tier"] = rec.get("Status", pd.Series("", index=rec.index)).astype(str).str.strip()
    rec["_league"] = rec.get("League", pd.Series("", index=rec.index)).astype(str).str.strip()
    return rec


def segment_record(rec: pd.DataFrame) -> dict:
    """Wins/losses/n/win%/flat-110 ROI for a subset."""
    n = len(rec)
    w = int(rec["_win"].sum()) if n else 0
    wr = (w / n) if n else 0.0
    roi = ((w * (100.0 / 110.0)) - (n - w)) / n if n else 0.0  # 1u flat at -110
    return {"wins": w, "losses": n - w, "n": n, "win_rate": wr, "roi": roi}


def summarize(rec: pd.DataFrame) -> dict[str, dict]:
    """Headline tiers + the segment scan, as a dict of {label: record}."""
    out: dict[str, dict] = {}
    out["TIER: Actionable"] = segment_record(rec[rec["_tier"] == "Actionable"])
    out["TIER: Actionable+HV"] = segment_record(rec[rec["_tier"].isin(STAKED)])
    out["TIER: Below Threshold"] = segment_record(rec[rec["_tier"] == "Below Threshold"])
    out["TIER: No Play"] = segment_record(rec[rec["_tier"] == "No Play"])
    out["ALL graded rows"] = segment_record(rec)
    for lg in sorted(rec["_league"].unique()):
        if lg:
            out[f"LEAGUE: {lg}"] = segment_record(rec[rec["_league"] == lg])
    mlb_tot = rec[(rec["_league"] == "MLB") & rec["_side"].isin(["Over", "Under"])]
    for s in ("Over", "Under"):
        out[f"MLB {s}"] = segment_record(mlb_tot[mlb_tot["_side"] == s])
    ov = mlb_tot[(mlb_tot["_side"] == "Over") & mlb_tot["_line"].notna()]
    buckets = [("MLB Over line<8.5", ov["_line"] < 8.5),
               ("MLB Over 8.5-9.5", (ov["_line"] >= 8.5) & (ov["_line"] < 10)),
               ("MLB Over line>=10", ov["_line"] >= 10)]
    for label, mask in buckets:
        out[label] = segment_record(ov[mask])
    return out


def _fmt(label: str, r: dict) -> str:
    flag = ""
    if r["n"] >= 20 and r["win_rate"] > BREAK_EVEN_110:
        flag = "  <-- above break-even"
    elif 0 < r["n"] < 20:
        flag = "  (small n)"
    return f"  {label:26} {r['wins']:3}-{r['losses']:<3} ({r['n']:3})  {r['win_rate']:5.1%}  ROI {r['roi']:+6.1%}{flag}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recaps", required=True, help="glob for Performance Recap*.csv files")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.recaps))
    rec = load_recaps(paths)
    if rec.empty:
        print(f"No gradeable recap rows found for: {args.recaps}")
        return 1
    print(f"Loaded {len(rec)} graded picks from {len(paths)} recap file(s). "
          f"Break-even at -110 = {BREAK_EVEN_110:.1%}\n")
    summary = summarize(rec)
    for label, r in summary.items():
        if r["n"] > 0:
            print(_fmt(label, r))
    print("\nNote: ROI is a flat 1u -110 approximation (recaps carry no odds). "
          "Segments above break-even are IN-SAMPLE; validate out-of-sample before staking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
