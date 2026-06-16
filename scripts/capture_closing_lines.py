#!/usr/bin/env python3
"""Capture closing lines for a slate and score CLV against the exported picks.

The pipeline records the line/odds at EXPORT time (hours before first pitch) but
never the CLOSE, so closing-line value — the best long-run edge predictor — is
currently unmeasurable. Run this ~10-15 min before the first game: it re-fetches
the live odds (the closing snapshot), matches them to the day's best_picks_export,
and writes app_exports/closing_lines_<run>.csv with open-vs-close line/price and
CLV per pick. Feed that file to scripts/backtest_signal_vs_outcome.py later.

The matching + CLV math lives in build_closing_snapshot(), a pure function over
two DataFrames, so it is unit-tested without the odds API. The CLI is a thin
wrapper that pulls the live snapshot via the pipeline's existing fetch and exits
cleanly (no fabricated data) when odds are unavailable.

Usage:
    python scripts/capture_closing_lines.py --export "app_exports/best_picks_export*.csv"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.clv import closing_line_value  # noqa: E402


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _open_line(row) -> float | None:
    for c in ("market_line_used", "total_line", "base_total_line"):
        if c in row and pd.notna(row.get(c)):
            v = _num(row.get(c))
            if v is not None:
                return v
    return None


def build_closing_snapshot(export_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    """Join exported picks to a live (closing) odds snapshot and score CLV.

    ``export_df`` rows need: matchup_id, market_type, best_pick, an open line
    (market_line_used/total_line) and odds_american. ``live_df`` is the
    fetch_live_odds_dataframe output (per-matchup novig_over/under point+price).
    Only totals picks are scored (CLV for spreads/h2h is a later extension).
    Returns one row per matched pick with open/close line+price and CLV fields.
    """
    live_by_id = {}
    if not live_df.empty and "matchup_id" in live_df.columns:
        live_by_id = {str(r["matchup_id"]): r for _, r in live_df.iterrows()}

    out = []
    for _, p in export_df.iterrows():
        mt = str(p.get("market_type", "")).lower()
        if "total" not in mt:
            continue
        side = "Over" if mt.endswith("over") or "over" in str(p.get("best_pick", "")).lower() else "Under"
        mid = str(p.get("matchup_id", ""))
        live = live_by_id.get(mid)
        if live is None:
            continue
        if side == "Over":
            close_line = _num(live.get("novig_over_point"))
            close_odds = _num(live.get("novig_over_price"))
            opp_close = _num(live.get("novig_under_price"))
        else:
            close_line = _num(live.get("novig_under_point"))
            close_odds = _num(live.get("novig_under_price"))
            opp_close = _num(live.get("novig_over_price"))
        clv = closing_line_value(
            side=side,
            open_line=_open_line(p),
            close_line=close_line,
            pick_open_odds=_num(p.get("odds_american")),
            pick_close_odds=close_odds,
            opp_close_odds=opp_close,
        )
        out.append({
            "matchup_id": mid,
            "league": p.get("league"),
            "Home": p.get("Home", p.get("home_team")),
            "Away": p.get("Away", p.get("away_team")),
            "best_pick": p.get("best_pick"),
            "Pick_Status": p.get("Pick_Status"),
            "open_line": _open_line(p),
            "close_line": close_line,
            "open_odds": _num(p.get("odds_american")),
            "close_odds": close_odds,
            "line_clv": clv["line_clv"],
            "price_clv": clv["price_clv"],
            "beat_close": clv["beat_close"],
            "export_run_id": p.get("export_run_id"),
        })
    return pd.DataFrame(out)


def _infer_sports(export_df: pd.DataFrame) -> list[str]:
    col = "league" if "league" in export_df.columns else None
    return sorted({str(x).upper() for x in export_df[col].dropna()}) if col else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, help="glob for the day's best_picks_export CSV")
    ap.add_argument("--out", default=None, help="output path (default app_exports/closing_lines_<run>.csv)")
    args = ap.parse_args(argv)

    matches = sorted(glob.glob(args.export))
    if not matches:
        print(f"No export matched: {args.export}", file=sys.stderr)
        return 1
    export_df = pd.read_csv(matches[-1], encoding="utf-8-sig")

    try:
        from core.streamlit_pipeline import fetch_live_odds_dataframe
        live_df = fetch_live_odds_dataframe(_infer_sports(export_df))
    except Exception as e:  # pragma: no cover - network/integration path
        print(f"Could not fetch live (closing) odds: {e}", file=sys.stderr)
        return 2
    if live_df is None or live_df.empty:
        print("Live odds fetch returned no rows (missing API key or off-hours) — "
              "no closing snapshot written.", file=sys.stderr)
        return 2

    snap = build_closing_snapshot(export_df, live_df)
    if snap.empty:
        print("No totals picks matched the live snapshot.", file=sys.stderr)
        return 3

    run = str(export_df.get("export_run_id", pd.Series(["unknown"])).iloc[0])
    out_path = args.out or os.path.join("app_exports", f"closing_lines_{run}.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    snap.to_csv(out_path, index=False)
    beat = snap["beat_close"].dropna()
    rate = beat.mean() if len(beat) else 0.0
    print(f"wrote {out_path}: {len(snap)} picks, beat-close rate {rate:.0%} ({int(beat.sum())}/{len(beat)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
