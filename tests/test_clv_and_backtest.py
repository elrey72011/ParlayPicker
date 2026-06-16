import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from core.clv import american_to_implied, no_vig_prob, line_clv, price_clv, closing_line_value
from scripts.backtest_signal_vs_outcome import load_recaps, summarize, segment_record
from scripts.capture_closing_lines import build_closing_snapshot


# ---- CLV pure math ----

def test_american_to_implied():
    assert abs(american_to_implied(-110) - 0.5238) < 1e-3
    assert abs(american_to_implied(+120) - 0.4545) < 1e-3
    assert american_to_implied(None) is None
    assert american_to_implied(0) is None


def test_no_vig_prob_strips_juice():
    # Two -110 sides -> 0.5 each after de-vig.
    assert abs(no_vig_prob(-110, -110) - 0.5) < 1e-6
    # Missing opposite price -> raw implied.
    assert abs(no_vig_prob(-110, None) - american_to_implied(-110)) < 1e-9


def test_line_clv_direction():
    # Over wants the lower number: close above entry is favorable (+).
    assert line_clv("Over 8.5", 8.5, 9.0) == 0.5
    assert line_clv("Over 8.5", 8.5, 8.0) == -0.5
    # Under wants the higher number.
    assert line_clv("Under 8.5", 8.5, 8.0) == 0.5
    assert line_clv("Under 8.5", 8.5, 9.0) == -0.5


def test_price_clv_positive_when_entry_cheaper():
    # Entry +120 (implied .4545), close -110 (.5238): we bought cheaper -> positive.
    pc = price_clv(120, -110)
    assert pc is not None and pc > 0


def test_closing_line_value_beat_close_flag():
    # Over 8.5 entered at 8.5/+120, closes 9.0/-110: line + price both favorable.
    r = closing_line_value("Over 8.5", open_line=8.5, close_line=9.0,
                           pick_open_odds=120, pick_close_odds=-110)
    assert r["beat_close"] is True
    # Mirror: closes worse on both -> did not beat close.
    r2 = closing_line_value("Over 8.5", open_line=8.5, close_line=8.0,
                            pick_open_odds=-110, pick_close_odds=120)
    assert r2["beat_close"] is False
    # No data -> None, not a crash.
    assert closing_line_value("Over 8.5")["beat_close"] is None


# ---- backtest harness aggregation ----

def _recap_frame():
    return pd.DataFrame({
        "League": ["MLB", "MLB", "MLB", "MLB"],
        "Home": ["A", "B", "C", "D"],
        "Away": ["a", "b", "c", "d"],
        "Pick Taken": ["Over 9.5", "Over 8.5", "Under 8.5", "Over 7.5"],
        "Outcome": ["WIN", "LOSS", "WIN", "N/A"],   # N/A is ungraded
        "Status": ["Actionable", "High Variance/Speculative", "Below Threshold", "No Play"],
    })


def test_load_recaps_drops_ungraded(tmp_path):
    p = tmp_path / "Performance Recap (x).csv"
    _recap_frame().to_csv(p, index=False)
    rec = load_recaps([str(p)])
    assert len(rec) == 3  # the N/A row dropped
    assert rec["_win"].sum() == 2


def test_segment_record_roi_flat_110():
    rec = load_recaps  # noqa: F841 (just exercising the math directly)
    df = pd.DataFrame({"_win": [True, False]})
    r = segment_record(df)
    assert r["n"] == 2 and r["wins"] == 1
    # 1 win @ -110 (+0.909) and 1 loss (-1.0) over 2 bets -> negative ROI.
    assert r["roi"] < 0


def test_summarize_separates_tiers(tmp_path):
    p = tmp_path / "Performance Recap (y).csv"
    _recap_frame().to_csv(p, index=False)
    rec = load_recaps([str(p)])
    s = summarize(rec)
    assert s["TIER: Actionable"]["n"] == 1
    assert s["TIER: Actionable+HV"]["n"] == 2
    assert s["ALL graded rows"]["n"] == 3


# ---- closing-snapshot join ----

def test_build_closing_snapshot_scores_clv():
    export = pd.DataFrame({
        "matchup_id": ["mlb::x"],
        "market_type": ["total_over"],
        "best_pick": ["Over 8.5"],
        "Pick_Status": ["High Variance/Speculative"],
        "total_line": [8.5],
        "odds_american": [120],
        "league": ["MLB"],
    })
    live = pd.DataFrame([{
        "matchup_id": "mlb::x",
        "novig_over_point": 9.0, "novig_over_price": -110,
        "novig_under_point": 9.0, "novig_under_price": -110,
    }])
    snap = build_closing_snapshot(export, live)
    assert len(snap) == 1
    row = snap.iloc[0]
    assert row["open_line"] == 8.5 and row["close_line"] == 9.0
    assert bool(row["beat_close"]) is True  # line up + bought at +120 vs -110 close


def test_build_closing_snapshot_skips_unmatched_and_non_totals():
    export = pd.DataFrame({
        "matchup_id": ["mlb::x", "mlb::y"],
        "market_type": ["spread_home", "total_under"],
        "best_pick": ["Team -1.5", "Under 7.5"],
        "total_line": [None, 7.5],
        "odds_american": [-110, -105],
        "league": ["MLB", "MLB"],
    })
    live = pd.DataFrame([{  # only has the spread game, not the totals one
        "matchup_id": "mlb::x",
        "novig_over_point": 8.0, "novig_over_price": -110,
        "novig_under_point": 8.0, "novig_under_price": -110,
    }])
    snap = build_closing_snapshot(export, live)
    assert snap.empty  # spread skipped (non-total); totals pick unmatched in live
