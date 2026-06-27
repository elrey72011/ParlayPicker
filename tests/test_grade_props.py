"""Prop-results grading: pure W/L + P&L logic and the running summary."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.grade_props import grade_card, grade_side, summarize


def test_grade_side_over_under_and_push():
    assert grade_side("over", 4.5, 6) == "WIN"      # 6 > 4.5
    assert grade_side("over", 5.5, 5) == "LOSS"     # 5 < 5.5
    assert grade_side("under", 4.5, 0) == "WIN"     # 0 < 4.5
    assert grade_side("under", 5.5, 9) == "LOSS"    # 9 > 5.5
    assert grade_side("over", 5.0, 5) == "PUSH"     # integer line, exact
    assert grade_side("over", 4.5, None) is None    # no data


def test_grade_card_filters_to_actionable_and_computes_pnl():
    card = pd.DataFrame([
        {"pitcher": "Will Warren", "market_type": "pitcher_strikeouts_under", "best_pick": "Will Warren Under 4.5 Ks",
         "line": 4.5, "expected_ks": 4.4, "odds_american": 116, "Kelly_Bet_Size": 5.0, "Pick_Status": "Actionable"},
        {"pitcher": "Skipped Guy", "market_type": "pitcher_strikeouts_over", "best_pick": "Skipped Guy Over 5.5 Ks",
         "line": 5.5, "expected_ks": 6.0, "odds_american": -110, "Kelly_Bet_Size": 0.0, "Pick_Status": "no_play"},
    ])
    ks = {"will warren": 0}  # actual: 0 strikeouts -> under wins
    rows = grade_card(card, "2026-06-26", name_to_id={"will warren": 1}, actual_ks_fn=lambda pid: ks_lookup(ks, pid))
    assert len(rows) == 1                       # non-Actionable filtered out
    r = rows[0]
    assert r["result"] == "WIN" and r["actual_ks"] == 0
    assert abs(r["profit"] - 5.0 * (1 + 116 / 100 - 1)) < 1e-6   # +116 -> profit = stake * 1.16


def ks_lookup(table, pid):
    # pid 1 -> will warren; mirrors the name_to_id mapping in the test
    return {1: table["will warren"]}.get(pid)


def test_summarize_record_and_roi():
    df = pd.DataFrame([
        {"result": "WIN", "stake": 5.0, "profit": 5.8},
        {"result": "LOSS", "stake": 5.0, "profit": -5.0},
        {"result": "WIN", "stake": 5.0, "profit": 7.65},
        {"result": "PUSH", "stake": 5.0, "profit": 0.0},   # excluded from record
    ])
    s = summarize(df)
    assert s["wins"] == 2 and s["losses"] == 1
    assert abs(s["pnl"] - (5.8 - 5.0 + 7.65)) < 1e-6
    assert abs(s["staked"] - 15.0) < 1e-6        # PUSH stake not counted
    assert abs(s["roi"] - (8.45 / 15.0)) < 1e-6
