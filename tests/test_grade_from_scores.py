"""Auto-grader: grade best-picks picks from final scores (ESPN) into the
backtest_exports format the calibration/bucket fitters consume."""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import scripts.grade_from_scores as g


def test_grade_pick_totals_and_spreads():
    # total: 8 runs (home 5, away 3)
    assert g._grade_pick("total_over", 7.5, 5, 3) == "WIN"
    assert g._grade_pick("total_over", 8.5, 5, 3) == "LOSS"
    assert g._grade_pick("total_under", 8.5, 5, 3) == "WIN"
    assert g._grade_pick("total_over", 8.0, 5, 3) == "PUSH"
    # spread_home -1.5: home wins by 2 -> covers
    assert g._grade_pick("spread_home", -1.5, 5, 3) == "WIN"
    assert g._grade_pick("spread_home", -1.5, 4, 3) == "LOSS"   # home by 1, doesn't cover -1.5
    # spread_away +1.5: away loses by 2 -> away +1.5 loses
    assert g._grade_pick("spread_away", 1.5, 5, 3) == "LOSS"
    assert g._grade_pick("spread_away", 1.5, 5, 4) == "WIN"     # away loses by 1, +1.5 covers
    # moneyline
    assert g._grade_pick("moneyline", None, 5, 3) == "WIN"
    assert g._grade_pick("moneyline", None, 3, 5) == "LOSS"


def test_line_from_pick_prefers_market_line_used_then_parses():
    assert g._line_from_pick("Over 8.5", 8.5) == 8.5
    assert g._line_from_pick("Over 8.5", None) == 8.5
    assert g._line_from_pick("Atlanta -1.5", None) == -1.5


def test_end_to_end_grades_and_excludes_no_play(tmp_path, monkeypatch):
    export = pd.DataFrame([
        # Gradeable over: NYY+CWS total -> need a final
        {"league": "MLB", "Home": "New York Yankees", "Away": "Chicago White Sox",
         "market_type": "total_over", "best_pick": "Over 8.5", "market_line_used": 8.5,
         "Pick_Status": "Below Threshold", "Kelly_Bet_Size": 100.0, "odds_american": -110,
         "WinProbability": 0.5, "effective_win_probability": 0.5, "effective_edge": 0.0,
         "effective_expected_value": 0.0, "consensus_agreement": "Neutral", "kalshi_probability": 0.49},
        # No Play row must be excluded
        {"league": "MLB", "Home": "Boston", "Away": "Toronto",
         "market_type": "total_under", "best_pick": "Under 9.5", "market_line_used": 9.5,
         "Pick_Status": "No Play", "Kelly_Bet_Size": 0.0, "odds_american": 100,
         "WinProbability": 0.48, "effective_win_probability": 0.48, "effective_edge": 0.0,
         "effective_expected_value": 0.0, "consensus_agreement": "Disagrees", "kalshi_probability": 0.47},
    ])
    export_csv = tmp_path / "export.csv"
    export.to_csv(export_csv, index=False)

    def fake_results(leagues, target_date=None):
        # NYY 6, CWS 4 -> total 10 > 8.5 -> Over WINS
        return pd.DataFrame([
            {"league": "MLB", "home_team": "New York Yankees", "away_team": "Chicago White Sox",
             "home_score": 6, "away_score": 4, "date": "2026-06-17"},
            {"league": "MLB", "home_team": "Boston", "away_team": "Toronto",
             "home_score": 2, "away_score": 1, "date": "2026-06-17"},
        ])

    monkeypatch.setattr(g, "fetch_espn_results", fake_results)
    out_csv = tmp_path / "out.csv"
    rc = g.grade_from_scores(export_csv, None, out_csv)
    assert rc == 0
    out = pd.read_csv(out_csv)
    # Only the gradeable NYY over row, graded WIN; No Play row excluded.
    assert len(out) == 1
    assert out.iloc[0]["Home"] == "New York Yankees"
    assert out.iloc[0]["W/L"] == "WIN"
    assert float(out.iloc[0]["Win Amount"]) > 100.0  # kelly * decimal odds
