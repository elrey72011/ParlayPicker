import pandas as pd
import pytest

from app.ui.results_dashboard import funded_prop_rows, summarize_prop_results
from scripts.grade_props import grade_card


def test_funded_prop_rows_excludes_blocked_actionable_research_rows():
    card = pd.DataFrame(
        [
            {"player": "Funded Player", "Pick_Status": "Actionable", "Stake_Status": "Funded", "Kelly_Bet_Size": 4.0},
            {"player": "Blocked Player", "Pick_Status": "Actionable", "Stake_Status": "Blocked - player exposure", "Kelly_Bet_Size": 4.0},
            {"player": "Research Player", "Pick_Status": "Below Threshold", "Stake_Status": "Research", "Kelly_Bet_Size": 0.0},
        ]
    )

    funded = funded_prop_rows(card)

    assert funded["player"].tolist() == ["Funded Player"]


def test_prop_summary_counts_push_stake_and_excludes_unresolved():
    results = pd.DataFrame(
        [
            {"result": "WIN", "stake": 5.0, "profit": 4.0},
            {"result": "LOSS", "stake": 4.0, "profit": -4.0},
            {"result": "PUSH", "stake": 3.0, "profit": 0.0},
            {"result": None, "stake": 10.0, "profit": None},
        ]
    )

    summary = summarize_prop_results(results)

    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["pushes"] == 1
    assert summary["unresolved"] == 1
    assert summary["staked"] == pytest.approx(12.0)
    assert summary["pnl"] == pytest.approx(0.0)
    assert summary["roi"] == pytest.approx(0.0)
    assert summary["win_rate"] == pytest.approx(0.5)


def test_prop_grader_honors_stake_status_when_export_has_it():
    card = pd.DataFrame(
        [
            {
                "player": "Funded Player",
                "Pick_Status": "Actionable",
                "Stake_Status": "Funded",
                "market_type": "pitcher_strikeouts_over",
                "best_pick": "Funded Player Over 5.5 Ks",
                "line": 5.5,
                "odds_american": -110,
                "Kelly_Bet_Size": 5.0,
            },
            {
                "player": "Blocked Player",
                "Pick_Status": "Actionable",
                "Stake_Status": "Blocked - probation exposure",
                "market_type": "pitcher_strikeouts_over",
                "best_pick": "Blocked Player Over 5.5 Ks",
                "line": 5.5,
                "odds_american": -110,
                "Kelly_Bet_Size": 5.0,
            },
        ]
    )

    rows = grade_card(
        card,
        "2026-07-12",
        {"funded player": 1, "blocked player": 2},
        lambda player_id: 7,
    )

    assert len(rows) == 1
    assert rows[0]["player"] == "Funded Player"
    assert rows[0]["result"] == "WIN"
