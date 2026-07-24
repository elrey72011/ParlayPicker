"""Best Overall Pick of the Day ranks funded production tickets only."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.pick_of_day import select_pick_of_the_day


def _games(**overrides) -> pd.DataFrame:
    base = {
        "league": ["MLB"],
        "home_team": ["Atlanta"],
        "away_team": ["New York Mets"],
        "best_pick": ["Over 8.5"],
        "market_type": ["total_over"],
        "Pick_Status": ["Below Threshold"],
        "production_eligible": [False],
        "Kelly_Bet_Size": [0.0],
        "odds_american": [-110],
        "empirical_win_probability": [0.49],
        "effective_win_probability": [0.52],
        "calibrated_probability": [0.55],
        "status_blocker_stage": [""],
        "game_already_started_flag": [False],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _props(**overrides) -> pd.DataFrame:
    base = {
        "league": ["MLB"],
        "pitcher": ["Ryan Feltner"],
        "matchup": ["San Francisco Giants @ Colorado Rockies"],
        "best_pick": ["Ryan Feltner Under 3.5 Ks"],
        "WinProbability": [0.6487],
        "odds_american": [-146],
        "Pick_Status": ["Actionable"],
        "production_eligible": [True],
        "Stake_Status": ["Funded"],
        "Kelly_Bet_Size": [9.23],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_prop_wins_when_games_card_is_benched():
    # The 3-Jul shape: every game Below Threshold at sub-50% empirical prob,
    # props Actionable at 65%. The day's pick must be the prop, at its Kelly.
    potd = select_pick_of_the_day(_games(), _props())
    assert potd is not None
    assert potd["board"] == "prop"
    assert potd["pick"] == "Ryan Feltner Under 3.5 Ks"
    assert potd["stake"] == 9.23
    assert not potd["below_floor"]


def test_game_wins_when_it_is_the_likelier_pick():
    games = _games(
        empirical_win_probability=[0.67],
        Pick_Status=["Actionable"],
        production_eligible=[True],
        Kelly_Bet_Size=[12.0],
    )
    potd = select_pick_of_the_day(games, _props())
    assert potd["board"] == "game"
    assert potd["pick"] == "Over 8.5"
    assert potd["runner_up"]["board"] == "prop"


def test_unfunded_research_board_returns_none():
    games = _games(empirical_win_probability=[0.51])
    props = _props(
        WinProbability=[0.89],
        Kelly_Bet_Size=[0.0],
        Pick_Status=["Qualified / No Stake"],
        Stake_Status=["Qualified / No Stake"],
        production_eligible=[False],
    )
    potd = select_pick_of_the_day(games, props)
    assert potd is None


def test_unfunded_high_confidence_prop_cannot_displace_funded_game():
    games = _games(
        empirical_win_probability=[0.61],
        Pick_Status=["Actionable"],
        production_eligible=[True],
        Kelly_Bet_Size=[8.0],
    )
    props = _props(
        WinProbability=[0.89],
        Kelly_Bet_Size=[0.0],
        Pick_Status=["Qualified / No Stake"],
        Stake_Status=["Qualified / No Stake"],
        production_eligible=[False],
    )
    potd = select_pick_of_the_day(games, props)
    assert potd["board"] == "game"
    assert potd["stake"] == 8.0


def test_disqualified_games_never_surface():
    started = _games(
        empirical_win_probability=[0.99],
        game_already_started_flag=[True],
    )
    wrong_game = _games(
        empirical_win_probability=[0.98],
        status_blocker_stage=["kalshi_wrong_game_title"],
    )
    proven_losing = _games(
        empirical_win_probability=[0.97],
        status_blocker_stage=["empirical_proven_losing_bucket"],
    )
    games = pd.concat([started, wrong_game, proven_losing], ignore_index=True)
    potd = select_pick_of_the_day(games, _props())
    assert potd["board"] == "prop"


def test_empty_board_returns_none():
    assert select_pick_of_the_day(pd.DataFrame(), pd.DataFrame()) is None
    assert select_pick_of_the_day(None, None) is None


def test_probability_basis_prefers_empirical_over_calibrated():
    # calibrated says 0.60 but empirical (bucket-truth) says 0.49 — the game must
    # rank on 0.49 and lose to a 0.55 prop.
    games = _games(
        empirical_win_probability=[0.49],
        effective_win_probability=[0.50],
        calibrated_probability=[0.60],
        Pick_Status=["Actionable"],
        production_eligible=[True],
        Kelly_Bet_Size=[4.0],
    )
    props = _props(WinProbability=[0.55])
    potd = select_pick_of_the_day(games, props)
    assert potd["board"] == "prop"
    assert potd["runner_up"]["win_probability"] == 0.49
