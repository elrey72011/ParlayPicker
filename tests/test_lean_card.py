"""All-games lean view: model's read on every game, tiered BET / LEAN / AVOID.

Pins the honest tiering so a near-efficient slate (mostly Below Threshold / No Play) still
produces a usable full-board read: the +EV-but-below-bar picks surface as LEAN, the proven
ones as BET, and the negative-EV / Kalshi-fading ones as AVOID.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.lean_card import build_all_games_lean_card, classify_lean_tier


def test_classify_tiers():
    assert classify_lean_tier("Actionable", -0.5, "Neutral") == "BET"        # actionable always bets
    assert classify_lean_tier("Below Threshold", 0.03, "Agrees") == "LEAN"   # +EV, not fading -> lean
    assert classify_lean_tier("Below Threshold", 0.03, "Disagrees") == "AVOID"  # fading Kalshi -> avoid
    assert classify_lean_tier("No Play", -0.02, "Neutral") == "AVOID"        # -EV -> avoid
    assert classify_lean_tier("Below Threshold", 0.0, "Neutral") == "AVOID"  # zero EV -> not a lean


def _df(rows):
    return pd.DataFrame(rows)


def test_lean_card_orders_and_labels_full_slate():
    df = _df([
        {"league": "MLB", "home_team": "A", "away_team": "B", "best_pick": "Over 8.5",
         "Pick_Status": "No Play", "effective_expected_value": -0.02,
         "effective_win_probability": 0.49, "effective_edge": 0.01,
         "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0},
        {"league": "MLB", "home_team": "C", "away_team": "D", "best_pick": "Under 8.5",
         "Pick_Status": "Below Threshold", "effective_expected_value": 0.03,
         "effective_win_probability": 0.53, "effective_edge": 0.035,
         "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0},
        {"league": "MLB", "home_team": "E", "away_team": "F", "best_pick": "Under 7.5",
         "Pick_Status": "Actionable", "effective_expected_value": 0.05,
         "effective_win_probability": 0.57, "effective_edge": 0.06,
         "consensus_agreement": "Agrees", "Kelly_Bet_Size": 8.0},
    ])
    card = build_all_games_lean_card(df)
    assert list(card["Tier"]) == ["BET", "LEAN", "AVOID"]          # ordered by tier
    assert card.iloc[0]["Matchup"] == "F @ E"
    assert float(card.iloc[0]["Suggested_Stake"]) == 8.0            # stake only on BET
    assert float(card.iloc[1]["Suggested_Stake"]) == 0.0           # LEAN is a read, no stake
    assert float(card.iloc[2]["Suggested_Stake"]) == 0.0


def test_lean_card_empty_when_no_games():
    assert build_all_games_lean_card(pd.DataFrame()).empty
    assert build_all_games_lean_card(None).empty


def test_lean_card_handles_export_column_names():
    # Post-export rename uses Home/Away/WinProbability instead of home_team/expected_value.
    df = _df([
        {"League": "MLB", "Home": "A", "Away": "B", "best_pick": "Under 8.5",
         "Pick_Status": "Below Threshold", "expected_value": 0.02,
         "WinProbability": 0.52, "edge": 0.03, "consensus_agreement": "Agrees",
         "Kelly_Bet_Size": 0.0},
    ])
    card = build_all_games_lean_card(df)
    assert len(card) == 1
    assert card.iloc[0]["Tier"] == "LEAN"
    assert card.iloc[0]["Matchup"] == "B @ A"
