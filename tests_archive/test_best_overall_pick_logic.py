import pandas as pd
import pytest
from app_core.new_summary_logic import build_game_summary_v2

def test_ml_fallback_no_spread_or_total():
    """
    Test that if a game only has a Moneyline market with probability > 0.50,
    it successfully outputs the ML pick as the Best Overall Pick.
    """
    data = [{
        "league": "NBA",
        "Home": "Lakers",
        "Away": "Warriors",
        "Commence (UTC)": "2024-03-01T00:00:00Z",
        "Market": "Moneyline",
        "Pick": "Lakers",
        "final_probability": 0.60,
        "Pick_Confidence": "LOW" # Doesn't matter for ML
    }]
    df = pd.DataFrame(data)
    summary = build_game_summary_v2(df)

    assert not summary.empty
    # Best Overall Pick should be the ML pick
    assert summary["Best Overall Pick"].iloc[0] == "Lakers"
    assert summary["Best Overall Prob"].iloc[0] == 0.60
    # Make sure ML pick and prob are populated correctly
    assert summary["ML Pick"].iloc[0] == "Lakers"
    assert summary["ML Prob"].iloc[0] == 0.60

def test_ml_fallback_fails_if_prob_too_low():
    """
    Test that if a game only has a Moneyline market with probability <= 0.50,
    it does NOT output it as the Best Overall Pick.
    """
    data = [{
        "league": "NBA",
        "Home": "Lakers",
        "Away": "Warriors",
        "Commence (UTC)": "2024-03-01T00:00:00Z",
        "Market": "Moneyline",
        "Pick": "Lakers",
        "final_probability": 0.49,
    }]
    df = pd.DataFrame(data)
    summary = build_game_summary_v2(df)

    assert not summary.empty
    assert summary["Best Overall Pick"].iloc[0] is None
    assert pd.isna(summary["Best Overall Prob"].iloc[0]) or summary["Best Overall Prob"].iloc[0] is None

def test_low_confidence_spread_is_accepted():
    """
    Test that a Spread pick with LOW confidence but > 0.50 probability is selected.
    """
    data = [{
        "league": "NBA",
        "Home": "Lakers",
        "Away": "Warriors",
        "Commence (UTC)": "2024-03-01T00:00:00Z",
        "Market": "Spread",
        "Pick": "Lakers",
        "Line": -5.5,
        "spread_pick_label": "Lakers -5.5",
        "final_probability": 0.55,
        "spread_prob_pick_final": 0.55,
        "Pick_Confidence": "LOW"
    }]
    df = pd.DataFrame(data)
    summary = build_game_summary_v2(df)

    assert not summary.empty
    assert summary["Best Overall Pick"].iloc[0] == "Lakers -5.5"
    assert summary["Best Overall Prob"].iloc[0] == 0.55

def test_no_valid_spread_or_total_warning_does_not_skip_ml():
    """
    Test that the "no_valid_spread_or_total" warning no longer skips the entire game.
    If it has a valid ML pick, it should fall back to it.
    """
    data = [
        {
            "league": "NBA",
            "Home": "Lakers",
            "Away": "Warriors",
            "Commence (UTC)": "2024-03-01T00:00:00Z",
            "Market": "Spread",
            "Warnings": "no_valid_spread_or_total",
            "final_probability": 0.0,
            "Pick_Confidence": "LOW"
        },
        {
            "league": "NBA",
            "Home": "Lakers",
            "Away": "Warriors",
            "Commence (UTC)": "2024-03-01T00:00:00Z",
            "Market": "Moneyline",
            "Pick": "Warriors",
            "final_probability": 0.58,
        }
    ]
    df = pd.DataFrame(data)
    summary = build_game_summary_v2(df)

    assert not summary.empty
    assert summary["Best Overall Pick"].iloc[0] == "Warriors"
    assert summary["Best Overall Prob"].iloc[0] == 0.58
