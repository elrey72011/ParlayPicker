import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import streamlit_pipeline as sp
from core.kelly_optimizer import kelly_fraction
from core.probability_engine import american_to_prob, remove_vig
from core.smart_parlay_engine import generate_smart_parlays
from parlaypicker.core.parlay_engine import parlay_probability
from parlaypicker.core.probability_engine import calibrated_probability


def test_header_mapping_mixed_case_to_canonical_columns():
    upload = pd.DataFrame(
        {
            "Sport": ["NBA"],
            "Home": ["Los Angeles Lakers"],
            "Away": ["Boston Celtics"],
            "Market Type": ["spread_home"],
            "WinProbability": [0.62],
            "Odds American": [-110],
            "Spread Line": [-3.5],
        }
    )

    out = sp._coerce_export_to_canonical(upload, selected_sports=["NBA"])

    assert list(out.columns) == sp.CANONICAL_BET_COLUMNS
    assert out.loc[0, "home_team"] == "Los Angeles Lakers"
    assert "Boston" in out.loc[0, "away_team"]
    assert out.loc[0, "theover_probability"] == 0.62


def test_coerce_export_permuted_columns_and_ignores_irrelevant_fields():
    upload = pd.DataFrame(
        {
            "Away Team": ["Team B"],
            "Some Irrelevant Col": ["ignore"],
            "Home Team": ["Team A"],
            "Sport": ["NBA"],
            "Market Type": ["total_over"],
            "Line": [223.5],
            "Odds": [-110],
            "Win Probability": [0.55],
        }
    )

    out = sp._coerce_export_to_canonical(upload, selected_sports=["NBA"])

    assert out.shape[0] == 1
    assert set(sp.CANONICAL_BET_COLUMNS).issubset(set(out.columns))
    assert "Some Irrelevant Col" not in out.columns


def test_normalize_upload_columns_random_case_and_symbols():
    upload = pd.DataFrame({"Home-Team": ["A"], "AWAY_team": ["B"], "Win Probability!!": [0.55]})
    normalized = sp._normalize_upload_columns(upload)
    assert list(normalized.columns) == ["home team", "away team", "win probability"]


def test_rounding_ev_and_edge_are_consistent_to_two_decimals():
    calibrated = 0.55
    market = 0.50
    decimal_odds = sp.american_to_decimal(-110)
    ev = calibrated * (decimal_odds - 1) - (1 - calibrated)
    edge = calibrated - market

    assert round(ev, 2) == 0.05
    assert round(edge, 2) == 0.05


def test_apply_analysis_calculations_rounds_ev_and_edge_to_four_decimals():
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["A"],
            "away_team": ["B"],
            "market_type": ["spread_home"],
            "odds_american": [-137],
            "theover_probability": [0.59337],
        }
    )
    out = sp._apply_analysis_calculations(df)

    ev = out.loc[0, "expected_value"]
    edge = out.loc[0, "edge"]
    assert np.isclose(ev, round(ev, 4))
    assert np.isclose(edge, round(edge, 4))


def test_zero_odds_convention_pipeline_does_not_crash():
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["A"],
            "away_team": ["B"],
            "market_type": ["spread_home"],
            "odds_american": [0],
            "theover_probability": [0.55],
        }
    )

    out = sp._apply_analysis_calculations(df)

    assert sp.american_to_decimal(0) == 2.0
    assert american_to_prob(0) == 0.5
    assert out.loc[0, "market_probability"] == 0.5


def test_remove_vig_edge_cases():
    assert remove_vig(0, 0) == (0.0, 0.0)
    home, away = remove_vig(1.2, -0.2)
    assert np.isclose(home, 1.2)
    assert np.isclose(away, -0.2)


def test_american_odds_edge_cases():
    assert american_to_prob(0) == 0.5
    assert american_to_prob(-100) == 0.5
    assert american_to_prob(100) == 0.5


def test_decimal_conversion_edge_cases():
    assert sp.american_to_decimal(-100) == 2.0
    assert sp.american_to_decimal(100) == 2.0
    assert sp.american_to_decimal(None) == 1.9091


def test_expected_value_with_positive_and_negative_odds():
    ev_pos = 0.5 * (sp.american_to_decimal(200) - 1) - (1 - 0.5)
    ev_neg = 0.5 * (sp.american_to_decimal(-200) - 1) - (1 - 0.5)
    assert round(ev_pos, 4) == 0.5
    assert round(ev_neg, 4) == -0.25


def test_parlay_probability_and_generate_smart_parlays_formula():
    assert parlay_probability([0.5, 0.6]) == 0.3
    assert parlay_probability([]) == 1.0
    assert parlay_probability([1.0, 0.0, 0.7]) == 0.0

    bets = pd.DataFrame(
        {
            "best_pick": ["Leg1", "Leg2", "Leg3"],
            "edge": [0.07, 0.06, -0.02],
            "calibrated_probability": [0.6, 0.6, 0.4],
            "decimal_odds": [2.0, 2.0, 1.5],
        }
    )
    parlays = generate_smart_parlays(bets)

    assert not parlays.empty
    top = parlays.iloc[0]
    expected_prob = 0.6 * 0.6
    expected_decimal = 2.0 * 2.0
    expected_ev = (expected_prob * (expected_decimal - 1)) - (1 - expected_prob)
    assert np.isclose(top["combined_probability"], expected_prob)
    assert np.isclose(top["combined_decimal_odds"], expected_decimal)
    assert np.isclose(top["parlay_ev"], expected_ev)


def test_kelly_fraction_edge_cases():
    assert kelly_fraction(np.nan, 2.0) == 0.0
    assert kelly_fraction(0.55, np.nan) == 0.0
    assert kelly_fraction(0.55, 1.0) == 0.0
    assert kelly_fraction(-0.10, 2.0) == 0.0
    assert kelly_fraction(1.0, 2.0) == 1.0
    assert kelly_fraction(0.5, 1.5) == 0.0


def test_compute_blended_probability_market_ml_only_and_with_kalshi():
    market = pd.Series([0.50, 0.50])
    kalshi = pd.Series([np.nan, 0.70])
    ml = pd.Series([0.60, 0.60])

    out = sp.compute_blended_probability(market, kalshi, ml)

    assert np.isclose(out.iloc[0], 0.75 * 0.50 + 0.25 * 0.60)
    consensus = (0.50 + 0.70) / 2
    assert np.isclose(out.iloc[1], 0.75 * consensus + 0.25 * 0.60)


def test_build_best_picks_df_selects_best_per_game_across_markets_and_has_schema():
    analysis_df = pd.DataFrame(
        {
            "league": ["NBA", "NBA", "NBA", "NBA"],
            "home_team": ["A", "A", "C", "C"],
            "away_team": ["B", "B", "D", "D"],
            "game_date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"], utc=True),
            "game_time_est": ["", "", "", ""],
            "market_type": ["spread_home", "total_over", "spread_home", "total_under"],
            "best_pick": ["A -3.5", "Over 220.5", "C -2.5", "Under 215.5"],
            "expected_value": [0.03, 0.08, 0.02, 0.01],
            "edge": [0.02, 0.07, 0.01, 0.005],
            "calibrated_probability": [0.52, 0.57, 0.51, 0.505],
            "market_probability": [0.50, 0.50, 0.50, 0.50],
            "ml_probability": [0.51, 0.55, 0.50, 0.50],
            "odds_american": [-110, -110, -110, -110],
            "odds_source": ["uploaded", "uploaded", "uploaded", "uploaded"],
            "consensus_agreement": ["⚪ No Kalshi", "⚪ No Kalshi", "⚪ No Kalshi", "⚪ No Kalshi"],
            "kalshi_probability": [np.nan, np.nan, np.nan, np.nan],
            "kalshi_match_status": ["", "", "", ""],
            "kalshi_match_reason": ["", "", "", ""],
            "has_signal_probability": [True, True, True, True],
        }
    )

    best = sp.build_best_picks_df(analysis_df)

    assert len(best) == 2
    game1 = best[(best["home_team"] == "A") & (best["away_team"] == "B")].iloc[0]
    assert game1["market_type"] == "total_over"
    for col in sp.BEST_PICK_COLUMNS:
        assert col in best.columns




def test_calibrated_probability_boundaries():
    assert calibrated_probability(0.0) == 0.01
    assert calibrated_probability(1.0) == 0.99
    assert calibrated_probability(-10) == 0.01

def test_prob_boundary_inputs_for_ev_formula():
    decimal = sp.american_to_decimal(-110)
    ev_zero = 0.0 * (decimal - 1) - (1 - 0.0)
    ev_one = 1.0 * (decimal - 1) - (1 - 1.0)

    assert np.isclose(ev_zero, -1.0)
    assert ev_one > 0
