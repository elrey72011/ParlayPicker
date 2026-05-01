import pandas as pd

from core.streamlit_pipeline import _string_series, optimize_portfolio_allocation


def test_string_series_handles_categorical_with_missing_values():
    df = pd.DataFrame({"Pick_Status": pd.Categorical(["Actionable", None, "No Play"])})
    out = _string_series(df, "Pick_Status")
    assert str(out.dtype) == "string"
    assert out.tolist() == ["Actionable", "", "No Play"]


def test_string_series_handles_categorical_default_not_in_categories():
    df = pd.DataFrame({"Pick_Status": pd.Categorical(["Actionable", None], categories=["Actionable", "No Play"])})
    out = _string_series(df, "Pick_Status", default="")
    assert out.iloc[1] == ""


def test_optimize_portfolio_allocation_handles_categorical_pick_status():
    df = pd.DataFrame([
        {"league": "MLB", "home_team": "A", "away_team": "B", "best_pick": "Over 8.5", "Pick_Status": "Actionable", "line_consistency_flag": True, "line_event_identity_match_flag": True, "market_line_source": "live", "line_provenance_warning": "", "market_line_used": 8.5, "calibrated_probability": 0.6, "decimal_odds": 2.0, "expected_value": 0.1, "edge": 0.1},
        {"league": "MLB", "home_team": "C", "away_team": "D", "best_pick": "Over 7.5", "Pick_Status": "No Play", "line_consistency_flag": True, "line_event_identity_match_flag": True, "market_line_source": "live", "line_provenance_warning": "", "market_line_used": 7.5, "calibrated_probability": 0.55, "decimal_odds": 2.0, "expected_value": 0.05, "edge": 0.05},
    ])
    df["Pick_Status"] = pd.Categorical(df["Pick_Status"], categories=["Actionable", "No Play"])
    out = optimize_portfolio_allocation(df, bankroll=1000)
    assert not out.empty
    assert out.loc[out["Pick_Status"] == "No Play", "production_bet_amount"].iloc[0] == 0
    assert out["production_bet_amount"].max() <= 40.0 + 1e-9


def test_string_series_object_column_regression():
    df = pd.DataFrame({"Pick_Status": ["Actionable", None]})
    out = _string_series(df, "Pick_Status")
    assert out.tolist() == ["Actionable", ""]
