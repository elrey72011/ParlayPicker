import pandas as pd
import pytest

from core.streamlit_pipeline import (
    _apply_analysis_calculations,
    _retire_game_winner_model_from_unsupported_markets,
)


def test_home_win_model_is_retired_from_totals_and_spreads_only():
    df = pd.DataFrame(
        {
            "market_type": ["total_over", "spread_home", "moneyline_home"],
            "ml_probability": [0.64, 0.61, 0.58],
            "model_status": ["OK", "OK", "OK"],
        }
    )
    diagnostics = {}

    out = _retire_game_winner_model_from_unsupported_markets(df, diagnostics)

    assert pd.isna(out.loc[0, "ml_probability"])
    assert pd.isna(out.loc[1, "ml_probability"])
    assert out.loc[2, "ml_probability"] == 0.58
    assert out.loc[0, "model_status"] == "Unsupported Target: home-win model"
    assert diagnostics["ml_target_mismatch_rows"] == 2
    assert diagnostics["ml_totals_retired_rows"] == 1
    assert diagnostics["ml_spreads_retired_rows"] == 1


def test_missing_ml_does_not_double_count_theover_in_blend(monkeypatch):
    captured = {}

    def fake_blend(**kwargs):
        captured.update(kwargs)
        return pd.Series([0.61])

    monkeypatch.setattr("core.streamlit_pipeline.compute_blended_probability", fake_blend)
    row = pd.DataFrame(
        {
            "odds_american": [-110],
            "theover_probability": [0.61],
            "ml_probability": [pd.NA],
            "kalshi_probability": [pd.NA],
            "sentiment_diff": [0.0],
            "league": ["MLB"],
            "market_type": ["total_over"],
        }
    )

    _apply_analysis_calculations(row)

    assert pd.isna(captured["p_ml"].iloc[0])
    assert captured["p_theover"].iloc[0] == pytest.approx(0.61)
