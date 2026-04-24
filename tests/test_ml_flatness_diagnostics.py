import pandas as pd

from core.streamlit_pipeline import _compute_ml_input_flatness_diagnostics


def test_flatness_diagnostics_populate_for_low_variance_inputs():
    df = pd.DataFrame(
        {
            "feature_home_win_pct": [0.5, 0.5, 0.5, 0.5],
            "feature_away_win_pct": [0.5, 0.5, 0.5, 0.5],
            "feature_diff_ppg": [0.0, 0.0, 0.0, 0.0],
            "feature_home_ppg": [100.0, 100.0, 100.0, 100.0],
            "feature_away_ppg": [100.0, 100.0, 100.0, 100.0],
        }
    )
    eligible = pd.Series([True, True, True, False], index=df.index)
    out = _compute_ml_input_flatness_diagnostics(df, eligible)

    assert out["ml_input_row_count"] == 4
    assert out["ml_feature_eligible_row_count"] == 3
    assert out["ml_rows_excluded_count"] == 1
    assert out["ml_zero_variance_feature_count"] >= 4
    assert out["ml_near_constant_feature_count"] >= out["ml_zero_variance_feature_count"]
    assert isinstance(out["ml_top_feature_nunique"], dict)
    assert isinstance(out["ml_top_missing_features"], dict)
    assert out["ml_flatness_root_cause_hint"] in {
        "too_few_ml_eligible_rows",
        "many_zero_variance_features",
        "many_near_constant_features",
    }

