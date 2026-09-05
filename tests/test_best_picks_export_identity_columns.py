import pandas as pd
import pytest

from core.streamlit_pipeline import PIPELINE_BUILD, ensure_best_pick_export_columns


def test_best_picks_export_identity_columns_present_without_regression():
    base = pd.DataFrame([
        {"league": "MLB", "home_team": "A", "away_team": "B", "game_date": "2026-05-01", "best_pick": "Over 8.5", "market_type": "total_over", "market_line_used": 8.5, "market_line_source": "live"}
    ])
    out = ensure_best_pick_export_columns(base)
    for col in ["export_run_id", "pick_id", "canonical_pick_key", "odds_feed_source", "market_line_used", "market_line_source", "line_consistency_flag", "line_event_identity_match_flag", "recent_regime_penalty_applied", "recent_regime_penalty_value", "recent_regime_penalty_reason"]:
        assert col in out.columns
    assert out.loc[0, "pick_id"].startswith("pick_")
    assert "mlb::a::b" in out.loc[0, "canonical_pick_key"]


def test_pipeline_build_identifies_current_export_contract():
    assert PIPELINE_BUILD == "2026-09-02d-gemini-prop-batching"


@pytest.mark.parametrize("column", [
    "best_available_selection_verified", "best_available_ranking_verified",
    "final_pick_valid", "qualified_pick", "controlled_card_recovery",
    "sellable_as_premium", "sellable_as_value_card", "best_available_only",
    "selection_probability_pair_normalized", "mlb_spread_finalist_penalty_applied",
    "recent_regime_penalty_applied", "best_available_value_override_applied",
    "degraded_feature_subset_flag", "totals_only_actionable_flag",
    "line_consistency_flag", "line_event_identity_match_flag",
])
def test_export_boolean_text_preserves_validation_and_qualification(column):
    values = ["False", " false ", "0", "no", "True", " YES ", "1", False, True, 0.0, 1.0, "unknown", ""]
    result = ensure_best_pick_export_columns(
        pd.DataFrame({column: values}), required_columns=[column]
    )
    assert result[column].tolist() == [False, False, False, False, True, True, True, False, True, False, True, False, False]


@pytest.mark.parametrize("dtype", ["object", "string", "boolean", "category"])
def test_export_boolean_nulls_keep_existing_column_defaults(dtype):
    values = pd.Series([True, False, None], dtype=dtype, index=[4, 8, 12])
    columns = ["line_consistency_flag", "best_available_only", "qualified_pick"]
    result = ensure_best_pick_export_columns(
        pd.DataFrame({column: values for column in columns}), required_columns=columns
    )
    assert result.index.tolist() == [4, 8, 12]
    assert result.line_consistency_flag.tolist() == [True, False, True]
    assert result.best_available_only.tolist() == [True, False, True]
    assert result.qualified_pick.tolist() == [True, False, False]
