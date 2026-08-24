import pandas as pd

from core.streamlit_pipeline import PIPELINE_BUILD, ensure_best_pick_export_columns


def test_best_picks_export_identity_columns_present_without_regression():
    base = pd.DataFrame([
        {"league": "MLB", "home_team": "A", "away_team": "B", "game_date": "2026-05-01", "best_pick": "Over 8.5", "market_type": "total_over", "market_line_used": 8.5, "market_line_source": "live"}
    ])
    out = ensure_best_pick_export_columns(base)
    for col in ["export_run_id", "pick_id", "canonical_pick_key", "market_line_used", "market_line_source", "line_consistency_flag", "line_event_identity_match_flag", "recent_regime_penalty_applied", "recent_regime_penalty_value", "recent_regime_penalty_reason"]:
        assert col in out.columns
    assert out.loc[0, "pick_id"].startswith("pick_")
    assert "mlb::a::b" in out.loc[0, "canonical_pick_key"]


def test_pipeline_build_identifies_current_export_contract():
    assert PIPELINE_BUILD == "2026-08-24b-precision-final-score"
