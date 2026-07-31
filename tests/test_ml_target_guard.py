import pandas as pd
import pytest
import streamlit_app as app
import core.streamlit_pipeline as sp

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


def test_target_specific_market_probability_survives_home_win_guard():
    df = pd.DataFrame(
        {
            "market_type": ["total_over", "spread_home"],
            "ml_probability": [0.57, 0.54],
            "ml_target": ["total_over", "spread_cover"],
            "model_status": ["Market Score Model", "Market Score Model"],
        }
    )

    out = _retire_game_winner_model_from_unsupported_markets(df, {})

    assert out["ml_probability"].tolist() == pytest.approx([0.57, 0.54])
    assert out["model_status"].eq("Market Score Model").all()


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

def test_post_kalshi_total_reblend_does_not_require_or_double_count_ml(monkeypatch):
    captured = {}

    def fake_blend(**kwargs):
        captured.update(kwargs)
        return pd.Series([0.61])

    monkeypatch.setattr("core.streamlit_pipeline.compute_blended_probability", fake_blend)
    row = pd.DataFrame(
        {
            "best_pick": ["Over 6.5"],
            "odds_american": [-110],
            "market_probability": [0.50],
            "kalshi_probability": [pd.NA],
            "model_probability": [0.91],
            "theover_probability": [0.61],
            "ml_probability": [pd.NA],
            "sentiment_diff": [0.0],
            "league": ["NHL"],
            "market_type": ["total_over"],
        }
    )

    out = app._recompute_consensus_from_kalshi(row, require_ml=True)

    assert pd.isna(captured["p_ml"].iloc[0])
    assert captured["p_theover"].iloc[0] == pytest.approx(0.61)
    assert pd.isna(out["blend_in_ml"].iloc[0])


def test_post_kalshi_moneyline_still_requires_ml_when_enabled():
    row = pd.DataFrame(
        {
            "odds_american": [-110],
            "market_probability": [0.50],
            "kalshi_probability": [pd.NA],
            "theover_probability": [pd.NA],
            "ml_probability": [pd.NA],
            "sentiment_diff": [0.0],
            "league": ["MLB"],
            "market_type": ["moneyline_home"],
        }
    )

    with pytest.raises(ValueError, match="failed to merge"):
        app._recompute_consensus_from_kalshi(row, require_ml=True)


def test_ml_sync_is_market_specific_and_never_restores_totals():
    analysis = pd.DataFrame(
        {
            "league": ["MLB", "MLB"],
            "home_team": ["Boston Red Sox", "Boston Red Sox"],
            "away_team": ["New York Yankees", "New York Yankees"],
            "game_date": ["2026-07-25", "2026-07-25"],
            "market_type": ["total_over", "moneyline_home"],
            "ml_probability": [pd.NA, pd.NA],
        }
    )
    best = pd.DataFrame(
        {
            "league": ["MLB", "MLB"],
            "home_team": ["Boston Red Sox", "Boston Red Sox"],
            "away_team": ["New York Yankees", "New York Yankees"],
            "game_date": ["2026-07-25", "2026-07-25"],
            "market_type": ["total_over", "moneyline_home"],
            "ml_probability": [0.80, 0.62],
        }
    )

    out = app._sync_ml_probabilities(analysis, best)

    assert pd.isna(out.loc[out["market_type"].eq("total_over"), "ml_probability"]).all()
    assert out.loc[out["market_type"].eq("moneyline_home"), "ml_probability"].iloc[0] == pytest.approx(0.62)


def test_run_pipeline_all_totals_continues_when_ml_is_enabled(monkeypatch):
    captured = {}

    def fake_run_analysis_pipeline(**kwargs):
        analysis = pd.DataFrame(
            [
                {
                    "league": "MLB",
                    "home_team": "Boston Red Sox",
                    "away_team": "New York Yankees",
                    "game_date": "2026-07-25",
                    "market_type": "total_over",
                    "ml_probability": pd.NA,
                    "theover_probability": 0.61,
                    "market_probability": 0.50,
                    "odds_american": -110,
                    "expected_value": 0.10,
                    "edge": 0.05,
                    "calibrated_probability": 0.60,
                    "line_consistency_flag": True,
                    "line_event_identity_match_flag": True,
                    "market_line_source": "live",
                    "line_provenance_warning": "",
                    "total_line": 8.5,
                }
            ]
        )
        return analysis, analysis.copy(), {}

    def fake_build_best_picks_df(analysis_df, diagnostics_out=None):
        return pd.DataFrame(
            [
                {
                    "league": "MLB",
                    "home_team": "Boston Red Sox",
                    "away_team": "New York Yankees",
                    "game_date": "2026-07-25",
                    "market_type": "total_over",
                    "best_pick": "Over 8.5",
                    "Pick_Status": "Actionable",
                    "expected_value": 0.10,
                    "edge": 0.05,
                    "effective_expected_value": 0.10,
                    "effective_edge": 0.05,
                    "calibrated_probability": 0.60,
                    "line_consistency_flag": True,
                    "line_event_identity_match_flag": True,
                    "market_line_source": "live",
                    "line_provenance_warning": "",
                    "market_line_used": 8.5,
                }
            ]
        )

    def fake_recompute(df, require_ml=False):
        captured["require_ml"] = require_ml
        return df

    monkeypatch.setattr(app, "run_analysis_pipeline", fake_run_analysis_pipeline)
    monkeypatch.setattr(sp, "build_best_picks_df", fake_build_best_picks_df)
    monkeypatch.setattr(app, "_enrich_with_kalshi_safe", lambda df: (df, None))
    monkeypatch.setattr(app, "_recompute_consensus_from_kalshi", fake_recompute)
    monkeypatch.setattr(app, "generate_parlays", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "optimize_portfolio_allocation", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "run_bankroll_simulation", lambda *args, **kwargs: {})

    controls = {
        "sports": ["MLB"],
        "use_ml": True,
        "theover_spreads": None,
        "theover_totals": None,
        "bankroll": 1000.0,
        "use_gemini": False,
    }
    state, warnings, errors = app._run_pipeline(controls)

    assert errors == []
    assert not state["analysis_df"].empty
    assert captured["require_ml"] is False
    assert state["diagnostics"]["ml_eligible_rows"] == 0
    assert any("no target-specific spread/total model probabilities" in warning for warning in warnings)


def test_market_specific_targets_are_ml_eligible_without_moneylines():
    frame = pd.DataFrame(
        {
            "market_type": ["spread_home", "total_over", "total_under", "spread_away"],
            "ml_target": ["spread_cover", "total_over", "total_under", "home_win"],
        }
    )

    mask = app._ml_eligible_market_mask(frame)

    assert mask.tolist() == [True, True, True, False]


def test_ml_sync_restores_market_specific_probability_with_target_metadata():
    analysis = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["Boston Red Sox"],
            "away_team": ["New York Yankees"],
            "game_date": ["2026-07-25"],
            "market_type": ["total_over"],
            "ml_probability": [pd.NA],
            "ml_target": [pd.NA],
        }
    )
    best = pd.DataFrame(
        {
            "league": ["MLB"],
            "home_team": ["Boston Red Sox"],
            "away_team": ["New York Yankees"],
            "game_date": ["2026-07-25"],
            "market_type": ["total_over"],
            "ml_probability": [0.61],
            "ml_probability_source": ["score-distribution-v1:mlb"],
            "ml_target": ["total_over"],
        }
    )

    out = app._sync_ml_probabilities(analysis, best)

    assert out.loc[0, "ml_probability"] == pytest.approx(0.61)
    assert out.loc[0, "ml_target"] == "total_over"
    assert out.loc[0, "ml_probability_source"] == "score-distribution-v1:mlb"

