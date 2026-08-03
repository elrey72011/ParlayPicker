"""Regression: realized bucket evidence must choose the game finalist, not only tier it."""
from __future__ import annotations

import pandas as pd

from core.streamlit_pipeline import build_best_picks_df
from core.empirical_tiers import empirical_selection_probabilities


def _candidate(market_type: str, probability: float, ev: float) -> dict:
    return {
        "game_id": "empirical-selection",
        "league": "MLB",
        "home_team": "Chicago Cubs",
        "away_team": "Pittsburgh Pirates",
        "game_date": pd.Timestamp("2026-07-25", tz="UTC"),
        "market_type": market_type,
        "total_line": 7.5,
        "spread_line": pd.NA,
        "calibrated_probability": probability,
        "model_probability": probability,
        "ml_probability": probability,
        "expected_value": ev,
        "edge": ev,
        "market_probability": 0.50,
        "kalshi_probability": 0.50,
        "consensus_agreement": "Neutral",
        "odds_american": -110,
        "odds_source": "test",
        "line_source": "live",
        "live_total_line": 7.5,
        "is_live_data": True,
        "used_stale_features": False,
    }


def test_empirical_bucket_blend_cannot_overturn_a_clearly_stronger_forecast(monkeypatch):
    stats = {
        "overall": {"n": 200, "win_rate": 0.50},
        "buckets": {
            "MLB:over:Neutral": {"n": 40, "wins": 28, "win_rate": 0.70},
            "MLB:under:Neutral": {"n": 40, "wins": 10, "win_rate": 0.25},
        },
    }
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: stats)
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    analysis = pd.DataFrame([
        _candidate("total_over", probability=0.60, ev=0.05),
        _candidate("total_under", probability=0.68, ev=0.12),
    ])
    diagnostics = {}
    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert len(best) == 1
    # Bucket history is evidence, not a replacement model. Even an extreme
    # 28-10 directional split may not overturn an eight-point forecast gap.
    assert best.iloc[0]["market_type"] == "total_under"
    assert best.iloc[0]["selection_probability_source"] == "empirical_bucket_blend"
    assert diagnostics["empirical_selection_candidate_count"] == 2

    candidates = analysis.copy()
    from core.empirical_tiers import empirical_selection_probabilities

    blended = empirical_selection_probabilities(candidates, stats)
    raw = pd.to_numeric(candidates["calibrated_probability"])
    movement = (blended - raw).abs()
    assert movement.gt(0).all()
    assert movement.max() < 0.04


def test_stale_bucket_stats_cannot_label_after_being_rejected_for_selection(monkeypatch):
    stale = {
        "overall": {"n": 200, "win_rate": 0.50},
        "buckets": {
            "MLB:over:Neutral": {"n": 80, "wins": 60, "win_rate": 0.75},
            "MLB:under:Neutral": {"n": 80, "wins": 20, "win_rate": 0.25},
        },
        "meta": {"fitted_on": "2026-01-01"},
    }
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: stale)
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    diagnostics = {}
    build_best_picks_df(
        pd.DataFrame(
            [
                _candidate("total_over", probability=0.58, ev=0.05),
                _candidate("total_under", probability=0.54, ev=0.02),
            ]
        ),
        diagnostics_out=diagnostics,
    )

    assert diagnostics["selection_bucket_stats_fresh"] is False
    assert diagnostics["empirical_tier_overlay"]["applied"] is False
    assert diagnostics["empirical_tier_overlay"]["reason"] == "stale_bucket_stats"


def test_side_candidates_use_same_bounded_empirical_selection_scale_as_totals():
    stats = {
        "overall": {"n": 200, "win_rate": 0.50},
        "buckets": {
            "MLB:side:Agrees": {"n": 40, "wins": 28, "win_rate": 0.70},
        },
    }
    frame = pd.DataFrame([{
        "league": "MLB",
        "market_type": "spread_home",
        "calibrated_probability": 0.60,
        "consensus_agreement": "Agrees",
        "kalshi_probability": 0.55,
        "odds_american": -110,
    }])

    adjusted = empirical_selection_probabilities(frame, stats)

    assert adjusted.iloc[0] > 0.60
    assert adjusted.iloc[0] < 0.62
