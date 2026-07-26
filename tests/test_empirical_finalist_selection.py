"""Regression: realized bucket evidence must choose the game finalist, not only tier it."""
from __future__ import annotations

import pandas as pd

from core.streamlit_pipeline import build_best_picks_df


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


def test_empirical_bucket_can_replace_higher_raw_probability_direction(monkeypatch):
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
    assert best.iloc[0]["market_type"] == "total_over"
    assert best.iloc[0]["selection_probability_source"] == "empirical_bucket_calibrated"
    assert diagnostics["empirical_selection_candidate_count"] == 2
