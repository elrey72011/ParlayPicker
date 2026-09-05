import pandas as pd
import pytest

from core.streamlit_pipeline import (
    _normalize_complementary_selection_probabilities,
    _selection_bucket_stats_are_fresh,
)
from core.empirical_tiers import (
    bucket_stats_are_fresh,
    recent_regime_stats_are_fresh,
)


def test_complementary_rank_probabilities_are_coherent_without_touching_order():
    frame = pd.DataFrame(
        {
            "matchup_id": ["game-1", "game-1"],
            "market_type": ["total_over", "total_under"],
            "total_line": [8.5, 8.5],
            "spread_line": [pd.NA, pd.NA],
            "_selection_probability": [0.48, 0.32],
        }
    )

    probability, normalized = _normalize_complementary_selection_probabilities(frame)

    assert normalized.all()
    assert float(probability.sum()) == pytest.approx(1.0)
    assert float(probability.iloc[0]) == pytest.approx(0.60)
    assert float(probability.iloc[1]) == pytest.approx(0.40)


def test_probability_normalization_skips_mismatched_market_lines():
    frame = pd.DataFrame(
        {
            "matchup_id": ["game-1", "game-1"],
            "market_type": ["total_over", "total_under"],
            "total_line": [8.5, 19.5],
            "spread_line": [pd.NA, pd.NA],
            "_selection_probability": [0.48, 0.32],
        }
    )

    probability, normalized = _normalize_complementary_selection_probabilities(frame)

    assert not normalized.any()
    assert probability.tolist() == [0.48, 0.32]


def test_dated_empirical_overlay_fails_closed_when_stale():
    now = pd.Timestamp("2026-07-30", tz="UTC")
    fresh = {"meta": {"fitted_on": "2026-07-25"}}
    stale = {"meta": {"fitted_on": "2026-07-09"}}
    future = {"meta": {"fitted_on": "2026-08-01"}}

    assert _selection_bucket_stats_are_fresh(fresh, now=now, max_age_days=14)
    assert not _selection_bucket_stats_are_fresh(stale, now=now, max_age_days=14)
    assert _selection_bucket_stats_are_fresh({"overall": {"n": 10}}, now=now)
    assert bucket_stats_are_fresh(fresh, now=now, max_age_days=14)
    assert not bucket_stats_are_fresh(stale, now=now, max_age_days=14)
    assert not bucket_stats_are_fresh(future, now=now, max_age_days=14)


def test_empirical_freshness_uses_newest_graded_slate_not_file_write_date():
    now = pd.Timestamp("2026-07-30", tz="UTC")
    rewritten_without_new_results = {
        "meta": {
            "fitted_on": "2026-07-30",
            "recency_anchor": "2026-07-10",
        }
    }

    assert not bucket_stats_are_fresh(
        rewritten_without_new_results,
        now=now,
        max_age_days=14,
    )


def test_recent_regime_window_expires_before_long_horizon_overlay():
    stats = {
        "meta": {
            "fitted_on": "2026-07-25",
            "recency_anchor": "2026-07-24",
        }
    }

    assert bucket_stats_are_fresh(
        stats,
        now=pd.Timestamp("2026-07-27", tz="UTC"),
    )
    assert not recent_regime_stats_are_fresh(
        stats,
        now=pd.Timestamp("2026-07-27", tz="UTC"),
    )
    assert recent_regime_stats_are_fresh(
        stats,
        now=pd.Timestamp("2026-07-25", tz="UTC"),
    )
