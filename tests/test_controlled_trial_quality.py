from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from app_core.controlled_trial import evaluate_candidates, select_review_candidates


NOW = datetime(2026, 9, 21, 19, 50, tzinfo=timezone.utc)


def row(**updates):
    value = {
        "candidate_id": "quality-check",
        "game_id": "quality-game",
        "league": "MLB",
        "market_type": "spread_home",
        "best_pick": "Home -1.5",
        "spread_line": -1.5,
        "odds_american": 167,
        "calibrated_probability": 0.40,
        "expected_value": 0.068,
        "game_start_utc": (NOW + timedelta(hours=2)).isoformat(),
        "odds_recorded_at": (NOW - timedelta(minutes=2)).isoformat(),
        "quote_binding_verified": True,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "critical_feature_error": False,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    "update",
    [
        {"used_stale_features": True},
        {"model_status": "statistical fallback"},
        {"stats_quality": "FALLBACK"},
        {"data_quality_status": "UNVERIFIED"},
    ],
)
def test_explicit_stale_or_fallback_quality_never_reaches_gemini(update):
    frame = pd.DataFrame([row(**update)])
    assert select_review_candidates(frame, now=NOW).empty
    assert "degraded or critical input state" in evaluate_candidates(frame, now=NOW).iloc[0]["controlled_trial_gate_reason"]
