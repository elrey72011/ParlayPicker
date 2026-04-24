import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.prediction_engine import (
    PredictionEngine,
    VERTEX_FEATURE_COLUMNS,
    _validate_feature_schema,
)


class _DummyModel:
    def predict_proba(self, df: pd.DataFrame):
        probs = np.array([0.51] * 5 + [0.52] * 5, dtype=float)
        probs = probs[: len(df)]
        if len(probs) < len(df):
            probs = np.pad(probs, (0, len(df) - len(probs)), constant_values=0.51)
        return np.column_stack([1 - probs, probs])


def _build_input(rows: int = 10) -> pd.DataFrame:
    base = {col: 0.0 for col in VERTEX_FEATURE_COLUMNS}
    out = []
    for i in range(rows):
        row = dict(base)
        row.update(
            {
                "league": "NBA",
                "home_team": f"Team {i}",
                "away_team": f"Opp {i}",
                "game_date": "2026-04-24",
                "market_type": "h2h_home",
                "feature_stats_fallback": bool(i % 3 == 0),
            }
        )
        if i >= rows // 2:
            row[VERTEX_FEATURE_COLUMNS[0]] = 1.0
        out.append(row)
    return pd.DataFrame(out)


def test_pre_and_post_prediction_flatness_diagnostics_populate():
    engine = PredictionEngine(model_path="models/does_not_exist.json")
    engine.model = _DummyModel()
    engine.use_fallback = False
    df = _build_input(10)

    probs = engine.predict_batch(df)
    metrics = engine._last_metrics

    assert len(probs) == 10
    assert metrics["ml_input_row_count"] == 10
    assert metrics["ml_feature_column_count"] == len(VERTEX_FEATURE_COLUMNS)
    assert metrics["ml_zero_variance_feature_count"] >= 1
    assert metrics["ml_near_constant_feature_count"] >= metrics["ml_zero_variance_feature_count"]
    assert metrics["rows_with_duplicate_feature_signature"] >= 1
    assert "raw_prediction_distribution" in metrics
    assert metrics["raw_prediction_distribution"]["unique_count"] >= 1
    assert "ml_flatness_root_cause_hint" in metrics


def test_duplicate_feature_row_detection_reports_ratio():
    engine = PredictionEngine(model_path="models/does_not_exist.json")
    engine.model = _DummyModel()
    engine.use_fallback = False
    df = _build_input(8)
    df.loc[:, VERTEX_FEATURE_COLUMNS] = 0.0

    engine.predict_batch(df)
    metrics = engine._last_metrics
    assert metrics["rows_with_duplicate_feature_signature"] >= 6
    assert metrics["duplicate_feature_row_ratio"] > 0.5
    assert isinstance(metrics["top_duplicate_feature_signatures"], list)


def test_schema_mismatch_is_detected_clearly():
    with pytest.raises(ValueError, match="Feature schema/order mismatch"):
        _validate_feature_schema(["a", "b", "c"], ["a", "c", "b"])

