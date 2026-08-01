import pandas as pd

from scripts.fit_calibration import validate_calibration_promotion


def test_calibration_promotion_uses_future_holdout_and_beats_baselines():
    dates = pd.date_range("2025-01-01", periods=200, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "slate_date": dates,
            "prob": ([0.20, 0.80] * 100),
            "win": ([0, 1] * 100),
        }
    )

    result = validate_calibration_promotion(frame, min_train_rows=100)

    assert result["promotable"] is True
    assert result["train_rows"] == 160
    assert result["test_rows"] == 40
    assert result["train_end"] < result["test_start"]


def test_calibration_promotion_rejects_missing_dates():
    frame = pd.DataFrame({"prob": [0.6, 0.4], "win": [1, 0]})
    result = validate_calibration_promotion(frame, min_train_rows=1)
    assert result["promotable"] is False


def test_calibration_promotion_rejects_insufficient_training_history():
    frame = pd.DataFrame(
        {
            "slate_date": pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC"),
            "prob": [0.6, 0.4] * 5,
            "win": [1, 0] * 5,
        }
    )

    result = validate_calibration_promotion(frame, min_train_rows=100)

    assert result["promotable"] is False
    assert "training rows" in result["reason"]
