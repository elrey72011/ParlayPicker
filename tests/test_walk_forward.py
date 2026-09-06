import unittest
import pandas as pd

from core.walk_forward import chronological_split, probability_metrics, walk_forward_evaluate


class TestWalkForward(unittest.TestCase):
    def test_split_is_chronological_without_shuffle(self):
        frame = pd.DataFrame({
            "date": ["2026-01-03", "2026-01-01", "2026-01-02", "2026-01-04", "2026-01-05"],
            "p": [0.9, 0.6, 0.7, 0.8, 0.5],
            "y": [1, 0, 1, 0, 1],
        })
        train, test = chronological_split(frame, "date", test_fraction=0.4, min_train_rows=2)
        self.assertEqual(train["date"].tolist(), ["2026-01-01", "2026-01-02", "2026-01-03"])
        self.assertEqual(test["date"].tolist(), ["2026-01-04", "2026-01-05"])

    def test_metrics_include_proper_scores_and_calibration(self):
        result = probability_metrics([0.9, 0.1], [1, 0])
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["brier"], 0.01)
        self.assertTrue(result["log_loss"] < 0.2)
        self.assertEqual(len(result["calibration"]), 2)

    def test_holdout_does_not_claim_unverified_training_provenance(self):
        frame = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=10, freq="D"),
            "p": [0.6] * 10,
            "y": [1, 0] * 5,
        })
        result = walk_forward_evaluate(frame, "date", "p", "y", min_train_rows=6, test_fraction=0.4)
        self.assertFalse(result["out_of_sample"])
        self.assertTrue(result["chronological_holdout"])
        self.assertEqual(result["train_rows"], 6)
        self.assertEqual(result["test_rows"], 4)
        self.assertLess(result["train_end"], result["test_start"])


if __name__ == "__main__":
    unittest.main()


def test_same_day_candidates_are_kept_together():
    frame = pd.DataFrame({"date": ["2026-01-01"] * 3 + ["2026-01-02"] * 4 + ["2026-01-03"] * 3})
    train, test = chronological_split(frame, "date", test_fraction=.5)
    assert set(train.date).isdisjoint(test.date)
    assert len(train) + len(test) == len(frame)


def test_single_slate_cannot_be_called_a_holdout():
    import pytest
    with pytest.raises(ValueError, match="distinct calendar slates"):
        chronological_split(pd.DataFrame({"date": ["2026-01-01"] * 10}), "date")
