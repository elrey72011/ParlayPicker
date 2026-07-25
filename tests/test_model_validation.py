import pytest

from core.model_validation import compare_candidate_to_market, select_best_candidate


def test_candidate_selection_uses_proper_scores_not_uniqueness():
    candidates = {
        "unique_but_bad": {
            "ratio": 1.0,
            "metrics": {"ll": 0.61, "brier": 0.22, "auc": 0.73},
        },
        "accurate": {
            "ratio": 0.50,
            "metrics": {"ll": 0.54, "brier": 0.18, "auc": 0.78},
        },
    }

    assert select_best_candidate(candidates) == "accurate"


def test_candidate_that_does_not_beat_market_is_blocked():
    result = compare_candidate_to_market(
        {"ll": 0.60, "brier": 0.21, "auc": 0.74},
        {"ll": 0.56, "brier": 0.18, "auc": 0.78},
    )

    assert result["promotable"] is False
    assert any("log loss" in reason for reason in result["reasons"])
    assert any("Brier" in reason for reason in result["reasons"])


def test_candidate_that_beats_market_on_proper_scores_can_promote():
    result = compare_candidate_to_market(
        {"ll": 0.52, "brier": 0.16, "auc": 0.80},
        {"ll": 0.56, "brier": 0.18, "auc": 0.78},
    )

    assert result["promotable"] is True
    assert result["log_loss_improvement"] == pytest.approx(0.04)
    assert result["brier_improvement"] == pytest.approx(0.02)
