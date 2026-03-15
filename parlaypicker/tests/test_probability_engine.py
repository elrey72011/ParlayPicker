import pytest
from parlaypicker.core.probability_engine import american_to_prob, remove_vig, ensemble_probability, calibrated_probability


def test_american_to_prob():
    assert round(american_to_prob(-110), 3) == 0.524
    assert american_to_prob(0) == 0.5
    assert american_to_prob(-100) == 0.5
    assert american_to_prob(100) == 0.5


def test_remove_vig():
    h, a = remove_vig(0.55, 0.55)
    assert round(h + a, 5) == 1.0


def test_remove_vig_zero_and_negative():
    assert remove_vig(0.0, 0.0) == (0.0, 0.0)
    h, a = remove_vig(-0.5, 1.0)
    assert pytest.approx(h + a, abs=1e-6) == 1.0


def test_ensemble_probability():
    row = {"market_prob": 0.5, "ml_prob": 0.6, "kalshi_prob": 0.55, "theover_prob": 0.5, "sentiment_prob": 0.5}
    val = ensemble_probability(row)
    assert 0 < val < 1


def test_calibrated_probability():
    assert calibrated_probability(1.2) == 0.99
    assert calibrated_probability(0.0) == 0.01
    assert calibrated_probability(1.0) == 0.99
    assert calibrated_probability(-10) == 0.01
