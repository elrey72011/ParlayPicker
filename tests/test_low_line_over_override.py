"""Tests for the sub-8.0 MLB total_over guardrail carve-out.

The carve-out (app_core/low_line_override) lets a strong, Kalshi-aligned
low-line MLB over keep its Actionable status instead of being demoted by the
blanket low_line_over_guardrail. These tests lock that it:
  * KEEPS a strong Agrees over (the segment that went 7-2 / +45% ROI in backtest);
  * does NOT rescue the documented losers (May-20 CHC/MIL, SD/LAD: Neutral/
    Disagrees at low win prob);
  * never promotes a Below Threshold pick or a >= 8.0 line; and
  * is fully gated by the feature flag.
"""
import importlib

import pytest

import app_core.weights_config as wc
import app_core.low_line_override as llo
from app_core.low_line_override import low_line_over_override_applies


@pytest.fixture(autouse=True)
def _enable_carveout(monkeypatch):
    """The carve-out ships DISABLED after the 6 Jun low-line-over regression
    (MLB_LOW_LINE_OVER_OVERRIDE_ENABLED=False). The tests below exercise the
    predicate's conditional *logic*, which is only reachable when the flag is on,
    so force-enable it here. The shipped default-off behavior is locked separately
    in ``test_ships_disabled_by_default``."""
    monkeypatch.setattr(llo, "MLB_LOW_LINE_OVER_OVERRIDE_ENABLED", True)


def _strong_agrees(**overrides):
    """A sub-8.0 over resembling the backtest's winning Agrees segment
    (e.g. 30-May Detroit/CWS Over 7.5: eff win 0.681, edge 0.195, Agrees)."""
    base = dict(
        league="MLB",
        market_type="total_over",
        total_line=7.5,
        status="Actionable",
        consensus="Agrees",
        effective_win_probability=0.68,
        effective_edge=0.19,
    )
    base.update(overrides)
    return base


def test_strong_agrees_sub8_over_keeps_actionable():
    assert low_line_over_override_applies(**_strong_agrees()) is True


def test_threshold_boundaries():
    # Exactly at the floors qualifies; just under does not.
    assert low_line_over_override_applies(
        **_strong_agrees(effective_win_probability=0.62, effective_edge=0.08)
    ) is True
    assert low_line_over_override_applies(
        **_strong_agrees(effective_win_probability=0.619)
    ) is False
    assert low_line_over_override_applies(
        **_strong_agrees(effective_edge=0.079)
    ) is False


def test_documented_losers_are_not_rescued():
    # May-20 Milwaukee/Chicago Cubs Over 6.5 — Neutral, eff win 0.576.
    assert low_line_over_override_applies(
        **_strong_agrees(total_line=6.5, consensus="Neutral",
                         effective_win_probability=0.576, effective_edge=0.088)
    ) is False
    # May-20 LAD/SD Over 7.5 — Disagrees, eff win 0.531.
    assert low_line_over_override_applies(
        **_strong_agrees(consensus="Disagrees",
                         effective_win_probability=0.531, effective_edge=0.053)
    ) is False


def test_non_agrees_consensus_blocked():
    for consensus in ("Neutral", "Disagrees", "No Kalshi", ""):
        assert low_line_over_override_applies(**_strong_agrees(consensus=consensus)) is False


def test_only_protects_actionable_never_promotes():
    # The carve-out prevents a demotion; it must not upgrade a Below Threshold pick.
    assert low_line_over_override_applies(**_strong_agrees(status="Below Threshold")) is False


def test_line_at_or_above_cap_not_affected():
    assert low_line_over_override_applies(**_strong_agrees(total_line=8.0)) is False
    assert low_line_over_override_applies(**_strong_agrees(total_line=8.5)) is False


def test_only_mlb_total_over():
    assert low_line_over_override_applies(**_strong_agrees(league="NBA")) is False
    assert low_line_over_override_applies(**_strong_agrees(market_type="total_under")) is False


def test_missing_metrics_blocked():
    assert low_line_over_override_applies(**_strong_agrees(effective_win_probability=None)) is False
    assert low_line_over_override_applies(**_strong_agrees(effective_edge=None)) is False
    assert low_line_over_override_applies(**_strong_agrees(total_line=None)) is False


def test_feature_flag_disables_everything(monkeypatch):
    # Overrides the autouse-enabled flag: a disabled carve-out vetoes even a strong
    # Agrees over. Patching the bound module global is enough (no reload needed).
    monkeypatch.setattr(llo, "MLB_LOW_LINE_OVER_OVERRIDE_ENABLED", False)
    assert low_line_over_override_applies(**_strong_agrees()) is False


def test_ships_disabled_by_default():
    # Lock the shipped default: the carve-out is OFF after the 6 Jun regression, so a
    # freshly imported module honors the disabled flag regardless of the test fixture.
    assert wc.MLB_LOW_LINE_OVER_OVERRIDE_ENABLED is False
    mod = importlib.reload(llo)
    try:
        assert mod.MLB_LOW_LINE_OVER_OVERRIDE_ENABLED is False
        assert mod.low_line_over_override_applies(**_strong_agrees()) is False
    finally:
        importlib.reload(llo)
