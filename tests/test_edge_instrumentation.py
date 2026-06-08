"""EV + Kalshi-direction instrumentation for the edge analysis.

Locks that the graded-pick parser extracts effective_expected_value into Pick.ev
and exposes Kalshi-direction, so scripts/edge_by_bucket.py can track which O/U
setups have edge (the basis for relaxing the EV / consensus / Kalshi-under gates).
"""
from scripts.backtest_low_line_over import Pick, _to_num


def _pick(**kw):
    base = dict(date="d", league="MLB", home="H", away="A", best_pick="Under 8.5",
                market_type="total_under", line=8.5, eff_win_prob=0.6, win_prob=0.6,
                edge=0.1, consensus="Disagrees", kelly=1.0, win_amount=None, result="WIN")
    base.update(kw)
    return Pick(**base)


def test_to_num_parses_ev_without_dividing():
    assert _to_num("0.22") == 0.22
    assert _to_num("-0.05") == -0.05
    assert _to_num("(0.05)") == -0.05      # parens => negative
    assert _to_num("22%") == 22.0          # strips %, does NOT treat as probability
    assert _to_num("1,234.5") == 1234.5
    assert _to_num("-") is None
    assert _to_num(None) is None
    assert _to_num("#REF!") is None


def test_pick_carries_ev():
    assert _pick(ev=0.24).ev == 0.24
    assert _pick().ev is None              # default when absent


def test_kalshi_opposes_direction():
    assert _pick(kalshi_prob=0.385).kalshi_opposes is True   # <0.50 = market override
    assert _pick(kalshi_prob=0.62).kalshi_opposes is False
    assert _pick(kalshi_prob=None).kalshi_opposes is False
