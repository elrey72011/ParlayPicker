"""Under win-prob floor relaxation (8 Jun, graded n=212).

Unders are the edge side (60.6% vs Overs 52.5%) and hold edge well below the old
0.62/0.63 floors, so the MLB under floor was lowered to 0.55 while OVERS stay strict
(relaxing overs bled to 46.7%). NBA/NHL unders must stay protected by their own,
higher league floors. See scripts/edge_by_bucket.py and weights_config.
"""
from app_core import weights_config as wc


def test_under_floor_relaxed_below_over_floor():
    # The asymmetry is the whole point: unders relaxed, overs kept strict.
    assert wc.TOTAL_UNDER_MIN_WIN_PROB == 0.55
    assert wc.MLB_TOTAL_UNDER_MIN_WIN_PROB == 0.55
    assert wc.MLB_OVER_ACTIONABLE_MIN_PROB == 0.65
    assert wc.MLB_TOTAL_UNDER_MIN_WIN_PROB < wc.MLB_OVER_ACTIONABLE_MIN_PROB


def test_non_mlb_unders_keep_higher_floors():
    # The effective MLB under floor is max(TOTAL_UNDER, MLB_TOTAL_UNDER); NBA/NHL pin
    # their own higher floors via max() downstream, so the relaxation cannot leak to them.
    mlb_floor = max(wc.TOTAL_UNDER_MIN_WIN_PROB, wc.MLB_TOTAL_UNDER_MIN_WIN_PROB)
    assert mlb_floor == 0.55
    assert wc.NHL_TOTAL_MIN_WIN_PROB_STRICT > mlb_floor   # NHL stays >= 0.58
    assert wc.NBA_TOTAL_MIN_WIN_PROB > mlb_floor          # NBA stays 0.65


def test_quality_gates_for_unders_unchanged():
    # The relaxation only touches the prob floor; the EV/edge quality backstop that keeps
    # weak unders out must remain (this is why lowering prob is safe).
    assert wc.TOTAL_UNDER_MIN_EV == 0.22
    assert wc.TOTAL_UNDER_MIN_EDGE == 0.13
