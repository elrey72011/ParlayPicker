"""MLB totals win-prob floor (28 Jun: symmetrized over/under).

The MLB under floor sits at 0.55. The over floor USED to stay strict at 0.65 (set when overs
bled 46.7%), but on the current graded sample (n=350) overs are ~51% — over:Agrees .509,
over:Neutral .510 — no worse than under:Neutral (.500) or under:Disagrees (.457). Direction is
no longer the signal (only under:Agrees .631 is truly good), and the empirical-tier overlay
gates staking by PROVEN bucket, so the asymmetric over penalty was removed. NBA/NHL unders
still keep their own higher league floors. See scripts/edge_by_bucket.py and weights_config.
"""
from app_core import weights_config as wc


def test_mlb_total_floor_is_symmetric_over_and_under():
    # The over/under asymmetry was removed: both MLB floors are 0.55, and the bucket overlay
    # (not a blanket over penalty) decides what actually stakes.
    assert wc.TOTAL_UNDER_MIN_WIN_PROB == 0.55
    assert wc.MLB_TOTAL_UNDER_MIN_WIN_PROB == 0.55
    assert wc.MLB_OVER_ACTIONABLE_MIN_PROB == 0.55
    assert wc.MLB_OVER_ACTIONABLE_MIN_PROB == wc.MLB_TOTAL_UNDER_MIN_WIN_PROB


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
