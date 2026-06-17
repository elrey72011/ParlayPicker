"""Earned-Actionable relaxation for proven Agrees-over buckets (17 Jun).

The over-prob shrink + market-anchored debias suppress MLB over calibrated probs
to ~0.50, so the empirical overlay's calibrated edge never cleared +4% even for
buckets with a strong realized win rate. assign_empirical_tiers now also trusts a
PROVEN bucket's own realized edge (smoothed rate vs break-even) to promote, as
long as the pick is not itself a negative-edge outlier and Kalshi agrees.
"""
import pandas as pd

from core.empirical_tiers import assign_empirical_tiers


def _bucket_stats():
    # Overall ~0.52; MLB:over:Agrees proven at smoothed ~0.59 over n=41.
    # MLB:over:Disagrees deliberately thin/below bar so it must NOT promote.
    return {
        "overall": {"win_rate": 0.52, "n": 400},
        "buckets": {
            "MLB:over:Agrees": {"n": 41, "wins": 25},
            "MLB:over:Disagrees": {"n": 41, "wins": 18},
        },
    }


def _row(consensus, win_prob=0.50, odds=-110):
    return {
        "league": "MLB",
        "market_type": "total_over",
        "consensus_agreement": consensus,
        "effective_win_probability": win_prob,
        "odds_american": odds,
        "Pick_Status": "Below Threshold",
        "Status_Reason": "Fails minimum Win Probability for Totals (65.0%)",
    }


def test_proven_agrees_over_bucket_promotes_despite_suppressed_calibrated_prob():
    # win_prob 0.52 -> calibrated edge ~+2.9% (HV band, below the +4% Actionable bar)
    # after the bucket tilt. On its own this would cap at High Variance, but the
    # proven MLB:over:Agrees bucket's realized edge carries it to Actionable.
    df = pd.DataFrame([_row("Agrees", win_prob=0.52)])
    out = assign_empirical_tiers(df, _bucket_stats(), calibration=None)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert "proven-bucket realized edge" in out.iloc[0]["Status_Reason"]


def test_coinflip_agrees_over_is_not_promoted_even_in_proven_bucket():
    # A genuine ~0.50 pick (calibrated edge below the HV bar) must stay blocked even
    # in a proven bucket — the bucket lifts viable edges, it does not rescue coin flips.
    df = pd.DataFrame([_row("Agrees", win_prob=0.50)])
    out = assign_empirical_tiers(df, _bucket_stats(), calibration=None)
    assert out.iloc[0]["Pick_Status"] != "Actionable"


def test_disagrees_over_is_not_promoted_even_in_a_decent_bucket():
    df = pd.DataFrame([_row("Disagrees", win_prob=0.50)])
    out = assign_empirical_tiers(df, _bucket_stats(), calibration=None)
    assert out.iloc[0]["Pick_Status"] != "Actionable"


def test_negative_edge_agrees_over_is_not_promoted_by_proven_bucket():
    # A pick whose own calibrated prob is clearly negative-edge (well below
    # break-even) is an outlier and must not ride the bucket to Actionable.
    df = pd.DataFrame([_row("Agrees", win_prob=0.30)])
    out = assign_empirical_tiers(df, _bucket_stats(), calibration=None)
    assert out.iloc[0]["Pick_Status"] != "Actionable"
