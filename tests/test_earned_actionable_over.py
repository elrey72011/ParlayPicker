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


def test_proven_agrees_over_bucket_promotes_despite_calibration_crush():
    # The real scenario: a strong over (pre-calibration prob 0.60, comfortably above
    # break-even) whose calibrated edge is crushed to ~0 by a down-mapping calibration.
    # On the calibrated number alone it would not be Actionable, but the proven
    # MLB:over:Agrees bucket's realized edge carries it — guarded by the raw 0.60.
    crush = [[0.3, 0.50], [0.9, 0.50]]  # map everything to 0.50 before the bucket tilt
    df = pd.DataFrame([_row("Agrees", win_prob=0.60)])
    out = assign_empirical_tiers(df, _bucket_stats(), calibration=crush)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert "proven-bucket realized edge" in out.iloc[0]["Status_Reason"]


def test_coinflip_agrees_over_is_not_promoted_even_in_proven_bucket():
    # A genuine ~0.50 pick sits below the -110 break-even (0.524), so the raw-prob
    # outlier guard blocks it even in a proven over bucket — the bucket lifts real
    # edges, it does not rescue coin flips.
    df = pd.DataFrame([_row("Agrees", win_prob=0.50)])
    out = assign_empirical_tiers(df, _bucket_stats(), calibration=None)
    assert out.iloc[0]["Pick_Status"] != "Actionable"


def test_disagrees_over_is_not_promoted_even_in_a_decent_bucket():
    df = pd.DataFrame([_row("Disagrees", win_prob=0.60)])
    out = assign_empirical_tiers(df, _bucket_stats(), calibration=None)
    assert out.iloc[0]["Pick_Status"] != "Actionable"


def test_negative_signal_agrees_over_is_not_promoted_by_proven_bucket():
    # A pick whose own pre-calibration prob is clearly below break-even is a
    # negative-signal outlier and must not ride the bucket to Actionable.
    df = pd.DataFrame([_row("Agrees", win_prob=0.30)])
    out = assign_empirical_tiers(df, _bucket_stats(), calibration=None)
    assert out.iloc[0]["Pick_Status"] != "Actionable"


def test_market_debias_exempts_kalshi_backed_overs():
    # On an over-heavy slate the de-bias pulls model overs toward the de-vig market,
    # EXCEPT overs Kalshi independently backs (k>=0.52), which keep their blended value.
    from core.streamlit_pipeline import apply_mlb_total_market_debias

    n = 8
    df = pd.DataFrame(
        {
            "league": ["MLB"] * n,
            "market_type": ["total_over"] * n,
            "kalshi_probability": [0.57, 0.45, 0.62, 0.40, 0.58, 0.44, 0.60, 0.43],
            "market_probability": [0.48] * n,
        }
    )
    cal = pd.Series([0.58] * n)
    out, bias = apply_mlb_total_market_debias(cal, df)
    assert bias > 0.0  # a real slate-mean over-bias was detected
    agrees = df["kalshi_probability"] >= 0.52
    # Kalshi-backed overs untouched; the rest pulled down toward the market.
    assert (out[agrees.values] == 0.58).all()
    assert (out[~agrees.values] < 0.58).all()
