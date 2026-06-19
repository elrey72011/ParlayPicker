"""Tighten thin Actionable overs in the empirical overlay (19 Jun).

Two guards on the earned-Actionable path, both seen live in the 18:24 export:
  Rule 1 — a proven bucket must not carry a pick whose CALIBRATED edge is <= 0
           (NYY Over 9.0 staked at calibrated -0.3% on the bucket's +8.8%).
  Rule 2 — a more-specific directional "the other side is the edge" demotion is not
           re-promoted by the coarse overlay bucket (Miami Over 8.0 re-promoted to
           Actionable while the mid-line-Over gate flagged the Under as the edge side).
"""
import pandas as pd

from core.empirical_tiers import assign_empirical_tiers


def _bucket_stats():
    return {
        "overall": {"win_rate": 0.52, "n": 400},
        "buckets": {"MLB:over:Agrees": {"n": 41, "wins": 25}},
    }


def _over_row(win_prob, odds=-110, blocker="none"):
    return {
        "league": "MLB", "market_type": "total_over", "consensus_agreement": "Agrees",
        "effective_win_probability": win_prob, "odds_american": odds,
        "Pick_Status": "Below Threshold", "status_blocker_stage": blocker,
        "Status_Reason": "Fails minimum Win Probability for Totals (65.0%)",
    }


def test_rule1_calibrated_negative_over_not_promoted():
    # Heavy down-calibration pushes the calibrated edge negative; the proven bucket must
    # no longer carry it (the NYY Over 9.0 case).
    crush = [[0.3, 0.40], [0.9, 0.40]]  # map to 0.40 -> empirical edge clearly negative
    out = assign_empirical_tiers(pd.DataFrame([_over_row(0.60)]), _bucket_stats(), calibration=crush)
    assert out.iloc[0]["Pick_Status"] != "Actionable"


def test_rule1_still_promotes_small_positive_calibrated_edge():
    # The legitimate earned case: calibrated edge crushed near zero but still POSITIVE.
    crush = [[0.3, 0.50], [0.9, 0.50]]
    out = assign_empirical_tiers(pd.DataFrame([_over_row(0.60)]), _bucket_stats(), calibration=crush)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert "proven-bucket realized edge" in out.iloc[0]["Status_Reason"]


def test_rule2_mid_line_over_demotion_is_not_repromoted():
    # The Miami Over 8.0 case: the mid-line-Over gate already demoted it ("the Under is
    # the edge side"); the proven Agrees-over bucket must not override that verdict.
    row = _over_row(0.60, blocker="mlb_over_mid_line_no_stake")
    out = assign_empirical_tiers(pd.DataFrame([row]), _bucket_stats(), calibration=[[0.3, 0.50], [0.9, 0.50]])
    assert out.iloc[0]["Pick_Status"] == "Below Threshold"
    assert out.iloc[0]["status_blocker_stage"] == "mlb_over_mid_line_no_stake"


def test_rule2_neutral_no_stake_demotion_is_not_repromoted():
    row = _over_row(0.60, blocker="mlb_total_neutral_no_stake")
    out = assign_empirical_tiers(pd.DataFrame([row]), _bucket_stats(), calibration=[[0.3, 0.50], [0.9, 0.50]])
    assert out.iloc[0]["Pick_Status"] == "Below Threshold"
