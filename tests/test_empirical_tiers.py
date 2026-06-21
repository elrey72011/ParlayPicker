"""Tests for the empirical tier overlay (core/empirical_tiers.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.empirical_tiers import (
    assign_empirical_tiers,
    bucket_key,
    empirical_win_probability,
    load_bucket_stats,
    smoothed_bucket_rate,
)

# Overall 53%; Agrees unders proven hot (60%, n=60), Neutral overs cold (48%, n=60),
# tiny bucket for shrink testing.
STATS = {
    "overall": {"n": 200, "win_rate": 0.53},
    "buckets": {
        "MLB:under:Agrees": {"n": 60, "wins": 36, "win_rate": 0.60},
        "MLB:over:Neutral": {"n": 60, "wins": 29, "win_rate": 0.483},
        "MLB:over:Agrees": {"n": 4, "wins": 4, "win_rate": 1.0},
    },
}


def _line_provenance_row(blocker):
    return pd.DataFrame({
        "league": ["MLB"], "market_type": ["total_under"],
        "consensus_agreement": ["Agrees"], "best_pick": ["Under 8.0"],
        "Pick_Status": ["High Variance/Speculative"], "Status_Reason": ["recovered"],
        "status_blocker_stage": [blocker],
        "effective_win_probability": [0.66], "odds_american": [-110],
    })


def test_overlay_does_not_promote_line_provenance_rows():
    # 21 Jun Miami Over 8.0: a live-rejected row recovered to the uploaded line still
    # carries corrupt odds / a wrong-game Kalshi match, so the overlay must NOT re-promote
    # it to Actionable off a bogus edge — even in a proven bucket (under:Agrees 60%).
    out = assign_empirical_tiers(_line_provenance_row("line_provenance"), STATS, calibration=None)
    assert out.iloc[0]["Pick_Status"] != "Actionable"


def test_overlay_still_promotes_clean_proven_bucket_row():
    # Control: identical numbers WITHOUT the line-provenance flag still promote, so the
    # guard above is the line-provenance check, not a broken bucket path.
    out = assign_empirical_tiers(_line_provenance_row("x"), STATS, calibration=None)
    assert out.iloc[0]["Pick_Status"] == "Actionable"


def test_bucket_key_normalization():
    assert bucket_key("mlb", "total_over", "Agrees") == "MLB:over:Agrees"
    assert bucket_key("MLB", "total_under", "Disagrees") == "MLB:under:Disagrees"
    assert bucket_key("NBA", "spread_home", "No Kalshi") == "NBA:side:Neutral"
    assert bucket_key("NHL", "moneyline", "") == "NHL:side:Neutral"


def test_empirical_probability_tilts_by_bucket_and_shrinks_small_samples():
    p_hot = empirical_win_probability(0.55, "MLB:under:Agrees", STATS)
    p_cold = empirical_win_probability(0.55, "MLB:over:Neutral", STATS)
    p_unknown = empirical_win_probability(0.55, "NHL:side:Agrees", STATS)
    assert p_hot > 0.55 > p_cold
    assert p_unknown == 0.55  # no bucket -> calibrated value untouched

    # A 4-0 bucket must NOT move the pick like a proven 60-pick bucket would:
    # the shrink weight is 4/54 vs 60/110.
    p_tiny = empirical_win_probability(0.55, "MLB:over:Agrees", STATS)
    assert p_tiny - 0.55 < p_hot - 0.55 + 0.05
    assert p_tiny < 0.60


def test_smoothed_bucket_rate_laplace():
    rate, n = smoothed_bucket_rate("MLB:under:Agrees", STATS)
    assert n == 60
    assert abs(rate - (36 + 0.53 * 10) / 70) < 1e-9
    rate0, n0 = smoothed_bucket_rate("missing", STATS)
    assert (rate0, n0) == (0.53, 0)


def _frame():
    return pd.DataFrame(
        {
            "league": ["MLB", "MLB", "MLB", "MLB"],
            "market_type": ["total_under", "total_over", "total_under", "total_under"],
            "consensus_agreement": ["Agrees", "Neutral", "Agrees", "Agrees"],
            "best_pick": ["Under 7.5", "Over 8.5", "Under 8.5", "Under 9.5"],
            "Pick_Status": ["Below Threshold", "Actionable", "Below Threshold", "No Play"],
            "Status_Reason": ["old", "old", "old", "No Play: suspicious data"],
            "status_blocker_stage": ["x", "x", "x", "suspicious_data_guardrail"],
            "effective_win_probability": [0.62, 0.60, 0.50, 0.62],
            "odds_american": [-110, -110, -110, -110],
        }
    )


def test_overlay_promotes_proven_bucket_and_demotes_cold_one():
    out = assign_empirical_tiers(_frame(), STATS, calibration=None)
    # Agrees under at 0.62 + hot-bucket tilt: edge over 0.524 breakeven > 4% -> Actionable
    assert out.loc[0, "Pick_Status"] == "Actionable"
    assert "empirical" in str(out.loc[0, "Status_Reason"])
    # Neutral over at 0.60 gets tilted DOWN by the cold bucket and loses Actionable
    assert out.loc[1, "Pick_Status"] != "Actionable"
    # Weak pick stays Below Threshold (negative empirical edge)
    assert out.loc[2, "Pick_Status"] == "Below Threshold"


def test_overlay_never_touches_safety_statuses():
    out = assign_empirical_tiers(_frame(), STATS, calibration=None)
    assert out.loc[3, "Pick_Status"] == "No Play"
    assert out.loc[3, "Status_Reason"] == "No Play: suspicious data"


def test_overlay_applies_calibration_before_bucket_tilt():
    # The calibration is applied to the probability BEFORE the bucket tilt, so a
    # suppressing table lowers empirical_win_probability vs the uncalibrated run for
    # the same row. (Tier outcome for a proven directional bucket is governed by the
    # earned path, tested separately — this asserts the calibration mechanic itself.)
    base = assign_empirical_tiers(_frame(), STATS, calibration=None)
    knots = [[0.4, 0.50], [0.9, 0.50]]
    supp = assign_empirical_tiers(_frame(), STATS, calibration=knots)
    assert supp.loc[0, "empirical_win_probability"] < base.loc[0, "empirical_win_probability"]


# A harsh global calibration that drags probabilities toward ~0.50.
_SUPPRESS_KNOTS = [[0.4, 0.40], [0.6, 0.50], [0.9, 0.80]]


def test_proven_under_promotes_on_bucket_edge_when_calibration_suppresses():
    # under:Agrees (60%, n=60) at pre-cal 0.60 — beats the -110 breakeven (0.524). The
    # global calibration crushes 0.60 -> 0.50, dropping the per-pick CALIBRATED edge below
    # the High Variance bar. Pre-20-Jun this earned path was over-only, so the under fell
    # to Below Threshold; now it earns Actionable on the proven bucket's realized edge,
    # guarded by its pre-calibration prob still beating breakeven (same as overs).
    df = pd.DataFrame({
        "league": ["MLB"], "market_type": ["total_under"],
        "consensus_agreement": ["Agrees"], "best_pick": ["Under 8.5"],
        "Pick_Status": ["Below Threshold"], "Status_Reason": ["old"],
        "status_blocker_stage": ["x"],
        "effective_win_probability": [0.60], "odds_american": [-110],
    })
    out = assign_empirical_tiers(df, STATS, calibration=_SUPPRESS_KNOTS)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert "realized edge" in str(out.iloc[0]["Status_Reason"]).lower()


def test_earned_under_guards_precalibration_below_breakeven():
    # Same proven bucket, but the pick's OWN pre-calibration prob (0.51) is below the
    # -110 breakeven (0.524). The earned path must NOT stake it on the bucket's back.
    df = pd.DataFrame({
        "league": ["MLB"], "market_type": ["total_under"],
        "consensus_agreement": ["Agrees"], "best_pick": ["Under 8.5"],
        "Pick_Status": ["Below Threshold"], "Status_Reason": ["old"],
        "status_blocker_stage": ["x"],
        "effective_win_probability": [0.51], "odds_american": [-110],
    })
    out = assign_empirical_tiers(df, STATS, calibration=_SUPPRESS_KNOTS)
    assert out.iloc[0]["Pick_Status"] != "Actionable"


def test_overlay_requires_proven_bucket_for_actionable():
    # Strong calibrated number in the tiny 4-0 bucket: n=4 < 25 -> capped at HV.
    df = _frame().iloc[[1]].copy()
    df["consensus_agreement"] = ["Agrees"]
    df["effective_win_probability"] = [0.68]
    out = assign_empirical_tiers(df, STATS, calibration=None)
    assert out.iloc[0]["Pick_Status"] == "High Variance/Speculative"


def test_thin_bucket_below_n25_floor_cannot_be_actionable():
    # The exact 12 Jun case: MLB:under:Disagrees at 60% but only n=16. Strong edge,
    # but the sample is too thin to carry Actionable under the raised n>=25 floor.
    stats = {
        "overall": {"n": 200, "win_rate": 0.53},
        "buckets": {"MLB:under:Disagrees": {"n": 16, "wins": 10, "win_rate": 0.625}},
    }
    df = pd.DataFrame(
        {
            "league": ["MLB"], "market_type": ["total_under"],
            "consensus_agreement": ["Disagrees"], "best_pick": ["Under 8.5"],
            "Pick_Status": ["Below Threshold"], "Status_Reason": ["old"],
            "status_blocker_stage": ["x"],
            "effective_win_probability": [0.66], "odds_american": [-110],
        }
    )
    out = assign_empirical_tiers(df, stats, calibration=None)
    # Positive empirical edge -> at least High Variance, but NOT Actionable (n<25).
    assert out.iloc[0]["Pick_Status"] == "High Variance/Speculative"


def test_load_bucket_stats_missing_returns_none(tmp_path):
    assert load_bucket_stats(tmp_path / "nope.json") is None


def test_fitted_bucket_stats_file_is_loadable():
    stats = load_bucket_stats(Path(__file__).resolve().parents[1] / "data/calibration/bucket_stats.json")
    assert stats is not None
    assert stats["overall"]["n"] >= 200
    assert "MLB:under:Agrees" in stats["buckets"]


def test_actionable_requires_market_agreement():
    # A non-Agrees bucket with a STRONG, well-sampled rate (60%, n=40) clears the
    # edge, sample, and rate floors — but WITHOUT market agreement it can no longer
    # carry a stake; it caps at High Variance instead of Actionable.
    stats = {
        "overall": {"n": 200, "win_rate": 0.53},
        "buckets": {"MLB:under:Disagrees": {"n": 40, "wins": 24, "win_rate": 0.60}},
    }
    df = pd.DataFrame(
        {
            "league": ["MLB"], "market_type": ["total_under"],
            "consensus_agreement": ["Disagrees"], "best_pick": ["Under 8.5"],
            "Pick_Status": ["Below Threshold"], "Status_Reason": ["old"],
            "status_blocker_stage": ["x"],
            "effective_win_probability": [0.66], "odds_american": [-110],
        }
    )
    out = assign_empirical_tiers(df, stats, calibration=None)
    assert out.iloc[0]["Pick_Status"] == "High Variance/Speculative"


def test_actionable_allows_agreement_bucket_with_same_metrics():
    # Identical strong metrics, but in an Agrees bucket -> Actionable, and the reason
    # cites market agreement.
    stats = {
        "overall": {"n": 200, "win_rate": 0.53},
        "buckets": {"MLB:under:Agrees": {"n": 40, "wins": 24, "win_rate": 0.60}},
    }
    df = pd.DataFrame(
        {
            "league": ["MLB"], "market_type": ["total_under"],
            "consensus_agreement": ["Agrees"], "best_pick": ["Under 8.5"],
            "Pick_Status": ["Below Threshold"], "Status_Reason": ["old"],
            "status_blocker_stage": ["x"],
            "effective_win_probability": [0.66], "odds_american": [-110],
        }
    )
    out = assign_empirical_tiers(df, stats, calibration=None)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert "agreement" in str(out.iloc[0]["Status_Reason"]).lower()
