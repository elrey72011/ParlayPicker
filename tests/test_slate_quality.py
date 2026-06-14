"""Tests for the slate-level data-quality guards (core/slate_quality.py),
exercised against the real 13 Jun 2026 corruption pattern."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import io

from core.slate_quality import (
    slate_direction_imbalanced,
    theover_feed_degraded,
    theover_upload_warning,
    totals_direction_share,
)

# The actual total_over TheOver P(Over) reads from
# best_picks_export - 2026-06-13T123558: five identical 0.692, plus 0.706/0.667,
# six default-0.5 no-reads, and one column-shift ("model_hit_rate" -> NaN).
JUN13_OVER_READS = [
    0.692, 0.706, 0.667, 0.692, 0.5, 0.5, float("nan"), 0.692,
    0.5, 0.692, 0.692, 0.5, 0.5, 0.5,
]


def test_theover_degraded_fires_on_jun13_clustering():
    degraded, reason = theover_feed_degraded(JUN13_OVER_READS)
    assert degraded is True
    assert "0.692" in reason
    # 5 of 7 real reads (0.5s and NaN excluded) are identical.
    assert "5/7" in reason


def test_theover_not_degraded_on_healthy_spread_of_reads():
    healthy = [0.58, 0.61, 0.47, 0.54, 0.63, 0.49, 0.56, 0.52]
    degraded, reason = theover_feed_degraded(healthy)
    assert degraded is False
    assert reason is None


def test_theover_no_false_positive_on_small_or_allnoread_slate():
    # Too few real reads -> cannot judge -> not degraded.
    assert theover_feed_degraded([0.69, 0.69, 0.5, 0.5])[0] is False
    # All no-reads -> no real reads -> not degraded.
    assert theover_feed_degraded([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])[0] is False


def test_theover_handles_nonnumeric_contamination():
    # A column-shift can drop a text label into the numeric field; it must coerce
    # to NaN and be ignored, not raise.
    reads = ["model_hit_rate", 0.692, 0.692, 0.692, 0.692, 0.692, 0.71]
    degraded, reason = theover_feed_degraded(reads)
    assert degraded is True  # 5/6 real reads identical


def test_direction_share_counts_only_totals():
    mt = ["total_over"] * 14 + ["spread_away"]
    direction, share, n = totals_direction_share(mt)
    assert direction == "over"
    assert n == 14
    assert share == 1.0


def test_slate_imbalanced_fires_on_jun13_all_over_card():
    mt = ["total_over"] * 14 + ["spread_away"]
    imbalanced, reason = slate_direction_imbalanced(mt)
    assert imbalanced is True
    assert "100%" in reason and "over" in reason


def test_slate_not_imbalanced_on_mixed_card():
    mt = ["total_over"] * 8 + ["total_under"] * 6 + ["spread_home"]
    imbalanced, reason = slate_direction_imbalanced(mt)
    assert imbalanced is False
    assert reason is None


def test_slate_not_imbalanced_below_min_games():
    # Even 100% one-sided, too few totals to be a confident fault signal.
    mt = ["total_over"] * 4
    assert slate_direction_imbalanced(mt)[0] is False


def test_slate_imbalance_allows_real_lopsided_slate():
    # 9/12 over (75%) is a plausible hot slate, not a fault — must not fire.
    mt = ["total_over"] * 9 + ["total_under"] * 3
    assert slate_direction_imbalanced(mt)[0] is False


# ---- upload-time content validation (theover_upload_warning) ----

def _upload_df(probs):
    teams = [(f"Home{i}", f"Away{i}") for i in range(len(probs))]
    return pd.DataFrame(
        {
            "home_team": [h for h, _ in teams],
            "away_team": [a for _, a in teams],
            "winprobability": probs,
            "total_line": [8.5] * len(probs),
        }
    )


def test_upload_warning_flags_clustering():
    # 5 identical reads among 7 real -> clustering warning at upload time.
    w = theover_upload_warning(_upload_df([0.692, 0.706, 0.667, 0.692, 0.692, 0.692, 0.692]))
    assert w is not None
    assert "corrupt" in w.lower() and "0.692" in w


def test_upload_warning_flags_nonnumeric_contamination():
    # The column-shift: a text label sits in the probability column.
    w = theover_upload_warning(_upload_df(["model_hit_rate", 0.61, 0.47, 0.55, 0.6, 0.52]))
    assert w is not None
    assert "non-numeric" in w.lower()


def test_upload_warning_clean_file_is_none():
    assert theover_upload_warning(_upload_df([0.58, 0.61, 0.47, 0.54, 0.63, 0.49, 0.52])) is None


def test_upload_warning_no_prob_column_is_none():
    assert theover_upload_warning(pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})) is None


def test_upload_warning_never_raises_on_garbage():
    assert theover_upload_warning(None) is None
    assert theover_upload_warning(pd.DataFrame()) is None


def test_load_theover_csv_surfaces_warning_and_keeps_data():
    from core.theover_loader import load_theover_csv

    df = _upload_df([0.692, 0.706, 0.667, 0.692, 0.692, 0.692, 0.692])
    buf = io.StringIO(df.to_csv(index=False))
    loaded, msg = load_theover_csv(buf)
    # Warning surfaced AND the data still loads (guards are the safety net).
    assert msg is not None and "corrupt" in msg.lower()
    assert not loaded.empty and len(loaded) == 7


def test_load_theover_csv_clean_file_no_warning():
    from core.theover_loader import load_theover_csv

    df = _upload_df([0.58, 0.61, 0.47, 0.54, 0.63, 0.49, 0.52])
    buf = io.StringIO(df.to_csv(index=False))
    loaded, msg = load_theover_csv(buf)
    assert msg is None
    assert len(loaded) == 7


# ---- market-anchored over-bias correction (market_anchored_over_bias) ----

from core.slate_quality import market_anchored_over_bias


def test_over_bias_measures_mean_model_minus_market_gap():
    model = [0.62, 0.58, 0.55, 0.60, 0.57, 0.59]   # mean 0.585
    market = [0.50, 0.48, 0.47, 0.49, 0.46, 0.50]  # mean 0.483
    bias = market_anchored_over_bias(model, market)
    assert abs(bias - (0.585 - 0.483333)) < 1e-4


def test_over_bias_clamped_to_max_shift():
    model = [0.90] * 6
    market = [0.40] * 6  # gap 0.50, far beyond cap
    assert market_anchored_over_bias(model, market, max_shift=0.15) == 0.15


def test_over_bias_zero_when_too_few_games():
    assert market_anchored_over_bias([0.7, 0.7, 0.7], [0.4, 0.4, 0.4]) == 0.0


def test_over_bias_preserves_genuine_market_lean():
    # Model agrees with the market (no systematic gap) even though the market
    # itself leans over -> bias ~0, so a real over-night is NOT flattened.
    model = [0.66, 0.64, 0.62, 0.63, 0.65, 0.61]
    market = [0.65, 0.64, 0.61, 0.64, 0.65, 0.60]
    assert abs(market_anchored_over_bias(model, market)) < 0.02


def test_over_bias_rebalances_jun13_card():
    # Real 13 Jun blended P(over) vs de-vig market P(over) on the 14 totals.
    model = [0.617, 0.576, 0.539, 0.592, 0.562, 0.549, 0.548, 0.570,
             0.575, 0.524, 0.540, 0.527, 0.614, 0.540]
    market = [0.494, 0.478, 0.438, 0.501, 0.459, 0.455, 0.463, 0.494,
              0.476, 0.445, 0.468, 0.457, 0.535, 0.484]
    bias = market_anchored_over_bias(model, market)
    assert bias > 0.07  # a real systematic over-lean is detected
    corrected = [p - bias for p in model]
    overs = sum(1 for p in corrected if p > 0.5)
    unders = len(corrected) - overs
    # Was 14/0 all-Over; after de-bias the all-over pathology is broken and the
    # card follows the (under-leaning) sharp market with a real spread of unders.
    assert overs < 14 and unders >= 4


def test_theover_degraded_catches_flip_split_constant_hitrate():
    # 14 Jun: TheOver applied a constant ~0.9 model hit-rate to every game. The
    # over/under flip splits the total_over rows' P(over) into 0.9 (over-picked)
    # and 0.1 (under-picked), which the old signed-value check missed (top value
    # only 46%). Folding to |p-0.5| collapses both to 0.4 -> caught.
    over_rows = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.72, 0.75]
    degraded, reason = theover_feed_degraded(over_rows)
    assert degraded is True
    assert "11/13" in reason  # eleven of thirteen real reads share |p-0.5|=0.4


def test_theover_not_degraded_when_magnitudes_vary():
    # A healthy feed has a real spread of confidence magnitudes, even with a
    # mix of over- and under-leaning reads -> not degraded.
    over_rows = [0.62, 0.41, 0.58, 0.47, 0.66, 0.38, 0.55, 0.44]
    assert theover_feed_degraded(over_rows)[0] is False


# ---- spread orientation guard (spread_moneyline_orientation_fault) ----

from core.slate_quality import spread_moneyline_orientation_fault


def test_orientation_fault_fires_on_flipped_away_spread():
    # 14 Jun: away pick shown -1.5 (spread favorite) but the moneyline favors the
    # home team (home -160, away +140) — the feed flipped the home/away spread.
    fault, reason = spread_moneyline_orientation_fault(
        "spread_away", -1.5, home_ml_price=-160, away_ml_price=140
    )
    assert fault is True
    assert "spread_orientation_fault" in reason and "home" in reason


def test_orientation_ok_when_spread_and_moneyline_agree():
    # Away pick +1.5 (spread underdog) with the home team favored on the moneyline
    # — consistent, no fault.
    fault, reason = spread_moneyline_orientation_fault(
        "spread_away", 1.5, home_ml_price=-160, away_ml_price=140
    )
    assert fault is False and reason is None


def test_orientation_fault_fires_on_flipped_home_spread():
    # Home pick -1.5 (spread favorite) but the away team is the moneyline favorite.
    fault, reason = spread_moneyline_orientation_fault(
        "spread_home", -1.5, home_ml_price=140, away_ml_price=-160
    )
    assert fault is True and "away" in reason


def test_orientation_ok_for_home_favorite_consistent():
    fault, _ = spread_moneyline_orientation_fault(
        "spread_home", -1.5, home_ml_price=-160, away_ml_price=140
    )
    assert fault is False


def test_orientation_no_fault_when_moneyline_missing():
    assert spread_moneyline_orientation_fault("spread_away", -1.5, None, None)[0] is False
    assert spread_moneyline_orientation_fault("spread_away", -1.5, -160, None)[0] is False


def test_orientation_no_fault_on_pickem_moneyline():
    # Near pick'em moneyline (-110/-110) is below the gap threshold, so even an
    # ambiguous spread sign is not treated as a fault.
    assert spread_moneyline_orientation_fault("spread_home", -1.5, -110, -110)[0] is False


def test_orientation_ignores_non_spread_and_missing_line():
    assert spread_moneyline_orientation_fault("total_over", -1.5, -160, 140)[0] is False
    assert spread_moneyline_orientation_fault("spread_away", None, -160, 140)[0] is False
    assert spread_moneyline_orientation_fault("spread_away", 0.0, -160, 140)[0] is False
