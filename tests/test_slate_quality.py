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
