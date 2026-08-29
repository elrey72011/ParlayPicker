"""Tests for the recency-weighted bucket-stats fit (scripts/fit_bucket_stats.py).

Motivation: the equal-weight fit let a bucket that decayed mid-season stay
"proven" on stale volume (over:Agrees read 51.9% cumulative on 30 Jun while the
last 21 days realized 44.4%). Picks are now exponentially decayed by age against
the newest slate in the exports.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.fit_bucket_stats import ROOT, _source_label, fit_bucket_stats


def _write_slate(path: Path, date: str, wl_list: list[str]) -> None:
    rows = [
        {
            "league": "MLB",
            "Home": "H",
            "Away": "A",
            "best_pick": "Over 8.5",
            "consensus_agreement": "Agrees",
            "W/L": wl,
        }
        for wl in wl_list
    ]
    pd.DataFrame(rows).to_csv(path / f"{date}.csv", index=False)


def test_recent_losses_outweigh_old_wins(tmp_path):
    # 10-0 sixty days ago vs 0-10 on the anchor date: equal-weight would say 50%,
    # recency weighting must land well below it (old wins decayed ~86% at hl=21).
    _write_slate(tmp_path, "2026-04-01", ["W"] * 10)
    _write_slate(tmp_path, "2026-05-31", ["L"] * 10)
    stats = fit_bucket_stats(tmp_path, half_life_days=21.0)
    b = stats["buckets"]["MLB:over:Agrees"]
    assert b["win_rate"] < 0.20
    # Kish effective n is smaller than the raw count when weights are unequal.
    assert b["n"] < b["raw_n"] == 20
    assert b["raw_wins"] == 10
    # wins/n stays consistent with win_rate for the overlay's Laplace smoothing.
    assert abs(b["wins"] / b["n"] - b["win_rate"]) < 0.05


def test_equal_dates_reproduce_unweighted_rates(tmp_path):
    # All picks on one date -> uniform weights -> plain averages and n == raw_n.
    _write_slate(tmp_path, "2026-06-01", ["W", "W", "L", "L"])
    stats = fit_bucket_stats(tmp_path, half_life_days=21.0)
    b = stats["buckets"]["MLB:over:Agrees"]
    assert b["win_rate"] == 0.5
    assert b["n"] == b["raw_n"] == 4
    assert stats["overall"]["win_rate"] == 0.5


def test_fit_persists_trailing_seven_day_bucket_evidence(tmp_path):
    _write_slate(tmp_path, "2026-04-01", ["W", "W"])
    _write_slate(tmp_path, "2026-05-25", ["W", "L"])
    _write_slate(tmp_path, "2026-05-31", ["L", "L", "L"])

    stats = fit_bucket_stats(tmp_path, half_life_days=21.0)
    bucket = stats["buckets"]["MLB:over:Agrees"]

    assert bucket["recent_n"] == 5
    assert bucket["recent_wins"] == 1
    assert bucket["recent_win_rate"] == 0.2
    assert stats["meta"]["recent_window_days"] == 7


def test_fit_persists_cross_league_market_family_evidence(tmp_path):
    rows = []
    for league, outcomes in (("MLB", ["W", "L", "L"]), ("NFL", ["L"] * 4)):
        rows.extend({
            "league": league,
            "Home": "H",
            "Away": "A",
            "best_pick": "Over 8.5",
            "consensus_agreement": "Neutral",
            "W/L": outcome,
        } for outcome in outcomes)
    pd.DataFrame(rows).to_csv(tmp_path / "2026-08-28.csv", index=False)

    stats = fit_bucket_stats(tmp_path, half_life_days=21.0)
    family = stats["families"]["over"]

    assert family["raw_n"] == family["recent_n"] == 7
    assert family["raw_wins"] == family["recent_wins"] == 1
    assert family["recent_win_rate"] == 1 / 7


def test_anchor_is_newest_slate_not_wall_clock(tmp_path):
    # Deterministic: the anchor comes from the exports, so refitting later with no
    # new slates yields identical numbers.
    _write_slate(tmp_path, "2026-05-01", ["W", "L"])
    _write_slate(tmp_path, "2026-05-15", ["W", "L"])
    a = fit_bucket_stats(tmp_path, half_life_days=21.0)
    b = fit_bucket_stats(tmp_path, half_life_days=21.0)
    assert a["buckets"] == b["buckets"]
    assert a["meta"]["recency_anchor"] == "2026-05-15"


def test_overall_n_is_raw_count(tmp_path):
    _write_slate(tmp_path, "2026-04-01", ["W"] * 5)
    _write_slate(tmp_path, "2026-06-01", ["L"] * 5)
    stats = fit_bucket_stats(tmp_path, half_life_days=21.0)
    assert stats["overall"]["n"] == 10
    # Overall rate is recency-weighted (recent losses dominate the old wins).
    assert stats["overall"]["win_rate"] < 0.30


def test_repo_calibration_source_is_portable():
    assert _source_label(ROOT / "data" / "backtest_exports") == "data/backtest_exports"
