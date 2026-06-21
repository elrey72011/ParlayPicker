"""MLB StatsAPI pitcher-form parsing (no network -- fixtures only)."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.mlb_pitcher_stats import (
    _innings_to_float,
    pitcher_form_from_gamelog,
    team_k_rate_from_stats,
)


def test_innings_notation_thirds():
    assert _innings_to_float("6.0") == 6.0
    assert abs(_innings_to_float("6.1") - (6 + 1 / 3)) < 1e-9
    assert abs(_innings_to_float("6.2") - (6 + 2 / 3)) < 1e-9
    assert _innings_to_float(5) == 5.0
    assert _innings_to_float("garbage") == 0.0


# A 3-start fixture: 7K/6IP, 9K/7IP, 5K/5IP  -> 21K over 18IP -> K/9 = 10.5, avg IP = 6.0
_SPLITS = [
    {"stat": {"strikeOuts": 7, "inningsPitched": "6.0"}},
    {"stat": {"strikeOuts": 9, "inningsPitched": "7.0"}},
    {"stat": {"strikeOuts": 5, "inningsPitched": "5.0"}},
]


def test_form_computes_k9_and_innings():
    form = pitcher_form_from_gamelog(_SPLITS, last_n=5)
    assert form["n_games"] == 3
    assert abs(form["k_per_9"] - 10.5) < 1e-9
    assert abs(form["avg_innings"] - 6.0) < 1e-9


def test_form_uses_only_last_n():
    form = pitcher_form_from_gamelog(_SPLITS, last_n=2)
    # last two: 9K/7IP + 5K/5IP = 14K/12IP -> K/9 = 10.5, avg IP = 6.0
    assert form["n_games"] == 2
    assert abs(form["k_per_9"] - 10.5) < 1e-9


def test_form_skips_zero_inning_relief_lines():
    splits = _SPLITS + [{"stat": {"strikeOuts": 0, "inningsPitched": "0.0"}}]
    form = pitcher_form_from_gamelog(splits, last_n=5)
    assert form["n_games"] == 3  # the 0 IP appearance is ignored


def test_form_empty_returns_none():
    assert pitcher_form_from_gamelog([]) is None
    assert pitcher_form_from_gamelog([{"stat": {"strikeOuts": 0, "inningsPitched": "0.0"}}]) is None


def test_team_k_rate():
    assert abs(team_k_rate_from_stats({"strikeOuts": 200, "plateAppearances": 1000}) - 0.20) < 1e-9
    # falls back to atBats when PA absent
    assert abs(team_k_rate_from_stats({"strikeOuts": 150, "atBats": 600}) - 0.25) < 1e-9
    assert team_k_rate_from_stats({"strikeOuts": 5}) is None
    assert team_k_rate_from_stats({}) is None
