"""Genuine MLB team features from StatsAPI standings -> model feature columns."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.mlb_team_stats import (
    enrich_mlb_model_features,
    fetch_mlb_team_stats,
    parse_standings,
)

_STANDINGS = {
    "records": [
        {"teamRecords": [
            {
                "team": {"id": 147, "name": "New York Yankees"},
                "wins": 50, "losses": 30, "gamesPlayed": 80,
                "runsScored": 400, "runsAllowed": 320,
                "streak": {"streakType": "wins", "streakNumber": 3},
                "records": {"splitRecords": [{"type": "lastTen", "wins": 7, "losses": 3}]},
            },
            {
                "team": {"id": 111, "name": "Boston Red Sox"},
                "wins": 36, "losses": 44, "gamesPlayed": 80,
                "runsScored": 340, "runsAllowed": 400,
                "streak": {"streakType": "losses", "streakNumber": 2},
                "records": {"splitRecords": [{"type": "lastTen", "wins": 4, "losses": 6}]},
            },
        ]}
    ]
}


def test_parse_standings_computes_per_game_and_signed_streak():
    s = parse_standings(_STANDINGS)
    nyy = s["newyorkyankees"]
    assert nyy["win_pct"] == 50 / 80
    assert nyy["ppg"] == 400 / 80          # 5.0 runs/game
    assert nyy["oppg"] == 320 / 80         # 4.0 allowed/game
    assert nyy["streak"] == 3              # W3 -> +3
    assert nyy["last10_pct"] == 0.7
    assert s["bostonredsox"]["streak"] == -2  # L2 -> -2


def test_parse_standings_defensive_on_missing_fields():
    s = parse_standings({"records": [{"teamRecords": [
        {"team": {"name": "Test Team"}, "wins": 1, "losses": 1}  # no runs/streak/lastTen
    ]}]})
    t = s["testteam"]
    assert t["win_pct"] == 0.5
    assert t["ppg"] is None and t["oppg"] is None
    assert t["streak"] is None and t["last10_pct"] is None


def test_enrich_sets_real_features_for_mlb_rows():
    stats = parse_standings(_STANDINGS)
    df = pd.DataFrame([{
        "league": "MLB", "home_team": "New York Yankees", "away_team": "Boston Red Sox",
        "feature_home_ppg": 4.5, "feature_away_ppg": 4.5, "feature_diff_ppg": 0.0,
    }])
    out = enrich_mlb_model_features(df, stats=stats)
    r = out.iloc[0]
    assert r["feature_home_ppg"] == 5.0          # replaced the 4.5 default
    assert r["feature_away_oppg"] == 5.0
    assert r["feature_diff_win_pct"] == (50 / 80) - (36 / 80)
    assert r["feature_diff_ppg"] == 5.0 - (340 / 80)
    assert r["feature_diff_streak"] == 3 - (-2)
    assert abs(r["feature_diff_last5"] - (0.7 - 0.4)) < 1e-9
    assert r["stats_source"] == "mlb_statsapi"


def test_enrich_resolves_by_nickname():
    stats = parse_standings(_STANDINGS)
    df = pd.DataFrame([{"league": "MLB", "home_team": "Yankees", "away_team": "Red Sox"}])
    out = enrich_mlb_model_features(df, stats=stats)
    assert out.iloc[0]["feature_home_ppg"] == 5.0


def test_resolver_disambiguates_red_sox_from_reds():
    # StatsAPI nicknames: "reds" is a substring of "bostonredsox" -> longest-match must win.
    stats = parse_standings({"records": [{"teamRecords": [
        {"team": {"name": "Reds"}, "wins": 1, "losses": 1, "gamesPlayed": 2, "runsScored": 6},
        {"team": {"name": "Red Sox"}, "wins": 1, "losses": 1, "gamesPlayed": 2, "runsScored": 10},
    ]}]})
    df = pd.DataFrame([{"league": "MLB", "home_team": "Boston Red Sox", "away_team": "Cincinnati Reds"}])
    out = enrich_mlb_model_features(df, stats=stats)
    assert out.iloc[0]["feature_home_ppg"] == 5.0   # Red Sox 10/2, NOT Reds 3.0
    assert out.iloc[0]["feature_away_ppg"] == 3.0   # Reds 6/2


def test_enrich_leaves_non_mlb_and_unresolved_untouched():
    stats = parse_standings(_STANDINGS)
    df = pd.DataFrame([
        {"league": "NBA", "home_team": "Lakers", "away_team": "Celtics", "feature_home_ppg": 9.9},
        {"league": "MLB", "home_team": "Nonexistent FC", "away_team": "Ghost Team", "feature_home_ppg": 4.5},
    ])
    out = enrich_mlb_model_features(df, stats=stats)
    assert out.iloc[0]["feature_home_ppg"] == 9.9   # NBA untouched
    assert out.iloc[1]["feature_home_ppg"] == 4.5   # unresolved MLB keeps default
    assert "stats_source" not in out.columns or pd.isna(out.iloc[1].get("stats_source"))


def test_enrich_never_raises_and_empty_stats_noops():
    df = pd.DataFrame([{"league": "MLB", "home_team": "New York Yankees", "away_team": "Boston Red Sox", "feature_home_ppg": 4.5}])
    assert enrich_mlb_model_features(df, stats={}).iloc[0]["feature_home_ppg"] == 4.5
    assert enrich_mlb_model_features(None, stats={}) is None


def test_fetch_returns_empty_on_http_error():
    def boom(*a, **k):
        raise __import__("requests").RequestException("down")
    assert fetch_mlb_team_stats(2026, http_get=boom) == {}
