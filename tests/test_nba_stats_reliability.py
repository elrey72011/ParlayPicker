import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app_core.feature_processing as fp


def test_nba_minnesota_resolves_to_timberwolves_not_wild():
    assert fp.normalize_team_for_stats("Minnesota", "NBA") == "MINNESOTA TIMBERWOLVES"


def test_nba_new_york_resolves_correctly():
    assert fp.normalize_team_for_stats("New York", "NBA") == "NEW YORK KNICKS"


def test_nba_toronto_cleveland_denver_resolve_correctly():
    assert fp.normalize_team_for_stats("Toronto", "NBA") == "TORONTO RAPTORS"
    assert fp.normalize_team_for_stats("Cleveland", "NBA") == "CLEVELAND CAVALIERS"
    assert fp.normalize_team_for_stats("Denver", "NBA") == "DENVER NUGGETS"


def test_league_specific_normalization_prevents_cross_league_alias_pollution():
    assert fp.normalize_team_for_stats("Minnesota", "NBA") != "MINNESOTA WILD"
    assert fp.normalize_team_for_stats("Minnesota", "NHL") == "MINNESOTA WILD"


def test_colorado_resolves_differently_by_league():
    assert fp.normalize_team_for_stats("Colorado", "NHL") == "COLORADO AVALANCHE"
    assert fp.normalize_team_for_stats("Colorado", "MLB") == "COLORADO ROCKIES"


def test_stats_index_canonicalization_and_resolver_matches_city_only_nba_mlb():
    stats_df = pd.DataFrame(
        [
            {"team_norm": "Minnesota", "league_key": "NBA"},
            {"team_norm": "Colorado", "league_key": "MLB"},
        ]
    )
    canonical = fp.canonicalize_stats_team_index(stats_df)
    nba_subset = canonical[canonical["league_key"] == "NBA"].set_index("stats_team_key")
    mlb_subset = canonical[canonical["league_key"] == "MLB"].set_index("stats_team_key")

    nba_maps = fp._build_stats_index_maps(nba_subset, "NBA")
    mlb_maps = fp._build_stats_index_maps(mlb_subset, "MLB")

    nba_direct = {idx: idx for idx in nba_subset.index}
    mlb_direct = {idx: idx for idx in mlb_subset.index}

    nba_match, nba_reason, nba_stage = fp.resolve_stats_team_match(
        "MINNESOTA TIMBERWOLVES",
        "NBA",
        nba_direct,
        nba_maps["canonical"],
        nba_maps["city_only"],
    )
    mlb_match, mlb_reason, mlb_stage = fp.resolve_stats_team_match(
        "COLORADO ROCKIES",
        "MLB",
        mlb_direct,
        mlb_maps["canonical"],
        mlb_maps["city_only"],
    )

    assert nba_match == "minnesota timberwolves"
    assert nba_reason == "direct"
    assert nba_stage == "resolved"
    assert mlb_match == "colorado rockies"
    assert mlb_reason == "direct"
    assert mlb_stage == "resolved"


def test_fetch_nba_stats_retries_before_fallback(monkeypatch):
    fp._NBA_STATS_RUNTIME_CACHE.clear()

    class FakeEndpoint:
        calls = 0

        def __init__(self, **kwargs):
            FakeEndpoint.calls += 1
            if FakeEndpoint.calls < 3:
                raise TimeoutError("stats timeout")

        def get_data_frames(self):
            return [
                pd.DataFrame(
                    [
                        {
                            "TEAM_NAME": "Atlanta Hawks",
                            "GP": 1,
                            "PTS": 120,
                            "PLUS_MINUS": 5,
                            "W_PCT": 0.6,
                            "TOV": 12,
                            "AST": 25,
                            "REB": 45,
                        }
                    ]
                )
            ]

    class FakeModule:
        LeagueDashTeamStats = FakeEndpoint

    monkeypatch.setattr(fp, "leaguedashteamstats", FakeModule)
    monkeypatch.setattr(fp.time, "sleep", lambda *_args, **_kwargs: None)

    stats = fp.fetch_nba_stats(2025)
    diag = fp.get_nba_fetch_diagnostics()

    assert len(stats) == 1
    assert FakeEndpoint.calls == 3
    assert diag["status"] == "ok"
    assert diag["retries_used"] == 2


def test_unresolved_nba_rows_marked_and_ml_ineligible(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(
            [
                {
                    "team_norm": "ATLANTA HAWKS",
                    "league_key": "NBA",
                    "win_pct": 0.6,
                    "home_win_pct": 0.6,
                    "away_win_pct": 0.6,
                    "points_per_game": 115.0,
                    "points_allowed_per_game": 110.0,
                    "turnovers": 12.0,
                    "streak": 0.0,
                    "last5_win_pct": 0.6,
                }
            ]
        )

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)
    fp._NBA_FETCH_DIAGNOSTICS.update({"status": "ok", "source": "live", "retries_used": 1, "last_error": ""})

    games = pd.DataFrame(
        [
            {
                "league": "NBA",
                "home_team": "Atlanta",
                "away_team": "New York",
                "market_type": "h2h_home",
                "decimal_odds": 1.91,
                "odds_american": -110,
            }
        ]
    )

    enriched = fp.enrich_with_model_features(games, api_clients={})

    assert "stats_source" in enriched.columns
    assert "stats_resolution_status" in enriched.columns
    assert "stats_fallback_reason" in enriched.columns
    assert "ml_feature_eligible" in enriched.columns
    assert "stats_fetch_retries_used" in enriched.columns

    assert enriched.loc[0, "stats_resolution_status"] == "unresolved"
    assert enriched.loc[0, "stats_source"] == "fallback"
    assert enriched.loc[0, "stats_fallback_reason"] == "team_mapping_unresolved"
    assert bool(enriched.loc[0, "ml_feature_eligible"]) is False
    assert bool(enriched.loc[0, "feature_stats_fallback"]) is True
    assert pd.isna(enriched.loc[0, "feature_home_win_pct"])
    assert pd.isna(enriched.loc[0, "feature_away_win_pct"])
    assert str(enriched.loc[0, "stats_resolution_stage_failure"]) != ""


def test_aggregated_stats_diagnostics_populate(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(
            [
                {
                    "team_norm": "ATLANTA HAWKS",
                    "league_key": "NBA",
                    "win_pct": 0.6,
                    "home_win_pct": 0.6,
                    "away_win_pct": 0.6,
                    "points_per_game": 115.0,
                    "points_allowed_per_game": 110.0,
                    "turnovers": 12.0,
                    "streak": 0.0,
                    "last5_win_pct": 0.6,
                }
            ]
        )

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)

    games = pd.DataFrame(
        [
            {"league": "NBA", "home_team": "Atlanta", "away_team": "New York", "market_type": "h2h_home", "decimal_odds": 1.91, "odds_american": -110},
            {"league": "NBA", "home_team": "Atlanta", "away_team": "Unknown Team", "market_type": "h2h_home", "decimal_odds": 1.91, "odds_american": -110},
        ]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={})

    assert "stats_unresolved_count_by_league" in enriched.columns
    assert "stats_ml_excluded_rows" in enriched.columns
    assert "stats_source_counts" in enriched.columns
    assert int(enriched.loc[0, "stats_unresolved_count_by_league"]) >= 1
    assert int(enriched.loc[0, "stats_ml_excluded_rows"]) >= 1
    assert "fallback" in str(enriched.loc[0, "stats_source_counts"])
    assert "stats_resolution_stage_failure_counts" in enriched.columns
    assert "after_fuzzy" in str(enriched.loc[0, "stats_resolution_stage_failure_counts"]) or "stats_index_lookup" in str(enriched.loc[0, "stats_resolution_stage_failure_counts"])
