import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app_core.feature_processing as fp
from app_core import weights_config


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


def test_nhl_boston_and_buffalo_and_mlb_arizona_normalize_to_expected_keys():
    assert fp.normalize_team_for_stats("Boston", "NHL") == "BOSTON BRUINS"
    assert fp.normalize_team_for_stats("Buffalo", "NHL") == "BUFFALO SABRES"
    assert fp.normalize_team_for_stats("Arizona", "MLB") == "ARIZONA DIAMONDBACKS"


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


def test_fetch_nba_stats_retries_before_fallback(monkeypatch, tmp_path):
    fp._NBA_STATS_RUNTIME_CACHE.clear()
    fp._NBA_STATS_SUCCESS_ARCHIVE.clear()
    monkeypatch.chdir(tmp_path)

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


def test_nba_fetch_failure_uses_same_day_cached_payload(monkeypatch):
    fp._NBA_STATS_RUNTIME_CACHE.clear()
    fp._NBA_STATS_SUCCESS_ARCHIVE.clear()
    slate_day = "2026-04-28"
    cached_payload = [{"team_norm": "ATLANTA HAWKS", "league_key": "NBA", "win_pct": 0.55, "home_win_pct": 0.55, "away_win_pct": 0.55, "points_per_game": 114.0, "points_allowed_per_game": 111.0, "assists_per_game": 24.0, "rebounds_per_game": 44.0, "turnovers": 12.0, "streak": 0.0, "last5_win_pct": 0.6}]
    fp._NBA_STATS_SUCCESS_ARCHIVE[2025] = {"slate_day": slate_day, "stats": cached_payload}

    class AlwaysFailEndpoint:
        def __init__(self, **kwargs):
            raise TimeoutError("stats timeout")

    class FakeModule:
        LeagueDashTeamStats = AlwaysFailEndpoint

    class FakeDateTime:
        @staticmethod
        def utcnow():
            return pd.Timestamp(slate_day)

    monkeypatch.setattr(fp, "leaguedashteamstats", FakeModule)
    monkeypatch.setattr(fp, "datetime", FakeDateTime)
    monkeypatch.setattr(fp.time, "sleep", lambda *_args, **_kwargs: None)

    stats = fp.fetch_nba_stats(2025)
    diag = fp.get_nba_fetch_diagnostics()
    assert stats == cached_payload
    assert diag["source"] == "runtime_cache"
    assert diag["status"] == "ok"


def test_nba_fetch_diagnostics_distinguish_live_cached_failed(monkeypatch, tmp_path):
    fp._NBA_STATS_RUNTIME_CACHE.clear()
    fp._NBA_STATS_SUCCESS_ARCHIVE.clear()
    monkeypatch.chdir(tmp_path)

    class SuccessEndpoint:
        def __init__(self, **kwargs):
            pass

        def get_data_frames(self):
            return [pd.DataFrame([{"TEAM_NAME": "Atlanta Hawks", "GP": 1, "PTS": 110, "PLUS_MINUS": 2, "W_PCT": 0.5, "TOV": 12, "AST": 20, "REB": 40}])]

    class SuccessModule:
        LeagueDashTeamStats = SuccessEndpoint

    monkeypatch.setattr(fp, "leaguedashteamstats", SuccessModule)
    assert fp.fetch_nba_stats(2025)
    assert fp.get_nba_fetch_diagnostics()["source"] == "live"
    assert fp.fetch_nba_stats(2025)
    assert fp.get_nba_fetch_diagnostics()["source"] == "runtime_cache"

    fp._NBA_STATS_RUNTIME_CACHE.clear()
    fp._NBA_STATS_SUCCESS_ARCHIVE.clear()

    class FailEndpoint:
        def __init__(self, **kwargs):
            raise TimeoutError("down")

    class FailModule:
        LeagueDashTeamStats = FailEndpoint

    class NextDayDateTime:
        @staticmethod
        def utcnow():
            return pd.Timestamp("2026-04-29")

    monkeypatch.setattr(fp, "datetime", NextDayDateTime)
    monkeypatch.setattr(fp, "leaguedashteamstats", FailModule)
    monkeypatch.setattr(fp.time, "sleep", lambda *_args, **_kwargs: None)
    recovered = fp.fetch_nba_stats(2026)
    diag_source = fp.get_nba_fetch_diagnostics()["source"]
    assert diag_source in {"failed", "disk_cache"}
    if diag_source == "failed":
        assert recovered == []
    else:
        assert recovered != []


def test_nba_fetch_failure_uses_same_day_disk_cache(monkeypatch, tmp_path):
    fp._NBA_STATS_RUNTIME_CACHE.clear()
    fp._NBA_STATS_SUCCESS_ARCHIVE.clear()
    slate_day = "2026-04-28"
    monkeypatch.chdir(tmp_path)

    class FakeDateTime:
        @staticmethod
        def utcnow():
            return pd.Timestamp(slate_day)

    class FailEndpoint:
        def __init__(self, **kwargs):
            raise TimeoutError("down")

    class FailModule:
        LeagueDashTeamStats = FailEndpoint

    disk_payload = [
        {
            "team_norm": "ATLANTA HAWKS",
            "league_key": "NBA",
            "win_pct": 0.55,
            "home_win_pct": 0.55,
            "away_win_pct": 0.55,
            "points_per_game": 114.0,
            "points_allowed_per_game": 111.0,
            "assists_per_game": 24.0,
            "rebounds_per_game": 44.0,
            "turnovers": 12.0,
            "streak": 0.0,
            "last5_win_pct": 0.6,
        }
    ]
    fp._save_nba_disk_cache(2025, slate_day, disk_payload)

    monkeypatch.setattr(fp, "datetime", FakeDateTime)
    monkeypatch.setattr(fp, "leaguedashteamstats", FailModule)
    monkeypatch.setattr(fp.time, "sleep", lambda *_args, **_kwargs: None)

    stats = fp.fetch_nba_stats(2025)
    diag = fp.get_nba_fetch_diagnostics()
    assert stats == disk_payload
    assert diag["source"] == "disk_cache"
    assert diag["status"] == "ok"


def test_nba_fetch_failure_can_recover_from_previous_season_same_day_archive(monkeypatch):
    fp._NBA_STATS_RUNTIME_CACHE.clear()
    fp._NBA_STATS_RUNTIME_CACHE_DAY.clear()
    fp._NBA_STATS_SUCCESS_ARCHIVE.clear()
    slate_day = "2026-04-28"
    cached_payload = [{"team_norm": "ATLANTA HAWKS", "league_key": "NBA", "win_pct": 0.55}]
    fp._NBA_STATS_SUCCESS_ARCHIVE[2025] = {"slate_day": slate_day, "stats": cached_payload}

    class FakeDateTime:
        @staticmethod
        def utcnow():
            return pd.Timestamp(slate_day)

    class FailEndpoint:
        def __init__(self, **kwargs):
            raise TimeoutError("down")

    class FailModule:
        LeagueDashTeamStats = FailEndpoint

    monkeypatch.setattr(fp, "datetime", FakeDateTime)
    monkeypatch.setattr(fp, "leaguedashteamstats", FailModule)
    monkeypatch.setattr(fp.time, "sleep", lambda *_args, **_kwargs: None)

    stats = fp.fetch_nba_stats(2026)
    diag = fp.get_nba_fetch_diagnostics()
    assert stats == cached_payload
    assert diag["source"] == "runtime_cache"
    assert int(diag.get("recovered_from_season", 0)) == 2025


def test_cached_success_does_not_inflate_unresolved_or_fallback(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(
            [
                {"team_norm": "BOSTON CELTICS", "league_key": "NBA", "win_pct": 0.61, "home_win_pct": 0.61, "away_win_pct": 0.61, "points_per_game": 118.0, "points_allowed_per_game": 111.0, "turnovers": 12.0, "streak": 0.0, "last5_win_pct": 0.6},
                {"team_norm": "PHILADELPHIA 76ERS", "league_key": "NBA", "win_pct": 0.58, "home_win_pct": 0.58, "away_win_pct": 0.58, "points_per_game": 114.0, "points_allowed_per_game": 109.0, "turnovers": 11.0, "streak": 0.0, "last5_win_pct": 0.6},
            ]
        )

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)
    fp._NBA_FETCH_DIAGNOSTICS.update({"status": "ok", "source": "runtime_cache", "retries_used": 3, "last_error": "timeout"})

    games = pd.DataFrame(
        [
            {"league": "NBA", "home_team": "Boston", "away_team": "Philadelphia", "market_type": "h2h_home", "decimal_odds": 1.91, "odds_american": -110}
        ]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={})

    assert enriched.loc[0, "stats_resolution_status"] == "resolved"
    assert enriched.loc[0, "stats_source"] == "cached"
    assert enriched.loc[0, "fallback_summary_by_league"] in {"{}", "{ }"}
    assert int(enriched.loc[0, "stats_unresolved_count_by_league"]) == 0


def test_nba_fetch_status_source_and_retries_reflect_final_source(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(
            [
                {"team_norm": "BOSTON CELTICS", "league_key": "NBA", "win_pct": 0.61, "home_win_pct": 0.61, "away_win_pct": 0.61, "points_per_game": 118.0, "points_allowed_per_game": 111.0, "turnovers": 12.0, "streak": 0.0, "last5_win_pct": 0.6},
                {"team_norm": "ATLANTA HAWKS", "league_key": "NBA", "win_pct": 0.52, "home_win_pct": 0.52, "away_win_pct": 0.52, "points_per_game": 114.0, "points_allowed_per_game": 113.0, "turnovers": 12.0, "streak": 0.0, "last5_win_pct": 0.5},
            ]
        )

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)
    fp._NBA_FETCH_DIAGNOSTICS.update({"status": "ok", "source": "disk_cache", "retries_used": 3, "last_error": "timeout"})

    games = pd.DataFrame(
        [
            {"league": "NBA", "home_team": "Boston", "away_team": "Atlanta", "market_type": "h2h_home", "decimal_odds": 1.91, "odds_american": -110}
        ]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={})

    assert enriched.loc[0, "nba_stats_fetch_status"] == "cached"
    assert enriched.loc[0, "nba_stats_fetch_source"] == "disk_cache"
    assert int(enriched.loc[0, "nba_stats_fetch_retries_used"]) == 3
    assert int(enriched.loc[0, "stats_unresolved_count_by_league"]) == 0
    assert "NBA" not in str(enriched.loc[0, "fallback_summary_by_league"])


def test_live_fail_cache_success_rows_show_cached_not_failed(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(
            [
                {"team_norm": "BOSTON CELTICS", "league_key": "NBA", "win_pct": 0.61, "home_win_pct": 0.61, "away_win_pct": 0.61, "points_per_game": 118.0, "points_allowed_per_game": 111.0, "turnovers": 12.0, "streak": 0.0, "last5_win_pct": 0.6},
                {"team_norm": "ATLANTA HAWKS", "league_key": "NBA", "win_pct": 0.52, "home_win_pct": 0.52, "away_win_pct": 0.52, "points_per_game": 114.0, "points_allowed_per_game": 113.0, "turnovers": 12.0, "streak": 0.0, "last5_win_pct": 0.5},
            ]
        )

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)
    fp._NBA_FETCH_DIAGNOSTICS.update({"status": "ok", "source": "runtime_cache", "retries_used": 3, "last_error": "timeout"})
    games = pd.DataFrame(
        [{"league": "NBA", "home_team": "Boston", "away_team": "Atlanta", "market_type": "h2h_home", "decimal_odds": 1.91, "odds_american": -110}]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={})
    assert enriched.loc[0, "nba_stats_fetch_status"] == "cached"
    assert enriched.loc[0, "nba_stats_fetch_source"] == "runtime_cache"
    assert int(enriched.loc[0, "stats_unresolved_count_by_league"]) == 0
    assert "NBA" not in str(enriched.loc[0, "fallback_summary_by_league"])


def test_live_fail_no_cache_rows_show_failed_with_warning(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(columns=["team_norm", "league_key"])

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)
    fp._NBA_FETCH_DIAGNOSTICS.update({"status": "failed", "source": "failed", "retries_used": 3, "last_error": "timeout"})
    games = pd.DataFrame(
        [
            {"league": "NBA", "home_team": "Boston", "away_team": "Philadelphia", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
            {"league": "NBA", "home_team": "New York", "away_team": "Atlanta", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
            {"league": "NBA", "home_team": "San Antonio", "away_team": "Portland", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
        ]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={})
    assert (enriched["nba_stats_fetch_status"].astype(str) == "failed").all()
    assert "NBA" in str(enriched.loc[0, "fallback_summary_by_league"])
    assert str(enriched.loc[0, "run_health_warning"]).strip() != ""


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
    assert "nba_stats_fetch_source" in enriched.columns

    assert enriched.loc[0, "stats_resolution_status"] == "unresolved"
    assert enriched.loc[0, "stats_source"] == "fallback"
    assert enriched.loc[0, "stats_fallback_reason"] == "team_mapping_unresolved"
    assert bool(enriched.loc[0, "ml_feature_eligible"]) is False
    assert bool(enriched.loc[0, "feature_stats_fallback"]) is True
    assert pd.isna(enriched.loc[0, "feature_home_win_pct"])
    assert pd.isna(enriched.loc[0, "feature_away_win_pct"])
    assert str(enriched.loc[0, "stats_resolution_stage_failure"]) != ""
    assert str(enriched.loc[0, "nba_stats_fetch_source"]) == "live"


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
    assert "stats_unresolved_count_by_league_detail" in enriched.columns
    assert "fallback_summary_by_league" in enriched.columns
    assert "stats_resolution_stage_failure_counts" in enriched.columns
    assert "after_fuzzy" in str(enriched.loc[0, "stats_resolution_stage_failure_counts"]) or "stats_index_lookup" in str(enriched.loc[0, "stats_resolution_stage_failure_counts"])


def test_nba_fetch_failure_marks_rows_unresolved_and_nans_features(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(columns=["team_norm", "league_key"])

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)
    fp._NBA_FETCH_DIAGNOSTICS.update({"status": "failed", "source": "live", "retries_used": 3, "last_error": "timeout"})

    games = pd.DataFrame(
        [
            {"league": "NBA", "home_team": "Atlanta", "away_team": "New York", "market_type": "h2h_home", "decimal_odds": 1.91, "odds_american": -110}
        ]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={})

    assert enriched.loc[0, "stats_resolution_status"] == "unresolved"
    assert enriched.loc[0, "stats_source"] == "failed"
    assert enriched.loc[0, "stats_fallback_reason"] == "nba_stats_fetch_failed"
    assert enriched.loc[0, "stats_resolution_stage_failure"] == "nba_stats_fetch_failed"
    assert bool(enriched.loc[0, "feature_stats_fallback"]) is True
    assert bool(enriched.loc[0, "ml_feature_eligible"]) is False
    assert pd.isna(enriched.loc[0, "feature_home_win_pct"])
    assert pd.isna(enriched.loc[0, "feature_away_win_pct"])


def test_nhl_boston_buffalo_and_mlb_arizona_resolve_without_fallback(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(
            [
                {"team_norm": "Boston Bruins", "league_key": "NHL", "win_pct": 0.6, "home_win_pct": 0.6, "away_win_pct": 0.6, "points_per_game": 3.2, "points_allowed_per_game": 2.6, "turnovers": 0.0, "streak": 0.0, "last5_win_pct": 0.6},
                {"team_norm": "Buffalo Sabres", "league_key": "NHL", "win_pct": 0.5, "home_win_pct": 0.5, "away_win_pct": 0.5, "points_per_game": 3.1, "points_allowed_per_game": 3.0, "turnovers": 0.0, "streak": 0.0, "last5_win_pct": 0.5},
                {"team_norm": "Arizona Diamondbacks", "league_key": "MLB", "win_pct": 0.55, "home_win_pct": 0.55, "away_win_pct": 0.55, "points_per_game": 4.8, "points_allowed_per_game": 4.3, "turnovers": 0.0, "streak": 0.0, "last5_win_pct": 0.6},
                {"team_norm": "Milwaukee Brewers", "league_key": "MLB", "win_pct": 0.52, "home_win_pct": 0.52, "away_win_pct": 0.52, "points_per_game": 4.6, "points_allowed_per_game": 4.4, "turnovers": 0.0, "streak": 0.0, "last5_win_pct": 0.5},
            ]
        )

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)
    games = pd.DataFrame(
        [
            {"league": "NHL", "home_team": "Buffalo", "away_team": "Boston", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
            {"league": "MLB", "home_team": "Milwaukee", "away_team": "Arizona", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
        ]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={})
    assert (enriched["stats_resolution_status"] == "resolved").all()
    assert not enriched["feature_stats_fallback"].any()


def test_run_health_warning_on_fallback_heavy_slate(monkeypatch):
    def fake_fetch_team_stats(_api_clients, season_year=None):
        return pd.DataFrame(columns=["team_norm", "league_key"])

    monkeypatch.setattr(fp, "fetch_team_stats", fake_fetch_team_stats)
    fp._NBA_FETCH_DIAGNOSTICS.update({"status": "failed", "source": "failed", "retries_used": 3, "last_error": "timeout"})
    games = pd.DataFrame(
        [
            {"league": "NBA", "home_team": "Boston", "away_team": "Philadelphia", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
            {"league": "NBA", "home_team": "New York", "away_team": "Atlanta", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
            {"league": "NHL", "home_team": "Buffalo", "away_team": "Boston", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
            {"league": "MLB", "home_team": "Milwaukee", "away_team": "Arizona", "market_type": "h2h_home", "decimal_odds": 1.9, "odds_american": -110},
        ]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={})
    assert bool(enriched.loc[0, "fallback_heavy_slate_flag"]) is True
    assert "Run health warning" in str(enriched.loc[0, "run_health_warning"])


def test_best_picks_calibration_constants_unchanged():
    # Re-baselined to the current intended calibration. These changed deliberately:
    # NBA_OVER_ACTIONABLE_BONUS was retired to 0.0 (Overs no longer get a side bonus),
    # and MLB_OVER_ACTIONABLE_MIN_PROB was raised 0.57 -> 0.65 to fix Over overconfidence.
    # The guard's job is to catch *accidental* drift from these values going forward.
    assert weights_config.NBA_SIDE_ACTIONABLE_BONUS == 0.01
    assert weights_config.NBA_OVER_ACTIONABLE_BONUS == 0.0
    assert weights_config.MLB_OVER_ACTIONABLE_MIN_PROB == 0.65
