import ast
import pandas as pd

from app_core import feature_processing as fp


def _mock_stats_frame():
    return pd.DataFrame(
        [
            {
                "team_norm": "MINNESOTA",
                "league_key": "NBA",
                "win_pct": 0.6,
                "home_win_pct": 0.62,
                "away_win_pct": 0.58,
                "points_per_game": 112.0,
                "points_allowed_per_game": 107.0,
                "turnovers": 12.0,
                "streak": 2.0,
                "last5_win_pct": 0.8,
                "source": "LIVE",
            },
            {
                "team_norm": "DENVER",
                "league_key": "NBA",
                "win_pct": 0.7,
                "home_win_pct": 0.72,
                "away_win_pct": 0.68,
                "points_per_game": 115.0,
                "points_allowed_per_game": 108.0,
                "turnovers": 11.0,
                "streak": 3.0,
                "last5_win_pct": 0.8,
                "source": "LIVE",
            },
            {
                "team_norm": "COLORADO",
                "league_key": "MLB",
                "win_pct": 0.5,
                "home_win_pct": 0.52,
                "away_win_pct": 0.48,
                "points_per_game": 4.2,
                "points_allowed_per_game": 4.7,
                "turnovers": 0.0,
                "streak": -1.0,
                "last5_win_pct": 0.4,
                "source": "LIVE",
            },
            {
                "team_norm": "SAN DIEGO",
                "league_key": "MLB",
                "win_pct": 0.55,
                "home_win_pct": 0.57,
                "away_win_pct": 0.53,
                "points_per_game": 4.8,
                "points_allowed_per_game": 4.1,
                "turnovers": 0.0,
                "streak": 1.0,
                "last5_win_pct": 0.6,
                "source": "LIVE",
            },
        ]
    )


def test_resolver_handles_minnesota_and_colorado_aliases():
    nba_subset = pd.DataFrame(index=["MINNESOTA", "DENVER"])
    nba_maps = fp._build_stats_index_maps(nba_subset, "NBA")
    nba_stats_map = {"minnesota timberwolves": "minnesota", "denver nuggets": "denver"}

    nba_match, nba_reason, nba_stage = fp.resolve_stats_team_match(
        "Minnesota",
        "NBA",
        nba_stats_map,
        nba_maps["canonical"],
        nba_maps["city_only"],
    )
    assert nba_match == "minnesota"
    assert nba_reason in {"direct", "canonical_alias", "city_alias"}
    assert nba_stage == "resolved"

    mlb_subset = pd.DataFrame(index=["COLORADO", "SAN DIEGO"])
    mlb_maps = fp._build_stats_index_maps(mlb_subset, "MLB")
    mlb_stats_map = {"colorado rockies": "colorado", "san diego padres": "san diego"}

    mlb_match, mlb_reason, mlb_stage = fp.resolve_stats_team_match(
        "Colorado",
        "MLB",
        mlb_stats_map,
        mlb_maps["canonical"],
        mlb_maps["city_only"],
    )
    assert mlb_match == "colorado"
    assert mlb_reason in {"direct", "canonical_alias", "city_alias"}
    assert mlb_stage == "resolved"


def test_enrich_normalizes_stats_index_and_resolves_city_inputs(monkeypatch):
    monkeypatch.setattr(fp, "fetch_team_stats", lambda *_args, **_kwargs: _mock_stats_frame())

    games = pd.DataFrame(
        [
            {"Home": "Minnesota", "Away": "Denver", "sport_title": "NBA"},
            {"Home": "Colorado", "Away": "San Diego", "sport_title": "MLB"},
        ]
    )

    enriched = fp.enrich_with_model_features(games, api_clients={}, season_year=2025)

    assert (enriched["stats_resolution_status"] == "resolved").all()
    assert (~enriched["feature_stats_fallback"]).all()
    assert enriched["ml_feature_eligible"].all()


def test_unresolved_rows_are_flagged_and_ml_stats_are_nan(monkeypatch):
    unresolved_stats = _mock_stats_frame()[_mock_stats_frame()["team_norm"] != "MINNESOTA"]
    monkeypatch.setattr(fp, "fetch_team_stats", lambda *_args, **_kwargs: unresolved_stats)

    games = pd.DataFrame(
        [
            {"Home": "Minnesota", "Away": "Denver", "sport_title": "NBA"},
            {"Home": "Colorado", "Away": "San Diego", "sport_title": "MLB"},
        ]
    )

    enriched = fp.enrich_with_model_features(games, api_clients={}, season_year=2025)

    nba_row = enriched.iloc[0]
    assert nba_row["stats_source"] == "fallback"
    assert nba_row["stats_resolution_status"] == "unresolved"
    assert nba_row["stats_fallback_reason"] == "team_mapping_unresolved"
    assert nba_row["ml_feature_eligible"] == False
    assert pd.isna(nba_row["feature_home_win_pct"])


def test_aggregated_diagnostics_populate(monkeypatch):
    unresolved_stats = _mock_stats_frame()[_mock_stats_frame()["team_norm"] != "MINNESOTA"]
    monkeypatch.setattr(fp, "fetch_team_stats", lambda *_args, **_kwargs: unresolved_stats)

    games = pd.DataFrame(
        [
            {"Home": "Minnesota", "Away": "Denver", "sport_title": "NBA"},
            {"Home": "Colorado", "Away": "San Diego", "sport_title": "MLB"},
        ]
    )

    enriched = fp.enrich_with_model_features(games, api_clients={}, season_year=2025)
    first = enriched.iloc[0]

    assert int(first["stats_ml_excluded_rows"]) >= 1
    assert int(first["stats_unresolved_count_by_league"]) >= 1

    source_counts = ast.literal_eval(str(first["stats_source_counts"]))
    assert source_counts["live"] >= 0
    assert source_counts["fallback"] >= 1
    assert "stats_resolution_stage_failure" in enriched.columns
    assert "stats_resolution_stage_failure_counts" in enriched.columns
