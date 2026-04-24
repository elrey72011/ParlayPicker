import ast
import pandas as pd
import pytest

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


@pytest.mark.parametrize(
    ("league", "raw_name", "expected"),
    [
        ("NHL", "Montreal", "MONTREAL CANADIENS"),
        ("NHL", "Anaheim", "ANAHEIM DUCKS"),
        ("NHL", "Edmonton", "EDMONTON OILERS"),
        ("NHL", "Utah", "UTAH HOCKEY CLUB"),
        ("MLB", "Houston", "HOUSTON ASTROS"),
        ("MLB", "Detroit", "DETROIT TIGERS"),
        ("MLB", "Baltimore", "BALTIMORE ORIOLES"),
        ("MLB", "Boston", "BOSTON RED SOX"),
        ("MLB", "Miami", "MIAMI MARLINS"),
        ("MLB", "Washington", "WASHINGTON NATIONALS"),
        ("MLB", "St Louis", "ST LOUIS CARDINALS"),
        ("MLB", "Saint Louis", "ST LOUIS CARDINALS"),
        ("MLB", "Texas", "TEXAS RANGERS"),
        ("MLB", "Milwaukee", "MILWAUKEE BREWERS"),
        ("MLB", "Pittsburgh", "PITTSBURGH PIRATES"),
        ("MLB", "Seattle", "SEATTLE MARINERS"),
        ("MLB", "Atlanta", "ATLANTA BRAVES"),
    ],
)
def test_city_forms_normalize_to_expected_full_team_names(league, raw_name, expected):
    assert fp.normalize_team_for_stats(raw_name, league) == expected


def test_canonical_fetched_stats_keys_bind_city_inputs_for_mlb_and_nhl(monkeypatch):
    stats_df = pd.DataFrame(
        [
            {"team_norm": "Montreal Canadiens", "league_key": "NHL", "win_pct": 0.55, "home_win_pct": 0.55, "away_win_pct": 0.55, "points_per_game": 3.2, "points_allowed_per_game": 2.9, "turnovers": 0.0, "streak": 1.0, "last5_win_pct": 0.6},
            {"team_norm": "Utah Hockey Club", "league_key": "NHL", "win_pct": 0.50, "home_win_pct": 0.50, "away_win_pct": 0.50, "points_per_game": 2.8, "points_allowed_per_game": 3.1, "turnovers": 0.0, "streak": -1.0, "last5_win_pct": 0.4},
            {"team_norm": "Houston Astros", "league_key": "MLB", "win_pct": 0.58, "home_win_pct": 0.58, "away_win_pct": 0.58, "points_per_game": 4.9, "points_allowed_per_game": 4.1, "turnovers": 0.0, "streak": 2.0, "last5_win_pct": 0.6},
            {"team_norm": "St. Louis Cardinals", "league_key": "MLB", "win_pct": 0.52, "home_win_pct": 0.52, "away_win_pct": 0.52, "points_per_game": 4.5, "points_allowed_per_game": 4.4, "turnovers": 0.0, "streak": 0.0, "last5_win_pct": 0.5},
        ]
    )
    monkeypatch.setattr(fp, "fetch_team_stats", lambda *_args, **_kwargs: stats_df)

    games = pd.DataFrame(
        [
            {"Home": "Montreal", "Away": "Utah", "sport_title": "NHL"},
            {"Home": "Houston", "Away": "Saint Louis", "sport_title": "MLB"},
        ]
    )
    enriched = fp.enrich_with_model_features(games, api_clients={}, season_year=2025)

    assert (enriched["stats_resolution_status"] == "resolved").all()
    assert (~enriched["feature_stats_fallback"]).all()
    assert enriched["ml_feature_eligible"].all()


def test_unresolved_rows_remain_out_of_ml_and_emit_binding_diagnostics(monkeypatch):
    stats_df = pd.DataFrame(
        [
            {"team_norm": "Houston Astros", "league_key": "MLB", "win_pct": 0.58, "home_win_pct": 0.58, "away_win_pct": 0.58, "points_per_game": 4.9, "points_allowed_per_game": 4.1, "turnovers": 0.0, "streak": 2.0, "last5_win_pct": 0.6},
        ]
    )
    monkeypatch.setattr(fp, "fetch_team_stats", lambda *_args, **_kwargs: stats_df)

    games = pd.DataFrame([{"Home": "Houston", "Away": "Seattle", "sport_title": "MLB"}])
    enriched = fp.enrich_with_model_features(games, api_clients={}, season_year=2025)
    row = enriched.iloc[0]

    assert row["stats_resolution_status"] == "unresolved"
    assert row["stats_source"] == "fallback"
    assert row["stats_fallback_reason"] == "team_mapping_unresolved"
    assert bool(row["ml_feature_eligible"]) is False
    assert pd.isna(row["feature_home_win_pct"])
    assert pd.isna(row["feature_away_win_pct"])
    assert "stats_binding_failures_by_league" in enriched.columns
    assert "stats_match_counts_by_method" in enriched.columns


def test_utah_nhl_variant_stats_row_binds_without_fallback(monkeypatch):
    stats_df = pd.DataFrame(
        [
            {"team_norm": "Utah HC", "league_key": "NHL", "win_pct": 0.50, "home_win_pct": 0.50, "away_win_pct": 0.50, "points_per_game": 2.9, "points_allowed_per_game": 3.0, "turnovers": 0.0, "streak": 0.0, "last5_win_pct": 0.5},
            {"team_norm": "Vegas Golden Knights", "league_key": "NHL", "win_pct": 0.62, "home_win_pct": 0.62, "away_win_pct": 0.62, "points_per_game": 3.4, "points_allowed_per_game": 2.7, "turnovers": 0.0, "streak": 1.0, "last5_win_pct": 0.7},
        ]
    )
    monkeypatch.setattr(fp, "fetch_team_stats", lambda *_args, **_kwargs: stats_df)

    games = pd.DataFrame([{"Home": "Utah", "Away": "Vegas", "sport_title": "NHL"}])
    enriched = fp.enrich_with_model_features(games, api_clients={}, season_year=2025)
    row = enriched.iloc[0]

    assert row["stats_resolution_status"] == "resolved"
    assert bool(row["feature_stats_fallback"]) is False
    assert bool(row["ml_feature_eligible"]) is True
