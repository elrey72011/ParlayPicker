import pandas as pd

from app_core.feature_processing import (
    canonicalize_stats_team_index,
    normalize_team_for_stats,
    resolve_stats_team_match,
    _build_stats_index_maps,
)


def test_nhl_pittsburgh_normalizes_to_penguins_keyspace():
    assert normalize_team_for_stats("Pittsburgh", "NHL") == "PITTSBURGH PENGUINS"


def test_nhl_pittsburgh_resolves_without_unresolved_fallback_when_stats_row_exists():
    stats_df = pd.DataFrame(
        [
            {"team_norm": "Pittsburgh Penguins", "league_key": "NHL"},
            {"team_norm": "Philadelphia Flyers", "league_key": "NHL"},
        ]
    )
    canonical = canonicalize_stats_team_index(stats_df)
    stats_subset = pd.DataFrame(index=canonical["stats_team_key"].tolist())
    stats_norm_map = {k: k for k in stats_subset.index}
    maps = _build_stats_index_maps(stats_subset, "NHL")

    match_key, method, phase = resolve_stats_team_match(
        "Pittsburgh",
        "NHL",
        stats_norm_map=stats_norm_map,
        canonical_alias_map=maps["canonical"],
        city_alias_map=maps["city_only"],
    )

    assert match_key == "pittsburgh penguins"
    assert method in {"direct", "canonical_alias", "city_alias"}
    assert phase == "resolved"
