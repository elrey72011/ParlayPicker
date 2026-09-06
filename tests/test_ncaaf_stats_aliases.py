import pandas as pd
import pytest

from app_core.feature_processing import normalize_team_for_stats, canonicalize_stats_team_index, resolve_stats_team_match


@pytest.mark.parametrize('short,full', [
    ('Notre Dame', 'Notre Dame Fighting Irish'),
    ('Wisconsin', 'Wisconsin Badgers'),
    ('Ole Miss', 'Ole Miss Rebels'),
    ('Louisville', 'Louisville Cardinals'),
    ('Florida Am', 'Florida A&M Rattlers'),
])
def test_football_aliases_resolve_directly_when_source_row_exists(short, full):
    stats = canonicalize_stats_team_index(pd.DataFrame([{'team_norm': full, 'league_key': 'NCAAF'}]))
    keys = {key: key for key in stats.stats_team_key}
    match, method, stage = resolve_stats_team_match(short, 'NCAAF', keys, {}, {})
    assert match == normalize_team_for_stats(short, 'NCAAF').lower()
    assert method == 'direct'
    assert stage == 'resolved'


def test_missing_notre_dame_stats_remain_unresolved():
    match, _, _ = resolve_stats_team_match('Notre Dame', 'NCAAF', {}, {}, {})
    assert match is None


def test_state_and_nonstate_teams_stay_distinct():
    assert normalize_team_for_stats('Washington State Cougars', 'NCAAF') != normalize_team_for_stats('Washington Huskies', 'NCAAF')
    assert normalize_team_for_stats('South Carolina State Bulldogs', 'NCAAF') != normalize_team_for_stats('South Carolina', 'NCAAF')


def test_upload_identity_is_stable_across_mascots_and_reversed_orientation():
    from core.streamlit_pipeline import _matchup_id, _canonical_matchup_key, _canonical_matchup_teams_key
    rows = pd.DataFrame([
        {"league": "NCAAF", "home_team": "Notre Dame", "away_team": "Wisconsin", "game_date": "2026-09-06"},
        {"league": "NCAAF", "home_team": "Wisconsin Badgers", "away_team": "Notre Dame Fighting Irish", "game_date": "2026-09-06"},
    ])
    for key in (_matchup_id, _canonical_matchup_key, _canonical_matchup_teams_key):
        assert key(rows).nunique() == 1
    rows.loc[1, "game_date"] = "2026-09-07"
    assert _matchup_id(rows).nunique() == 2
