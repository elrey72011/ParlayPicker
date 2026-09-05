import pandas as pd
import pytest

from core.team_mapper import normalize_team_name


@pytest.mark.parametrize("aliases,canonical", [
    (["Houston Baptist", "Houston Baptist Huskies", "Houston Christian", "Houston Christian Huskies"], "Houston Christian"),
    (["Citadel", "Citadel Bulldogs", "The Citadel", "The Citadel Bulldogs"], "Citadel"),
    (["Nicholls", "Nicholls Colonels", "Nicholls State", "Nicholls State Colonels", "Nicholls St Colonels"], "Nicholls State"),
    (["SE Louisiana", "SE Louisiana Lions", "Southeastern Louisiana", "Southeastern Louisiana Lions"], "Southeastern Louisiana"),
])
def test_college_feed_aliases_share_a_stable_matchup_identity(aliases, canonical):
    from core.streamlit_pipeline import _canonical_matchup_key

    assert {normalize_team_name(alias) for alias in aliases} == {canonical}
    assert normalize_team_name(canonical) == canonical
    frame = pd.DataFrame({
        "league": "NCAAF", "home_team": "Rice Owls", "away_team": aliases,
        "game_date": "2026-09-05",
    })
    assert _canonical_matchup_key(frame).nunique() == 1


def test_nba_mappings():
    assert normalize_team_name("Detroit Pistons") == "Detroit"
    assert normalize_team_name("Memphis Grizzlies") == "Memphis"
    assert normalize_team_name("Golden State Warriors") == "Golden State"

def test_nhl_mappings():
    assert normalize_team_name("Toronto Maple Leafs") == "Toronto"
    assert normalize_team_name("Utah Hockey Club") == "Utah"
    assert normalize_team_name("Boston Bruins") == "Boston"

def test_ncaab_mappings():
    assert normalize_team_name("Prairie View A&M Panthers") == "Prairie View A&M"
    assert normalize_team_name("Utah Tech Trailblazers") == "Utah Tech"
