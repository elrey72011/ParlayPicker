from core.team_mapper import normalize_team_name

def test_nba_mappings():
    assert normalize_team_name("Detroit Pistons") == "Detroit"
    assert normalize_team_name("Memphis Grizzlies") == "Memphis"
    assert normalize_team_name("Golden State Warriors") == "Golden State"

def test_nhl_mappings():
    assert normalize_team_name("Toronto Maple Leafs") == "Toronto"
    assert normalize_team_name("Utah Hockey Club") == "Utah"
    assert normalize_team_name("Boston Bruins") == "Boston"

def test_ncaab_mappings():
    assert normalize_team_name("Prairie View A&M Panthers") == "Prairie View A&M Panthers"
    assert normalize_team_name("Utah Tech Trailblazers") == "Utah Tech"
