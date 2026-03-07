from app_core.kalshi_integrator import _det_team_code


def test_ncaab_alias_overrides_examples():
    assert _det_team_code("NCAAB", "Manhattan") == "MAN"
    assert _det_team_code("NCAAB", "Wagner") == "WAG"
    assert _det_team_code("NCAAB", "Princeton") == "PRIN"
    assert _det_team_code("NCAAB", "Vermont") == "UVM"
    assert _det_team_code("NCAAB", "Washington State") == "WSU"
    assert _det_team_code("NCAAB", "Seton Hall") == "HALL"


def test_ncaab_alias_st_johns_and_state_variants():
    assert _det_team_code("NCAAB", "St John's") == "SJU"
    assert _det_team_code("NCAAB", "Saint Johns") == "SJU"
    assert _det_team_code("NCAAB", "Idaho St") == "IDST"
    assert _det_team_code("NCAAB", "Sam Houston State") == "SHSU"
    assert _det_team_code("NCAAB", "Florida Gulf Coast") == "FGCU"
    assert _det_team_code("NCAAB", "Kennesaw State") == "KENN"


def test_ncaab_alias_matchup_examples_map_to_expected_codes():
    assert _det_team_code("NCAAB", "Manhattan") == "MAN"
    assert _det_team_code("NCAAB", "Wagner") == "WAG"

    assert _det_team_code("NCAAB", "Princeton") == "PRIN"
    assert _det_team_code("NCAAB", "Vermont") == "UVM"

    assert _det_team_code("NCAAB", "Washington St") == "WSU"
    assert _det_team_code("NCAAB", "Seton Hall") == "HALL"

    assert _det_team_code("NCAAB", "Idaho St") == "IDST"
    assert _det_team_code("NCAAB", "Sam Houston St") == "SHSU"

    assert _det_team_code("NCAAB", "FGCU") == "FGCU"
    assert _det_team_code("NCAAB", "Kennesaw St") == "KENN"


def test_ncaab_alias_additional_mappings():
    assert _det_team_code("NCAAB", "Columbia") == "CLMB"
    assert _det_team_code("NCAAB", "Saint Mary's") == "SMC"
