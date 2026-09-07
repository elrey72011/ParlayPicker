from copy import deepcopy
import pytest
from app_core.ncaaf_identity import normalize_ncaaf_team
from app_core.ncaaf_prospective import _match

PAIRS = [
    ("Kansas Jayhawks", "Missouri Tigers", "Kansas", "Missouri"),
    ("East Carolina Pirates", "Appalachian State Mountaineers", "East Carolina", "App State"),
    ("Army Black Knights", "South Florida Bulls", "Army", "South Florida"),
    ("Illinois Fighting Illini", "Duke Blue Devils", "Illinois", "Duke"),
    ("Vanderbilt Commodores", "Delaware Blue Hens", "Vanderbilt", "Delaware"),
    ("Liberty Flames", "Gardner-Webb Runnin Bulldogs", "Liberty", "Gardner-Webb"),
    ("Nebraska Cornhuskers", "Bowling Green Falcons", "Nebraska", "Bowling Green"),
    ("Kennesaw State Owls", "Georgia State Panthers", "Kennesaw State", "Georgia State"),
    ("Houston Cougars", "Southern University Jaguars", "Houston", "Southern"),
    ("Sam Houston State Bearkats", "Tulsa Golden Hurricane", "Sam Houston", "Tulsa"),
    ("Auburn Tigers", "Southern Mississippi Golden Eagles", "Auburn", "Southern Miss"),
    ("TCU Horned Frogs", "Grambling State Tigers", "TCU", "Grambling"),
    ("San Jose State Spartans", "Cal Poly Mustangs", "San José State", "Cal Poly"),
    ("Air Force Falcons", "North Dakota State Bison", "Air Force", "North Dakota State"),
]


@pytest.mark.parametrize('oh,oa,ch,ca', PAIRS)
def test_exact_provider_pairs_and_gates(oh,oa,ch,ca):
    event=dict(home_team=oh,away_team=oa,commence_time='2026-10-10T12:00:00Z')
    game=dict(homeTeam=ch,awayTeam=ca,startDate=event['commence_time'],startTimeTBD=False,completed=False,id=1,homeId=2,awayId=3)
    assert _match(event,[game])==game
    assert _match(event,[game,deepcopy(game)]) is None
    assert _match(event,[dict(game,startDate='2026-10-10T12:01:01Z')]) is None
    assert _match(event,[dict(game,completed=True)]) is None
    assert _match(event,[dict(game,startTimeTBD=True)]) is None
    assert _match(dict(event,home_team=oa,away_team=oh),[game]) is None


@pytest.mark.parametrize('a,b', [('Southern','Southern Miss'),('North Dakota','North Dakota State'),('Sam Houston','Houston'),('Missouri','Missouri State')])
def test_distinct_schools_not_collapsed(a,b):
    assert normalize_ncaaf_team(a)!=normalize_ncaaf_team(b)


def test_alias_idempotence_and_missing_name():
    for pair in PAIRS:
        for name in pair:
            canonical=normalize_ncaaf_team(name)
            assert normalize_ncaaf_team(canonical)==canonical
    assert normalize_ncaaf_team(None)==''
