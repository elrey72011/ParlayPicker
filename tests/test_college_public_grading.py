"""College final-score matching must not reuse cross-sport lock identities."""
from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import pytest

from app_core.public_history import grade_leg, grading_team_name, event_key, fetch_scores


@pytest.mark.parametrize("saved,provider", [
    ("Florida", "Florida Gators"),
    ("Colorado State", "Colorado State Rams"),
    ("Tennessee State", "Tennessee State Tigers"),
    ("Tennessee Tech", "Tennessee Tech Golden Eagles"),
    ("San Jose State", "San José State Spartans"),
    ("Southern Mississippi Golden", "Southern Miss Golden Eagles"),
    ("Prairie View A&M", "Prairie View A&M Panthers"),
    ("Ul Monroe", "UL Monroe Warhawks"),
    ("Western Kentucky", "Western Kentucky Hilltoppers"),
    ("Ball State", "Ball State Cardinals"),
    ("Delaware", "Delaware Blue Hens"),
    ("Louisiana", "Louisiana Ragin' Cajuns"),
    ("Western Illinois", "Western Illinois Leathernecks"),
    ("Wisconsin", "Wisconsin Badgers"),
    ("Umass", "Massachusetts Minutemen"),
    ("Massachusetts", "Massachusetts Minutemen"),
    ("Stephen F Austin", "Stephen F. Austin Lumberjacks"),
    ("Bethunecookman", "Bethune-Cookman Wildcats"),
    ("North Dakota", "North Dakota Fighting Hawks"),
    ("East Texas Am Lions", "East Texas A&M Lions"),
    ("Arkansaspine Bluff", "Arkansas-Pine Bluff Golden Lions"),
    ("Nicholls State", "Nicholls Colonels"),
    ("Southeastern Louisiana", "SE Louisiana Lions"),
    ("Morehead State", "Morehead State Eagles"),
    ("Portland State", "Portland State Vikings"),
])
def test_college_exact_provider_alias_grades_original_spread(saved, provider):
    leg = dict(sport="NCAAF", game=f"{saved} at Auburn", pick=f"{saved} +3.5",
               market="spread_away", start="2026-09-12T23:00:00+00:00")
    score = dict(sport="NCAAF", event_id="fixture", away=provider, home="Auburn Tigers",
                 away_score=20, home_score=23, start=leg["start"])
    original = deepcopy(leg)
    identity = event_key(leg)
    assert grade_leg(leg, [score])[0] == "WIN"
    assert leg == original and event_key(leg) == identity
    leg.update(pick=f"{saved} +2.5")
    assert grade_leg(leg, [score])[0] == "LOSS"
    leg.update(pick="Under 43", market="total_under")
    assert grade_leg(leg, [score])[0] == "PUSH"
    leg.update(pick=saved, market="moneyline_away")
    assert grade_leg(leg, [score])[0] == "LOSS"


@pytest.mark.parametrize("a,b", [("Tennessee", "Tennessee State"), ("Tennessee", "Tennessee Tech"),
    ("Colorado", "Colorado State"), ("Florida", "Florida International"),
    ("Miami", "Miami (OH)"), ("North Dakota", "North Dakota State")])
def test_distinct_college_programs_never_share_result_identity(a,b):
    assert grading_team_name(a,"NCAAF") != grading_team_name(b,"NCAAF")


def delayed_game():
    leg=dict(sport="NCAAF", game="Georgia Southern at Clemson", pick="Under 50.5",
             market="total_under", start="2026-09-12T23:00:00+00:00")
    score=dict(sport="NCAAF", event_id="delayed", away="Georgia Southern Eagles",
               home="Clemson Tigers", away_score=7, home_score=22,
               start="2026-09-13T01:30:00+00:00")
    return leg,score


def test_delayed_college_kickoff_matches_same_eastern_day_only():
    leg,score=delayed_game()
    assert grade_leg(leg,[score])[0] == "WIN"
    assert grade_leg(leg,[score,{**score,"event_id":"ambiguous"}])[0] == "PENDING"
    assert grade_leg(leg,[{**score,"start":"2026-09-14T01:30:00+00:00"}])[0] == "PENDING"
    assert grade_leg(leg,[{**score,"away":score["home"],"home":score["away"]}])[0] == "PENDING"
    assert grade_leg({**leg,"pick":"Clemson line unresolved","market":"spread_home"},[score])[0] == "PENDING"


def test_baseball_doubleheader_time_guard_is_unchanged():
    leg=dict(sport="MLB",game="Seattle at Boston",pick="Over 7.5",market="total_over",
             start="2026-09-12T17:00:00+00:00")
    score=dict(sport="MLB",event_id="late-game",away="Seattle Mariners",home="Boston Red Sox",
               away_score=6,home_score=4,start="2026-09-12T23:00:00+00:00")
    assert grade_leg(leg,[score])[0] == "PENDING"


def test_fetch_preserves_college_source_names_and_requires_final(monkeypatch):
    games=[]
    for identity,status in [("final","STATUS_FINAL"),("suspended","STATUS_SUSPENDED")]:
        games.append(dict(id=identity,competitions=[dict(date="2026-09-12T23:00:00Z",
            status=dict(type=dict(completed=True,state="post",name=status)),competitors=[
                dict(homeAway="away",score="20",team=dict(displayName="Tennessee State Tigers")),
                dict(homeAway="home",score="23",team=dict(displayName="Colorado State Rams"))])]))
    monkeypatch.setattr("requests.get", lambda *a,**k: SimpleNamespace(
        raise_for_status=lambda: None,json=lambda:dict(events=games)))
    scores=fetch_scores(date(2026,9,12),{"NCAAF"})["scores"]
    assert len(scores)==1  # The default and FCS views repeat the same final.
    assert scores[0]["away"]=="Tennessee State Tigers"
    assert scores[0]["home"]=="Colorado State Rams"


def test_new_score_revision_repairs_saved_history_without_replacing_pick():
    from test_public_history import pub
    from app_core.public_history import digest, report
    original=pub()
    original['package']['games']={'overall':[dict(
        sport='NCAAF',game='Campbell at Florida',pick='Under 60.5',market='total_under',
        odds=-110,status='PASS',start='2026-09-09T20:00:00+00:00',
        as_of='2026-09-09T19:55:00+00:00')]}
    original['package_hash']=digest(original['package'])
    preserved=deepcopy(original)
    old_score=dict(sport='NCAAF',event_id='same-event',away='CAMPBELL FIGHTING CAMELS',
                   home='FLORIDA GATORS',away_score=6,home_score=44,
                   start='2026-09-09T21:30:00+00:00')
    old_revision=dict(recorded_at='2026-09-10T01:00:00Z',scores=[old_score])
    corrected={**old_score,'away':'Campbell Fighting Camels','home':'Florida Gators'}
    new_revision=dict(recorded_at='2026-09-10T02:00:00Z',scores=[corrected])
    rows=report([original],[old_revision,new_revision])
    assert len(rows)==1 and rows[0]['outcome']=='WIN'
    assert rows[0]['picks']=='Campbell at Florida: Under 60.5'
    assert rows[0]['odds']=='-110'
    assert original==preserved
