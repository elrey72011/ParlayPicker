"""Retained catalog projection; event/quote price fixtures are SYNTHETIC."""
from collections import defaultdict
from contextlib import closing
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch
import pytest
from app_core import football_stage1 as stage,football_stage1_cycle as cycle

NOW=datetime(2026,10,6,9,tzinfo=timezone.utc)
START=datetime(2026,10,10,18,tzinfo=timezone.utc)
FIXTURE=Path(__file__).parent/'fixtures/football/cfbd-alternate-names-retained.json'

def teams():return [r['catalog_projection'] for r in json.loads(FIXTURE.read_text())['teams'].values()]
def game():return dict(id=401866440,homeId=113,awayId=193,homeTeam='Massachusetts',awayTeam='Miami (OH)',startDate=START.isoformat(),season=2026,week=6,seasonType='regular',neutralSite=False,completed=False)
def nfl():return dict(id='healthy-nfl',date=START.isoformat(),season=dict(year=2026,type=2),week=dict(number=6),status=dict(type=dict(completed=False)),competitions=[dict(date=START.isoformat(),neutralSite=False,competitors=[dict(homeAway='home',team=dict(id='8',displayName='Atlanta Falcons'),score='0'),dict(homeAway='away',team=dict(id='9',displayName='Green Bay Packers'),score='0')])])
def offer(sport='NCAAF',home='UMass Minutemen',away='Miami (OH) RedHawks'):
    return dict(id='SYNTHETIC-'+sport,sport_key=stage.SPORT_KEYS[sport],commence_time=START.isoformat(),home_team=home,away_team=away,bookmakers=[dict(key='SYNTHETIC_BOOK',markets=[dict(key='spreads',last_update=(NOW-timedelta(minutes=1)).isoformat(),outcomes=[dict(name=home,point=-1.5,price=-110),dict(name=away,point=1.5,price=-110)]),dict(key='totals',last_update=(NOW-timedelta(minutes=1)).isoformat(),outcomes=[dict(name='Over',point=44.5,price=-110),dict(name='Under',point=44.5,price=-110)])])])

class Response:
    status_code=200
    headers={}
    def __init__(self,payload):self.payload=payload
    def json(self):return self.payload

def run(path,*,catalog=None,ncaaf_offer=None,games=None):
    calls=[]
    def get(url,*,params,headers=None,timeout,allow_redirects):
        assert timeout==15 and allow_redirects is False
        calls.append(url)
        if url==cycle.ESPN:return Response({'events':[nfl()] if params['dates']=='20261010' else []})
        if url==cycle.CFBD+'/games':return Response([game()] if games is None else games)
        if url==cycle.CFBD+'/teams/fbs':return Response(teams() if catalog is None else catalog)
        if url.endswith('/americanfootball_nfl/odds'):return Response([offer('NFL','Atlanta Falcons','Green Bay Packers')])
        if url.endswith('/americanfootball_ncaaf/odds'):return Response([offer() if ncaaf_offer is None else ncaaf_offer])
        raise AssertionError(url)
    def sync(*args,**kwargs):return dict(records_restored=0,remote_records_read=0,records_verified=0,new_records_verified=0)
    with patch.object(cycle.prospective_remote,'sync',side_effect=sync):
        report=cycle.run_cycle(path,'SYNTHETIC_FOLDER',object(),'SYNTHETIC_KEY','SYNTHETIC_KEY',now=NOW,get=get)
    assert len(calls)==19 and calls.count(cycle.CFBD+'/teams/fbs')==1
    assert report['sports']['NFL']['requested_slate_success']
    return report

def test_retained_alternate_names_through_actual_cycle(tmp_path):
    path=tmp_path/'synthetic.sqlite3';report=run(path)
    assert report['requested_slate_success']
    diag=report['sports']['NCAAF']['provider_events'][0]
    assert diag['canonical_match']=='ncaaf:cfbd:401866440' and diag['classification']=='MATCHED_TARGET'
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        quotes=db.execute("SELECT game_id,selection,line,american_odds,provider_last_update,observed_at,identity_mapping_hash FROM prospective_football_quote WHERE sport='NCAAF'").fetchall()
        assert len(quotes)==4 and {q[0] for q in quotes}=={'ncaaf:cfbd:401866440'}
        assert all(q[4]==(NOW-timedelta(minutes=1)).isoformat() and q[5]==NOW.isoformat() for q in quotes)
        assert {q[1] for q in quotes}=={'UMass Minutemen','Miami (OH) RedHawks','Over','Under'}
        assert all(q[6] for q in quotes)
        for table in ('prospective_prediction','prospective_model','prospective_calibration','prospective_deployment_review'):
            assert db.execute('SELECT count(*) FROM '+table).fetchone()[0]==0

def test_repeat_cycle_preserves_existing_quotes_and_clocks(tmp_path):
    path=tmp_path/'synthetic.sqlite3'
    first=run(path)
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        before=db.execute('SELECT * FROM prospective_football_quote ORDER BY quote_id').fetchall()
    second=run(path)
    with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        assert db.execute('SELECT * FROM prospective_football_quote ORDER BY quote_id').fetchall()==before
    assert first['requested_slate_success'] and second['requested_slate_success']
    assert second['sports']['NCAAF']['denominator']['games'][0]['status']=='HORIZON_ALREADY_CAPTURED'

def test_collision_does_not_force_match(tmp_path):
    catalog=teams()+[dict(id=999,school='Distinct University',mascot='Minutemen',alternateNames=['UMass'])]
    report=run(tmp_path/'synthetic.sqlite3',catalog=catalog)
    assert not report['requested_slate_success']
    assert report['sports']['NCAAF']['provider_events'][0]['classification']=='TEAM_IDENTITY_MISMATCH'

@pytest.mark.parametrize('value',['UMass',{'name':'UMass'},[None,False,{},['UMass']]])
def test_malformed_alternate_names_are_not_aliases(tmp_path,value):
    catalog=teams()
    next(t for t in catalog if t['id']==113)['alternateNames']=value
    report=run(tmp_path/'synthetic.sqlite3',catalog=catalog)
    assert not report['sports']['NCAAF']['requested_slate_success']

@pytest.mark.parametrize('change,classification',[
    ('kickoff','KICKOFF_TIME_REVISION'),('wrong_sport','PROVIDER_DATA_INVALID'),
    ('unlisted_school','TEAM_IDENTITY_MISMATCH'),('reversed','SCHEDULE_EVENT_MISSING'),
])
def test_identity_and_kickoff_guards_remain(tmp_path,change,classification):
    quote=offer()
    if change=='kickoff':quote['commence_time']=(START+timedelta(hours=1)).isoformat()
    elif change=='wrong_sport':quote['sport_key']='americanfootball_nfl'
    elif change=='unlisted_school':quote['home_team']='Unverified University Minutemen'
    else:quote['home_team'],quote['away_team']=quote['away_team'],quote['home_team']
    report=run(tmp_path/'synthetic.sqlite3',ncaaf_offer=quote)
    diag=report['sports']['NCAAF']['provider_events'][0]
    assert diag['classification']==classification and not diag['persisted']
    assert not report['sports']['NCAAF']['requested_slate_success']
    if change=='kickoff':assert diag['kickoff_delta_seconds']==3600

@pytest.mark.parametrize('change',['bad_price','future','missing_clock'])
def test_changed_prices_and_clocks_remain_rejected(tmp_path,change):
    quote=offer()
    for market in quote['bookmakers'][0]['markets']:
        if change=='bad_price':market['outcomes'][0]['price']=0
        elif change=='future':market['last_update']=(NOW+timedelta(minutes=1)).isoformat()
        else:market.pop('last_update')
    report=run(tmp_path/'synthetic.sqlite3',ncaaf_offer=quote)
    diag=report['sports']['NCAAF']['provider_events'][0]
    assert diag['classification']=='MATCHED_TARGET' and not diag['pregame_valid']
    assert not report['sports']['NCAAF']['requested_slate_success']

def test_ambiguous_games_remain_rejected(tmp_path):
    second=dict(game(),id=401866441)
    report=run(tmp_path/'synthetic.sqlite3',games=[game(),second])
    assert report['sports']['NCAAF']['provider_events'][0]['classification']=='AMBIGUOUS_MATCH'
    assert not report['sports']['NCAAF']['requested_slate_success']

def test_unlisted_hawaii_spelling_remains_unverified():
    catalog=teams()
    def get(url,**kwargs):return Response([game()] if url.endswith('/games') else catalog)
    _,_,_,aliases,_=cycle._ncaaf_schedule(NOW,'SYNTHETIC',get=get,ledger=defaultdict(lambda:defaultdict(int)))
    assert stage._name('NCAAF','Hawaii Rainbow Warriors') not in aliases['62']
    assert stage._name('NCAAF',"Hawai'i Rainbow Warriors") in aliases['62']

def test_legacy_catalog_shape_still_works():
    catalog=[dict(id=113,school='Massachusetts',alt_name='UMass',mascot='Minutemen')]
    def get(url,**kwargs):return Response([game()] if url.endswith('/games') else catalog)
    _,_,_,aliases,_=cycle._ncaaf_schedule(NOW,'SYNTHETIC',get=get,ledger=defaultdict(lambda:defaultdict(int)))
    assert stage._name('NCAAF','Massachusetts Minutemen') in aliases['113']
    assert stage._name('NCAAF','UMass Minutemen') in aliases['113']
