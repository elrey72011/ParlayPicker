"""Explicit synthetic dependency injection; never imported by production UI."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo
import csv
import io
import json

from app_core.nfl_private_workflow import Workflow, authenticate
from app_core.nfl_workflow_store import Store

TOKEN = 'SYNTHETIC-owner-token-private-workflow-only'
KEY = b'SYNTHETIC-store-signing-key-no-production-authority'


class Stream:
    def __init__(self, body, content_type):
        self.body=body;self.status=200
        self.headers={'content-type':content_type,'content-length':str(len(body))}
    def chunks(self):
        yield self.body
    def close(self):
        pass


class SyntheticProvider:
    def __init__(self, *, anchor=None):
        self.calls=[];self.line=2.5;self.price=-110;self.wrong=False;self.stale=False;self.fail=False
        current=anchor or datetime.now(timezone.utc)
        self.rows=[];self.targets=[]
        for index,(home,away) in enumerate([('DAL','BUF'),('PHI','KC')]):
            for past in range(6):
                day=(current-timedelta(days=7*(past+1))).date().isoformat()
                for code,opp in [(home,'HOU'),(away,'NE')]:
                    self.rows.append(dict(game_id='SYNTHETIC-HISTORY-'+code+'-'+str(past),season=str(current.year),
                        gameday=day,gametime='13:00',home_team=code,away_team=opp,home_score=str(24+past),
                        away_score='20',result=str(4+past),location='Home',stadium='SYNTHETIC venue'))
            local=(current+timedelta(hours=12+index*2)).astimezone(ZoneInfo('America/New_York')).replace(second=0,microsecond=0)
            row=dict(game_id='SYNTHETIC-UNCATALOGUED-'+home+'-'+away,season=str(current.year),gameday=local.date().isoformat(),
                gametime=local.strftime('%H:%M'),home_team=home,away_team=away,home_score='',away_score='',result='',location='Home',stadium='SYNTHETIC venue')
            self.rows.append(row);self.targets.append((row,local.astimezone(timezone.utc)))
        self.names={'DAL':'Dallas Cowboys','BUF':'Buffalo Bills','PHI':'Philadelphia Eagles','KC':'Kansas City Chiefs'}

    def scores(self,request,limits,deadline,monotonic):
        self.calls.append(('scores',request))
        if self.fail:raise ConnectionRefusedError('SYNTHETIC retained failure')
        text=io.StringIO(newline='');writer=csv.DictWriter(text,fieldnames=list(self.rows[0]));writer.writeheader();writer.writerows(self.rows)
        return Stream(text.getvalue().encode(),'text/csv')

    def __call__(self,request,limits,deadline,monotonic):
        self.calls.append(('market',request))
        if self.fail:raise ConnectionRefusedError('SYNTHETIC retained failure')
        kickoff=datetime.fromisoformat(request['params']['commenceTimeFrom'].replace('Z','+00:00'))+timedelta(seconds=1)
        row,start=next((row,start) for row,start in self.targets if start==kickoff)
        home=self.names[row['home_team']];away=self.names[row['away_team']]
        if self.wrong:home,away=away,home
        updated=datetime.now(timezone.utc)-timedelta(minutes=20 if self.stale else 0)
        body=json.dumps([dict(id='SYNTHETIC-ODDS-'+row['game_id'],sport_key='americanfootball_nfl',
            commence_time=start.strftime('%Y-%m-%dT%H:%M:%SZ'),home_team=home,away_team=away,
            bookmakers=[dict(key='novig',markets=[dict(key='spreads',last_update=updated.isoformat(),
                outcomes=[dict(name=home,point=self.line,price=self.price),dict(name=away,point=-self.line,price=self.price)])])])]).encode()
        return Stream(body,'application/json')

    def validate_body(self,body):
        assert b'SYNTHETIC' in body


def make_workflow(path, *, provider=None):
    provider=provider or SyntheticProvider()
    path=Path(path).resolve()
    if path.exists():
        def factory(readonly):
            return sqlite3.connect(path.as_uri()+('?mode=ro' if readonly else '?mode=rw'),uri=True,timeout=5)
        store=Store(factory,KEY,synthetic=True)
    else:
        store=Store.synthetic_fixture(path,KEY)
    access=authenticate(TOKEN,TOKEN)
    service=Workflow(store,access,allow_acquisition=True,allow_inference=True,
        score_transport=provider.scores,market_transport=provider)
    return service,provider


def capture_source(service):
    operation=service.source_operation()
    end=(datetime.now(timezone.utc)+timedelta(days=3)).isoformat()
    return service.capture_source(operation['id'],private_retention=True,private_research=True,permission_end=end)


def capture_two(service):
    capture_source(service)
    ids=[row['event']['canonical_event_id'] for row in service.games()['games']]
    prepared=service.prepare(ids)
    operation_ids=[row['operation']['id'] for row in prepared]
    service.capture_quotes(operation_ids,remaining_credits=2,
        reset_at=(datetime.now(timezone.utc)+timedelta(days=10)).isoformat(),private_use=True,no_overage=True)
    return service.stage(operation_ids)
