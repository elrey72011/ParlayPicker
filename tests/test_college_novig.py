from copy import deepcopy
from app_core.college_novig import recover_college_novig


def game():
    return dict(id='event1',home_team='Kansas',away_team='Missouri',commence_time='2099-09-11T23:00:00Z',bookmakers=[{'key':'draftkings','markets':[]}])


def test_recovers_only_exact_event_novig_and_preserves_original():
    original=game();payload=deepcopy(original)
    payload['bookmakers']=[{'key':'novig','last_update':'2099-09-11T22:00:00Z','markets':[{'key':'spreads','outcomes':[{'name':'Kansas','point':4.5,'price':-110}]}]}]
    calls=[]
    class Response:
        status_code=200
        def json(self):return payload
    def get(url,**kw):calls.append(kw);return Response()
    result=recover_college_novig([original],'test',get=get)
    assert len(original['bookmakers'])==1
    assert result[0]['bookmakers'][1]==payload['bookmakers'][0]
    assert calls[0]['params']['bookmakers']=='novig'
    payload['home_team']='Other'
    assert recover_college_novig([original],'test',get=get)==[original]


def test_bounded_missing_response_and_existing_complete_quotes():
    calls=[]
    class Response:
        status_code=404
    def get(*a,**k):calls.append(1);return Response()
    games=[dict(game(),id='event'+str(i)) for i in range(8)]
    assert recover_college_novig(games,'test',get=get)==games
    assert len(calls)==5
    complete=game();complete['bookmakers']=[{'key':'novig','markets':[{'key':k,'outcomes':[{}]} for k in ('spreads','totals')]}]
    calls.clear();assert recover_college_novig([complete],'test',get=get)==[complete]
    assert not calls
