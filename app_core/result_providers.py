"""Public final scores with provider provenance and MLB schedule fallback."""
import math
import requests
from app_core.result_reconciliation import stamp


def fetch_results(day, sports):
    from app_core.public_history import now
    from app_core.espn_results import ESPN_ENDPOINTS, _scoreboard_urls
    at = now()
    scores, events, errors = {}, {}, []
    for sport in sorted(set(sports)):
        if sport not in ESPN_ENDPOINTS:
            errors.append({'sport':sport,'source':'none','reason':'UNSUPPORTED_SPORT'})
            continue
        for url in _scoreboard_urls(sport, day.strftime('%Y%m%d')):
            try:
                response = requests.get(url,timeout=10); response.raise_for_status()
                for event in response.json().get('events',[]):
                    for game in event.get('competitions',[]):
                        status = game.get('status',{}).get('type',{})
                        completed = bool(status.get('completed') and status.get('state') == 'post' and str(status.get('name','')).startswith('STATUS_FINAL'))
                        teams = {t.get('homeAway'):t for t in game.get('competitors',[])}
                        start = game.get('date') or event.get('date')
                        if set(teams) != {'away','home'} or not stamp(start): continue
                        row = dict(sport=sport,event_id=str(event['id']),provider_event_id=str(event['id']),result_source='ESPN',provider_recorded_at=at,
                                   start=stamp(start).isoformat(),away=teams['away']['team']['displayName'],home=teams['home']['team']['displayName'],completed=completed)
                        events[(sport,event['id'])] = row
                        if completed:
                            try:
                                a,h = (float(teams[k]['score']) for k in ('away','home'))
                                if not all(math.isfinite(x) and x >= 0 and x.is_integer() for x in (a,h)): raise ValueError()
                            except (ValueError,TypeError,KeyError):
                                errors.append({'sport':sport,'source':'ESPN','reason':'FINAL_SCORE_INVALID'}); continue
                            scores[(sport,event['id'])] = dict(row,away_score=int(a),home_score=int(h))
            except (requests.RequestException, ValueError, KeyError, TypeError):
                errors.append({'sport':sport,'source':'ESPN','reason':'PROVIDER_FAILURE'})
    # One official MLB schedule request on explicit MLB grading also supplies
    # unfinished doubleheader events, preventing a unique-final false match.
    if 'MLB' in sports:
        try:
            response=requests.get('https://statsapi.mlb.com/api/v1/schedule',params={'sportId':1,'date':day.isoformat()},timeout=10)
            response.raise_for_status()
            for group in response.json().get('dates',[]):
                for game in group.get('games',[]):
                    if not stamp(game.get('gameDate')): continue
                    complete=game.get('status',{}).get('abstractGameState') == 'Final'
                    row=dict(sport='MLB',event_id=str(game['gamePk']),provider_event_id=str(game['gamePk']),result_source='MLB',provider_recorded_at=at,
                             start=stamp(game['gameDate']).isoformat(),away=game['teams']['away']['team']['name'],home=game['teams']['home']['team']['name'],completed=complete,game_number=game.get('gameNumber'),official_date=game.get('officialDate') or group.get('date'))
                    events[('MLB', 'mlb:'+str(game['gamePk']))]=row
                    if complete:
                        a,h=(game['teams'][k].get('score') for k in ('away','home'))
                        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 or int(x)!=x for x in (a,h)):
                            errors.append({'sport':'MLB','source':'MLB','reason':'FINAL_SCORE_INVALID'});continue
                        scores[('MLB','mlb:'+str(game['gamePk']))]=dict(row,away_score=int(a),home_score=int(h))
        except (requests.RequestException,ValueError,KeyError,TypeError):
            errors.append({'sport':'MLB','source':'MLB','reason':'PROVIDER_FAILURE'})
    return {'recorded_at':at,'scores':list(scores.values()),'events':list(events.values()),'errors':errors}
