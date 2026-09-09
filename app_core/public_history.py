"""Immutable, site-scoped public publication records and conservative result grading."""
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app_core.result_team_names import normalize_result_team


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


class History:
    def __init__(self, site, folder, client=None):
        from app_core.netlify_publishing import identifier
        from app_core.evidence_drive import DriveStore
        if not folder:
            raise ValueError('Configure Shared Drive storage before tracking public publications.')
        self.client = client or DriveStore(folder)
        self.prefix = 'parlaypicker/public-history-v1/' + identifier(site) + '/'

    def read(self, key):
        body=self.client.get_object(Key=self.prefix+key)['Body']
        try:
            return json.loads(body.read())
        finally:
            body.close()

    def put(self, key, value, first=False):
        raw=encoded(value)
        try:
            self.client.put_object(Key=self.prefix+key,Body=raw,IfNoneMatch='*')
        except Exception as exc:
            if getattr(exc,'response',{}).get('Error',{}).get('Code') not in {'412','PreconditionFailed'}:
                raise
        saved=self.read(key)
        if not first and saved!=value:
            raise ValueError('Public history conflict; preserve the existing record.')
        return saved

    def archive(self, package):
        from app_core.public_board import validate_package
        validate_package(package)
        key=digest(package)
        self.put('packages/'+key+'.json',package)
        return key

    def submitted(self, deploy_id, package_hash):
        from app_core.netlify_publishing import identifier
        return self.put('deployments/'+identifier(deploy_id)+'.json',
                        {'deploy_id':deploy_id,'package_hash':package_hash})

    def confirm(self, deploy_id, package_hash, confirmed_at=None):
        from app_core.netlify_publishing import identifier
        value={'package_hash':package_hash,'confirmed_at':confirmed_at or now()}
        saved=self.put('confirmed/'+identifier(deploy_id)+'.json',value,first=True)
        if saved['package_hash']!=package_hash:
            raise ValueError('Deployment history conflict')
        return saved

    def all(self, kind):
        values=[]
        for page in self.client.get_paginator('list_objects_v2').paginate(Prefix=self.prefix+kind+'/'):
            for item in page.get('Contents',[]):
                values.append(self.read(item['Key'][len(self.prefix):]))
        return values

    def publications(self):
        result=[]
        for receipt in self.all('confirmed'):
            package=self.read('packages/'+receipt['package_hash']+'.json')
            if digest(package)!=receipt['package_hash']:
                raise ValueError('Public history hash mismatch')
            result.append({**receipt,'package':package})
        return result


def event_key(leg):
    teams=re.split(r'\s+(?:at|@)\s+',leg['game'],flags=re.I)
    if len(teams)!=2 or not leg.get('start'):
        return None
    return (leg['sport'].upper(), *(normalize_result_team(t) for t in teams), datetime.fromisoformat(leg['start']).isoformat())


def eligible(leg, confirmed):
    try:
        at=datetime.fromisoformat(leg['as_of']);start=datetime.fromisoformat(leg['start'])
        return event_key(leg) is not None and at<=confirmed<start and (confirmed-at).total_seconds()<=900 and leg.get('market') in {'spread_home','spread_away','total_over','total_under','moneyline_home','moneyline_away','h2h_home','h2h_away'} and leg.get('odds') is not None and abs(leg['odds'])>=100
    except (TypeError,ValueError):
        return False


def selections(publications):
    """First eligible publication per category/event, never first winning revision."""
    chosen={}
    for pub in sorted(publications,key=lambda p:(p['confirmed_at'],p['package_hash'])):
        confirmed=datetime.fromisoformat(pub['confirmed_at'])
        package=pub['package']
        entries=[(family,[leg],leg['status']=='APPROVED') for family,rows in package['games'].items() for leg in rows]
        entries += [('parlays',p['legs'],False) for p in package.get('parlays',[])]
        for category,legs,approved in entries:
            if not legs or not all(eligible(leg,confirmed) for leg in legs):
                continue
            if len({datetime.fromisoformat(x['start']).astimezone(ZoneInfo('America/New_York')).date() for x in legs})!=1:
                continue
            identity=(category,tuple(sorted((*event_key(leg)[:3], datetime.fromisoformat(leg['start']).astimezone(ZoneInfo('America/New_York')).date().isoformat()) for leg in legs)))
            if identity in chosen:
                continue
            date=min(datetime.fromisoformat(x['start']) for x in legs).astimezone(ZoneInfo('America/New_York')).date().isoformat()
            chosen[identity]={'id':digest(identity),'category':category,'date':date,'group':'Approved' if approved else 'Research',
                              'published_at':pub['confirmed_at'],'legs':legs}
    return list(chosen.values())


def grade_leg(leg, scores):
    key=event_key(leg)
    if not key:
        return 'PENDING',None
    matches=[]
    for score in scores:
        if (score['sport'],score['away'],score['home'])!=key[:3]:
            continue
        # Exact teams plus a narrow start-time tolerance disambiguate doubleheaders.
        if abs((datetime.fromisoformat(score['start'])-datetime.fromisoformat(leg['start'])).total_seconds())<=1800:
            matches.append(score)
    unique={s['event_id']:s for s in matches}
    if len(unique)!=1:
        return 'PENDING',None
    score=next(iter(unique.values()))
    a,h=score['away_score'],score['home_score']
    market=leg['market'];pick=leg['pick']
    if market.startswith('total_'):
        match=re.fullmatch(r'(Over|Under)\s+(\d+(?:\.\d+)?)',pick,re.I)
        if not match or match[1].lower()!=market.split('_')[1]:return 'PENDING',None
        margin=(a+h-float(match[2]))*(1 if market=='total_over' else -1)
    elif market.startswith('spread_'):
        match=re.fullmatch(r'(.+)\s+([+-]\d+(?:\.\d+)?)',pick)
        team=score['home'] if market=='spread_home' else score['away']
        if not match or normalize_result_team(match[1])!=team:return 'PENDING',None
        margin=(h-a if market=='spread_home' else a-h)+float(match[2])
    elif market in {'moneyline_home','h2h_home','moneyline_away','h2h_away'}:
        team=score['home'] if market.endswith('home') else score['away']
        name=re.sub(r'\s+(?:ML|Moneyline)$','',pick,flags=re.I)
        if normalize_result_team(name)!=team:return 'PENDING',None
        margin=h-a if market.endswith('home') else a-h
    else:
        return 'PENDING',None
    return ('WIN' if margin>0 else 'LOSS' if margin<0 else 'PUSH'),f'{a}–{h} (away–home)'


def report(publications, revisions):
    latest={}
    for revision in sorted(revisions,key=lambda x:x['recorded_at']):
        for score in revision['scores']:
            latest[(score['sport'],score['event_id'])]=score
    rows=[]
    for item in selections(publications):
        graded=[grade_leg(leg,list(latest.values())) for leg in item['legs']]
        outcomes=[x[0] for x in graded]
        # Wait for every leg; pushed/voided tickets excluded from win percentage.
        outcome='PENDING' if 'PENDING' in outcomes else 'LOSS' if 'LOSS' in outcomes else 'PUSH' if 'PUSH' in outcomes else 'WIN'
        rows.append({**{k:v for k,v in item.items() if k!='legs'},'outcome':outcome,
                     'picks':' + '.join(x['game']+': '+x['pick'] for x in item['legs']),
                     'odds':' / '.join(str(x['odds']) for x in item['legs']),
                     'final_score':' / '.join(x[1] or 'Pending' for x in graded)})
    return rows


def fetch_scores(day, sports):
    """Explicit one-day grading; no paid API and no background fetch on rendering."""
    import requests
    from app_core.espn_results import ESPN_ENDPOINTS, _scoreboard_urls
    scores={}
    for sport in sorted(set(sports)):
        if sport not in ESPN_ENDPOINTS:continue
        for url in _scoreboard_urls(sport,day.strftime('%Y%m%d')):
            response=requests.get(url,timeout=10)
            response.raise_for_status()
            for event in response.json().get('events',[]):
                for game in event.get('competitions',[]):
                    status=game.get('status',{}).get('type',{})
                    if not status.get('completed') or status.get('state')!='post' or not str(status.get('name','')).startswith('STATUS_FINAL'):continue
                    teams={t.get('homeAway'):t for t in game.get('competitors',[])}
                    if set(teams)!={'home','away'}:continue
                    try:
                        a=float(teams['away']['score']);h=float(teams['home']['score'])
                        start=game.get('date') or event['date']
                        datetime.fromisoformat(start.replace('Z','+00:00'))
                        if not all(math.isfinite(n) and n>=0 and n.is_integer() for n in (a,h)):continue
                    except (ValueError,TypeError,KeyError):continue
                    scores[(sport,event['id'])]={'sport':sport,'event_id':event['id'],'start':start.replace('Z','+00:00'),
                        'away':normalize_result_team(teams['away']['team']['displayName']),
                        'home':normalize_result_team(teams['home']['team']['displayName']),
                        'away_score':int(a),'home_score':int(h)}
    return {'recorded_at':now(),'scores':list(scores.values())}
