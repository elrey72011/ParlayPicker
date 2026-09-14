"""Immutable, site-scoped public publication records and conservative result grading."""
from app_core.public_quote_policy import supported_quote
from app_core.quote_freshness import QUOTE_MAX_AGE_MINUTES, package_age_minutes
from contextlib import contextmanager
from time import perf_counter
import logging
import hashlib
import json
import math
import re
from functools import lru_cache
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from app_core.result_team_names import normalize_result_team


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def lock_stage(name, records=0):
    started = perf_counter()
    outcome = 'error'
    try:
        yield
        outcome = 'ok'
    finally:
        logging.getLogger(__name__).warning(
            'PERFORMANCE lock_stage=%s seconds=%.3f records=%d outcome=%s',
            name, perf_counter() - started, records, outcome)


class History:
    def __init__(self, site, folder, client=None):
        from app_core.netlify_publishing import identifier
        from app_core.evidence_drive import DriveStore
        if not folder:
            raise ValueError('Configure Shared Drive storage before tracking public publications.')
        self.client = client or DriveStore(folder)
        self.prefix = 'parlaypicker/public-history-v1/' + identifier(site) + '/'

    def read(self, key, *, client=None):
        client = self.client if client is None else client
        body=client.get_object(Key=self.prefix+key)['Body']
        try:
            return json.loads(body.read())
        finally:
            body.close()

    def put(self, key, value, first=False, *, client=None):
        client = self.client if client is None else client
        raw=encoded(value)
        try:
            client.put_object(Key=self.prefix+key,Body=raw,IfNoneMatch='*')
        except Exception as exc:
            if getattr(exc,'response',{}).get('Error',{}).get('Code') not in {'412','PreconditionFailed'}:
                raise
        saved=self.read(key, client=client)
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

    def _all(self, kind):
        if hasattr(self.client, 'read_objects'):
            return [json.loads(raw) for _, raw in self.client.read_objects(Prefix=self.prefix+kind+'/')]
        values=[]
        for page in self.client.get_paginator('list_objects_v2').paginate(Prefix=self.prefix+kind+'/'):
            for item in page.get('Contents',[]):
                values.append(self.read(item['Key'][len(self.prefix):]))
        return values

    def all(self, kind):
        values = self._all(kind)
        if kind == 'locks':
            removed = {r['lock_hash'] for r in self._all('lock_removals')}
            values = [r for r in values if digest(r) not in removed]
        return values

    def remove_locks(self, selected_hashes, reason):
        """Append owner corrections; never delete original locks or remove later relocks."""
        if not reason.strip() or not selected_hashes:
            raise ValueError('Select locks and provide a correction reason.')
        originals = {digest(r): r for r in self._all('locks')}
        if not set(selected_hashes) <= originals.keys():
            raise ValueError('Lock history changed. Restore history before correcting it.')
        for key in sorted(set(selected_hashes)):
            self.put('lock_removals/'+key+'.json',
                     {'lock_hash':key, 'lock_id':originals[key]['id'],
                      'removed_at':now(), 'reason':reason.strip(), 'lock':originals[key]}, first=True)
        return self.all('locks')

    def lock_picks(self, package, selected_ids, *, progress=None):
        from app_core.locked_picks import lock_candidates
        # One authoritative server acceptance time, unchanged by I/O completion.
        at = now()
        choices = {row['id']: row for row in lock_candidates(package, at)}
        requested = set(selected_ids)
        if not requested or not requested <= choices.keys():
            raise ValueError('Selections changed, started or became stale. Rebuild the preview before locking.')
        def update(label, done=0, total=0):
            if progress:
                progress(label, done, total)
        update('Saving reviewed board')
        with lock_stage('archive_board'):
            self.archive(package)
        update('Reading existing locks')
        with lock_stage('read_existing_locks'):
            originals = self._all('locks')
            removals = self._all('lock_removals')
            removed = {r['lock_hash'] for r in removals}
            active = {r['id']: r for r in originals if digest(r) not in removed}
        pending = []
        for identity in sorted(requested):
            # First write wins, including concurrent clicks and later previews.
            if identity in active:
                continue
            generation = sorted(r['lock_hash'] for r in removals if r['lock_id']==identity)
            suffix = '-'+digest(generation) if generation else ''
            pending.append(('locks/' + identity + suffix + '.json', choices[identity]))
        update('Saving and verifying locks', 0, len(pending))
        with lock_stage('save_verified_locks', len(pending)):
            if hasattr(self.client, 'run_parallel'):
                saved = self.client.run_parallel(
                    lambda worker, item: self.put(item[0], item[1], first=True, client=worker),
                    pending, progress=lambda done, total: update('Saving and verifying locks', done, total))
            else:
                saved = []
                for key, value in pending:
                    saved.append(self.put(key, value, first=True))
                    update('Saving and verifying locks', len(saved), len(pending))
        active.update({r['id']: r for r in saved})
        return [active[identity] for identity in sorted(requested)]

    def publications(self):
        result=[]
        for receipt in self.all('confirmed'):
            package=self.read('packages/'+receipt['package_hash']+'.json')
            if digest(package)!=receipt['package_hash']:
                raise ValueError('Public history hash mismatch')
            result.append({**receipt,'package':package})
        return result


def team_name(value, sport):
    # The generic alias table maps bare Seattle to Seattle University.
    # Scope this explicit alias to MLB; never change college team identity.
    if sport.upper() == 'NCAAF':
        compact = re.sub(r'[^a-z0-9]', '', str(value).casefold())
        if compact in {'floridaam', 'floridaamrattlers', 'famu', 'famurattlers'}:
            return 'FLORIDA A&M'
    if sport.upper() == 'MLB' and str(value).strip().casefold() == 'seattle':
        value = 'Seattle Mariners'
    return normalize_result_team(value)


def event_key(leg):
    teams=re.split(r'\s+(?:at|@)\s+',leg['game'],flags=re.I)
    if len(teams)!=2 or not leg.get('start'):
        return None
    return (leg['sport'].upper(), *(team_name(t,leg['sport']) for t in teams), datetime.fromisoformat(leg['start']).isoformat())


def resolved_pick(leg):
    pick = str(leg.get('pick') or '').strip()
    market = leg.get('market', '')
    if not pick or any(token in pick.lower() for token in ('unresolved', 'unavailable', 'no line', 'no bet')):
        return False
    if market.startswith('spread_'):
        return bool(re.fullmatch(r'.+\s+[+-]\d+(?:\.\d+)?', pick))
    if market.startswith('total_'):
        match = re.fullmatch(r'(Over|Under)\s+\d+(?:\.\d+)?', pick, re.I)
        return bool(match and match[1].lower() == market.split('_')[1])
    return True


def eligible(leg, confirmed, *, max_age_minutes=QUOTE_MAX_AGE_MINUTES):
    try:
        if 'quote_time_basis' in leg and (not supported_quote(leg) or leg.get('status') != 'PASS' or datetime.fromisoformat(leg.get('quote_time')) > datetime.fromisoformat(leg.get('as_of'))):
            return False
        if 'quote_source' in leg:
            if not supported_quote(leg) or not leg.get('quote_time'):
                return False
            age=(confirmed-datetime.fromisoformat(leg['quote_time'])).total_seconds()
            if not 0 <= age <= max_age_minutes * 60:
                return False
        at=datetime.fromisoformat(leg['as_of']);start=datetime.fromisoformat(leg['start'])
        return event_key(leg) is not None and at<=confirmed<start and (confirmed-at).total_seconds()<=max_age_minutes * 60 and leg.get('market') in {'spread_home','spread_away','total_over','total_under','moneyline_home','moneyline_away','h2h_home','h2h_away'} and leg.get('odds') is not None and abs(leg['odds'])>=100
    except (TypeError,ValueError):
        return False


def selections(publications):
    """First eligible publication per category/event, never first winning revision."""
    chosen={}
    for pub in sorted(publications,key=lambda p:(p['confirmed_at'],p['package_hash'])):
        confirmed=datetime.fromisoformat(pub['confirmed_at'])
        package=pub['package']
        entries=[(family,[leg],leg['status']=='APPROVED') for family,rows in package['games'].items() for leg in rows]
        entries += [('parlays',p['legs'],False) for p in [*package.get('parlays',[]), *package.get('research_parlays',[])]]
        for category,legs,approved in entries:
            if not legs or not all(eligible(leg,confirmed,max_age_minutes=package_age_minutes(package)) for leg in legs):
                continue
            if len({datetime.fromisoformat(x['start']).astimezone(ZoneInfo('America/New_York')).date() for x in legs})!=1:
                continue
            identity=(category,tuple(sorted((*event_key(leg)[:3], datetime.fromisoformat(leg['start']).astimezone(ZoneInfo('America/New_York')).date().isoformat()) for leg in legs)))
            if identity in chosen:
                continue
            date=min(datetime.fromisoformat(x['start']) for x in legs).astimezone(ZoneInfo('America/New_York')).date().isoformat()
            chosen[identity]={'id':digest(identity),'category':category,'date':date,'group':'Approved' if approved else 'Research',
                              'published_at':pub['confirmed_at'],'legs':legs}
    from app_core.top_ten_history import top_ten_selections
    return list(chosen.values()) + top_ten_selections(publications)


@lru_cache(maxsize=8192)
def grading_team_name(value, sport):
    """Resolve result names without changing immutable publication/lock identities."""
    if sport.upper() == 'NFL':
        from app_core.nfl_identity import nfl_result_name
        return nfl_result_name(value)
    if sport != 'NCAAF':
        return team_name(value, sport)
    from app_core.ncaaf_identity import normalize_ncaaf_team, _key
    # Exact variants seen in archived picks and ESPN. Resolve before AND after
    # the provider mapper: some legacy aliases map in opposite directions.
    groups = (
        ('massachusetts', 'umass', 'massachusetts minutemen', 'umass minutemen'),
        ('wisconsin', 'wisconsin badgers'),
        ('southern miss', 'southern mississippi golden', 'southern miss golden'),
        ('florida am', 'florida a m', 'florida a m rattlers'),
    )
    aliases = {_key(alias): group[0] for group in groups for alias in group}
    raw = _key(value)
    if raw in aliases:
        return aliases[raw]
    mapped = normalize_ncaaf_team(value)
    return aliases.get(mapped, mapped)


def grade_leg(leg, scores, *, imported=False):
    from app_core.result_reconciliation import match_result
    score, reason = match_result(leg, scores)
    if reason:
        return 'PENDING', None
    a,h=score['away_score'],score['home_score']
    market=leg['market'];pick=leg['pick']
    if market.startswith('total_'):
        match=re.fullmatch(r'(Over|Under)\s+(\d+(?:\.\d+)?)',pick,re.I)
        if not match or match[1].lower()!=market.split('_')[1]:return 'PENDING',None
        margin=(a+h-float(match[2]))*(1 if market=='total_over' else -1)
    elif market.startswith('spread_'):
        match=re.fullmatch(r'(.+)\s+([+-]\d+(?:\.\d+)?)',pick)
        team=grading_team_name(score['home'] if market=='spread_home' else score['away'],leg['sport'])
        if not match or grading_team_name(match[1],leg['sport'])!=team:return 'PENDING',None
        margin=(h-a if market=='spread_home' else a-h)+float(match[2])
    elif market in {'moneyline_home','h2h_home','moneyline_away','h2h_away'}:
        team=grading_team_name(score['home'] if market.endswith('home') else score['away'],leg['sport'])
        name=re.sub(r'\s+(?:ML|Moneyline)$','',pick,flags=re.I)
        if grading_team_name(name,leg['sport'])!=team:return 'PENDING',None
        margin=h-a if market.endswith('home') else a-h
    else:
        return 'PENDING',None
    return ('WIN' if margin>0 else 'LOSS' if margin<0 else 'PUSH'),f'{a}–{h} (away–home)'


def original_estimate(leg):
    """Read only the saved probability, never a current quote or ranking score."""
    value = leg.get('win_estimate')
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        return {}
    return {'original_win_estimate': value}


def report(publications, revisions, imports=None, locks=None):
    from app_core.result_reconciliation import latest_scores
    latest = latest_scores(revisions)
    rows=[]
    from app_core.imported_recaps import imported_selections
    from app_core.locked_picks import locked_selections
    for item in selections(publications) + imported_selections(imports or []) + locked_selections(locks or []):
        graded=[grade_leg(leg,latest,imported=item['group']=='Imported research') for leg in item['legs']]
        outcomes=[x[0] for x in graded]
        # Wait for every leg; pushed/voided tickets excluded from win percentage.
        outcome='PENDING' if 'PENDING' in outcomes else 'LOSS' if 'LOSS' in outcomes else 'PUSH' if 'PUSH' in outcomes else 'WIN'
        diagnostics = []
        from app_core.mlb_event_matcher import match_mlb_event
        for leg in item['legs']:
            if leg['sport'] == 'MLB':
                matched = match_mlb_event(leg, latest)
                if not matched.status.startswith('MATCHED_'):
                    diagnostics.append({'event_match_status':matched.status, 'grading_reason':matched.reason, 'event_match_method':matched.identity_method, 'settlement_review_required':matched.settlement_review_required})
        rows.append({**{k:v for k,v in item.items() if k!='legs'},'outcome':outcome,
                     **({'grading_diagnostics':diagnostics} if diagnostics else {}),
                     'picks':' + '.join(x['game']+': '+x['pick'] for x in item['legs']),
                     'odds':' / '.join(str(x['odds']) for x in item['legs']),
                     **({k:item['legs'][0].get(k, '') for k in ('sport', 'market')} if len(item['legs']) == 1 else {}),
                     **(original_estimate(item['legs'][0]) if len(item['legs']) == 1 and item['group'] != 'Imported research' else {}),
                     **({'quote_source':item['legs'][0]['quote_source']} if item['group']=='Locked' and item['legs'][0].get('quote_source') else {}),
                     **({k:item['legs'][0][k] for k in ('quote_time', 'quote_time_basis')} if item['group']=='Locked' and item['legs'][0].get('quote_time_basis') == 'espn_observed' else {}),
                     'final_score':' / '.join(x[1] or 'Pending' for x in graded)})
    return rows


def fetch_scores(day, sports):
    """Explicit authoritative retrieval; never called by passive rendering."""
    from app_core.result_providers import fetch_results
    return fetch_results(day, sports)
