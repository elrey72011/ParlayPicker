"""Owner-maintained exposure records. This module cannot submit wagers."""
from contextlib import closing
import hashlib
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from core.wager_decisions import finite, aware

LIMITS = ('total_cap','daily_cap','weekly_cap','game_cap','team_cap')

def now_utc(): return datetime.now(timezone.utc)
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def connect(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path,timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, recorded_at TEXT NOT NULL, payload TEXT NOT NULL)')
    for action in ('UPDATE','DELETE'):
        db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{action} BEFORE {action} ON events BEGIN SELECT RAISE(ABORT,'ledger is append-only'); END")
    return db

def events(path):
    if not Path(path).exists(): return []
    with closing(connect(path)) as db, db: rows=db.execute('SELECT event_id,payload FROM events ORDER BY rowid').fetchall()
    out=[]
    for key,raw in rows:
        value=json.loads(raw)
        if digest(value)!=key: raise ValueError('Ledger hash mismatch')
        out.append(dict(value,ledger_event_id=key))
    return out

def append(path,event,*,confirmed=False,now=None):
    now=now or now_utc();value=dict(event)
    if not confirmed: raise ValueError('Explicit owner confirmation required')
    status=value.get('status')
    history=events(path)
    if status=='CONFIGURED':
        if not all(finite(value.get(k)) is not None and finite(value[k])>0 for k in ('bankroll','unit_value')) or not value.get('currency'):
            raise ValueError('Explicit bankroll, units and currency required')
        if not all(finite(value.get(k)) is not None and 0<=finite(value[k])<=1 for k in LIMITS): raise ValueError('Explicit exposure limits required')
    elif status in {'RECOMMENDED','COMMITTED'}:
        if not value.get('bet_id') or not value.get('source_snapshot_id') or not value.get('sportsbook'): raise ValueError('Missing wager identity')
        if any(e.get('bet_id')==value['bet_id'] and e['status']=='COMMITTED' for e in history) and status=='COMMITTED': raise ValueError('Already committed; append a correction under a new ID')
        if finite(value.get('stake_dollars')) is None or finite(value['stake_dollars'])<=0: raise ValueError('Positive confirmed stake required')
        legs=value.get('legs')
        if not isinstance(legs,list) or not legs: raise ValueError('Underlying event/team identities required')
        for leg in legs:
            teams=leg.get('team_ids')
            if not leg.get('sport') or not leg.get('game_id') or not isinstance(teams,list) or len(teams)!=2 or len(set(teams))!=2 or not all(isinstance(x,str) and x for x in teams): raise ValueError('Invalid underlying identity')
            if not leg.get('market') or not leg.get('selection') or finite(leg.get('odds')) is None or abs(finite(leg['odds']))<100 or finite(leg.get('line')) is None: raise ValueError('Actual market/line/price required')
        cfg=next((e for e in reversed(history) if e['status']=='CONFIGURED'),None)
        if not cfg: raise ValueError('Configure bankroll first')
        value['bankroll_at_commit']=cfg['bankroll'];value['stake_units']=float(value['stake_dollars'])/float(cfg['unit_value'])
    elif status in {'SETTLED','VOID','CANCELLED'}:
        related=[e for e in history if e.get('bet_id')==value.get('bet_id') and e['status']!='RECOMMENDED']
        if not related or related[-1]['status']!='COMMITTED': raise ValueError('No open commitment')
    else: raise ValueError('Invalid ledger event')
    value['recorded_at']=now.isoformat();key=digest(value)
    with closing(connect(path)) as db, db: db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?)',(key,value['recorded_at'],json.dumps(value,sort_keys=True,allow_nan=False)))
    return key

def snapshot(path,*,now=None):
    now=now or now_utc();history=events(path)
    cfg=next((e for e in reversed(history) if e['status']=='CONFIGURED'),None)
    if not cfg: raise ValueError('BANKROLL_AND_LIMITS_NOT_CONFIGURED')
    current={}
    for e in history:
        if e.get('bet_id') and e['status']!='RECOMMENDED': current[e['bet_id']]=e
    used={'total':0.,'daily':0.,'weekly':0.}
    today=now.astimezone(ZoneInfo('America/New_York')).date()
    # Daily/weekly turnover remains consumed after settlement; open exposure clears.
    for e in history:
        if e['status']!='COMMITTED': continue
        at=aware(e['recorded_at'])
        if at is None or at>now: raise ValueError('Invalid ledger time')
        day=at.astimezone(ZoneInfo('America/New_York')).date();fraction=float(e['stake_dollars'])/float(cfg['bankroll'])
        if day==today: used['daily']+=fraction
        if day.isocalendar()[:2]==today.isocalendar()[:2]: used['weekly']+=fraction
        if current[e['bet_id']]['status']!='COMMITTED': continue
        used['total']+=fraction
        keys=set()
        for leg in e['legs']:
            keys.update([f"sport:{leg['sport']}",f"game:{leg['sport']}:{leg['game_id']}"])
            keys.update(f"team:{leg['sport']}:{t}" for t in leg['team_ids'])
            # The same exact market consumes leg and overlapping-ticket risk
            # whether it was placed as a straight or within a parlay. Candidate
            # IDs are deliberately excluded: they can change on republishing.
            market_key=digest({'sport':leg['sport'],'game_id':leg['game_id'],
                               'market_type':leg['market'],'selection':leg['selection'],
                               'line':leg['line']})
            keys.update((f'leg:{market_key}',f'overlap:{market_key}'))
        if e.get('parlay_id'):
            keys.add(f"parlay:{e['parlay_id']}")
        if e.get('product_type')=='SAME_GAME_PARLAY' or e.get('sgp_component_ids'):
            keys.add('sgp:total')
        for key in keys: used[key]=used.get(key,0)+fraction
    result={'as_of':now.isoformat(),'bankroll':float(cfg['bankroll']),'unit_value':float(cfg['unit_value']),'currency':cfg['currency'],'committed':used,'ledger_hash':digest(history),**{k:float(cfg[k]) for k in LIMITS}}
    result['snapshot_hash']=digest(result)
    return result

def verify_snapshot(value,*,now=None):
    now=now or now_utc();at=aware(value.get('as_of'))
    if digest({k:v for k,v in value.items() if k!='snapshot_hash'})!=value.get('snapshot_hash'): raise ValueError('Exposure hash mismatch')
    if at is None or not 0<=(now-at).total_seconds()<=1800: raise ValueError('Exposure snapshot stale')
    if not all(finite(value.get(k)) is not None and 0<=finite(value[k])<=1 for k in LIMITS): raise ValueError('Invalid limits')
    if not all(finite(value.get(k)) is not None and finite(value[k])>0 for k in ('bankroll','unit_value')): raise ValueError('Invalid bankroll')
    if not isinstance(value.get('committed'),dict) or not all(finite(x) is not None and finite(x)>=0 for x in value['committed'].values()): raise ValueError('Invalid exposure')
    return value
