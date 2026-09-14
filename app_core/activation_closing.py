"""Verified immutable closing observations in the existing evidence database."""
import json
from contextlib import closing
from core.exposure_ledger import digest
from core.wager_decisions import aware,finite,decimal_price
from core.clv import line_clv,price_clv

def record_close(database,candidate,quote,*,captured_at):
    start=aware(candidate.get('game_start_utc'));at=aware(quote.get('quote_recorded_at'));capture=aware(captured_at)
    if None in (start,at,capture) or not at<=capture<start or not 0<(start-at).total_seconds()<=1800 or (capture-at).total_seconds()>1800: raise ValueError('Not a verified pregame closing observation')
    if candidate.get('provider_namespace') not in {'odds_api','espn','mlb'} or candidate.get('provider_namespace')!=quote.get('provider_namespace'): raise ValueError('Closing provider namespace missing or different')
    for key in ('game_id','sport','market_type','sportsbook','provider_event_id'):
        source=candidate.get('quote_bookmaker') if key=='sportsbook' else candidate.get(key)
        if not source or source!=quote.get(key):raise ValueError('Closing identity/book mismatch')
    # Exact direction is market_type, selected-team identity is immutable game ID + direction.
    if decimal_price(quote.get('price')) is None or finite(quote.get('line')) is None:raise ValueError('Invalid close')
    opening=finite(candidate.get('market_line_used'))
    value={'snapshot_id':candidate['snapshot_id'],'candidate_id':candidate['candidate_id'],'quote':quote,'closing_capture_at':capture.isoformat(),
        'line_clv':line_clv(candidate['market_type'],opening,quote['line']),
        'price_clv':price_clv(candidate.get('odds_american'),quote['price']) if opening==quote['line'] else None,'quote_verified':True}
    value['beat_close']=value['line_clv']>0 if value['line_clv'] not in (None,0) else value['price_clv']>0 if value['price_clv'] is not None else None
    key=digest(value)
    from app_core.prediction_evidence import connect
    with closing(connect(database)) as db,db:db.execute('INSERT OR IGNORE INTO closing_observations VALUES (?,?,?,?)',(key,value['snapshot_id'],value['candidate_id'],json.dumps(value,sort_keys=True)))
    return key

def observations(database):
    from app_core.prediction_evidence import connect
    with closing(connect(database)) as db: rows=db.execute('SELECT observation_id,payload FROM closing_observations').fetchall()
    out=[]
    for key,raw in rows:
        value=json.loads(raw)
        if digest(value)!=key:raise ValueError('Closing payload changed')
        out.append(value)
    return out


def capture_live(database, *, fetch=None, now=None):
    """Explicit job; never called by rendering. Missing provider identity blocks."""
    from datetime import datetime, timezone
    from app_core.prediction_evidence import load_snapshots
    current = now or datetime.now(timezone.utc)
    candidates = []
    for _, audit, _ in load_snapshots(database):
        for r in audit.astype(object).where(audit.notna(), None).to_dict('records'):
            start = aware(r.get('game_start_utc'))
            if start and 0 < (start-current).total_seconds() <= 1800:
                candidates.append(r)
    if not candidates:
        return {'verified':0, 'unavailable':0, 'reason':'no_candidates_in_closing_window'}
    if fetch is None:
        from core.streamlit_pipeline import fetch_live_odds_dataframe
        fetch = fetch_live_odds_dataframe
    live = fetch(sorted({r.get('sport',r.get('league')) for r in candidates}))
    saved = 0; blocked = {}
    for r in candidates:
        matching = []
        for row in live.to_dict('records'):
            if row.get('matchup_id') != r.get('matchup_id'): continue
            for q in json.loads(row.get('provider_quotes') or '[]'):
                if (q.get('provider_namespace') and q.get('provider_namespace') == r.get('provider_namespace')
                    and q.get('provider_event_id') == r.get('provider_event_id')
                    and q.get('book') == r.get('quote_bookmaker') and q.get('market_type') == r.get('market_type')):
                    matching.append(q)
        if len(matching) != 1:
            blocked['missing_or_ambiguous_exact_quote'] = blocked.get('missing_or_ambiguous_exact_quote',0)+1
            continue
        q = matching[0]
        quote = dict(game_id=r['game_id'], sport=r['sport'], market_type=r['market_type'],
                     sportsbook=q['book'], provider_event_id=q['provider_event_id'],
                     provider_namespace=q['provider_namespace'], quote_recorded_at=q.get('recorded_at'),
                     line=q.get('point'), price=q.get('price'))
        try:
            record_close(database,r,quote,captured_at=current.isoformat()); saved += 1
        except ValueError:
            blocked['invalid_or_stale_close'] = blocked.get('invalid_or_stale_close',0)+1
    return {'verified':saved, 'unavailable':sum(blocked.values()), 'reasons':blocked}
