"""Observed pregame closing proxies; no historical or in-play substitution."""
import json
import math
from app_core import ncaaf_prospective as prospective
from app_core import ncaaf_prospective_store as store
from app_core.ncaaf_history import timestamp
from app_core.ncaaf_identity import normalize_ncaaf_team
from app_core.prediction_evidence import provider_quotes


def decimal(price):
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or abs(price)<100:
        return None
    return 1 + (price/100 if price>0 else 100/abs(price))


def capture(key, *, get=None, path=None):
    events = prospective.fetch("https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds", key,
        {"regions":"us", "markets":"h2h,spreads,totals", "oddsFormat":"american"}, cfbd=False, get=get)
    observed = prospective.utcnow()
    saved=[]
    for e in events:
        start=timestamp(e.get("commence_time"))
        if not start or not 0 < (start-observed).total_seconds() <= 1800 or not e.get("id"):
            continue
        if not e.get("home_team") or not e.get("away_team"):
            continue
        try:
            quotes=json.loads(provider_quotes(e))
        except (ValueError, TypeError, AttributeError, KeyError):
            continue
        valid=[]
        for q in quotes:
            t=timestamp(q.get("recorded_at"))
            if not t or not 0 <= (observed-t).total_seconds() <= 900 or not 0 < (start-t).total_seconds() <= 1800:
                continue
            if decimal(q.get("price")) is None:
                continue
            line=q.get("point")
            if not q['market_type'].startswith('moneyline') and (isinstance(line,bool) or not isinstance(line,(float,int)) or not math.isfinite(line)):
                continue
            valid.append(q)
        if valid:
            saved.append({"event_id":e['id'],"home":e['home_team'],"away":e['away_team'],"start":start.isoformat(),"quotes":valid})
    finished=prospective.utcnow()
    saved=[e for e in saved if timestamp(e['start'])>finished]
    key=store.save('closing',{"observed_at":finished.isoformat(),"events":saved,
        "policy":"Latest observed same-book exact-line quote in final 30 minutes before kickoff; closing proxy only."},path)
    record=next(r for r in store.records(path) if r['id']==key)
    return {"saved_events":len(record['data']['events']),"record_id":key}


def report(path=None):
    records=store.records(path)
    models={r['id']:r for r in records if r['kind']=='model'}
    first={}
    for r in sorted((r for r in records if r['kind']=='capture'),key=lambda r:(r['data']['captured_at'],r['id'])):
        model=models.get(r['data']['model_id'])
        at=timestamp(r['data']['captured_at'])
        if not model or timestamp(model['created_at'])>at:
            continue
        for e in r['data']['events']:
            if at<timestamp(e['start']):
                first.setdefault((r['data']['model_id'],e['cfbd_id']),(r,e))
    closing=[r for r in records if r['kind']=='closing']
    rows=[]
    for (cohort,gid),(entry,e) in first.items():
        at=timestamp(entry['data']['captured_at']);start=timestamp(e['start'])
        samples=[]
        for r in closing:
            observed=timestamp(r['data']['observed_at'])
            if not observed or not at<=observed<start or (start-observed).total_seconds()>1800:
                continue
            for c in r['data']['events']:
                if (c['event_id']==e['event_id'] and timestamp(c['start'])==start
                    and all(normalize_ncaaf_team(c[s])==normalize_ncaaf_team(e[s]) for s in ('home','away'))):
                    samples.append((observed,r['id'],c))
        for name,m in e['models'].items():
            q=m['selected']
            row={"model_id":cohort,"model":name,"game_id":gid,"market":q['market_type'],"book":q['book'],
                 "entry_line":q.get('point'),"entry_price":q['price'],"price_clv":None,"status":"no_comparable_closing_proxy"}
            relevant=[s for s in samples if any(x["book"]==q["book"] and x["market_type"]==q["market_type"] for x in s[2]["quotes"])]
            if relevant:
                # Take the latest market observation first; never cherry-pick an
                # earlier quote just because the original line still existed.
                observed,record_id,c=max(relevant,key=lambda item:(item[0],item[1]))
                market=[x for x in c['quotes'] if x['book']==q['book'] and x['market_type']==q['market_type']]
                exact=[x for x in market if q['market_type'].startswith('moneyline') or x.get('point')==q.get('point')]
                row.update(closing_observed_at=observed.isoformat(),closing_record_id=record_id)
                if len(exact)==1:
                    close=exact[0];qt=timestamp(close.get('recorded_at'));entry_t=timestamp(q.get('recorded_at'))
                    d=decimal(close['price']);ed=decimal(q['price'])
                    if qt and entry_t and entry_t<=qt<=observed and 0<(start-qt).total_seconds()<=1800 and (observed-qt).total_seconds()<=900 and d and ed:
                        row.update(status='exact_line_closing_proxy',closing_price=close['price'],closing_quote_at=qt.isoformat(),price_clv=ed/d-1)
                elif len(exact)>1:
                    row['status']='ambiguous_closing_quote'
                elif market:
                    row['status']='line_changed_no_price_clv'
                    row['observed_lines']=[x.get('point') for x in market]
            rows.append(row)
    comparable=[r for r in rows if r['price_clv'] is not None]
    return {"closing_records":len(closing),"selected_markets":len(rows),"comparable_markets":len(comparable),
            "rows":rows,"limitations":["Closing proxy is the latest manually observed pregame quote, not a guaranteed final close.",
            "Price CLV = entry decimal odds / closing decimal odds - 1; positive means a better entry payout at the same line and book.",
            "No price CLV is assigned across changed lines, missing quotes, different books or rescheduled starts.",
            "This is raw price comparison, not no-vig CLV, realized profit or wagering approval."]}
