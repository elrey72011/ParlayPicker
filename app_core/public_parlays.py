"""Deterministic, disjoint two-leg research combinations from public game picks."""
from app_core.quote_freshness import QUOTE_MAX_AGE_MINUTES
from core.market_policy import production_market
import itertools
import math
import re
from datetime import datetime, timezone


def _legacy_build_parlays(rows, now=None, *, qualified_only=False, max_age_minutes=QUOTE_MAX_AGE_MINUTES):
    now = now or datetime.now(timezone.utc)
    candidates = []
    for row in rows:
        if not production_market(row.get('market')):
            continue
        if qualified_only:
            from app_core.recommendation_quality import positive_price_edge
            if row.get('status') != 'APPROVED' or not row.get('quote_time') or row.get('quote_source') != 'Novig' or not positive_price_edge(row.get('win_estimate'), row.get('odds'), row.get('ev')):
                continue
        teams = re.split(r'\s+(?:at|@)\s+', row['game'], flags=re.I)
        if len(teams) != 2:
            continue
        teams = {(row['sport'].casefold(), re.sub(r'[^a-z0-9]', '', t.casefold().replace('saint ', 'st '))) for t in teams}
        if len(teams) != 2:
            continue
        try:
            if 'quote_source' in row:
                if row['quote_source'] != 'Novig' or not row.get('quote_time'):
                    continue
                if not 0 <= (now-datetime.fromisoformat(row['quote_time'])).total_seconds() <= max_age_minutes * 60:
                    continue
            age = (now-datetime.fromisoformat(row['as_of'])).total_seconds()
            start = datetime.fromisoformat(row['start'])
            p, odds = row['win_estimate'], row['odds']
            if not (0 <= age <= max_age_minutes * 60 and start > now and 0 < p < 1 and abs(odds) >= 100):
                continue
            decimal = 1 + (odds/100 if odds > 0 else 100/abs(odds))
            if not math.isfinite(decimal):
                continue
        except (TypeError, ValueError):
            continue
        candidates.append((row, teams, decimal))
    pairs = []
    for a, b in itertools.combinations(candidates, 2):
        if a[1] & b[1]:
            continue
        p = a[0]['win_estimate'] * b[0]['win_estimate']
        decimal = a[2]*b[2]
        approved_legs = all(x[0]['status']=='APPROVED' for x in (a,b))
        pairs.append((approved_legs, p, decimal, a, b))
    pairs.sort(key=lambda x: (-x[0], -x[1], x[3][0]['game'], x[4][0]['game']))
    used = set()
    result = []
    for approved, p, decimal, a, b in pairs:
        teams = a[1] | b[1]
        if used & teams:
            continue
        used.update(teams)
        result.append({'legs':[a[0].copy(), b[0].copy()], 'win_estimate':p,
                       'decimal_odds_estimate':decimal, 'ev_estimate':p*decimal-1,
                       'approved_legs':approved, 'status':'RESEARCH ONLY'})
        if len(result)==3:
            break
    return result


def build_research_parlays(rows, now=None, *, qualified_parlays=(), max_age_minutes=QUOTE_MAX_AGE_MINUTES):
    """Positive-edge research tickets at one sportsbook; never wager approval."""
    from collections import Counter
    from zoneinfo import ZoneInfo
    from app_core.public_history import eligible, event_key, resolved_pick
    from app_core.public_quote_policy import supported_quote
    from app_core.recommendation_quality import positive_price_edge
    now = now or datetime.now(timezone.utc)
    eastern = ZoneInfo('America/New_York')
    def teams(row):
        key = event_key(row)
        return {(key[0], key[1]), (key[0], key[2])} if key else set()
    # Duplicate aliases must not choose a different line by input ordering.
    identities = Counter(frozenset(teams(row)) for row in rows if row.get('start'))
    used = set()
    for ticket in qualified_parlays:
        for leg in ticket['legs']:
            used.update(teams(leg))
    candidates = []
    for row in rows:
        if not production_market(row.get('market')):
            continue
        try:
            names = teams(row)
            if len(names) != 2 or identities[frozenset(names)] != 1 or used & names:
                continue
            if row.get('status') not in {'PASS', 'APPROVED'} or not supported_quote(row) or not row.get('quote_time'):
                continue
            if not resolved_pick(row) or not eligible(row, now, max_age_minutes=max_age_minutes):
                continue
            if not positive_price_edge(row.get('win_estimate'), row.get('odds'), row.get('ev')):
                continue
            day = datetime.fromisoformat(row['start']).astimezone(eastern).date()
            if day != now.astimezone(eastern).date():
                continue
            odds = row['odds']
            decimal = 1 + (odds/100 if odds > 0 else 100/abs(odds))
            candidates.append((row, names, decimal))
        except (TypeError, ValueError):
            continue
    pairs = []
    for a, b in itertools.combinations(candidates, 2):
        if a[1] & b[1] or a[0]['quote_source'] != b[0]['quote_source']:
            continue
        a, b = sorted((a, b), key=lambda x: (x[0]['game'], x[0]['pick']))
        probability = a[0]['win_estimate'] * b[0]['win_estimate']
        decimal = a[2] * b[2]
        pairs.append((probability, decimal, a, b))
    pairs.sort(key=lambda x: (-x[0], -(x[0]*x[1]-1), x[2][0]['game'], x[3][0]['game']))
    result = []
    for probability, decimal, a, b in pairs:
        names = a[1] | b[1]
        if used & names:
            continue
        used.update(names)
        result.append({'legs': [a[0].copy(), b[0].copy()], 'win_estimate': probability,
                       'decimal_odds_estimate': decimal, 'ev_estimate': probability*decimal-1,
                       'approved_legs': all(x[0]['status']=='APPROVED' for x in (a,b)),
                       'status': 'RESEARCH ONLY'})
        if len(result) == 3:
            break
    return result


def parlay_funnel(rows, now=None, *, max_age_minutes=QUOTE_MAX_AGE_MINUTES):
    """Owner-only sequential counts and independent exclusion reasons."""
    from collections import Counter
    from app_core.public_history import eligible, event_key, resolved_pick
    from app_core.public_quote_policy import supported_quote
    from app_core.recommendation_quality import positive_price_edge
    now = now or datetime.now(timezone.utc)
    counts = Counter(input=len(rows)); reasons = Counter(); candidates = []
    for row in rows:
        failed=[]
        market = production_market(row.get('market'))
        price = isinstance(row.get('odds'),(int,float)) and not isinstance(row.get('odds'),bool) and math.isfinite(row['odds']) and 100 <= abs(row['odds']) <= 10000
        value = positive_price_edge(row.get('win_estimate'),row.get('odds'),row.get('conservative_ev',row.get('ev')))
        book = supported_quote(row) and bool(row.get('quote_time')) and not row.get('quote_time_basis')
        fresh = eligible(row,now,max_age_minutes=max_age_minutes) and resolved_pick(row)
        maturity = row.get('maturity')
        mature = maturity in {'STANDARD','PREMIUM'} if maturity else row.get('status') == 'APPROVED'
        approved = row.get('status') == 'APPROVED' and row.get('production_eligible',True) is True
        gemini = row.get('gemini_review_status','')
        if gemini in {'HOLD','OPPOSE','ABSTAIN','LOW_CONFIDENCE'}: failed.append('gemini_hard_veto')
        if gemini in {'OUTAGE_CAPPED','UNAVAILABLE'}: failed.append('gemini_unavailable')
        stages=[('spread_total',market,'moneyline_or_invalid_market'),('valid_price',price,'invalid_price'),('positive_ev',value,'nonpositive_ev'),('fresh_pregame',fresh,'stale_or_unresolved_quote'),('supported_book',book,'unsupported_book'),('production_eligible',approved,'not_approved'),('standard_premium',mature,'provisional_straight_only' if maturity=='PROVISIONAL' else 'research_maturity')]
        reached=True
        for stage,ok,reason in stages:
            if not ok: failed.append(reason)
            reached = reached and ok
            if reached: counts[stage]+=1
        key=event_key(row)
        names={(key[0],key[1]),(key[0],key[2])} if key else set()
        if len(names)!=2: failed.append('team_identity')
        reasons.update(set(failed))
        if not failed: candidates.append((row,names))
    pairs=[]; partners=set()
    for i,(a,at) in enumerate(candidates):
        for j,(b,bt) in enumerate(candidates[i+1:],i+1):
            if at & bt:
                reasons['same_game_conflict' if at==bt else 'same_team_conflict']+=1;continue
            if a['quote_source'] != b['quote_source']: continue
            pairs.append((a,b));partners.update((i,j))
    reasons['no_same_book_partner']=len(candidates)-len(partners)
    counts['same_book_compatible']=len(partners);counts['valid_pairs']=len(pairs)
    return {'counts':dict(counts),'exclusions':dict(reasons),'pairs':pairs}


def build_parlays(rows, now=None, *, qualified_only=False, max_age_minutes=QUOTE_MAX_AGE_MINUTES, legacy=False):
    if legacy or not qualified_only:
        return _legacy_build_parlays(rows,now,qualified_only=qualified_only,max_age_minutes=max_age_minutes)
    from app_core.public_history import event_key
    funnel=parlay_funnel(rows,now,max_age_minutes=max_age_minutes)
    def ranking(pair):
        return (-pair[0]['win_estimate']*pair[1]['win_estimate'], pair[0]['game'],pair[1]['game'])
    used=set();tickets=[]
    for a,b in sorted(funnel['pairs'],key=ranking):
        names={(event_key(r)[0],name) for r in (a,b) for name in event_key(r)[1:3]}
        if used & names:continue
        used.update(names)
        probability=a['win_estimate']*b['win_estimate']
        decimal=math.prod(1+(r['odds']/100 if r['odds']>0 else 100/abs(r['odds'])) for r in (a,b))
        tickets.append({'legs':[a.copy(),b.copy()],'win_estimate':probability,'decimal_odds_estimate':decimal,'ev_estimate':probability*decimal-1,'approved_legs':True,'status':'RESEARCH ONLY'})
        if len(tickets)==3:break
    return tickets
