"""Deterministic, disjoint two-leg research combinations from public game picks."""
from app_core.quote_freshness import QUOTE_MAX_AGE_MINUTES
import itertools
import math
import re
from datetime import datetime, timezone


def build_parlays(rows, now=None, *, qualified_only=False, max_age_minutes=QUOTE_MAX_AGE_MINUTES):
    now = now or datetime.now(timezone.utc)
    candidates = []
    for row in rows:
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
