"""Deterministic, disjoint two-leg research combinations from public game picks."""
import itertools
import math
import re
from datetime import datetime, timezone


def build_parlays(rows, now=None):
    now = now or datetime.now(timezone.utc)
    candidates = []
    for row in rows:
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
                if not 0 <= (now-datetime.fromisoformat(row['quote_time'])).total_seconds() <= 900:
                    continue
            age = (now-datetime.fromisoformat(row['as_of'])).total_seconds()
            start = datetime.fromisoformat(row['start'])
            p, odds = row['win_estimate'], row['odds']
            if not (0 <= age <= 900 and start > now and 0 < p < 1 and abs(odds) >= 100):
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
