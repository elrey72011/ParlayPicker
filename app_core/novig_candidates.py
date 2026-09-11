"""Preserve MLB half-run quotes before probability calculation."""
import json
import math


def preserve_half_run_quote(candidate):
    # Half-run spreads are valid MLB offers, not corrupted standard 1.5 lines.
    if str(candidate.get('league','')).upper() != 'MLB':
        return candidate
    kind = candidate.get('market_type')
    if kind not in {'spread_home','spread_away'}:
        return candidate
    try:
        quotes=json.loads(candidate.get('provider_quotes') or '[]')
        pair={}
        for side in ('spread_home','spread_away'):
            matches=[q for q in quotes if q.get('book')=='novig' and q.get('market_type')==side]
            if len(matches)!=1:
                return candidate
            q=matches[0]
            point=float(q['point']);price=float(q['price'])
            if abs(point)!=0.5 or not math.isfinite(price) or not 100<=abs(price)<=10000 or not q.get('recorded_at'):
                return candidate
            pair[side]=(point,price)
        if pair['spread_home'][0]+pair['spread_away'][0]!=0:
            return candidate
    except (ValueError, TypeError, KeyError, AttributeError):
        return candidate
    result=dict(candidate)
    point,price=pair[kind]
    other='spread_away' if kind=='spread_home' else 'spread_home'
    result.update(spread_line=point,live_spread_line=point,odds_american=price,
                  opposing_odds_american=pair[other][1],opposing_odds_source='novig',
                  odds_source='odds_api',line_source='novig_exact_half_run_quote')
    uploaded=result.get('uploaded_spread_line')
    try:result['line_delta']=point-float(uploaded)
    except (TypeError,ValueError):pass
    return result
