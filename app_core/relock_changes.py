"""Saved-fact comparisons only; never selects or reprices a candidate."""
import re
from app_core.public_history import digest


def latest_removed(removals, identity):
    rows = [r for r in removals if r.get('lock_id') == identity]
    for r in rows:
        if digest(r['lock']) != r['lock_hash'] or r['lock']['id'] != identity:
            raise ValueError('Invalid removed lock history')
    return max(rows, key=lambda r: (r['removed_at'], r['lock_hash'])) if rows else None


def saved_favorite(leg):
    """Only a unique paired, same-book/event/time moneyline quote proves favorite."""
    import json
    import math
    try:
        quotes = leg.get('provider_quotes', [])
        if isinstance(quotes, str):
            quotes = json.loads(quotes)
        namespace, event = leg.get('provider_namespace'), leg.get('provider_event_id')
        if not namespace or not event or not leg.get('quote_time') or not leg.get('quote_source'):
            return None
        quotes = [q for q in quotes if q.get('provider_namespace') == namespace and q.get('provider_event_id') == event
                  and q.get('book', '').casefold() == leg['quote_source'].casefold() and q.get('recorded_at') == leg['quote_time']]
        home = [q for q in quotes if q.get('market_type') == 'moneyline_home']
        away = [q for q in quotes if q.get('market_type') == 'moneyline_away']
        if len(home) != 1 or len(away) != 1:
            return None
        h, a = float(home[0]['price']), float(away[0]['price'])
        if not all(math.isfinite(v) and abs(v) >= 100 for v in (h,a)):
            return None
        probability = lambda v: -v / (100-v) if v < 0 else 100 / (100+v)
        teams = leg['game'].split(' at ')
        if len(teams) != 2 or probability(h) == probability(a):
            return None
        return teams[1] if probability(h) > probability(a) else teams[0]
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def facts(row):
    leg = row['legs'][0]
    market = leg.get('market', '')
    family = 'Spread' if market.startswith('spread_') else 'Total' if market.startswith('total_') else market
    pick = leg.get('pick', '')
    match = re.fullmatch(r'(.+)\s+([+-]\d+(?:\.\d+)?)', pick) if family == 'Spread' else None
    total = re.fullmatch(r'(Over|Under)\s+(\d+(?:\.\d+)?)', pick, re.I) if family == 'Total' else None
    return {'Pick': pick, 'Market': market, 'Market family': family,
            'Selected team': match[1] if match else None,
            'Direction': total[1].lower() if total else None, 'Market favorite': saved_favorite(leg),
            'Line': float(match[2]) if match else float(total[2]) if total else None,
            'Odds': leg.get('odds'), 'Sportsbook': leg.get('quote_source'),
            'Analysis': leg.get('as_of'), 'Export run': leg.get('export_run_id'),
            'Prediction timestamp': leg.get('prediction_generated_at'), 'Quote time': leg.get('quote_time'),
            'Provider namespace': leg.get('provider_namespace'), 'Provider event ID': leg.get('provider_event_id')}


def compare(previous, current):
    a, b = facts(previous), facts(current)
    changed = lambda k: a[k] is not None and b[k] is not None and a[k] != b[k]
    team = changed('Selected team')
    family = changed('Market family')
    same = a['Market family'] == b['Market family'] and (
        bool(a['Selected team']) and a['Selected team'] == b['Selected team'] or
        bool(a['Direction']) and a['Direction'] == b['Direction'] or a['Pick'] == b['Pick'])
    flags = {'same_selection': bool(same), 'line_changed': changed('Line'), 'price_changed': changed('Odds'),
             'market_family_changed': family, 'selected_team_changed': team,
             'market_favorite_changed': changed('Market favorite') if a['Market favorite'] and b['Market favorite'] else None, 'analysis_run_changed': changed('Export run') or changed('Analysis')}
    severity = 'CRITICAL' if team else 'HIGH' if family else 'WARNING' if not same else 'NORMAL'
    label = 'Selected team reversed' if team else 'Market changed' if family else 'Selection changed' if not same else 'Price changed' if changed('Odds') or changed('Line') else 'No change'
    return {'previous': a, 'current': b, 'flags': flags, 'severity': severity, 'label': label,
            'differences': [k for k in a if changed(k)],
            'recorded_analysis': {k: [previous['legs'][0].get(k), current['legs'][0].get(k)]
                for k in ('win_estimate', 'best_available_rank', 'market_probability', 'kalshi_probability', 'ml_probability')
                if previous['legs'][0].get(k) is not None and current['legs'][0].get(k) is not None}}


def review_token(removal, candidate):
    return digest({'removed': removal, 'candidate_id': candidate['id'], 'legs': candidate['legs']})


def acknowledged(change, checked=False, typed=''):
    return change['severity'] == 'NORMAL' or bool(checked and (change['severity'] != 'CRITICAL' or typed == 'RELOCK'))


def verify_review(removals, choices, requested, tokens):
    for identity in requested:
        prior = latest_removed(removals, identity)
        expected = review_token(prior, choices[identity]) if prior else None
        if tokens.get(identity) != expected:
            raise ValueError('Re-lock review changed; review the current selection and archived lock again.')
