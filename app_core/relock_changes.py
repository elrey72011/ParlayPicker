"""Immutable saved-fact comparisons; no ranking, quote binding or pricing decisions."""
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
import json
import math
import re
from types import MappingProxyType
from app_core.public_history import digest


class RelockReviewExpired(ValueError):
    """Pre-write review invalidation: no new lock was written."""


class RelockAlreadyLocked(ValueError):
    def __init__(self, *, after_write=False):
        super().__init__('This game was already locked elsewhere')
        self.after_write = after_write


@dataclass(frozen=True)
class RelockComparison(Mapping):
    lock_id: str
    previous_lock_hash: str
    previous_removed_at: str | None
    previous_removal_reason: str | None
    previous: Mapping
    current: Mapping
    flags: Mapping
    severity: str
    change_code: str
    state: str
    label: str
    differences: tuple
    recorded_analysis: Mapping

    def __post_init__(self):
        for name in ('previous', 'current', 'flags', 'recorded_analysis'):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    def __getitem__(self, name):
        if name not in {f.name for f in fields(self)}:
            raise KeyError(name)
        return getattr(self, name)

    def __iter__(self):
        return iter(f.name for f in fields(self))

    def __len__(self):
        return len(fields(self))

    def to_dict(self):
        return {k: dict(v) if isinstance(v, Mapping) else v for k, v in self.items()}


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Missing timestamp timezone')
    return parsed


def latest_removed(removals, identity):
    rows = [r for r in removals if r.get('lock_id') == identity]
    for r in rows:
        if digest(r['lock']) != r['lock_hash'] or r['lock']['id'] != identity:
            raise ValueError('Invalid removed lock history')
    return max(rows, key=lambda r: (timestamp(r['removed_at']), timestamp(r['lock']['published_at']), r['lock_hash'])) if rows else None


def teams(leg):
    home, away = leg.get('home_team'), leg.get('away_team')
    # Legacy public legs store the matchup as a single field, not team columns.
    pair = str(leg.get('game', '')).split(' at ')
    if len(pair) == 2:
        away, home = away or pair[0], home or pair[1]
    return home, away


def family(market):
    return 'spread' if market in ('spread_home', 'spread_away') else 'total' if market in ('total_over', 'total_under') else 'unknown'


def saved_favorite(leg):
    """Only an unambiguous saved opposing pair for this event/book/time."""
    try:
        quotes = leg.get('provider_quotes', [])
        if isinstance(quotes, str):
            quotes = json.loads(quotes)
        namespace, event = leg.get('provider_namespace'), leg.get('provider_event_id')
        if not namespace or not event or not leg.get('quote_time') or not leg.get('quote_source'):
            return None
        quotes = [q for q in quotes if q.get('provider_namespace') == namespace and q.get('provider_event_id') == event
                  and q.get('book', '').casefold() == leg['quote_source'].casefold()
                  and timestamp(q.get('recorded_at')) == timestamp(leg['quote_time'])]
        home_team, away_team = teams(leg)
        if not home_team or not away_team:
            return None
        home = [q for q in quotes if q.get('market_type') == 'moneyline_home']
        away = [q for q in quotes if q.get('market_type') == 'moneyline_away']
        if home or away:
            if len(home) != 1 or len(away) != 1:
                return None
            h, a = float(home[0]['price']), float(away[0]['price'])
            if not all(math.isfinite(v) and abs(v) >= 100 for v in (h, a)):
                return None
            probability = lambda v: -v / (100-v) if v < 0 else 100 / (100+v)
            return None if probability(h) == probability(a) else home_team if probability(h) > probability(a) else away_team
        # Standard spreads only, never a lone/alternate selected line.
        home = [q for q in quotes if q.get('market_type') == 'spread_home' and not q.get('is_alternate')]
        away = [q for q in quotes if q.get('market_type') == 'spread_away' and not q.get('is_alternate')]
        if len(home) != 1 or len(away) != 1:
            return None
        h, a = float(home[0]['point']), float(away[0]['point'])
        if not all(math.isfinite(v) for v in (h,a)) or h == 0 or abs(h+a) > 1e-9:
            return None
        return home_team if h < 0 else away_team
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def facts(row):
    leg = row['legs'][0]
    market = leg.get('market', '')
    kind = family(market)
    pick = leg.get('pick', '')
    home, away = teams(leg)
    selected = (home if market == 'spread_home' else away) if kind == 'spread' else None
    # Explicitly limited legacy fallback: structured orientation/team wins.
    match = re.fullmatch(r'(.+)\s+([+-]\d+(?:\.\d+)?)', pick) if kind == 'spread' else None
    total = re.fullmatch(r'(Over|Under)\s+(\d+(?:\.\d+)?)', pick, re.I) if kind == 'total' else None
    if kind == 'spread' and not selected and match:
        selected = match[1]
    line = leg.get('spread_line' if kind == 'spread' else 'total_line')
    if line is None:
        line = float(match[2]) if match else float(total[2]) if total else None
    return {'Pick': pick, 'Market': market, 'Market family': kind, 'Selected team': selected,
            'Direction': market.split('_')[1] if kind == 'total' else None, 'Market favorite': saved_favorite(leg),
            'Line': line, 'Odds': leg.get('odds'), 'Sportsbook': leg.get('quote_source'),
            'Analysis': leg.get('as_of'), 'Export run': leg.get('export_run_id'),
            'Prediction timestamp': leg.get('prediction_generated_at'), 'Quote time': leg.get('quote_time'),
            'Provider namespace': leg.get('provider_namespace'), 'Provider event ID': leg.get('provider_event_id')}


def compare(previous, current, removal=None):
    a, b = facts(previous), facts(current)
    changed = lambda k: a[k] not in (None, '') and b[k] not in (None, '') and a[k] != b[k]
    team = a['Market family'] == b['Market family'] == 'spread' and changed('Selected team')
    market_change = changed('Market') if 'unknown' in (a['Market family'], b['Market family']) else changed('Market family')
    same = a['Market family'] == b['Market family'] and (
        bool(a['Selected team']) and a['Selected team'] == b['Selected team'] or
        bool(a['Direction']) and a['Direction'] == b['Direction'] or a['Pick'] == b['Pick'])
    provider_known = all(x[k] for x in (a,b) for k in ('Provider namespace','Provider event ID'))
    flags = {'same_selection': bool(same), 'line_changed': changed('Line'), 'price_changed': changed('Odds'),
             'sportsbook_changed': changed('Sportsbook'), 'quote_time_changed': changed('Quote time'),
             'market_family_changed': market_change, 'selected_team_changed': team,
             'provider_identity_changed': bool(provider_known and (changed('Provider namespace') or changed('Provider event ID'))),
             'market_favorite_changed': changed('Market favorite') if a['Market favorite'] and b['Market favorite'] else None,
             'analysis_run_changed': changed('Export run') or changed('Analysis')}
    price = any(flags[k] for k in ('line_changed', 'price_changed', 'sportsbook_changed', 'quote_time_changed'))
    severity, code, state, label = (
        ('CRITICAL','SELECTED_TEAM_REVERSED','REMOVED_LOCK_TEAM_REVERSAL','Selected team reversed') if team else
        ('HIGH','MARKET_CHANGED','REMOVED_LOCK_MARKET_CHANGE','Market changed') if market_change else
        ('WARNING','SELECTION_CHANGED','REMOVED_LOCK_SELECTION_CHANGE','Selection changed') if not same else
        ('NORMAL','PRICE_CHANGED','REMOVED_LOCK_PRICE_CHANGE','Price changed') if price else
        ('NORMAL','NO_MATERIAL_CHANGE','REMOVED_LOCK_SAME','No material change'))
    return RelockComparison(current['id'], digest(previous), removal.get('removed_at') if removal else None,
        removal.get('reason') if removal else None, a, b, flags, severity, code, state, label,
        tuple(k for k in a if changed(k)),
        {k: (previous['legs'][0][k], current['legs'][0][k])
         for k in ('win_estimate','best_available_rank','market_probability','kalshi_probability','ml_probability')
         if previous['legs'][0].get(k) is not None and current['legs'][0].get(k) is not None})


def review_token(removal, candidate):
    return digest({'removed': removal, 'candidate_id': candidate['id'], 'legs': candidate['legs']})


def requirements(change):
    severity = change['severity']
    return {'checkbox': severity != 'NORMAL', 'typed': severity == 'CRITICAL', 'dialog': severity != 'NORMAL',
            'button': {'NORMAL':'Re-lock at current price','WARNING':'Confirm changed re-lock',
                       'HIGH':'Confirm market-change re-lock','CRITICAL':'Re-lock opposite side'}[severity]}


def acknowledged(change, checked=False, typed=''):
    return change['severity'] == 'NORMAL' or bool(checked and (change['severity'] != 'CRITICAL' or typed == 'RELOCK'))


def verify_review(removals, choices, requested, tokens):
    for identity in requested:
        prior = latest_removed(removals, identity)
        expected = review_token(prior, choices[identity]) if prior else None
        if tokens.get(identity) != expected:
            raise RelockReviewExpired('Re-lock review changed; review the current selection and archived lock again.')


def validate_candidates(package, selected, reviewed, at):
    from app_core.locked_picks import lock_candidates
    current = {r['id']:r for r in lock_candidates(package, at)}
    if not selected or not set(selected) <= current.keys():
        raise RelockReviewExpired('Candidate is no longer eligible')
    if any(i not in reviewed or current[i]['legs'] != reviewed[i]['legs'] for i in selected):
        raise RelockReviewExpired('Candidate changed')
    return current


def clear_review_state(session):
    for key in list(session):
        if key.startswith(('relock_ack_', 'relock_type_', 'relock_pending', 'relock_context')):
            del session[key]


def log_review(identity, change, outcome):
    import logging
    logging.getLogger(__name__).info('relock_review lock_id=%s severity=%s change_code=%s old_family=%s new_family=%s outcome=%s',
        identity, change['severity'], change['change_code'], change['previous']['Market family'], change['current']['Market family'], outcome)
