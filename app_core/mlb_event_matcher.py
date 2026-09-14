"""Pure MLB event identity matching, separate from sportsbook settlement."""
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo
import math
import re


@dataclass(frozen=True)
class EventMatch:
    status: str
    event: dict | None = None
    reason: str | None = None
    candidate_count: int = 0
    identity_method: str | None = None
    settlement_review_required: bool = False


def scheduled_eastern_date(record):
    for key in ('official_date', 'scheduled_date', 'game_date', 'event_date',
                'scheduled_start', 'start', 'commence_time'):
        value = record.get(key)
        if value is None:
            continue
        try:
            if isinstance(value, date) and not isinstance(value, datetime):
                return value
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
                return date.fromisoformat(value)
            dt = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.astimezone(ZoneInfo('America/New_York')).date() if dt.utcoffset() is not None else None
        except (ValueError, TypeError, AttributeError):
            return None
    return None


def provider_ids(record):
    def namespace(value):
        return {'mlb': 'mlb', 'espn': 'espn', 'odds_api': 'odds_api',
                'odds api': 'odds_api'}.get(str(value).lower())
    result = {}
    for key, value in (record.get('provider_ids') or {}).items():
        if namespace(key) and value is not None:
            result[namespace(key)] = str(value)
    for key, provider in [('mlb_game_pk', 'mlb'), ('espn_event_id', 'espn'), ('odds_event_id', 'odds_api')]:
        if record.get(key) is not None:
            result[provider] = str(record[key])
    source = namespace(record.get('game_id_provider') or record.get('provider') or record.get('source_provider') or record.get('result_source'))
    if source:
        for key in ('provider_event_id', 'event_id', 'game_id'):
            if record.get(key) is not None:
                result.setdefault(source, str(record[key]))
                break
    return result


def game_number(record):
    for key in ('game_number', 'gameNumber', 'doubleheader_game', 'provider_game_number'):
        value = record.get(key)
        try:
            if not isinstance(value, bool) and float(value).is_integer() and float(value) > 0:
                return int(float(value))
        except (ValueError, TypeError, OverflowError):
            pass
    return None


def _teams(record):
    from app_core.public_history import grading_team_name
    away, home = record.get('away'), record.get('home')
    if not away or not home:
        parts = re.split(r'\s+(?:at|@)\s+', record.get('game', ''), flags=re.I)
        if len(parts) != 2:
            return None
        away, home = parts
    return tuple(grading_team_name(t, 'MLB') for t in (away, home))


def match_mlb_event(saved, events):
    names, day = _teams(saved), scheduled_eastern_date(saved)
    if not names or day is None:
        return EventMatch('INVALID_SAVED_EVENT', reason='INVALID_SAVED_EVENT')
    pool = [e for e in events if str(e.get('sport', '')).upper() == 'MLB']
    ids = provider_ids(saved)
    identified = [e for e in pool if any(provider_ids(e).get(k) == v for k, v in ids.items())]
    method = 'provider_id' if identified else 'teams_scheduled_date'
    if identified:
        if any(_teams(e) != names or any(k in provider_ids(e) and provider_ids(e)[k] != v for k, v in ids.items()) for e in identified):
            return EventMatch('PROVIDER_ID_CONFLICT', reason='PROVIDER_ID_CONFLICT', candidate_count=len(identified))
        candidates = identified
    else:
        same_teams = [e for e in pool if _teams(e) == names]
        candidates = [e for e in same_teams if scheduled_eastern_date(e) == day]
        if not candidates:
            status = ('INVALID_RESULT_EVENT' if any(scheduled_eastern_date(e) is None for e in same_teams)
                      else 'DATE_MISMATCH' if same_teams else 'PROVIDER_ID_NOT_FOUND' if ids else 'NO_MATCH')
            return EventMatch(status, reason=status if status != 'NO_MATCH' else 'NO_FINAL_PROVIDER_RESULT')
    # Repeated identical observations are harmless; conflicting revisions remain ambiguous.
    candidates = [e for i, e in enumerate(candidates) if not provider_ids(e) or e not in candidates[:i]]
    sources = {}
    for event in candidates:
        source = tuple(sorted(provider_ids(event))) or ('unscoped',)
        sources.setdefault(source, []).append(event)
    selected = []
    for group in sources.values():
        if len(group) > 1:
            number = game_number(saved)
            if number is None:
                return EventMatch('DOUBLEHEADER_AMBIGUOUS', reason='DOUBLEHEADER_AMBIGUOUS', candidate_count=len(candidates))
            group = [e for e in group if game_number(e) == number]
            if len(group) != 1:
                return EventMatch('DOUBLEHEADER_AMBIGUOUS', reason='DOUBLEHEADER_AMBIGUOUS', candidate_count=len(candidates))
            if not identified:
                method = 'game_number'
        selected.extend(group)
    if any(scheduled_eastern_date(e) is None for e in selected):
        return EventMatch('INVALID_RESULT_EVENT', reason='INVALID_RESULT_EVENT', candidate_count=len(candidates))
    if identified and any(scheduled_eastern_date(e) != day for e in selected):
        return EventMatch('RESCHEDULED_NEEDS_REVIEW', selected[0], 'RESCHEDULED_SETTLEMENT_REVIEW', len(candidates), method, True)
    finals = [e for e in selected if e.get('final', e.get('completed', True))]
    if not finals:
        return EventMatch('NO_MATCH', reason='NO_FINAL_PROVIDER_RESULT', candidate_count=len(candidates))
    for event in finals:
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 or int(v) != v
               for v in (event.get('away_score'), event.get('home_score'))):
            return EventMatch('INVALID_RESULT_EVENT', reason='FINAL_SCORE_INVALID', candidate_count=len(candidates))
    if len({(e['away_score'], e['home_score']) for e in finals}) > 1:
        return EventMatch('MULTIPLE_MATCHING_EVENTS', reason='PROVIDER_SCORE_CONFLICT', candidate_count=len(candidates))
    status = {'provider_id': 'MATCHED_PROVIDER_ID', 'game_number': 'MATCHED_GAME_NUMBER', 'teams_scheduled_date': 'MATCHED_UNIQUE_EVENT'}[method]
    return EventMatch(status, finals[0], candidate_count=len(candidates), identity_method=method)
