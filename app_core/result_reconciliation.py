"""Deterministic result identity matching; no network or historical mutation."""
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import re


def stamp(value):
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed if parsed.tzinfo else None
    except (ValueError, TypeError, AttributeError):
        return None


def match_result(leg, scores):
    if str(leg.get('sport', '')).upper() == 'MLB':
        from app_core.mlb_event_matcher import match_mlb_event
        match = match_mlb_event(leg, scores)
        return (match.event, None) if match.status.startswith('MATCHED_') else (None, match.reason or match.status)
    from app_core.public_history import grading_team_name
    sport = str(leg.get('sport', '')).upper()
    from app_core.espn_results import ESPN_ENDPOINTS
    if sport not in ESPN_ENDPOINTS:
        return None, 'UNSUPPORTED_SPORT'
    teams = re.split(r'\s+(?:at|@)\s+', leg.get('game', ''), flags=re.I)
    start = stamp(leg.get('start'))
    if len(teams) != 2 or start is None:
        return None, 'TEAM_IDENTITY_MISMATCH'
    names = tuple(grading_team_name(t, sport) for t in teams)
    day = start.astimezone(ZoneInfo('America/New_York')).date()
    pool = [s for s in scores if s.get('sport', '').upper() == sport]
    ids = {provider: str(leg[key]) for provider, key in [('ESPN','espn_event_id'), ('MLB','mlb_game_pk')] if leg.get(key)}
    if leg.get('result_source') in {'ESPN','MLB'} and leg.get('provider_event_id'):
        ids[leg['result_source']] = str(leg['provider_event_id'])
    # Generic sportsbook event IDs are not ESPN/MLB IDs.
    if ids:
        pool = [s for s in pool if s.get('result_source', 'ESPN') in ids and str(s.get('provider_event_id', s.get('event_id'))) == ids[s.get('result_source','ESPN')]]
        if not pool:
            return None, 'EVENT_ID_NOT_FOUND'
    candidates = []
    for score in pool:
        when = stamp(score.get('start'))
        if tuple(grading_team_name(score.get(k,''), sport) for k in ('away','home')) != names:
            continue
        if when and (ids or when.astimezone(ZoneInfo('America/New_York')).date() == day):
            candidates.append(score)
    if not candidates:
        return None, 'TEAM_IDENTITY_MISMATCH' if ids else 'NO_FINAL_PROVIDER_RESULT'
    # Collapse repeat revisions by provider ID, never by teams alone.
    unique = {(s.get('result_source','ESPN'), str(s.get('provider_event_id',s.get('event_id')))): s for s in candidates}
    candidates = list(unique.values())
    by_source = {}
    for score in candidates:
        by_source.setdefault(score.get('result_source','ESPN'), []).append(score)
    selected = []
    for source, events in by_source.items():
        if len(events) > 1:
            number = leg.get('game_number')
            if number is not None:
                events = [s for s in events if str(s.get('game_number')) == str(number)]
            else:
                events = [s for s in events if abs((stamp(s['start']) - start).total_seconds()) <= 1800]
            if len(events) != 1:
                return None, 'DOUBLEHEADER_AMBIGUOUS' if sport == 'MLB' else 'MULTIPLE_MATCHING_EVENTS'
        selected.extend(events)
    finals = [s for s in selected if s.get('completed', True)]
    if not finals:
        return None, 'NO_FINAL_PROVIDER_RESULT'
    for s in finals:
        values = [s.get('away_score'), s.get('home_score')]
        if any(isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or n < 0 or int(n) != n for n in values):
            return None, 'FINAL_SCORE_INVALID'
    if len({(s['away_score'],s['home_score']) for s in finals}) > 1:
        return None, 'PROVIDER_SCORE_CONFLICT'
    return sorted(finals, key=lambda s: (s.get('result_source','ESPN') != 'ESPN', str(s.get('event_id'))))[0], None


def pending_diagnostics(entries, revisions):
    from app_core.public_history import grade_leg
    scores = latest_scores(revisions)
    result = []
    for entry in entries:
        for leg in entry['legs']:
            score, reason = match_result(leg, scores)
            if reason is None and grade_leg(leg, scores)[0] == 'PENDING':
                reason = 'SELECTION_IDENTITY_MISMATCH'
            if reason:
                failures = [e for r in revisions for e in r.get('errors',[]) if e.get('sport') == leg['sport']]
                if reason == 'NO_FINAL_PROVIDER_RESULT' and failures:
                    reason = 'PROVIDER_FAILURE'
                metadata = {}
                if leg['sport'] == 'MLB':
                    from app_core.mlb_event_matcher import match_mlb_event
                    match = match_mlb_event(leg, scores)
                    metadata = {'event_match_status':match.status, 'event_match_method':match.identity_method, 'candidate_count':match.candidate_count, 'settlement_review_required':match.settlement_review_required}
                result.append({**metadata, 'id':entry['id'],'date':entry['date'],'game':leg['game'],'pick':leg['pick'],'reason':reason})
    return result


def latest_scores(revisions):
    latest = {}
    for revision in sorted(revisions, key=lambda r:r['recorded_at']):
        for score in [*revision.get('events',[]), *revision['scores']]:
            key = (score['sport'],score.get('result_source','ESPN'),str(score['event_id']))
            # An unfinished schedule observation cannot erase a verified final.
            if ('away_score' in score and 'home_score' in score) or 'away_score' not in latest.get(key,{}):
                latest[key] = score
    return list(latest.values())


def revision_signature(revision):
    from app_core.public_history import digest
    return digest({k: sorted([{n:v for n,v in row.items() if n != 'provider_recorded_at'} for row in revision.get(k,[])], key=lambda r: str(sorted(r.items()))) for k in ('scores','events','errors')})
