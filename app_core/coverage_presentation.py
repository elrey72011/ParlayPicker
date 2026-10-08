"""Canonical presentation after exact named-side verification; no authority.

Original producer descriptors remain in publication input and private exports.
No original facts, schedule matching policy or wagering gates are repaired here.
"""
from app_core.slate_coverage import _match, clock, digest, text

CODE = 'COVERAGE_PUBLIC_EVENT_BINDING_CONFLICT'
TEAM_FIELDS = (('home_team', 'Home'), ('away_team', 'Away'))
START_FIELDS = ('game_start_utc', 'commence_time', 'start', 'Commence (Local)', 'game_time_est')
DESCRIPTOR_FIELDS = ('league', 'League', 'sport', 'canonical_event_id', 'schedule_event_id',
    'matchup_id', 'home_team', 'Home', 'away_team', 'Away', *START_FIELDS)


class CoverageBindingConflict(ValueError):
    def __init__(self, decision, category, field, expected, actual):
        super().__init__(CODE)
        self.diagnostic = dict(version='coverage-preview-conflict-v1', code=CODE,
            stage=_safe(category), board_category=category if category in {'overall', 'sides', 'totals'} else 'overall',
            canonical_event_id=_safe(decision.get('canonical_event_id', '')),
            inventory_id=_safe(decision.get('inventory_id', '')), run_id=_safe(decision.get('run_id', '')),
            field=field, expected=_safe(expected), actual=_safe(actual))


def _safe(value):
    if value is None: return None
    if not isinstance(value, str): return '[unsupported descriptor omitted]'
    if '://' in value or any(ord(c) < 32 for c in value) or len(value) > 240:
        return '[unsupported descriptor omitted]'
    return value


def original_descriptors(row):
    return {key: descriptor(row, key) for key in DESCRIPTOR_FIELDS if descriptor(row, key)}


def descriptor(row, key):
    from datetime import datetime
    from app_core.candidate_evidence_schema import missing
    value = row.get(key)
    if missing(value): return ''
    return value.isoformat() if isinstance(value, datetime) else text(row, key)


def instant(value):
    if not value: return None
    from app_core.public_board import timestamp
    from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
    try: return clock(timestamp(value))
    except (ValueError, TypeError, AmbiguousTimeError, NonExistentTimeError): return None


def team_key(league, value):
    if league == 'NCAAF':
        from app_core.ncaaf_identity import normalize_ncaaf_team
        return normalize_ncaaf_team(value)
    if league == 'NFL':
        from app_core.nfl_identity import nfl_result_name
        return nfl_result_name(value)
    return value.casefold()


def verify(row, decision, events, category, *, placeholder=False):
    """Check every supplied descriptor, including those ignored by precedence."""
    def reject(field, expected, actual):
        raise CoverageBindingConflict(decision, category, field, expected, actual)
    for field in ('canonical_event_id', 'schedule_event_id'):
        value = text(row, field)
        if value and value != decision['canonical_event_id']:
            reject(field, decision['canonical_event_id'], value)
    rid = text(row, 'matchup_id')
    if rid in {e['canonical_event_id'] for e in events} and rid != decision['canonical_event_id']:
        reject('matchup_id', decision['canonical_event_id'], rid)
    for field in ('league', 'League', 'sport'):
        value = text(row, field)
        if value and value.upper() != decision['league']:
            reject(field, decision['league'], value)
    status = text(row, 'football_identity_status')
    schedule_status = text(row, 'schedule_match_status')
    if status == 'CONFLICT' or schedule_status in {'AMBIGUOUS_PROVIDER_ID', 'AMBIGUOUS', 'KICKOFF_OR_IDENTITY_CONFLICT'}:
        reject('event_match', 'one unambiguous named-side event', status if status == 'CONFLICT' else schedule_status)
    if team_key(decision['league'], decision['home_team']) == team_key(decision['league'], decision['away_team']):
        reject('event_match', 'distinct named home and away teams', 'ambiguous normalized schedule labels')
    for fields, canonical in zip(TEAM_FIELDS, ('home_team', 'away_team')):
        if not any(text(row, f) for f in fields): reject(canonical, decision[canonical], None)
        for field in fields:
            value = text(row, field)
            if value and team_key(decision['league'], value) != team_key(decision['league'], decision[canonical]):
                reject(field, decision[canonical], value)
    values = [(f, descriptor(row, f)) for f in START_FIELDS if descriptor(row, f)]
    if not values and not placeholder: reject('game_start_utc', decision['original_start'], None)
    for field, value in values:
        if instant(value) is None or instant(value) != clock(decision['original_start']):
            reject(field, decision['original_start'], value)
    if placeholder: return
    comparable = dict(row, league=decision['league'],
        game_start_utc=instant(values[0][1]).isoformat() if values else '')
    eid, status, matches = _match(comparable, events)
    if eid != decision['canonical_event_id']:
        reject('event_match', 'one unambiguous named-side event', status + ':' + ','.join(matches))


def resolve(row, events):
    """Resolve without daily-name joins. Known conflicting rows remain failures."""
    comparable = dict(row)
    start = next((descriptor(row, f) for f in START_FIELDS if descriptor(row, f)), '')
    parsed = instant(start)
    comparable['game_start_utc'] = parsed.isoformat() if parsed else ''
    eid, status, matches = _match(comparable, events)
    if eid:
        decision = next(d for d in events if d['canonical_event_id'] == eid)
        verify(row, decision, events, 'publication_rows')
        return decision
    claimed = [d for d in events if d['canonical_event_id'] in
        {text(row, k) for k in ('canonical_event_id', 'schedule_event_id', 'matchup_id')}]
    if len(claimed) == 1:
        if not start:
            # Missing original clocks support no evaluated selection. Preserve
            # the independent schedule placeholder, never borrow its clock for
            # this candidate or carry any candidate price/probability/stake.
            verify(row, claimed[0], events, 'publication_rows', placeholder=True)
            return None
        verify(row, claimed[0], events, 'publication_rows')
    # Diagnosis only: this never substitutes the schedule clock for original input.
    named = [d for d in events if text(row, 'league', 'sport', 'League').upper() == d['league']
        and all(team_key(d['league'], text(row, *fields)) == team_key(d['league'], d[canonical])
            for fields, canonical in zip(TEAM_FIELDS, ('home_team', 'away_team')))]
    if len(named) == 1:
        if not start:
            verify(row, named[0], events, 'publication_rows', placeholder=True)
            return None
        verify(row, named[0], events, 'publication_rows')
    if claimed or len(named) > 1 or status == 'IDENTITY_AMBIGUOUS':
        # An ambiguous row has no verified canonical event. Do not label the
        # first possible game as its identity, even in a private diagnostic.
        d = claimed[0] if len(claimed) == 1 else dict(canonical_event_id='', inventory_id='', run_id='')
        raise CoverageBindingConflict(d, 'publication_rows', 'event_match',
            'one unambiguous named-side event', status + ':' + ','.join(matches))
    return None  # Genuine orphans stay outside the schedule denominator.


def decision_for(row, report):
    import json
    value = row.get('coverage_decision')
    if not isinstance(value, str): return None
    decision = json.loads(value)
    expected = next((d for d in report['decisions'] if d['canonical_event_id'] == decision['canonical_event_id']), None)
    if expected != decision:
        raise CoverageBindingConflict(decision, 'publication_rows', 'coverage_decision',
            digest(expected), digest(decision))
    return decision
