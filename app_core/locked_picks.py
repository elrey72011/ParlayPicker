"""Immutable owner-selected overall picks, distinct from published/approved history."""
from datetime import datetime
from zoneinfo import ZoneInfo
from app_core.public_history import eligible, event_key, digest
from app_core.quote_freshness import package_age_minutes


def lock_candidates(package, at):
    from app_core.public_board import validate_package
    validate_package(package)
    clock = datetime.fromisoformat(at)
    today = clock.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    rows = {}
    for leg in package['games']['overall']:
        if not eligible(leg, clock, max_age_minutes=package_age_minutes(package)):
            continue
        date = datetime.fromisoformat(leg['start']).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        if date != today:
            continue
        identity = digest(('locked-overall', *event_key(leg)[:3], date))
        if identity in rows:
            raise ValueError('Multiple games have the same teams and date; lock needs an unambiguous game.')
        rows[identity] = {'id': identity, 'category': 'overall', 'date': date, 'group': 'Locked',
                          'published_at': at, 'legs': [dict(leg)]}
    return list(rows.values())


def locked_selections(locks):
    result = {}
    for row in sorted(locks, key=lambda item: item['published_at']):
        if row.get('category') != 'overall' or row.get('group') != 'Locked' or len(row.get('legs', [])) != 1:
            raise ValueError('Invalid locked pick record')
        leg = row['legs'][0]
        at = datetime.fromisoformat(row['published_at'])
        key = event_key(leg)
        if not key or not at.tzinfo or not eligible(leg, at):
            raise ValueError('Invalid locked pick timing or game')
        day = datetime.fromisoformat(leg['start']).astimezone(ZoneInfo('America/New_York')).date().isoformat()
        if row['date'] != day or at.astimezone(ZoneInfo('America/New_York')).date().isoformat() != day:
            raise ValueError('Locked pick date mismatch')
        expected = digest(('locked-overall', *key[:3], row['date']))
        if expected != row['id'] or not eligible(leg, at):
            raise ValueError('Locked pick identity or timing mismatch')
        if row['id'] in result and result[row['id']] != row:
            raise ValueError('Conflicting locked picks')
        result[row['id']] = row
    return list(result.values())


def lock_audit(package, at, locks=()):
    """Explain every current board row without changing eligibility or saved locks."""
    from app_core.public_board import validate_package
    from app_core.public_quote_policy import supported_quote
    validate_package(package)
    clock = datetime.fromisoformat(at)
    eastern = ZoneInfo('America/New_York')
    today = clock.astimezone(eastern).date().isoformat()
    limit = package_age_minutes(package)
    existing = {r['id'] for r in locks}
    seen, result = set(), []
    def parsed(value):
        try:
            value = datetime.fromisoformat(value)
            return value if value.tzinfo else None
        except (TypeError, ValueError):
            return None
    def display(value):
        return value.astimezone(eastern).strftime('%Y-%m-%d %I:%M %p') if value else 'Unavailable'
    for leg in package['games']['overall']:
        start, analysis, quote = (parsed(leg.get(k)) for k in ('start', 'as_of', 'quote_time'))
        key = event_key(leg) if start else None
        day = start.astimezone(eastern).date().isoformat() if start else ''
        identity = digest(('locked-overall', *key[:3], day)) if key else None
        event = (*key[:3], start) if key else None
        if event and event in seen:
            status, detail = 'Duplicate game entry', 'Same normalized teams and start time already appear in this board.'
        elif identity in existing:
            status, detail = 'Already locked', 'Original selection and odds remain saved; no new lock is needed.'
        elif not start or not key:
            status, detail = 'Missing game timing', 'A valid start time and identifiable away/home teams are required.'
        elif day != today:
            status, detail = 'Other date', 'Locks are available only on the game date in Eastern time.'
        elif start <= clock:
            status, detail = 'Started', 'Scheduled start time has passed; the app cannot create a pregame lock.'
        elif 'quote_source' in leg and (not supported_quote(leg) or not quote):
            status, detail = 'Quote unavailable', leg.get('quote_reason') or 'No verified supported sportsbook quote is saved for this pick.'
        elif quote and quote > clock:
            status, detail = 'Future quote timestamp', 'The saved sportsbook timestamp is ahead of the current time.'
        elif quote and (clock - quote).total_seconds() > limit * 60:
            status, detail = 'Stale quote', f'Sportsbook quote is older than {limit} minutes; refresh game picks.'
        elif not analysis or analysis > clock:
            status, detail = 'Invalid analysis time', 'Analysis time is missing or ahead of the current time.'
        elif (clock - analysis).total_seconds() > limit * 60:
            status, detail = 'Stale analysis', f'Game analysis is older than {limit} minutes; refresh game picks.'
        elif not eligible(leg, clock, max_age_minutes=limit):
            status, detail = 'Invalid market or odds', 'The saved market or price is not supported for locking.'
        else:
            status, detail = 'Eligible now', 'Select this game below to save its current pick and price.'
        if event:
            seen.add(event)
        result.append({'League':leg['sport'], 'Game':leg['game'], 'Lock status':status,
                       'Reason':detail, 'Pick':leg['pick'], 'Sportsbook':leg.get('quote_source','Not recorded'),
                       'Start (Eastern)':display(start), 'Analysis (Eastern)':display(analysis),
                       'Quote (Eastern)':display(quote), 'Time basis':'Observed via ESPN; sportsbook update unknown' if leg.get('quote_time_basis') == 'espn_observed' else 'Provider update'})
    return result
