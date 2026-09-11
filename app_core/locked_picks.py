"""Immutable owner-selected overall picks, distinct from published/approved history."""
from datetime import datetime
from zoneinfo import ZoneInfo
from app_core.public_history import eligible, event_key, digest


def lock_candidates(package, at):
    from app_core.public_board import validate_package
    validate_package(package)
    clock = datetime.fromisoformat(at)
    today = clock.astimezone(ZoneInfo('America/New_York')).date().isoformat()
    rows = {}
    for leg in package['games']['overall']:
        if not eligible(leg, clock):
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
