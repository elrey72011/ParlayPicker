"""Daily Top 10 cohorts derived only from immutable confirmed publications."""
import math
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo('America/New_York')
POLICY = 'first-publication-v1'


def ranked_picks(package, at):
    from app_core.public_history import eligible
    from app_core.public_quote_policy import supported_quote
    from app_core.quote_freshness import package_age_minutes
    if package.get('selection_policy') != 'qualified-v1':
        return []
    day = at.astimezone(EASTERN).date()
    def qualifies(row):
        probability, ev = row.get('win_estimate'), row.get('ev')
        return (row.get('status') == 'APPROVED' and supported_quote(row)
                and isinstance(probability, (int, float)) and not isinstance(probability, bool)
                and math.isfinite(probability) and 0 < probability <= 1
                and isinstance(ev, (int, float)) and not isinstance(ev, bool) and math.isfinite(ev) and ev > 0
                and eligible(row, at, max_age_minutes=package_age_minutes(package))
                and datetime.fromisoformat(row['start']).astimezone(EASTERN).date() == day)
    rows = [r for r in package['games']['overall'] if qualifies(r)]
    rows.sort(key=lambda r: (-r['win_estimate'], -r['ev'], r['game']))
    return rows[:10]


def top_ten_selections(publications):
    """First non-empty daily cohort wins; later publications cannot refill or replace it."""
    from app_core.public_history import digest, event_key
    days, entries = set(), []
    for pub in sorted(publications, key=lambda p: (datetime.fromisoformat(p['confirmed_at']), p['package_hash'])):
        package = pub['package']
        if package.get('top_ten_policy') != POLICY:
            continue  # Never reconstruct a record for releases without tracking.
        at = datetime.fromisoformat(pub['confirmed_at'])
        day = at.astimezone(EASTERN).date().isoformat()
        if day in days:
            continue
        rows = ranked_picks(package, at)
        if not rows:
            continue
        days.add(day)
        for row in rows:
            entries.append({'id': digest(('top10', day, event_key(row))),
                            'category': 'top10', 'date': day, 'group': 'Approved',
                            'published_at': pub['confirmed_at'], 'legs': [row]})
    return entries
