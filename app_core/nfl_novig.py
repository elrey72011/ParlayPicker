"""One bounded bulk retry when the NFL feed omits Novig markets."""
from copy import deepcopy
from datetime import datetime, timezone
import logging
import requests

log = logging.getLogger(__name__)


def recover_nfl_novig(games, api_key, *, get=None, now=None):
    result = deepcopy(games)
    now = now or datetime.now(timezone.utc)
    missing = {}
    for game in result:
        try:
            if datetime.fromisoformat(game['commence_time'].replace('Z', '+00:00')) <= now:
                continue
        except (KeyError, ValueError, TypeError):
            continue
        present = {m.get('key') for b in game.get('bookmakers', [])
                   if b.get('key') in {'novig', 'novig_us'}
                   for m in b.get('markets', []) if m.get('outcomes')}
        absent = {'spreads', 'totals'} - present
        if absent and game.get('id'):
            missing[game['id']] = (game, absent)
    if not missing or not api_key:
        return result
    # One request per slate, not one request per missing game.
    params = {'apiKey': api_key, 'bookmakers': 'novig', 'markets': 'spreads,totals',
              'oddsFormat': 'american', 'dateFormat': 'iso'}
    starts = [g['commence_time'] for g, _ in missing.values()]
    params.update(commenceTimeFrom=min(starts), commenceTimeTo=max(starts))
    recovered_count = 0
    try:
        response = (get or requests.get)(
            'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds',
            params=params, timeout=5)
        response.raise_for_status()
        recovered = response.json()
        if not isinstance(recovered, list):
            return result
        counts = {}
        for row in recovered:
            if isinstance(row, dict):
                counts[row.get('id')] = counts.get(row.get('id'), 0) + 1
        for offer in recovered:
            if not isinstance(offer, dict) or offer.get('id') not in missing or counts[offer['id']] != 1:
                continue
            game, absent = missing[offer['id']]
            if any(offer.get(k) != game.get(k) for k in ('home_team', 'away_team', 'commence_time')):
                continue
            for book in offer.get('bookmakers', []):
                if book.get('key') not in {'novig', 'novig_us'}:
                    continue
                markets = [m for m in book.get('markets', []) if m.get('key') in absent and m.get('outcomes')]
                if markets:
                    game.setdefault('bookmakers', []).append({**book, 'markets': markets})
                    recovered_count += len(markets)
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        log.warning('NFL Novig recovery failed; original quotes retained')
    log.warning('NFL Novig recovery missing_games=%s recovered_markets=%s requests=1', len(missing), recovered_count)
    return result
