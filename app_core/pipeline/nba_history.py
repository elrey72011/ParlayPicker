"""NBA completed-game history for the shared pipeline (roadmap M4.1).

History = the shipped baseline (``data/nba/espn-games.csv``: the D9 download,
2024-25 through 2026-11-04) plus each later completed day from ESPN's public
scoreboard. Each day is one *claimed* acquisition operation, so a day is
downloaded at most once per successful attempt and a restart never repeats it.
Only days strictly before the as-of date are fetched (their games are final).
Only games between the 30 NBA franchises are kept, without duplicates.

No Odds API credits are used.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

from app_core.pipeline.adapter import AdapterError, instant
from app_core.pipeline.operations import CANCELLED, COMPLETED, EXPIRED, FAILED, Operations
from app_core.pipeline.store import StoreError

BASELINE = Path(__file__).resolve().parents[2] / 'data' / 'nba' / 'espn-games.csv'
ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
ATTEMPTS_PER_DAY = 3
MAX_DAYS_PER_CALL = 30
WINDOW = timedelta(minutes=10)

# The 30 NBA franchises as ESPN names them. All-Star teams ("Team Shaq", "World")
# and exhibition opponents are excluded.
NBA_FRANCHISES = frozenset({
    'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers',
    'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets',
    'Indiana Pacers', 'LA Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks',
    'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic',
    'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs',
    'Toronto Raptors', 'Utah Jazz', 'Washington Wizards'})
# Odds API (and other feeds) name -> ESPN name, where they differ.
ALIASES = {'Los Angeles Clippers': 'LA Clippers'}


def team(name):
    """ESPN franchise name for a feed's team name, or None if it is not an NBA franchise."""
    name = ALIASES.get(str(name).strip(), str(name).strip())
    return name if name in NBA_FRANCHISES else None


def franchise_games(games):
    """Only games between two NBA franchises, with exact duplicates removed."""
    seen, out = set(), []
    for game in games:
        key = (game['game_date'], game['home'], game['away'])
        if game['home'] in NBA_FRANCHISES and game['away'] in NBA_FRANCHISES and key not in seen:
            seen.add(key)
            out.append(game)
    return out


def games_from_espn(payload, day=None):
    """Completed, non-neutral regular-season games from one ESPN scoreboard payload.

    ``day`` is the requested scoreboard date (US local). It is the game date, because
    ESPN's own ``date`` field is UTC and would move evening games to the next day.
    """
    games = []
    for event in payload.get('events', []):
        if (event.get('season') or {}).get('type') != 2:
            continue
        competition = (event.get('competitions') or [{}])[0]
        if competition.get('neutralSite') or not competition.get('status', {}).get('type', {}).get('completed'):
            continue
        sides = {c.get('homeAway'): c for c in competition.get('competitors', [])}
        if set(sides) != {'home', 'away'}:
            continue
        try:
            hs, as_ = int(sides['home']['score']), int(sides['away']['score'])
        except (KeyError, TypeError, ValueError):
            continue
        games.append(dict(game_date=day or str(event['date'])[:10], home=sides['home']['team']['displayName'],
                          away=sides['away']['team']['displayName'], home_score=hs, away_score=as_,
                          espn_event_id=str(event.get('id'))))
    return games


def espn_day(day: date):
    """One ESPN scoreboard request for a single US date."""
    import requests
    response = requests.get(ESPN_SCOREBOARD, params={'dates': day.strftime('%Y%m%d')}, timeout=20)
    response.raise_for_status()
    return games_from_espn(response.json(), day.isoformat())


def load_baseline(path=None):
    with Path(path or BASELINE).open(newline='') as fh:
        return [dict(r, home_score=int(r['home_score']), away_score=int(r['away_score'])) for r in csv.DictReader(fh)]


def _claimed_day(ops: Operations, day: date, now, fetch):
    subject = dict(source='espn-nba-scoreboard', day=day.isoformat())
    predecessor = None
    for _ in range(ATTEMPTS_PER_DAY):
        op = ops.create('NBA', 'acquisition', subject, not_after=(now + WINDOW).isoformat(), review_required=False,
                        predecessor=predecessor)
        status = ops.reconcile('NBA', op['id'])
        if status['state'] == COMPLETED:
            return ops.result('NBA', op['id'])['output']['games']
        if status['state'] in {FAILED, EXPIRED, CANCELLED}:
            predecessor = op['id']
            continue
        try:
            return ops.run('NBA', op['id'], lambda operation: dict(games=fetch(day)))['output']['games']
        except StoreError:
            raise
        except Exception:
            predecessor = op['id']
    raise AdapterError('NBA_HISTORY_DAY_UNAVAILABLE:' + day.isoformat())


def history(ops: Operations, as_of_utc: str, *, fetch=espn_day, baseline=None, max_days=MAX_DAYS_PER_CALL):
    """Baseline plus every completed day after it and before the as-of date."""
    now = instant(as_of_utc)
    games = load_baseline(baseline)
    last = max(date.fromisoformat(g['game_date']) for g in games)
    missing = []
    day = last + timedelta(days=1)
    while day < now.date():
        missing.append(day)
        day += timedelta(days=1)
    if len(missing) > max_days:
        raise AdapterError(f'NBA_HISTORY_BACKLOG:{len(missing)}_DAYS')  # refresh the shipped baseline instead
    for day in missing:
        games.extend(_claimed_day(ops, day, now, fetch))
    return franchise_games(sorted(games, key=lambda g: (g['game_date'], g['home'])))
