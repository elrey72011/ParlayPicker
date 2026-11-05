"""Fit the NBA score model and save its artifact (roadmap M4.1).

Usage:
    python scripts/train_nba_model.py --seasons 2024-25 2025-26 2026-27 [--espn] [--through YYYY-MM-DD]
                                      [--csv games.csv] [--save-games games.csv] [--output models/nba-score-ridge-v1.json]

Data comes from the free public NBA stats endpoint via ``nba_api`` (the source
ordinary Refresh already uses), or from a CSV with columns game_date, home, away,
home_score, away_score. No Odds API credits are used. Running it downloads
training data, which needs the owner's approval (see roadmap decisions).

The script prints the time-ordered holdout report. It writes the artifact only
with --output; committing that artifact is what makes NBA a supported sport.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app_core.pipeline.nba import fit  # noqa: E402


def games_from_game_finder(rows):
    """Completed games from LeagueGameFinder team rows (two per game), keyed by full team name."""
    by_game = {}
    for row in rows:
        by_game.setdefault(str(row['GAME_ID']), []).append(row)
    games = []
    for pair in by_game.values():
        if len(pair) != 2:
            continue
        home = [r for r in pair if ' vs. ' in str(r['MATCHUP'])]
        away = [r for r in pair if ' @ ' in str(r['MATCHUP'])]
        if len(home) != 1 or len(away) != 1:
            continue  # neutral or malformed rows are left out, not guessed
        h, a = home[0], away[0]
        try:
            hs, as_ = int(h['PTS']), int(a['PTS'])
        except (TypeError, ValueError):
            continue
        games.append(dict(game_date=str(h['GAME_DATE'])[:10], home=str(h['TEAM_NAME']), away=str(a['TEAM_NAME']),
                          home_score=hs, away_score=as_))
    return sorted(games, key=lambda g: (g['game_date'], g['home']))


# The 30 NBA franchises as ESPN names them. Anything else (All-Star teams such as
# "Team Shaq" or "World", exhibition opponents) is excluded, never trained on.
NBA_FRANCHISES = frozenset({
    'Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets', 'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers',
    'Dallas Mavericks', 'Denver Nuggets', 'Detroit Pistons', 'Golden State Warriors', 'Houston Rockets',
    'Indiana Pacers', 'LA Clippers', 'Los Angeles Lakers', 'Memphis Grizzlies', 'Miami Heat', 'Milwaukee Bucks',
    'Minnesota Timberwolves', 'New Orleans Pelicans', 'New York Knicks', 'Oklahoma City Thunder', 'Orlando Magic',
    'Philadelphia 76ers', 'Phoenix Suns', 'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs',
    'Toronto Raptors', 'Utah Jazz', 'Washington Wizards'})


def franchise_games(games):
    """Only games between two NBA franchises, with exact duplicates removed."""
    seen, out = set(), []
    for game in games:
        key = (game['game_date'], game['home'], game['away'])
        if game['home'] in NBA_FRANCHISES and game['away'] in NBA_FRANCHISES and key not in seen:
            seen.add(key)
            out.append(game)
    return out


ESPN_SCOREBOARD = 'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard'
# Regular-season date windows (inclusive); the current season ends at the as-of date.
SEASON_WINDOWS = {'2024-25': ('2024-10-22', '2025-04-13'), '2025-26': ('2025-10-21', '2026-04-12'),
                  '2026-27': ('2026-10-20', '2027-04-11')}


def games_from_espn(payload, day=None):
    """Completed, non-neutral regular-season games from one ESPN scoreboard payload.

    ``day`` is the requested scoreboard date (US local). It is the game date, because
    ESPN's own ``date`` field is UTC and would move evening games to the next day.
    """
    games = []
    for event in payload.get('events', []):
        if (event.get('season') or {}).get('type') != 2:
            continue  # regular season only
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


def fetch_espn(seasons, *, through=None, pause=0.25):
    """One ESPN request per calendar day in each season window (free, public; no Odds API credits)."""
    from datetime import date, timedelta
    import time
    import requests
    through = date.fromisoformat(through) if through else date.today() - timedelta(days=1)
    games, seen = [], set()
    for season in seasons:
        start, end = (date.fromisoformat(d) for d in SEASON_WINDOWS[season])
        day = start
        while day <= min(end, through):
            response = requests.get(ESPN_SCOREBOARD, params={'dates': day.strftime('%Y%m%d')}, timeout=20)
            response.raise_for_status()
            for game in games_from_espn(response.json(), day.isoformat()):
                if game['espn_event_id'] not in seen:
                    seen.add(game['espn_event_id'])
                    games.append(game)
            day += timedelta(days=1)
            time.sleep(pause)
    return sorted(games, key=lambda g: (g['game_date'], g['home']))


def fetch(seasons):
    from nba_api.stats.endpoints import leaguegamefinder
    rows = []
    for season in seasons:
        frame = leaguegamefinder.LeagueGameFinder(season_nullable=season, league_id_nullable='00',
                                                  season_type_nullable='Regular Season').get_data_frames()[0]
        rows.extend(frame.to_dict('records'))
    return games_from_game_finder(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--seasons', nargs='*', default=[])
    parser.add_argument('--csv', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--espn', action='store_true', help='use the ESPN public scoreboard instead of nba_api')
    parser.add_argument('--through', help='last date to include (default: yesterday)')
    parser.add_argument('--save-games', type=Path, help='also write the downloaded games to this CSV')
    args = parser.parse_args(argv)
    if args.csv:
        with args.csv.open(newline='') as fh:
            games = [dict(r, home_score=int(r['home_score']), away_score=int(r['away_score'])) for r in csv.DictReader(fh)]
    elif args.seasons:
        games = fetch_espn(args.seasons, through=args.through) if args.espn else fetch(args.seasons)
        if args.save_games:
            with args.save_games.open('w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=['game_date', 'home', 'away', 'home_score', 'away_score'],
                                        extrasaction='ignore')
                writer.writeheader()
                writer.writerows(games)
    else:
        parser.error('give --seasons or --csv')
    downloaded = len(games)
    games = franchise_games(games)
    model = fit(games)
    print(json.dumps(dict(games=len(games), excluded_non_franchise_or_duplicate=downloaded - len(games), train_games=model.train_games, holdout_games=model.holdout_games,
                          trained_through=model.trained_through, holdout=model.holdout), indent=2))
    if args.output:
        model.save(args.output)
        print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
