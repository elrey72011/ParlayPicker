"""Fit the NBA score model and save its artifact (roadmap M4.1).

Usage:
    python scripts/train_nba_model.py --seasons 2024-25 2025-26 [--csv games.csv] [--output models/nba-score-ridge-v1.json]

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
    args = parser.parse_args(argv)
    if args.csv:
        with args.csv.open(newline='') as fh:
            games = [dict(r, home_score=int(r['home_score']), away_score=int(r['away_score'])) for r in csv.DictReader(fh)]
    elif args.seasons:
        games = fetch(args.seasons)
    else:
        parser.error('give --seasons or --csv')
    model = fit(games)
    print(json.dumps(dict(games=len(games), train_games=model.train_games, holdout_games=model.holdout_games,
                          trained_through=model.trained_through, holdout=model.holdout), indent=2))
    if args.output:
        model.save(args.output)
        print(f'Wrote {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
