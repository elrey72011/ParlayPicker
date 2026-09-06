"""Replay the direction guard on a saved candidate audit, never refit on outcomes.

Usage: python scripts/replay_model_direction.py GRADED_CANDIDATE_AUDIT.csv
This is a fixed-candidate replay, not a historical feature/pipeline reconstruction.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.model_direction import guard_model_direction
from app_core.espn_ncaaf_odds import merge_missing_ncaaf_games, ESPN_FALLBACK_SOURCE


def replay(frame: pd.DataFrame) -> dict:
    f = frame.copy()
    # Apply primary/fallback event identity before replaying selection.
    events = f.drop_duplicates('matchup_id')
    college = events[events.league.eq('NCAAF')]
    kept = merge_missing_ncaaf_games(
        college[~college.odds_feed_source.eq(ESPN_FALLBACK_SOURCE)].to_dict('records'),
        college[college.odds_feed_source.eq(ESPN_FALLBACK_SOURCE)].to_dict('records'))
    ids = {r['matchup_id'] for r in kept} | set(events.loc[~events.league.eq('NCAAF'), 'matchup_id'])
    f = f[f.matchup_id.isin(ids)].copy()
    # Audit preserves the exact signed offered line in the pick text.
    number = pd.to_numeric(f.best_pick.str.extract(r'([+-]?\d+(?:\.\d+)?)\s*$', expand=False), errors='coerce')
    f['total_line'] = number.where(f.market_type.str.startswith('total_'))
    f['spread_line'] = number.where(f.market_type.str.startswith('spread_'))
    labels = f.candidate_outcome.copy()
    ranking = guard_model_direction(f.drop(columns=['candidate_outcome']))
    ranking['candidate_outcome'] = labels
    old = f[f.best_available_selected.astype('string').str.lower().eq('true')]
    ordered = ranking.sort_values(
        ['final_family_score', 'selection_probability_used', 'tier_score', 'expected_value', 'edge'],
        ascending=[False, False, True, False, False], kind='stable', na_position='last')
    new = ordered.drop_duplicates('matchup_id')
    def record(rows):
        w = int(rows.candidate_outcome.eq('WIN').sum())
        l = int(rows.candidate_outcome.eq('LOSS').sum())
        return dict(wins=w, losses=l, ungraded=len(rows)-w-l, accuracy=w/(w+l) if w+l else None)
    changed = new.merge(old[['matchup_id','best_pick','candidate_outcome']], on='matchup_id', suffixes=('', '_old'))
    changed = changed[changed.best_pick.ne(changed.best_pick_old)]
    return dict(evaluation='fixed-candidate retrospective replay; not independent holdout',
                before=record(old), after=record(new),
                by_league={k:record(g) for k,g in new.groupby('league')},
                adjusted_candidates=int(ranking.model_direction_guard_applied.sum()),
                changes=changed[['home_team','away_team','best_pick_old','candidate_outcome_old',
                                 'best_pick','candidate_outcome','ml_probability']].to_dict('records'))


if __name__ == '__main__':
    print(json.dumps(replay(pd.read_csv(sys.argv[1])), indent=2))
