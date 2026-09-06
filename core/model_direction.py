"""Keep resolved independent model direction intact through ranking overlays."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app_core.precision_card import PRECISION_CARD_MIN_WIN_PROBABILITY


def guard_model_direction(frame: pd.DataFrame) -> pd.DataFrame:
    """Cap reversed ranking scores without modifying probabilities or stakes.

    Only unique, same-line MLB pairs above the existing precision confidence floor from the resolved score-distribution model
    qualify. Missing, mismatched, ambiguous and coin-flip models are unchanged.
    This selection policy consumes no outcomes; its accuracy needs forward testing.
    """
    out = frame.copy()
    out['model_direction_pre_guard_score'] = pd.to_numeric(
        out.get('final_family_score', pd.Series(float('nan'), index=out.index)), errors='coerce'
    )
    out['model_direction_guard_applied'] = False
    out['model_direction_guard_penalty'] = 0.0
    out['model_direction_guard_reason'] = ''
    required = {'matchup_id', 'market_type', 'league', 'ml_probability',
                'ml_probability_source', 'ml_target', 'ml_feature_quality',
                'final_family_score'}
    if out.empty or not required.issubset(out.columns):
        return out

    def text(col):
        return out[col].astype('string').fillna('').str.strip().str.lower()

    p = pd.to_numeric(out.ml_probability, errors='coerce')
    score = pd.to_numeric(out.final_family_score, errors='coerce')
    market, target = text('market_type'), text('ml_target')
    supported = (text('league').eq('mlb')
                 & text('ml_probability_source').eq('score-distribution-v1:mlb')
                 & text('ml_feature_quality').eq('resolved_team_scoring_stats')
                 & p.between(0, 1) & np.isfinite(score))
    for matchup, indices in out.groupby('matchup_id', dropna=False).groups.items():
        if pd.isna(matchup) or not str(matchup).strip():
            continue
        for types, line_col in [({'total_over', 'total_under'}, 'total_line'),
                                ({'spread_home', 'spread_away'}, 'spread_line')]:
            pair = market.loc[indices][market.loc[indices].isin(types)].index
            if len(pair) != 2 or set(market.loc[pair]) != types or not supported.loc[pair].all():
                continue
            targets = market.loc[pair] if line_col == 'total_line' else pd.Series('spread_cover', index=pair)
            if not target.loc[pair].eq(targets).all() or line_col not in out:
                continue
            lines = pd.to_numeric(out.loc[pair, line_col], errors='coerce')
            if not np.isfinite(lines).all():
                continue
            a, b = map(float, lines)
            if not np.isclose(a, b if line_col == 'total_line' else -b, rtol=0, atol=1e-6):
                continue
            if not np.isclose(p.loc[pair].sum(), 1.0, rtol=0, atol=1e-6):
                continue
            preferred, opposed = pair[p.loc[pair].gt(0.5)], pair[p.loc[pair].lt(0.5)]
            if len(preferred) != 1 or len(opposed) != 1:
                continue
            winner, loser = preferred[0], opposed[0]
            if p.loc[winner] < PRECISION_CARD_MIN_WIN_PROBABILITY:
                continue
            if score.loc[loser] < score.loc[winner]:
                continue
            capped = float(np.nextafter(float(score.loc[winner]), -np.inf))
            penalty = float(score.loc[loser]) - capped
            out.loc[loser, 'final_family_score'] = capped
            if 'final_family_score_no_mlb_spread_penalty' in out:
                out.loc[loser, 'final_family_score_no_mlb_spread_penalty'] -= penalty
            out.loc[loser, 'model_direction_guard_applied'] = True
            out.loc[loser, 'model_direction_guard_penalty'] = penalty
            out.loc[loser, 'model_direction_guard_reason'] = (
                'Independent model favors the same-line opposite; ranking overlay cannot reverse it.'
            )
    return out
