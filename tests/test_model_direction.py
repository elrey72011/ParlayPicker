import pandas as pd
import pytest

from core.model_direction import guard_model_direction


def pair():
    return pd.DataFrame([
        dict(matchup_id='game', league='MLB', market_type=market, total_line=8.5,
             ml_probability=p, ml_target=market, final_family_score=score,
             ml_probability_source='score-distribution-v1:mlb',
             ml_feature_quality='resolved_team_scoring_stats',
             odds_american=-110, expected_value=.02, Play_Stake=0)
        for market, p, score in [('total_over', .64, .49), ('total_under', .36, .51)]
    ])


def test_guard_changes_rank_without_changing_probability_price_or_stake():
    source = pair()
    result = guard_model_direction(source)
    assert result.sort_values('final_family_score', ascending=False).iloc[0].market_type == 'total_over'
    assert result.model_direction_guard_applied.tolist() == [False, True]
    for col in ['ml_probability', 'odds_american', 'expected_value', 'Play_Stake']:
        pd.testing.assert_series_equal(source[col], result[col])
    pd.testing.assert_frame_equal(guard_model_direction(result), result.assign(
        model_direction_pre_guard_score=result.final_family_score,
        model_direction_guard_applied=False, model_direction_guard_penalty=0., model_direction_guard_reason=''))


@pytest.mark.parametrize('col,value', [
    ('total_line', 9.5), ('ml_probability', .30), ('ml_probability', 1.1),
    ('ml_probability', float('nan')), ('ml_target', 'home_win'),
    ('ml_feature_quality', 'fallback'), ('ml_probability_source', 'home-win-xgboost'),
    ('league', 'NCAAF'), ('final_family_score', float('nan')),
])
def test_guard_leaves_unreliable_or_mismatched_pairs_unchanged(col, value):
    source = pair(); source.loc[1, col] = value
    result = guard_model_direction(source)
    assert not result.model_direction_guard_applied.any()
    pd.testing.assert_series_equal(result.final_family_score, source.final_family_score)


def test_guard_does_not_use_outcomes_or_ambiguous_pairs():
    source = pair()
    a = guard_model_direction(source.assign(candidate_outcome=['WIN', 'LOSS']))
    b = guard_model_direction(source.assign(candidate_outcome=['LOSS', 'WIN']))
    pd.testing.assert_series_equal(a.final_family_score, b.final_family_score)
    duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    assert not guard_model_direction(duplicated).model_direction_guard_applied.any()
    source.ml_probability = .5
    assert not guard_model_direction(source).model_direction_guard_applied.any()


def test_guard_requires_opposite_signed_spread_lines():
    source = pair().assign(market_type=['spread_home', 'spread_away'],
                           ml_target='spread_cover', spread_line=[-1.5, 1.5])
    assert guard_model_direction(source).model_direction_guard_applied.tolist() == [False, True]
    source.spread_line = 1.5
    assert not guard_model_direction(source).model_direction_guard_applied.any()


def test_weak_independent_direction_does_not_override_other_evidence():
    source = pair()
    source.ml_probability = [.59, .41]
    assert not guard_model_direction(source).model_direction_guard_applied.any()
    source.ml_probability = [.60, .40]
    assert guard_model_direction(source).model_direction_guard_applied.tolist() == [False, True]
