import pandas as pd
import pytest
from scripts.analyze_candidate_weights import run_analysis


def fixture_frame():
    rows=[]
    for day in range(1,7):
        for market,p in [('total_over',.6),('total_under',.4)]:
            rows.append(dict(game_date=f'2026-08-{day:02d}',
                game_time_est=f'2026-08-{day:02d} 7:00 PM ET',
                export_run_id=f'202608{day:02d}T120000Z',
                matchup_id=f'2026-08-{day:02d}|A|B',league='MLB',home_team='A',away_team='B',
                odds_feed_source='the_odds_api',market_type=market,
                best_pick='Over 8.5' if p>.5 else 'Under 8.5',
                candidate_outcome='WIN' if p>.5 else 'LOSS',
                market_probability=p,ml_probability=p,theover_probability=float('nan'),
                selection_probability_used=p,best_available_selected=p>.5,
                best_available_rank=1 if p>.5 else 2))
    return pd.DataFrame(rows)


def test_regression_never_fits_on_later_labels_and_ignores_repeated_downloads(tmp_path):
    frame=fixture_frame();p=tmp_path/'audit.csv';frame.to_csv(p,index=False)
    a=run_analysis([str(p),str(p)],'2026-08-03')
    assert a['train']['events']==3 and a['test']['events']==3
    frame.loc[frame.game_date.gt('2026-08-03'),'candidate_outcome'] = frame.loc[
        frame.game_date.gt('2026-08-03'),'candidate_outcome'].map({'WIN':'LOSS','LOSS':'WIN'})
    frame.to_csv(p,index=False)
    b=run_analysis([str(p)],'2026-08-03')
    assert a['models']['nonnegative_blend']['weights']==b['models']['nonnegative_blend']['weights']
    assert a['models']['ridge_signals']['coefficients']==b['models']['ridge_signals']['coefficients']
    assert a['models']['existing_selector']['wins']==3
    assert b['models']['existing_selector']['wins']==0


def test_regression_excludes_started_and_incompletely_graded_games(tmp_path):
    frame=fixture_frame()
    frame.loc[frame.game_date.eq('2026-08-05'),'export_run_id']='20260806T010000Z'
    frame.loc[(frame.game_date.eq('2026-08-06')) & frame.market_type.eq('total_over'),'candidate_outcome']='N/A'
    p=tmp_path/'audit.csv';frame.to_csv(p,index=False)
    result=run_analysis([str(p)],'2026-08-03')
    assert result['test']['events']==1
    assert result['data']['not_pregame_events']==1
    assert result['data']['incomplete_events']==1


def test_regression_requires_a_later_evaluation_window(tmp_path):
    p=tmp_path/'audit.csv';fixture_frame().to_csv(p,index=False)
    with pytest.raises(ValueError,match='Both chronological'):
        run_analysis([str(p)],'2026-08-31')
