import pandas as pd
import pytest
from core.walk_forward import compare_by_league


def test_later_slates_paired_market_comparison_separates_leagues():
    rows=[]
    for league in ('MLB','NCAAF'):
        for day in range(1,5):
            for i in range(2):
                rows.append(dict(league=league,date=f'2026-09-{day:02}',p=.8 if league=='MLB' else .2,market=.5,y='WIN'))
    frame=pd.DataFrame(rows)
    reports=compare_by_league(frame,'date','p','y','market',min_train_rows=4)
    assert set(reports)=={'MLB','NCAAF'}
    for report in reports.values():
        assert report['train_end']<report['test_start']
        assert report['model']['n']==report['market']['n']==2
        assert report['out_of_sample'] is False
    assert reports['MLB']['model_minus_market']['brier']<0
    assert reports['NCAAF']['model_minus_market']['brier']>0


def test_missing_market_and_unsettled_are_excluded_and_one_day_cannot_validate():
    frame=pd.DataFrame([dict(league='MLB',date='2026-09-11',p=.6,market=.5,y='WIN'),
                        dict(league='MLB',date='2026-09-11',p=.6,market=None,y='LOSS'),
                        dict(league='MLB',date='2026-09-11',p=.6,market=.5,y='PENDING')])
    result=compare_by_league(frame,'date','p','y','market',min_train_rows=1)['MLB']
    assert result['status']=='insufficient_history'
    assert result['paired_rows']==1 and result['excluded_rows']==2
