import json
import pandas as pd
from app_core.novig_candidates import preserve_half_run_quote
from core.streamlit_pipeline import _expand_live_odds_to_bet_rows, _filter_preselection_line_integrity


def feed():
    quotes=[{'book':'novig','market_type':'spread_away','point':.5,'price':122,'recorded_at':'2026-09-11T21:05:54Z'},
            {'book':'novig','market_type':'spread_home','point':-.5,'price':-133,'recorded_at':'2026-09-11T21:05:54Z'}]
    return dict(league='MLB',away_team='Baltimore',home_team='Toronto',game_date='2026-09-11',matchup_id='g',
        commence_time_raw='2026-09-11T23:07:00Z',provider_quotes=json.dumps(quotes),
        novig_away_point=.5,novig_home_point=-.5,novig_away_price=122,novig_home_price=-133,
        novig_h2h_away_price=125,novig_h2h_home_price=-127,
        fanduel_away_point=1.5,fanduel_home_point=-1.5,fanduel_away_price=-200,fanduel_home_price=164,
        fanduel_h2h_away_price=114,fanduel_h2h_home_price=-134)


def test_half_run_survives_candidate_generation_and_integrity():
    rows,_=_expand_live_odds_to_bet_rows(pd.DataFrame([feed()]))
    spreads=rows[rows.market_type.str.startswith('spread')]
    away=spreads[spreads.market_type=='spread_away'].iloc[0]
    assert away.spread_line==.5 and away.odds_american==122
    assert away.opposing_odds_american==-133
    assert away.line_source=='novig_exact_half_run_quote'
    assert len(_filter_preselection_line_integrity(spreads))==2


def test_incomplete_ambiguous_or_other_magnitude_not_repaired():
    row=dict(feed(),market_type='spread_away',spread_line=1.5,odds_american=-200)
    for quotes in ([],json.loads(row['provider_quotes'])[:1],json.loads(row['provider_quotes'])*2):
        candidate=dict(row,provider_quotes=json.dumps(quotes))
        assert preserve_half_run_quote(candidate)==candidate
    quotes=json.loads(row['provider_quotes']);quotes[0]['point']=2.5
    candidate=dict(row,provider_quotes=json.dumps(quotes))
    assert preserve_half_run_quote(candidate)==candidate
