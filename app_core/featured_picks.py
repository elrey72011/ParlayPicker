"""Presentation ranking of finalized game selections; never authorizes a wager."""
import numpy as np
import pandas as pd


def featured_picks(board, family='overall'):
    if board is None or board.empty:
        return pd.DataFrame()
    data=board.copy()
    market=data.get('market_type',pd.Series('',index=data.index)).astype('string').fillna('').str.lower().str.strip()
    sides=market.isin(['moneyline_home','moneyline_away','h2h_home','h2h_away','spread_home','spread_away','spreads_home','spreads_away'])
    totals=market.isin(['total_over','total_under','totals_over','totals_under'])
    if family not in ('overall','sides','totals'):
        raise ValueError('Unknown featured market family')
    keep=(sides|totals) if family=='overall' else sides if family=='sides' else totals
    probability=pd.to_numeric(data.get('production_win_probability',pd.Series(np.nan,index=data.index)),errors='coerce')
    odds=pd.to_numeric(data.get('odds_american',pd.Series(np.nan,index=data.index)),errors='coerce')
    keep &= probability.between(0,1) & np.isfinite(odds) & odds.abs().ge(100)
    picks=data.get('best_pick',pd.Series('',index=data.index)).astype('string').fillna('').str.strip()
    keep &= picks.ne('') & ~picks.str.contains('unresolved|rejected',case=False,regex=True)
    for c in ('Started','started','is_started','game_already_started_flag'):
        if c in data:
            keep &= ~data[c].astype('string').fillna('').str.lower().isin(['true','1','yes'])
    for c in ('Bet_Decision','Play_Tier'):
        if c in data:
            keep &= ~data[c].astype('string').fillna('').str.upper().eq('STARTED')
    if 'status_blocker_stage' in data:
        keep &= ~data.status_blocker_stage.astype('string').fillna('').isin(['game_already_started','kalshi_wrong_game_title','line_provenance_unresolved','extreme_price_guard','empirical_proven_losing_bucket'])
    data=data.loc[keep].copy()
    if data.empty:
        return data
    data['_featured_probability']=probability.loc[data.index]
    for c in ('production_edge','production_expected_value'):
        values=pd.to_numeric(data.get(c,pd.Series(np.nan,index=data.index)),errors='coerce')
        data['_featured_'+c]=values.where(np.isfinite(values))
    data['_featured_approved']=data.get('Bettable',pd.Series(False,index=data.index)).astype('string').fillna('').str.lower().isin(['true','1','yes'])
    data['_featured_key']=data.get('canonical_pick_key',data.get('best_pick')).astype(str)
    return data.sort_values(['_featured_approved','_featured_probability','_featured_production_edge','_featured_production_expected_value','_featured_key'],ascending=[False,False,False,False,True],kind='stable',na_position='last')
