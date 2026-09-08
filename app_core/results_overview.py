"""Read-only summaries of a loaded game recap; never infer actual execution."""
import numpy as np
import pandas as pd


def _flag(series):
    return series.astype('string').fillna('').str.strip().str.lower().isin(['true', '1', 'yes', 'y'])


def approved_wager_mask(frame):
    if frame is None or frame.empty:
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)
    data = frame.copy()
    stakes = [pd.to_numeric(data[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0) for c in
              ('Play_Stake', 'production_bet_amount', 'Kelly_Bet_Size', 'recommended_bet', 'Suggested_Stake') if c in data]
    funded = pd.concat(stakes, axis=1).max(axis=1).gt(0) if stakes else pd.Series(False, index=data.index)
    # A positive suggested amount alone is not evidence of final approval.
    approved = pd.Series(False, index=data.index)
    for c in ('Bettable', 'production_eligible', 'wager_approved', 'Wager_Approved'):
        if c in data:
            approved |= _flag(data[c])
    if 'Bet_Decision' in data:
        approved |= data.Bet_Decision.astype('string').str.upper().eq('BET').fillna(False)
    for c in ('Bettable', 'production_eligible', 'wager_approved', 'Wager_Approved'):
        if c in data:
            approved &= _flag(data[c])
    if 'Bet_Decision' in data:
        approved &= data.Bet_Decision.astype('string').str.upper().eq('BET').fillna(False)
    if 'Stake_Status' in data:
        approved &= data.Stake_Status.astype('string').fillna('').str.strip().str.lower().eq('funded')
    approved &= funded
    return approved


def summarize_results(frame):
    if frame is None or frame.empty:
        return pd.DataFrame()
    data = frame.copy()
    approved = approved_wager_mask(data)
    data['_scope'] = np.where(approved, 'App-approved (paper)', 'Research / unapproved')
    data['_sport'] = data.get('league', data.get('League', pd.Series('Unknown', index=data.index))).fillna('Unknown').astype(str)
    outcome = data.get('Outcome', data.get('Pick_Outcome', pd.Series('', index=data.index)))
    data['_outcome'] = outcome.astype('string').fillna('').str.strip().str.upper().replace({'W':'WIN', 'L':'LOSS', 'P':'PUSH'})
    data['_odds'] = pd.to_numeric(data.get('odds_american', pd.Series(np.nan, index=data.index)), errors='coerce')
    rows = []
    for (sport, scope), group in data.groupby(['_sport', '_scope'], sort=True):
        outcomes = group['_outcome']
        wins, losses, pushes = [int(outcomes.eq(o).sum()) for o in ('WIN', 'LOSS', 'PUSH')]
        voids = int(outcomes.isin(['VOID','DNP','CANCELLED','CANCELED']).sum())
        settled = wins + losses + pushes
        prices = group['_odds']
        valid = np.isfinite(prices) & prices.abs().ge(100)
        priced = group.loc[valid & outcomes.isin(['WIN','LOSS','PUSH'])]
        profit = sum((float(r['_odds']) / 100 if r['_odds'] > 0 else 100 / abs(float(r['_odds']))) if r['_outcome']=='WIN' else -1 if r['_outcome']=='LOSS' else 0 for _, r in priced.iterrows())
        rows.append({'Sport': sport, 'Record': scope, 'Selections': len(group), 'Wins': wins, 'Losses': losses,
                     'Pushes': pushes, 'Void / DNP': voids, 'Pending / unresolved': len(group)-settled-voids,
                     'Decisions (W+L)': wins+losses, 'Win rate': wins/(wins+losses) if wins+losses else None,
                     'Priced settled rows': len(priced), 'Unpriced settled rows': settled-len(priced),
                     'Paper profit (units)': profit if len(priced) else None,
                     'Paper ROI': profit/len(priced) if len(priced) else None})
    return pd.DataFrame(rows)
