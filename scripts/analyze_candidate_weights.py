"""Research-only chronological candidate weight analysis; never writes live weights.

Example: python scripts/analyze_candidate_weights.py --audits "downloads/graded_candidate_audit*.csv" --train-through 2026-09-02 --output report.json
Earlier inspected slates are development evidence, not a pristine future holdout.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize
from app_core.espn_ncaaf_odds import merge_missing_ncaaf_games, ESPN_FALLBACK_SOURCE


def run_analysis(audit_paths, train_through):
    frames=[]
    for path in sorted(audit_paths):
        f=pd.read_csv(path)
        if 'ml_probability' in f and f.ml_probability.notna().any():frames.append(f)
    if not frames:
        raise ValueError('No candidate audits with independent model inputs were supplied')
    f=pd.concat(frames,ignore_index=True)
    # Repeated downloads must not count as independent games. Keep the latest run.
    f=f.sort_values('export_run_id',kind='stable')
    latest=f.groupby('matchup_id').export_run_id.transform('max')
    f=f[f.export_run_id.eq(latest)].drop_duplicates(['matchup_id','market_type','best_pick'],keep='last')
    f['day']=pd.to_datetime(f.game_date,utc=True).dt.strftime('%Y-%m-%d')
    # Exact time comparison: no final scores or grades are used to determine eligibility.
    f['kickoff']=pd.to_datetime(f.game_time_est.str.replace(' ET','',regex=False),format='%Y-%m-%d %I:%M %p',errors='coerce').dt.tz_localize('America/New_York').dt.tz_convert('UTC')
    f['run']=pd.to_datetime(f.export_run_id,format='%Y%m%dT%H%M%SZ',utc=True,errors='coerce')
    stats={'raw_rows':len(f),'raw_events':f.matchup_id.nunique(),'not_pregame_events':f.loc[~f.run.lt(f.kickoff),'matchup_id'].nunique()}
    f=f[f.run.lt(f.kickoff)].copy()
    # Remove duplicate fallback events using the actual production merge identity.
    keep=[]
    for day,g in f.groupby('day'):
        e=g.drop_duplicates('matchup_id'); c=e[e.league.eq('NCAAF')]
        merged=merge_missing_ncaaf_games(c[~c.odds_feed_source.eq(ESPN_FALLBACK_SOURCE)].to_dict('records'),c[c.odds_feed_source.eq(ESPN_FALLBACK_SOURCE)].to_dict('records'))
        keep+=list(e.loc[~e.league.eq('NCAAF'),'matchup_id'])+[r['matchup_id'] for r in merged]
    f=f[f.matchup_id.isin(keep)].copy()
    # A game only contributes to evaluation when every offered candidate can be graded.
    complete=f.groupby('matchup_id').candidate_outcome.transform(lambda s:s.isin(['WIN','LOSS']).all())
    stats['incomplete_events']=f.loc[~complete,'matchup_id'].nunique()
    f=f[complete].copy().reset_index(drop=True)
    f['y']=f.candidate_outcome.eq('WIN').astype(int)
    train=f[f.day.le(train_through)].copy(); test=f[f.day.gt(train_through)].copy()
    if train.empty or test.empty:
        raise ValueError('Both chronological training and evaluation windows need complete pregame games')
    assert train.day.max()<test.day.min()
    assert not set(train.matchup_id)&set(test.matchup_id)
    signals=['market_probability','ml_probability','theover_probability']
    
    def features(d,rank=False):
        out=pd.DataFrame(index=d.index)
        for col in signals+(['selection_probability_used'] if rank else []):
            p=pd.to_numeric(d[col],errors='coerce').where(lambda x:x.between(0,1))
            out[col+'_logit']=np.log(p.fillna(.5).clip(.01,.99)/(1-p.fillna(.5).clip(.01,.99)))
            out[col+'_missing']=p.isna().astype(float)
        out['total']=d.market_type.str.startswith('total_').astype(float)
        out['mlb']=d.league.eq('MLB').astype(float)
        return out
    
    def balanced(d):return 1/d.groupby('matchup_id').matchup_id.transform('size')
    def blend(d,w):
        x=d[signals].apply(pd.to_numeric,errors='coerce').to_numpy()
        valid=np.isfinite(x)&(x>=0)&(x<=1); denom=(valid*w).sum(axis=1)
        return np.divide((np.where(valid,x,0)*w).sum(axis=1),denom,out=np.full(len(d),.5),where=denom>0)
    def loss(d,p):
        p=np.clip(p,1e-6,1-1e-6); y=d.y.to_numpy();w=balanced(d).to_numpy()
        return float(np.average(-(y*np.log(p)+(1-y)*np.log(1-p)),weights=w))
    def metrics(d,p,selected=False):
        work=d.copy();work['p']=p
        if selected: chosen=work[work.best_available_selected.astype('string').str.lower().eq('true')]
        else: chosen=work.sort_values(['p','best_available_rank'],ascending=[False,True],kind='stable').drop_duplicates('matchup_id')
        w=int(chosen.y.sum());n=len(chosen)
        record={'wins':w,'losses':n-w,'accuracy':w/n if n else None}
        record['log_loss']=loss(d,np.asarray(p))
        record['brier']=float(np.average((np.asarray(p)-d.y.to_numpy())**2,weights=balanced(d)))
        record['by_league']={k:{'wins':int(g.y.sum()),'n':len(g)} for k,g in chosen.groupby('league')}
        record['confidence_bands']={str(t):{'n':int(chosen.p.ge(t).sum()),'wins':int(chosen.loc[chosen.p.ge(t),'y'].sum())} for t in [.6,.65,.7,.75]}
        return record
    
    result={'data':stats,'train':{'days':sorted(train.day.unique()),'events':train.matchup_id.nunique(),'candidates':len(train)},'test':{'days':sorted(test.day.unique()),'events':test.matchup_id.nunique(),'candidates':len(test)},'models':{}}
    result['models']['existing_selector']=metrics(test,test.selection_probability_used.to_numpy(),True)
    result['models']['market_only']=metrics(test,test.market_probability.fillna(.5).to_numpy())
    for name,rank in [('ridge_signals',False),('ridge_signals_and_rank',True)]:
        x=features(train,rank); model=LogisticRegression(C=1.,max_iter=2000)
        model.fit(x,train.y,sample_weight=balanced(train))
        p=model.predict_proba(features(test,rank))[:,1]
        result['models'][name]=metrics(test,p)
        result['models'][name]['coefficients']=dict(zip(x.columns,model.coef_[0].tolist()))
        result['models'][name]['intercept']=float(model.intercept_[0])
    opt=minimize(lambda w:loss(train,blend(train,w)),np.ones(3)/3,method='SLSQP',bounds=[(0,1)]*3,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],options={'maxiter':500,'ftol':1e-10})
    assert opt.success,opt.message
    result['models']['nonnegative_blend']=metrics(test,blend(test,opt.x))
    result['models']['nonnegative_blend']['weights']=dict(zip(signals,opt.x.tolist()))
    result['availability']={part:{c:float(g[c].notna().mean()) for c in signals} for part,g in [('train',train),('test',test)]}
    return result

if __name__ == '__main__':
    import argparse
    import glob
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audits',required=True)
    parser.add_argument('--train-through',required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    cutoff=pd.Timestamp(args.train_through).strftime('%Y-%m-%d')
    result=run_analysis(glob.glob(args.audits),cutoff)
    rendered=json.dumps(result,indent=2,default=str)
    if args.output: args.output.write_text(rendered,encoding='utf-8')
    print(rendered)
