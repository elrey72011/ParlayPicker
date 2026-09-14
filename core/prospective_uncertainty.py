"""Prospective sport-local uncertainty; frozen settings, prior completed slates only."""
import math
from core.wager_decisions import aware,finite
from core.exposure_ledger import digest


def forecast(candidate, records, settings):
    p=finite(candidate.get('calibrated_probability'));at=aware(candidate.get('prediction_generated_at'))
    names=('historical_prior_decay','historical_effective_sample_cap','current_season_weight','parent_weight','uncertainty_quantile')
    cfg={k:finite(settings.get(k)) for k in names}
    if p is None or not 0<p<1 or at is None or any(v is None for v in cfg.values()):return {'uncertainty_status':'MISSING_ORIGINAL_INPUTS'}
    if not 0<cfg['uncertainty_quantile']<.5 or not 0<=cfg['historical_prior_decay']<=1 or any(cfg[k]<0 for k in ('historical_effective_sample_cap','current_season_weight','parent_weight')):raise ValueError('Invalid frozen uncertainty settings')
    season=finite(candidate.get('season'))
    if season is None:return {'uncertainty_status':'MISSING_SEASON'}
    usable=[];seen=set()
    for r in records:
        end=aware(r.get('outcome_recorded_at'));past_season=finite(r.get('season'))
        key=r.get('game_id')
        if not key or key in seen or r.get('sport')!=candidate.get('sport') or r.get('slate_id')==candidate.get('slate_id'):continue
        if end is None or end>=at or past_season is None or past_season>season:continue
        if any(r.get(k)!=candidate.get(k) for k in ('model_version','calibration_version')):continue
        if str(r.get('market_type','')).split('_')[0]!=str(candidate.get('market_type','')).split('_')[0] or r.get('candidate_outcome') not in {'WIN','LOSS'}:continue
        reference=finite(r.get('calibrated_probability'))
        if reference is None or not 0<reference<1:continue
        seen.add(key)
        child=r['market_type']==candidate['market_type'] and int(reference*10)==int(p*10)
        weight=(cfg['current_season_weight'] if past_season==season else cfg['historical_prior_decay']**(season-past_season))*(1 if child else cfg['parent_weight'])
        usable.append((r,weight,reference,past_season==season))
    mass=sum(w for r,w,p,c in usable if not c);scale=min(1,cfg['historical_effective_sample_cap']/mass) if mass else 1
    usable=[(r,w if c else w*scale,q,c) for r,w,q,c in usable if w>0]
    if not usable:return {'uncertainty_status':'NO_PRIOR_ADMISSIBLE_SLATES'}
    from scipy.stats import beta
    def distribution(pool):
        total=sum(w for r,w,q,c in pool)
        if not total:return None
        reference=sum(w*q for r,w,q,c in pool)/total
        offset=math.log(p/(1-p))-math.log(reference/(1-reference))
        a=.5+sum(w for r,w,q,c in pool if r['candidate_outcome']=='WIN');b=.5+sum(w for r,w,q,c in pool if r['candidate_outcome']=='LOSS')
        def transform(q):
            q=min(1-1e-12,max(1e-12,float(q)))
            return 1/(1+math.exp(-(math.log(q/(1-q))+offset)))
        return [transform(beta.ppf(q,a,b)) for q in (cfg['uncertainty_quantile'],.5,1-cfg['uncertainty_quantile'])]
    mixed=distribution(usable);old=distribution([x for x in usable if not x[3]]);current=distribution([x for x in usable if x[3]])
    conflict=bool(old and current and (current[2]<old[0] or old[2]<current[0]))
    # Conflict widens, never narrows, the conservative interval.
    low=min(mixed[0],current[0]) if conflict else mixed[0];high=max(mixed[2],current[2]) if conflict else mixed[2]
    total=sum(w for r,w,q,c in usable);historical=sum(w for r,w,q,c in usable if not c)
    return {'uncertainty_status':'PROSPECTIVE_ESTIMATE_NOT_DEPLOYMENT_AUTHORITY','hierarchical_probability':mixed[1],
        'conservative_probability':min(p,low),'probability_interval_lower':min(p,low),'probability_interval_upper':max(p,high),
        'calibration_uncertainty':high-low,'historical_prior_probability':old[1] if old else None,'current_season_probability':current[1] if current else None,
        'historical_prior_weight':historical/total,'current_season_weight':1-historical/total,'current_regime_conflict':conflict,
        'effective_evidence_size':min(total,total**2/sum(w*w for r,w,q,c in usable)),
        'evidence_snapshot_id':digest({'settings':settings,'records':[(r['candidate_id'],w,r['candidate_outcome']) for r,w,q,c in usable]}),
        'evidence_frozen_at':at.isoformat(),'hierarchy_level_used':'sport/market/direction/probability-band'}


def prepare_live(frame, *, database=None, plan_dir='data/validation_plans', now=None):
    """Called only after live analysis, never when replaying a saved publication."""
    from pathlib import Path
    import json
    import pandas as pd
    from datetime import datetime,timezone
    from app_core.prediction_evidence import materialize,connect
    from core.activation_validation import reasons
    from app_core.candidate_evidence_schema import project
    from contextlib import closing
    clock=now or datetime.now(timezone.utc)
    past,_=materialize(database)
    records=[]
    if not past.empty:
        for r in past.astype(object).where(past.notna(),None).to_dict('records'):
            # Seeds still require original model/calibration/quote/outcome facts;
            # a conservative forecast is the output being bootstrapped here.
            if not set(reasons(r))-{'missing_conservative_probability'}:records.append(r)
    out=project(frame)
    for idx,row in out.iterrows():
        path=Path(plan_dir,str(row['sport'])+'.json')
        if not path.exists():continue
        plan=json.loads(path.read_text())
        with closing(connect(database)) as db:
            receipt=db.execute('SELECT payload FROM validation_plans WHERE plan_id=?',(plan.get('plan_hash'),)).fetchone()
        if receipt is None or json.loads(receipt[0])!=plan:continue
        r=row.to_dict()
        # This timestamps a newly computed prospective forecast, not a saved row.
        r['prediction_generated_at']=clock.isoformat()
        if any(r.get(k)!=v for k,v in plan.get('versions',{}).items()):continue
        result=forecast(r,records,plan.get('evidence_parameters',{}))
        for key,value in result.items():out.at[idx,key]=value
        out.at[idx,'prediction_generated_at']=clock.isoformat()
    return out
