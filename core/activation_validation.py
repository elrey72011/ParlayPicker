"""Strict activation evidence and frozen-criteria, slate-separated validation.

This module reports missing evidence. It never invents timestamps or promotes a
sport from a pooled calibration flag. Validation choices must predate holdout.
"""
import json
import math
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
from core.exposure_ledger import digest
from core.wager_decisions import aware,finite,decimal_price
from core.market_policy import production_market
from core.sport_policy import SPORTS

VERSIONS=('model_version','calibration_version','selection_policy_version','sport_policy_version','evidence_version')
FIELDS='candidate_id game_id sport season slate_id event_date home_team_id away_team_id calibration_version calibration_available_at evidence_version evidence_frozen_at selection_policy_version sport_policy_version raw_model_probability sport_calibrated_probability hierarchical_probability conservative_probability fair_market_probability conservative_edge conservative_ev calibration_uncertainty effective_evidence_size historical_prior_weight current_season_weight ml_context_probability ml_spread_alignment gemini_review_status gemini_reviewed_at gemini_input_hash identity_verified data_quality_status push_semantics_verified candidate_maturity candidate_rank_before_gate candidate_rank_after_gate selected_as_best_pick best_available_candidate_count'.split()

def enrich(frame):
    from app_core.candidate_evidence_schema import project
    return project(frame)


def reasons(row):
    r=row;bad=[]
    def check(ok,why):
        if not ok: bad.append(why)
    def present(k): return isinstance(r.get(k),str) and bool(r[k].strip())
    pred,start=aware(r.get('prediction_generated_at')),aware(r.get('game_start_utc'))
    check(pred is not None,'missing_prediction_timestamp')
    check(pred is not None and start is not None and pred<start,'prediction_after_start')
    for key in VERSIONS: check(present(key),'missing_'+key)
    train=aware(r.get('model_trained_through'));available=aware(r.get('model_available_at'))
    check(train is not None,'missing_training_cutoff')
    check(train is not None and pred is not None and train<pred and r.get('training_cutoff_basis')!='frozen_artifact_information_upper_bound','unverified_training_cutoff')
    check(available is not None and pred is not None and available<=pred,'model_not_available')
    for field,label in [('calibration_available_at','calibration_not_available'),('evidence_frozen_at','evidence_not_available')]:
        at=aware(r.get(field));check(at is not None and pred is not None and at<=pred,label)
    quote=aware(r.get('odds_recorded_at'))
    check(quote is not None,'missing_quote_timestamp')
    check(quote is not None and pred is not None and quote<=pred,'quote_after_prediction')
    check(r.get('quote_binding_verified') is True or str(r.get('quote_binding_verified')).lower()=='true','quote_not_verified')
    check(r.get('identity_verified') is True or str(r.get('identity_verified')).lower()=='true','identity_unverified')
    check(present('slate_id'),'missing_slate_id')
    check(r.get('sport') in SPORTS,'unsupported_sport')
    check(production_market(r.get('market_type')),'unsupported_market')
    p=finite(r.get('conservative_probability'));check(p is not None and 0<p<1,'missing_conservative_probability')
    check(r.get('candidate_outcome') in {'WIN','LOSS','PUSH','VOID'},'missing_outcome' if r.get('candidate_outcome') not in {'NEEDS_REVIEW'} else 'ambiguous_outcome')
    outcome_at=aware(r.get('outcome_recorded_at'))
    check(outcome_at is not None and start is not None and start<=outcome_at<=datetime.now(timezone.utc),'missing_or_future_outcome_timestamp')
    return sorted(set(bad))

def read_dataset(database):
    from app_core.prediction_evidence import materialize
    frame,_=materialize(database)
    if frame.empty:return [],[]
    frame=frame.astype(object).where(pd.notna(frame),None)
    accepted=[];excluded=[]
    # Freeze one whole latest pregame snapshot per game/version cohort. Repeated
    # runs must not multiply evidence or mix opposing candidates across snapshots.
    cohorts={}
    for _,g in frame.groupby(['snapshot_id','matchup_id'],dropna=False):
        first=g.iloc[0].to_dict()
        key=tuple(str(first.get(k)) for k in ('sport','matchup_id',*VERSIONS))
        pred=aware(first.get('prediction_generated_at'))
        start=aware(first.get("game_start_utc"))
        if pred is not None and start is not None and pred<start:
            prior=cohorts.get(key)
            candidate=(pred,str(first['snapshot_id']))
            if prior is None or candidate>prior:cohorts[key]=candidate
    for identity,group in frame.groupby(['snapshot_id','matchup_id'], dropna=False):
        from app_core.candidate_evidence_schema import pool_status
        complete=pool_status(group)
        for r in group.to_dict('records'):
            bad=reasons(r)
            key=tuple(str(r.get(k)) for k in ('sport','matchup_id',*VERSIONS))
            if key in cohorts and str(r['snapshot_id'])!=cohorts[key][1]:bad.append('superseded_pregame_snapshot')
            if not complete:bad.append('candidate_pool_incomplete')
            (excluded if bad else accepted).append(dict(r,exclusion_reasons=bad))
    return accepted,excluded

def metrics(rows):
    settled=[r for r in rows if r['candidate_outcome'] in {'WIN','LOSS'}]
    result={'n':len(settled),'effective_n':len({(r['sport'],r['game_id']) for r in settled})}
    if not settled:return result
    for field,label in [('raw_model_probability','raw'),('calibrated_probability','calibrated'),('fair_market_probability','market'),('conservative_probability','conservative')]:
        pairs=[(finite(r.get(field)),int(r['candidate_outcome']=='WIN')) for r in settled]
        if any(p is None or not 0<p<1 for p,y in pairs):continue
        result[label+'_brier']=sum((p-y)**2 for p,y in pairs)/len(pairs)
        result[label+'_log_loss']=-sum(y*math.log(p)+(1-y)*math.log(1-p) for p,y in pairs)/len(pairs)
        result[label+'_mean']=sum(p for p,y in pairs)/len(pairs)
    result['realized_mean']=sum(r['candidate_outcome']=='WIN' for r in settled)/len(settled)
    result['conservative_calibration_gap']=result['realized_mean']-result.get('conservative_mean',0)
    slate_returns=defaultdict(list)
    for r in settled:
        d=decimal_price(r.get('odds_american'))
        if d is None:continue
        slate_returns[r['slate_id']].append(d-1 if r['candidate_outcome']=='WIN' else -1.)
    values=[sum(v)/len(v) for v in slate_returns.values()]
    result['slates']=len(values)
    if values:
        from scipy.stats import t
        mean=sum(values)/len(values);sd=(sum((x-mean)**2 for x in values)/(len(values)-1))**.5 if len(values)>1 else None
        result['flat_roi']=mean
        result['roi_lower_95']=mean-float(t.ppf(.975,len(values)-1))*sd/(len(values)**.5) if sd is not None else None
        wealth=peak=1.;drawdown=0.
        for x in values:
            wealth+=x;peak=max(peak,wealth);drawdown=max(drawdown,peak-wealth)
        result['flat_drawdown_units']=drawdown
    # Interval coverage cannot be inferred from individual binary outcomes.
    # Report observed probability-band calibration; missing intervals stay absent.
    bins=[]
    for lo in range(0,10):
        band=[r for r in settled if lo/10<=float(r['conservative_probability'])<(lo+1)/10]
        if band:bins.append({'lower':lo/10,'n':len(band),'predicted':sum(float(r['conservative_probability']) for r in band)/len(band),'realized':sum(r['candidate_outcome']=='WIN' for r in band)/len(band)})
    result['calibration_bins']=bins
    return result

def validate(database,sport,plan=None):
    accepted,excluded=read_dataset(database)
    rows=[r for r in accepted if r['sport']==sport]
    if plan and plan.get('versions'):
        incompatible=[r for r in rows if any(r.get(k)!=plan['versions'].get(k) for k in VERSIONS) or r['market_type'].split('_')[0]!=plan.get('market_family')]
        excluded.extend(dict(r,exclusion_reasons=['version_mismatch']) for r in incompatible)
        rows=[r for r in rows if r not in incompatible]
    blockers=[];hold=[];cohorts=defaultdict(list)
    for r in rows:cohorts[tuple(r.get(k) for k in VERSIONS)+(r['market_type'].split('_')[0],)].append(r)
    if not rows:blockers.append('no_strict_admissible_evidence')
    if plan:
        from app_core.prediction_evidence import connect
        from contextlib import closing
        with closing(connect(database)) as db:
            receipt=db.execute('SELECT payload FROM validation_plans WHERE plan_id=?',(plan.get('plan_hash'),)).fetchone()
        if receipt is None or json.loads(receipt[0])!=plan:raise ValueError('Development choices have no immutable freeze receipt')
    if not plan:blockers.append('missing_frozen_development_plan')
    else:
        if digest({k:v for k,v in plan.items() if k!='plan_hash'})!=plan.get('plan_hash'):raise ValueError('Development plan hash mismatch')
        freeze,cut=aware(plan.get('frozen_at')),aware(plan.get('development_through'))
        if plan.get('sport')!=sport or freeze is None or cut is None or freeze<cut:raise ValueError('Invalid development plan')
        for pool in cohorts.values():
            dev=[r for r in pool if aware(r['game_start_utc'])<=cut];future=[r for r in pool if aware(r['game_start_utc'])>cut]
            if {r['slate_id'] for r in dev}&{r['slate_id'] for r in future}:raise ValueError('Same-slate leakage')
            if any(aware(r['prediction_generated_at'])<=freeze for r in future):raise ValueError('Holdout predates frozen choices')
            hold.extend(future)
    from app_core.activation_closing import observations
    closes={(x['snapshot_id'],x['candidate_id']):x for x in observations(database)}
    for r in hold:
        close=closes.get((r['snapshot_id'],r['candidate_id']))
        if close:r['closing']=close
    measured=metrics(hold)
    from core.activation_studies import studies, strategy_comparison
    study=studies(hold,plan or {})
    selected=[r for r in hold if str(r.get('selected_as_best_pick',r.get('best_available_selected'))).lower()=='true']
    selected_metrics=metrics(selected)
    measured['selected_strategy']=selected_metrics
    # The full opposing-side pool assesses calibration, not an investable card.
    for key in ('flat_roi','roi_lower_95','flat_drawdown_units'):
        measured[key]=selected_metrics.get(key)
    from scipy.stats import t
    def lower(values):
        if len(values)<2:return None
        mean=sum(values)/len(values);sd=(sum((x-mean)**2 for x in values)/(len(values)-1))**.5
        return mean-float(t.ppf(.975,len(values)-1))*sd/len(values)**.5
    by_slate=defaultdict(list)
    for r in hold:by_slate[r['slate_id']].append(r)
    residuals=[];price=[];line=[]
    for slate in by_slate.values():
        settled=[r for r in slate if r['candidate_outcome'] in {'WIN','LOSS'}]
        if settled:residuals.append(sum(int(r['candidate_outcome']=='WIN')-float(r['conservative_probability']) for r in settled)/len(settled))
        for field,target in [('price_clv',price),('line_clv',line)]:
            vals=[r.get('closing',{}).get(field) for r in slate]
            if vals and all(finite(v) is not None for v in vals):target.append(sum(vals)/len(vals))
    measured['conservative_gap_lower_95']=lower(residuals)
    measured['price_clv_lower_95']=lower(price);measured['line_clv_lower_95']=lower(line)
    measured['closing_n']=sum('closing' in r for r in hold)
    measured['calibration_excess_brier']=(measured['calibrated_brier']-measured['market_brier']) if all(k in measured for k in ('calibrated_brier','market_brier')) else None
    measured['historical_prior_contribution']=sum(finite(r.get('historical_prior_weight')) or 0 for r in hold)
    measured['current_season_contribution']=sum(finite(r.get('current_season_weight')) or 0 for r in hold)
    if hold and any('closing' not in r for r in hold):blockers.append('missing_closing_quote')
    if len(cohorts)>1:blockers.append('version_or_market_cohort_mismatch')
    # A frozen plan evaluates a single sport/market/version cohort. Parameters
    # are part of the frozen cohort; no tuning is performed on this holdout.
    if plan and rows:
        expected=plan.get('versions',{})
        if any(any(r.get(k)!=expected.get(k) for k in VERSIONS) for r in rows):blockers.append('version_mismatch')
        for k in ('historical_prior_weight','current_season_weight','calibration_uncertainty'):
            if any(finite(r.get(k)) is None for r in hold):blockers.append('missing_'+k)
    earned='UNVALIDATED';supported_policy=None;tier_results={}
    comparisons={'effective_n':('min_effective_n',lambda a,b:a>=b),
      'roi_lower_95':('min_roi_lower_95',lambda a,b:a>=b),
      'conservative_gap_lower_95':('min_conservative_gap_lower_95',lambda a,b:a>=b),
      'calibration_excess_brier':('max_calibration_excess_brier',lambda a,b:a<=b),
      'price_clv_lower_95':('min_price_clv_lower_95',lambda a,b:a>=b),
      'flat_drawdown_units':('max_drawdown_units',lambda a,b:a<=b)}
    for state in ('PROVISIONAL_VALIDATED','STANDARD_VALIDATED','PREMIUM_VALIDATED'):
        spec=(plan or {}).get('tiers',{}).get(state)
        if not spec:continue
        failures=list(blockers)
        failures.extend(study['uncertainty']['blockers'])
        failures.extend(study['prior_validation']['blockers'])
        prior=study['prior_validation']
        if not prior['blockers']:
            excess=finite(spec.get('max_hierarchical_excess_brier'))
            if excess is None or excess>0 or prior['hierarchical']['brier']-min(prior['historical_only']['brier'],prior['current_only']['brier'])>excess: failures.append('hierarchical_incremental_value_unproven')
        coverage=finite(study['uncertainty']['lower_coverage'])
        threshold=finite(spec.get('min_lower_coverage'))
        if coverage is None or threshold is None or not 0<threshold<=1 or coverage<threshold: failures.append('uncertainty_coverage')
        if (plan or {}).get('uses_ml_context'):
            with_ml,without=study['ml_ablation']['with_ml'],study['ml_ablation']['without_ml']
            if 'brier' not in with_ml or 'brier' not in without or with_ml['brier']>=without['brier'] or with_ml['log_loss']>=without['log_loss']: failures.append('ml_context_incremental_value_unproven')
        for metric,(threshold,compare) in comparisons.items():
            a,b=finite(measured.get(metric)),finite(spec.get(threshold))
            if a is None or b is None or not compare(a,b):failures.append(metric)
        # Nonpositive return/CLV support cannot be made passing by negative limits.
        if (finite(spec.get('min_roi_lower_95')) or 0)<0 or (finite(spec.get('min_price_clv_lower_95')) or 0)<0 or (finite(spec.get('max_calibration_excess_brier')) or 0)>0:failures.append('invalid_promotion_threshold')
        policy=spec.get('policy',{})
        from core.sport_policy import SportPolicy
        try:
            parsed=SportPolicy(**policy)
            if parsed.sport!=sport or parsed.deployment_state!=state:failures.append('policy_state_mismatch')
        except (TypeError,ValueError):failures.append('invalid_policy')
        comparison=strategy_comparison(hold,policy)
        risk=comparison['maturity_capped_kelly']
        for metric,threshold,compare in [('probability_profit','min_probability_profit',lambda a,b:a>=b),('max_drawdown','max_strategy_drawdown',lambda a,b:a<=b)]:
            a,b=finite(risk[metric]),finite(spec.get(threshold))
            if a is None or b is None or not compare(a,b):failures.append('strategy_'+metric)
        measured[state+'_risk']=risk
        measured[state+'_strategy_comparison']=comparison
        if not spec.get('maturity_rules') or not spec.get('exposure_limits'):failures.append('missing_frozen_runtime_rules')
        tier_results[state]=sorted(set(failures))
        if not failures:earned=state;supported_policy=spec
    if earned=='UNVALIDATED':blockers=sorted(set(blockers+['no_tier_passed_frozen_holdout_criteria']))
    result={'sport':sport,'deployment_state':earned,'strict_n':len(rows),'effective_n':len({r['game_id'] for r in rows}),'spread_n':sum(r['market_type'].startswith('spread') for r in rows),'total_n':sum(r['market_type'].startswith('total') for r in rows),'metrics':measured,'cohort_count':len(cohorts),'blockers':blockers,'tier_results':tier_results,'supported_policy':supported_policy,'exclusions':dict(Counter(x for r in excluded if r.get('sport',r.get('league'))==sport for x in r['exclusion_reasons'])),'evidence_hash':digest(rows),'plan_hash':plan.get('plan_hash') if plan else None,'expires_at':plan.get('expires_at') if plan else None,'versions':(plan or {}).get('versions',{}),'market_family':(plan or {}).get('market_family'),'validation_slates':sorted({r['slate_id'] for r in hold}),'validation_through':max((r.get('outcome_recorded_at','') for r in hold),default=None)}
    result.update(study)
    result['validation_schema']='activation-validation-v2'
    result['development_through']=(plan or {}).get('development_through')
    result['validation_from']=min((r['prediction_generated_at'] for r in hold),default=None)
    result['provenance_pass']=bool(hold) and not blockers
    result['walk_forward']=[]
    previous=[]
    for slate in sorted(by_slate,key=lambda key:min(aware(r['game_start_utc']) for r in by_slate[key])):
        fold=by_slate[slate];origin=min(aware(r['prediction_generated_at']) for r in fold)
        available=[r for r in previous if aware(r.get('outcome_recorded_at')) is not None and aware(r['outcome_recorded_at'])<origin]
        result['walk_forward'].append({'validation_slate':slate,'origin':origin.isoformat(),'previous_completed_slates':sorted({r['slate_id'] for r in available}),'metrics':metrics(fold),'tuning_on_holdout':False})
        previous.extend(fold)
    result['validation_hash']=digest(result)
    return result


def freeze_plan(database,choices,*,now=None):
    from app_core.prediction_evidence import connect
    from contextlib import closing
    now=now or datetime.now(timezone.utc)
    if choices.get('sport') not in SPORTS or choices.get('market_family') not in {'spread','total'}:raise ValueError('Sport and market required')
    cutoff=aware(choices.get('development_through'))
    if cutoff is None or cutoff>=now:raise ValueError('Development must end before freezing')
    value=dict(choices,frozen_at=now.isoformat());value.pop('plan_hash',None);value['plan_hash']=digest(value)
    with closing(connect(database)) as db,db:db.execute('INSERT OR IGNORE INTO validation_plans VALUES (?,?,?)',(value['plan_hash'],value['sport'],json.dumps(value,sort_keys=True)))
    return value
