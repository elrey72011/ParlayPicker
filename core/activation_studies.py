"""Frozen, future-cohort studies; absent inputs are explicit blockers."""
from collections import defaultdict
import math
from statistics import median
from core.wager_decisions import finite, decimal_price


def score(rows, field):
    pairs = [(finite(r.get(field)), int(r['candidate_outcome']=='WIN')) for r in rows if r.get('candidate_outcome') in {'WIN','LOSS'}]
    if not pairs or any(p is None or not 0<p<1 for p,y in pairs):
        return {'n':len(pairs), 'blocker':'missing_'+field}
    return {'n':len(pairs), 'brier':sum((p-y)**2 for p,y in pairs)/len(pairs),
            'log_loss':-sum(y*math.log(p)+(1-y)*math.log(1-p) for p,y in pairs)/len(pairs),
            'hit_rate':sum(y for p,y in pairs)/len(pairs)}


def studies(rows, settings):
    settled=[r for r in rows if r.get('candidate_outcome') in {'WIN','LOSS'}]
    bands=defaultdict(list)
    for r in settled: bands[(r['sport'],r['market_type'].split('_')[0],int(float(r['conservative_probability'])*10))].append(r)
    coverage=[];missing=[]
    for (sport,market,band), group in sorted(bands.items()):
        intervals=[(finite(r.get('probability_interval_lower')),finite(r.get('probability_interval_upper'))) for r in group]
        if any(a is None or b is None or not 0<=a<=b<=1 for a,b in intervals):
            missing.append('missing_probability_intervals');continue
        observed=sum(r['candidate_outcome']=='WIN' for r in group)/len(group)
        low=sum(a for a,b in intervals)/len(group);high=sum(b for a,b in intervals)/len(group)
        coverage.append({'sport':sport,'market':market,'band':band/10,'n':len(group),
            'mean_lower':low,'mean_upper':high,'observed':observed,'interval_width':high-low,
            'band_mean_covered':low<=observed<=high,'lower_bound_covered':observed>=low})
    uncertainty={'method':'heldout_probability_band_mean_coverage_not_individual_binary_coverage',
        'configured_lower_quantile':settings.get('uncertainty_quantile'), 'bands':coverage,
        'coverage':sum(x['band_mean_covered'] for x in coverage)/len(coverage) if coverage else None,
        'lower_coverage':sum(x['lower_bound_covered'] for x in coverage)/len(coverage) if coverage else None,
        'blockers':sorted(set(missing+([] if coverage else ['no_coverage_observations'])))}
    prior={name:score(settled,field) for name,field in [('historical_only','historical_prior_probability'),('current_only','current_season_probability'),('hierarchical','hierarchical_probability')]}
    prior['parameters']=settings.get('evidence_parameters')
    prior['blockers']=[v['blocker'] for v in prior.values() if isinstance(v,dict) and v.get('blocker')]
    if not prior['parameters']:prior['blockers'].append('missing_frozen_evidence_parameters')
    ablation={'without_ml':score(settled,'probability_without_ml_context'),'with_ml':score(settled,'probability_with_ml_context')}
    premium=[r for r in settled if r.get('candidate_maturity',r.get('maturity'))=='PREMIUM']
    wins=sum(r['candidate_outcome']=='WIN' for r in premium);n=len(premium)
    from scipy.stats import beta
    premium_report={'n':n,'wins':wins,'losses':n-wins,'hit_rate':wins/n if n else None,
        'interval_95':[float(beta.ppf(.025,wins,n-wins+1)) if wins else 0.,float(beta.ppf(.975,wins+1,n-wins)) if wins<n else 1.] if n else None,
        'average_odds':sum(float(r['odds_american']) for r in premium)/n if n else None,'target':.75,'target_is_activation_gate':False}
    closes=[r['closing'] for r in rows if r.get('closing')]
    clv={key:sum(float(x[key]) for x in closes if finite(x.get(key)) is not None)/sum(finite(x.get(key)) is not None for x in closes) if any(finite(x.get(key)) is not None for x in closes) else None for key in ('price_clv','line_clv')}
    clv['beat_close_rate']=sum(x['beat_close'] is True for x in closes)/len(closes) if closes else None
    return {'uncertainty':uncertainty,'prior_validation':prior,'ml_ablation':ablation,'premium':premium_report,'clv':clv}


def strategy_comparison(rows, policy):
    """Compare frozen caps; resample whole slates, never select using outcomes."""
    import random
    modes={};cap=finite(policy.get('provisional_stake_cap')) or 0
    if policy.get('deployment_state')=='STANDARD_VALIDATED':cap=finite(policy.get('standard_stake_cap')) or 0
    if policy.get('deployment_state')=='PREMIUM_VALIDATED':cap=finite(policy.get('premium_stake_cap')) or 0
    # Only the pregame selected candidate per game, never both opposing sides.
    selected=[r for r in rows if str(r.get('selected_as_best_pick')).lower()=='true' and r.get('candidate_outcome') in {'WIN','LOSS'}]
    for mode in ('flat_small_units','fractional_kelly','maturity_capped_kelly'):
        slates=defaultdict(float);seen=set()
        for r in selected:
            key=(r['sport'],r['game_id'])
            if key in seen:continue
            seen.add(key);d=decimal_price(r.get('odds_american'));p=finite(r.get('conservative_probability'))
            if d is None or p is None:continue
            k=max(0,(p*d-1)/(d-1))*(finite(policy.get('kelly_fraction')) or 0)
            fraction=cap if mode=='flat_small_units' else k if mode=='fractional_kelly' else min(cap,k)
            slates[r['slate_id']]+=fraction*(d-1 if r['candidate_outcome']=='WIN' else -1)
        returns=list(slates.values());sim=[];rng=random.Random(0)
        if len(returns)>=2:
            for _ in range(500):
                wealth=peak=1.;dd=0.
                for _ in returns:
                    wealth*=max(0,1+rng.choice(returns));peak=max(peak,wealth);dd=max(dd,1-wealth/peak)
                sim.append((wealth-1,dd))
        modes[mode]={'slates':len(returns),'median_return':median([v for v,d in sim]) if sim else None,
            'probability_profit':sum(v>0 for v,d in sim)/len(sim) if sim else None,
            'max_drawdown':max((d for v,d in sim),default=None),
            **{f'drawdown_{int(q*100)}_probability':sum(d>=q for v,d in sim)/len(sim) if sim else None for q in (.1,.25,.5,.9)}}
    return modes
