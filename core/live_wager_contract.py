"""Live adapter to the existing decision engine. No network or order execution."""
from datetime import datetime, timezone
import json
import os
import pandas as pd
from core.sport_policy import SportPolicy, research_policies
from core.wager_decisions import candidate_decision, allocate_exposure, finite, aware
from core.market_policy import production_market

VERSION = 'live-v1'
MATURITIES = {'RESEARCH','QUALIFIED','PROVISIONAL','STANDARD','PREMIUM'}
PUBLIC_FIELDS = ('wager_contract_version game_id matchup_id sport market_type selection line odds sportsbook quote_timestamp start raw_model_probability calibrated_probability sport_calibrated_probability hierarchical_probability conservative_probability market_probability fair_market_probability edge conservative_edge expected_value conservative_ev data_quality_status identity_verified quote_verified quote_fresh gemini_review_status gemini_gate_reason gemini_stake_multiplier gemini_outage_capped maturity evidence_maturity_score production_eligible production_gate_reason raw_kelly production_bet_amount recommended_units model_version calibration_version sport_policy_version evidence_version strategic_action').split()


def load_configuration(now):
    """Only explicit, unexpired policy artifacts can authorize live exposure."""
    path = os.environ.get('PARLAYPICKER_WAGER_POLICY_PATH','data/policies/active_wager_policy.json')
    fallback = (research_policies(), {}, 'No validated live sport policy configured')
    try:
        from core.activation_policy import verify_sport
        from core.exposure_ledger import snapshot as ledger_snapshot
        with open(path,encoding='utf-8') as file: config=json.load(file)
        if config.get('schema')!='activation-v1' or not config.get('activation'):return fallback
        policies=research_policies()
        for sport,values in config['sports'].items():
            try:
                verify_sport(config,sport,now=now)
                policies[sport]=SportPolicy(**values)
            except (ValueError,TypeError,KeyError):
                continue  # This sport has zero authority; strict expiry is mandatory.
        try:
            config['exposure']=ledger_snapshot(os.environ.get('PARLAYPICKER_EXPOSURE_LEDGER','data/exposure/exposure.sqlite3'),now=now)
            config['unit_value']=config['exposure']['unit_value']
        except ValueError:config['exposure']={}
        config['automatic_maturity']=True
        return policies,config,''
    except (OSError,ValueError,TypeError,KeyError):return fallback



def _value(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and not isinstance(value, (dict, list, tuple)) and not pd.isna(value):
            return value
    return None


def _clean(row):
    return {k:(None if not isinstance(v,(dict,list,tuple)) and pd.isna(v) else v) for k,v in row.items()}


def _ticket(row):
    return tuple(str(_value(row, *keys) or '') for keys in [('league','sport'),('away_team','Away'),('home_team','Home'),('market_type',),('best_pick','selection'),('odds_american','odds')])


def adapt_candidate(row, review=None):
    row = _clean(row)
    out = dict(row)
    aliases = {'sport':('sport','league'), 'game_id':('game_id','matchup_id'), 'selection':('selection','best_pick'),
               'book':('book','quote_bookmaker','quote_source','odds_source'), 'quote_time':('quote_time','odds_recorded_at','quote_timestamp'),
               'start':('start','game_start_utc','commence_time','game_time_est'),
               'line':('line','market_line_used','spread_line' if 'spread' in str(row.get('market_type')) else 'total_line'),
               'mean_probability':('mean_probability','sport_calibrated_probability','calibrated_probability')}
    for field, keys in aliases.items():
        out[field] = _value(row,*keys)
    for target,source in [('exact_quote_verified','quote_binding_verified'),('evidence_effective_sample_size','effective_evidence_size')]:
        if out.get(target) is None and row.get(source) is not None:out[target]=row[source]
    # Do not copy a selected side's review to its opposite or another market.
    review = review or {}
    status = review.get('gemini_review_status', 'UNAVAILABLE')
    error = str(review.get('gemini_error','')).upper()
    if status in {'UNAVAILABLE','OUTAGE_CAPPED'}:
        status = 'TIMEOUT' if 'TIMEOUT' in error else 'SERVICE_ERROR' if '5XX' in error or '503' in error or '504' in error else 'UNAVAILABLE'
    status = {'HOLD':'HARD_VETO','OPPOSE':'ABSTAIN','LOW_CONFIDENCE':'ABSTAIN','CONFIRM':'APPROVE'}.get(status,status)
    out['gemini_status'] = status
    out['gemini_stake_multiplier'] = review.get('gemini_stake_multiplier',1)
    out['gemini_gate_reason'] = review.get('gemini_gate_reason','Secondary review unavailable')
    if row.get('line_consistency_flag') is False or row.get('line_event_identity_match_flag') is False or row.get('degraded_feature_subset_flag') is True:
        out['critical_feature_error'] = True
    out['maturity'] = row.get('maturity') if row.get('maturity') in MATURITIES else 'RESEARCH'
    from app_core.public_quote_policy import supported_quote
    if not supported_quote({'sport':out['sport'] or '', 'quote_source':out['book']}):
        out['exact_quote_verified'] = False
    return out


def snapshot(row, now, unit_value=None):
    result = {key: None for key in PUBLIC_FIELDS}
    for key in result:
        value = row.get(key)
        if isinstance(value,(str,bool,int,float)) and not (isinstance(value,float) and not pd.notna(value)):
            result[key] = value
    amount = finite(row.get('recommended_stake')) or 0.0
    result.update(wager_contract_version=VERSION, game_id=str(row.get('game_id') or ''),
        matchup_id=str(row.get('matchup_id') or row.get('game_id') or ''), sport=str(row.get('sport') or ''),
        sportsbook=row.get('book'), odds=finite(row.get('odds_american')), quote_timestamp=row.get('quote_time'),
        quote_verified=row.get('exact_quote_verified') is True, identity_verified=row.get('identity_verified') is True,
        quote_fresh=bool(aware(row.get('quote_time')) and 0 <= (now-aware(row['quote_time'])).total_seconds() <= 1800),
        production_eligible=bool(row.get('production_eligible') is True and amount > 0), production_bet_amount=amount,
        recommended_units=amount/unit_value if unit_value and unit_value > 0 else None,
        evidence_version=row.get('evidence_snapshot_id'), data_quality_status='VERIFIED' if row.get('critical_feature_error') is False else 'UNVERIFIED')
    return result


def finalize_live_wagers(candidates, best, bankroll, *, now=None, policies=None, config=None, reviews=None):
    """Evaluate all candidates before selecting a funded winner; retain research separately."""
    now = now or datetime.now(timezone.utc)
    if policies is None:
        policies, config, configuration_reason = load_configuration(now)
    else:
        config = config or {}
        configuration_reason = ''
    reviews = reviews if reviews is not None else best
    review_map = {_ticket(r):r.to_dict() for _,r in reviews.iterrows()}
    from core.streamlit_pipeline import _format_best_pick
    from collections import defaultdict
    grouped = defaultdict(list)
    for _,series in candidates.iterrows():
        raw = _clean(series.to_dict())
        if not production_market(raw.get('market_type')):
            continue
        raw['best_pick'] = _format_best_pick(series)
        # Use existing matchup keys only; never treat generated labels as verification.
        if not raw.get('matchup_id'):
            same = best[(best['league']==raw.get('league')) & (best['home_team']==raw.get('home_team')) & (best['away_team']==raw.get('away_team'))]
            if len(same)==1: raw['matchup_id'] = same.iloc[0].get('matchup_id')
        row = adapt_candidate(raw, review_map.get(_ticket(raw)))
        policy = policies.get(row['sport'])
        if policy is None: continue
        runtime=config.get('validation_results',{}).get(row['sport'],{}).get('supported_policy') or {}
        if config.get('automatic_maturity'):
            from core.candidate_maturity import assign
            validation=config.get('validation_results',{}).get(row['sport'],{})
            # The loader verifies active policy artifacts. Bind the study again
            # to this candidate; candidate-supplied family is never authority.
            row['validated_evidence_family'] = None
            from core.exposure_ledger import digest
            study_hash = validation.get('validation_hash')
            expires = aware(validation.get('expires_at'))
            if (validation.get('sport') != row['sport'] or policy.sport != row['sport']
                    or validation.get('deployment_state') != policy.deployment_state
                    or not study_hash or study_hash != policy.validation_id
                    or digest({k:v for k,v in validation.items() if k != 'validation_hash'}) != study_hash
                    or expires is None or expires <= now):
                row['critical_feature_error'] = True
            versions=validation.get('versions',{})
            validated_at=aware(validation.get('validation_through'))
            if validated_at is None or validated_at>now or not row.get('slate_id') or not validation.get('validation_slates') or row.get('slate_id') in validation.get('validation_slates',[]):row['critical_feature_error']=True
            row['prior_clv_lower']=validation.get('metrics',{}).get('price_clv_lower_95')
            if any(not versions.get(k) or row.get(k)!=versions[k] for k in ('model_version','calibration_version','selection_policy_version','sport_policy_version','evidence_version')):
                row['critical_feature_error']=True
            if str(row.get('market_type','')).split('_')[0]!=config.get('validation_results',{}).get(row['sport'],{}).get('market_family'):
                row['critical_feature_error']=True
            generated=aware(row.get('prediction_generated_at'));trained=aware(row.get('model_trained_through'));available=aware(row.get('model_available_at'));cal_available=aware(row.get('calibration_available_at'))
            if any(x is None for x in (generated,trained,available,cal_available,validated_at)) or not trained<generated<=now or not available<=generated or not cal_available<=generated or not validated_at<generated:
                row['critical_feature_error']=True
            if row.get('critical_feature_error') is False:
                row['validated_evidence_family'] = validation.get('market_family')
            row=assign(row,policy,runtime.get('maturity_rules',{}),now)
        decision = candidate_decision(row, policy, now, outage_policy=runtime.get('gemini_outage',config.get('gemini_outage')))
        grouped[(row['sport'],str(row.get('game_id') or ''))].append(decision)
    selected=[]; templates=[]
    for _,template in best.iterrows():
        pool=grouped.get((template.get('league'),str(_value(template,'game_id','matchup_id') or '')),[])
        eligible=[r for r in pool if r['production_eligible']]
        eligible.sort(key=lambda r: (-(r['conservative_ev'] or 0),-(r.get('conservative_probability') or 0),str(r.get('selection'))))
        if eligible:
            row=eligible[0]
        else:
            exact=[r for r in pool if str(r.get('selection'))==str(template.get('best_pick'))]
            row=dict(exact[0] if exact else adapt_candidate(template.to_dict()))
            reasons=sorted({reason for r in pool for reason in r['reason_for_pass']})
            row.update(recommended_fraction=0,production_eligible=False,strategic_action='PASS',production_gate_reason='; '.join(reasons) or configuration_reason or 'No valid candidate evidence')
        selected.append(row);templates.append(template.to_dict())
    # Missing committed exposure/caps cannot be interpreted as unused bankroll.
    caps=config.get('exposure',{})
    caps = dict(caps) if isinstance(caps,dict) else {}
    if config.get('automatic_maturity') and finite(caps.get('bankroll'))!=finite(bankroll): caps={}
    if config.get('automatic_maturity') and caps:
        for key in ('total_cap','daily_cap','weekly_cap','game_cap','team_cap'):
            validated=[finite((config.get('validation_results',{}).get(s,{ }).get('supported_policy') or {}).get('exposure_limits',{}).get(key)) or 0 for s,p in policies.items() if p.deployment_state!='UNVALIDATED']
            caps[key]=min([finite(caps.get(key)) or 0]+validated) if validated else 0
    committed=caps.get('committed')
    valid_exposure=isinstance(committed,dict) and aware(caps.get('as_of')) is not None and 0 <= (now-aware(caps['as_of'])).total_seconds() <= 1800
    valid_exposure = valid_exposure and all(finite(caps.get(k,0)) is not None and 0<=finite(caps.get(k,0))<=1 for k in ('total_cap','daily_cap','weekly_cap','game_cap','team_cap')) and all(finite(v) is not None and finite(v)>=0 for v in (committed or {}).values())
    if valid_exposure:
        caps = dict(caps, **{k:finite(caps.get(k,0)) for k in ('total_cap','daily_cap','weekly_cap','game_cap','team_cap')})
        committed = {k:finite(v) for k,v in committed.items()}
    allocated=allocate_exposure(selected,bankroll,total_cap=caps.get('total_cap',0) if valid_exposure else 0,
        daily_cap=caps.get('daily_cap',0) if valid_exposure else 0,
        weekly_cap=caps.get('weekly_cap',0) if valid_exposure else 0,
        game_cap=caps.get('game_cap',0) if valid_exposure else 0,team_cap=caps.get('team_cap',0) if valid_exposure else 0,
        sport_caps={s:p.sport_exposure_cap for s,p in policies.items()},committed=committed if valid_exposure else {})
    lookup={(r.get('sport'),str(r.get('game_id'))):r for r in allocated}
    output=[]
    for template, decision in zip(templates,selected):
        row=lookup.get((decision.get('sport'),str(decision.get('game_id'))),dict(decision,recommended_stake=0))
        if not valid_exposure:
            row['production_gate_reason']=(row.get('production_gate_reason','')+'; committed portfolio exposure unavailable').strip('; ')
        contract=snapshot(row,now,finite(config.get('unit_value')))
        if contract['production_bet_amount']<=0:
            contract['strategic_action']='PASS'
        result=dict(template)
        if row.get('candidate_id'):
            # Preserve the selected exact candidate's line/book and evidence.
            result.update(row)
        for key in ('team_ids','maturity_reason','maturity_policy_version','maturity_inputs_hash','deployment_state','validated_evidence_family'):
            result[key]=row.get(key)
        if contract['production_eligible']:
            result.update(best_pick=row['selection'],market_type=row['market_type'],odds_american=row['odds_american'],market_line_used=row['line'],odds_source=row['book'], production_win_probability=row.get('conservative_probability'), production_expected_value=row.get('conservative_ev'),production_edge=row.get('conservative_edge'))
        from core.streamlit_pipeline import _build_canonical_pick_key
        result['canonical_pick_key']=_build_canonical_pick_key(pd.Series(result))
        result['wager_contract']=contract
        output.append(result)
    return enforce_frame(pd.DataFrame(output)), [dict(snapshot(r,now),**{k:r.get(k) for k in ('maturity_reason','maturity_policy_version','maturity_inputs_hash','deployment_state','validated_evidence_family','candidate_id')}) for pool in grouped.values() for r in pool]


def enforce_frame(frame):
    """Final compatibility projection. Legacy labels cannot grant new authority."""
    if frame is None or frame.empty or 'wager_contract' not in frame:
        return frame
    out=frame.copy()
    for idx,row in out.iterrows():
        c=row.get('wager_contract')
        if not isinstance(c,dict) or c.get('wager_contract_version')!=VERSION:
            continue
        funded=c.get('production_eligible') is True and (finite(c.get('production_bet_amount')) or 0)>0 and production_market(c.get('market_type'))
        stake=c['production_bet_amount'] if funded else 0.0
        for key in ('production_bet_amount','Kelly_Bet_Size','Play_Stake','Suggested_Stake','recommended_bet'):
            out.at[idx,key]=stake
        for key in ('production_eligible','wager_approved','Wager_Approved','All_Row_Bet','Bettable'):
            out.at[idx,key]=funded
        for key in ('maturity','conservative_probability','conservative_ev','conservative_edge','gemini_review_status','production_gate_reason'):
            out.at[idx,key]=c.get(key)
        out.at[idx,'sellable_as_premium']=funded and c.get('maturity')=='PREMIUM'
        out.at[idx,'sellable_as_value_card']=False
        out.at[idx,'controlled_card_recovery']=False
        out.at[idx,'commercial_tier']=c.get('maturity')
        out.at[idx,'Export_Scope']='PRODUCTION BET' if funded else 'COVERAGE / RESEARCH'
        out.at[idx,'Wager_Instruction']='BET - '+str(c.get('maturity'))+(' - OUTAGE CAPPED; SECONDARY REVIEW NOT COMPLETED' if c.get('gemini_outage_capped') else '') if funded else 'DO NOT BET - $0 PASS / RESEARCH'
        out.at[idx,'Play_Units']=c.get('recommended_units') if funded else 0.0
        out.at[idx,'Pick_Status']='Actionable' if funded else 'PASS'
        out.at[idx,'Bet_Decision']='BET' if funded else 'PASS'
        reason=c.get('production_gate_reason') or ('Validated canonical wager' if funded else 'No allocation remains after portfolio limits')
        for key in ('Production_Gate_Reason','qualification_reason','Status_Reason'):
            out.at[idx,key]=reason
    return out


def validate_snapshot(c):
    import math
    if set(c)!=set(PUBLIC_FIELDS) or c.get('wager_contract_version')!=VERSION:
        raise ValueError('Invalid wager contract schema')
    for key,value in c.items():
        if value is not None and not isinstance(value,(str,int,float,bool)):
            raise ValueError('Invalid wager contract field')
        if isinstance(value,float) and not math.isfinite(value):
            raise ValueError('Nonfinite wager contract metric')
    if c.get('maturity') not in MATURITIES:
        raise ValueError('Invalid wager maturity')
    if c.get('production_eligible') is True:
        if not production_market(c.get('market_type')) or c.get('maturity') not in {'PROVISIONAL','STANDARD','PREMIUM'} or (finite(c.get('production_bet_amount')) or 0)<=0 or (finite(c.get('conservative_ev')) or 0)<=0 or c.get('identity_verified') is not True or c.get('quote_verified') is not True:
            raise ValueError('Invalid funded wager contract')
