"""Automatic evidence-based maturity. Neither outcomes nor current closes are inputs."""
from core.exposure_ledger import digest
from core.wager_decisions import candidate_decision,finite
from core.sport_policy import DEPLOYMENT_STATES

INPUTS='sport game_id market_type line odds_american book start quote_time identity_verified exact_quote_verified model_validated model_version calibration_validated calibration_version evidence_snapshot_id evidence_frozen_at critical_feature_error conservative_probability mean_probability evidence_effective_sample_size calibration_uncertainty prior_clv_lower current_regime_conflict validated_evidence_family push_probability alternate alternate_quote_verified'.split()

def assign(row,policy,rules,now,*,canonical_quote_verified=False):
    inputs={k:row.get(k) for k in INPUTS}
    # Only a completed, validated earlier-slate CLV aggregate may enter rules.
    reason=['missing_validated_maturity_rules'];tier='RESEARCH'
    for candidate in ('PREMIUM','STANDARD','PROVISIONAL'):
        spec=rules.get(candidate) if isinstance(rules,dict) else None
        if not spec:continue
        if DEPLOYMENT_STATES[policy.deployment_state]<{'PROVISIONAL':1,'STANDARD':2,'PREMIUM':3}[candidate]:continue
        failures=[]
        for field,limit,op in [('calibration_uncertainty','max_uncertainty',lambda x,y:x<=y),('prior_clv_lower','min_prior_clv',lambda x,y:x>=y)]:
            a,b=finite(inputs.get(field)),finite(spec.get(limit))
            if a is None or b is None or not op(a,b):failures.append(field)
        if inputs.get('current_regime_conflict') is not False:failures.append('regime_unverified_or_conflicting')
        if inputs.get('validated_evidence_family')!=str(inputs.get('market_type','')).split('_')[0]:failures.append('evidence_family_mismatch')
        check=candidate_decision(dict(inputs,maturity=candidate,gemini_status='CONFIRM'),policy,now,
                                 canonical_quote_verified=canonical_quote_verified)
        failures.extend(check['reason_for_pass'])
        if not failures:tier=candidate;reason=['validated_candidate_evidence'];break
        reason=failures
    if tier=='RESEARCH' and (finite(row.get('conservative_ev')) or 0)>0:tier='QUALIFIED'
    return dict(row,maturity=tier,maturity_reason='; '.join(sorted(set(reason))),maturity_policy_version=policy.version,maturity_inputs_hash=digest(inputs))
