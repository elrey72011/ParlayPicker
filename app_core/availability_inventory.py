"""Static retained-row availability inventory; no inference, acquisition or repair.

Candidate rows are not an independent schedule denominator. Current hosted
availability remains UNKNOWN unless a supplied current package establishes it.
"""
from collections import Counter
from copy import deepcopy
import json

VERSION='research-availability-inventory-v1'


def _json(value):
    if isinstance(value,dict):return value
    try:return json.loads(value) if isinstance(value,str) else {}
    except (ValueError,TypeError):return {}


def _value(value):
    if isinstance(value,dict) and value.get('state')=='VALUE':return value.get('value')
    return None if isinstance(value,dict) else value


def inventory(candidates=(), *, coverage=None, traces=(), reference, hosted_current=False):
    """Preserve original rejection order/facts; no guessed gate outcomes.

    Private output intentionally excludes raw metadata, features and bodies.
    Schedule-only rows stay visible. Conflicting candidate identities are not
    deduplicated into an apparently healthy row.
    """
    rows=[];trace_by={}
    for t in traces:
        key=t.get('candidate_id')
        if key:
            if key in trace_by and trace_by[key]!=t:raise ValueError('AVAILABILITY_CONFLICTING_TRACE')
            trace_by[key]=t
    for i,c in enumerate(candidates):
        t=trace_by.get(c.get('candidate_id'),_json(c.get('research_estimate_trace')))
        m=_json(c.get('ml_estimate_metadata')) or _json(_value((t.get('source') or {}).get('ml_estimate_metadata')))
        producer=m.get('producer_contract',{});offer=producer.get('offer',{});event=producer.get('event',{})
        display=t.get('display') or _json(c.get('research_display'));origin=t.get('origin') or {}
        first=t.get('first_rejection') or t.get('first_observed_failure') or {}
        gates=t.get('gate_results',[])
        if not first: first=next((g for g in gates if g.get('status')=='FAIL'),{})
        stage=first.get('gate') or first.get('stage') or t.get('first_rejection_stage')
        source_codes=origin.get('source_contract_diagnostics',[])
        code=first.get('code') or t.get('rejection_code') or c.get('ml_unavailable_reason') or (
            source_codes[0] if stage=='quote.source_contract' and source_codes else display.get('availability_reason') if stage else None)
        message=m.get('reason') or c.get('Status_Reason') or None
        league=c.get('league') or c.get('League') or event.get('sport')
        market=c.get('market_type') or offer.get('market')
        identity=c.get('canonical_event_id')
        observed=c.get('provider_event_id') or event.get('provider_event_id')
        # An unordered legacy matchup key is retained as a label, never promoted
        # into an accepted schedule identity.
        probability=_value(m.get('probability'))
        inference_status=m.get('inference_status') or c.get('ml_inference_status')
        reason=str(code or message or '')
        if any(s in reason.upper() for s in ('MODEL','ARTIFACT','RUNTIME','FEATURE','INPUT','DEPENDENCY','ORIGINAL_PACKET')):
            need='authentic_inputs_or_compatible_artifact';owner='evidence custodian / model reviewer'
            action='Supply the exact original target inputs, dependency bytes/clocks and consumed compatible artifact; then independently review.'
        elif any(s in reason.upper() for s in ('SOURCE','PERIOD','SETTLEMENT','RIGHT','LISTING','PRODUCT','REVIEW','ACCEPT')):
            need='independent_source_evidence';owner='source owner / independent reviewer'
            action='Bind this exact event/offer to applicable listing/product, period, settlement, clock meaning and account rights; preserve original review timing.'
        elif any(s in reason.upper() for s in ('STALE','FUTURE','START','QUOTE','ODDS')):
            need='fresh_authentic_observation';owner='owner / evidence custodian'
            action='Approve a bounded prospective plan after source admission; acquire a genuinely new quote and complete review before inference.'
        else:
            need='missing_diagnostic_or_acceptance';owner='owner / evidence custodian'
            action='Provide the exact existing private trace for this row; identify the first actual gate before proposing a correction.'
        rows.append(dict(row_reference=f'{reference}#/rows/{i}',canonical_event_id=identity,canonical_identity_status='RECORDED' if identity else 'UNKNOWN',
            provider_event_id=observed,legacy_matchup_id=c.get('matchup_id'),league=league,home_team=c.get('home_team') or event.get('home'),
            away_team=c.get('away_team') or event.get('away'),market=market,side=offer.get('side') or (str(market).split('_')[-1] if market else None),
            line=offer.get('line',c.get('spread_line') if str(market).startswith('spread') else c.get('total_line')),
            quote_clock=offer.get('source_time') or c.get('odds_recorded_at'),observation_clock=c.get('quote_observed_at'),
            inference_clock=producer.get('inference_time') or m.get('generated_at'),first_actual_rejection_code=code,
            first_actual_rejection_stage=stage or ('recorded_predictor' if inference_status=='unavailable' else 'UNKNOWN'),
            recorded_rejection_message=message,observed_gate_results=deepcopy(gates),required_evidence=need,responsible_role=owner,next_action=action,
            missing_fields=deepcopy(origin.get('missing_fields',[])),source_contract_codes=deepcopy(source_codes),
            period=offer.get('period'),settlement_rules=offer.get('rules'),product=offer.get('product'),listing=offer.get('listing_id'),
            consumed_predictor=_value(m.get('predictor_id')),raw_probability=_value(m.get('probability')),
            original_blend=c.get('calibrated_probability'),ui_refresh='UNRECORDED_UNLESS_EXPLICIT_RECEIPT_SUPPLIED',
            research_probability=dict(status='RECORDED_NOT_DISPLAY_ACCEPTED' if probability is not None else 'UNAVAILABLE_OR_UNRECORDED',value=probability),
            display_probability=deepcopy(display.get('probability')),value_display=dict(status=display.get('value_reason','UNKNOWN'),ev=display.get('ev'),edge=display.get('edge')),
            wagering=dict(status='UNKNOWN',qualification='NOT_ESTABLISHED_BY_RESEARCH',new_authority=False,new_stake=0),
            current_hosted_status='SUPPLIED_CURRENT_RECORD_ONLY' if hosted_current else 'UNKNOWN',
            run_id=c.get('export_run_id'),snapshot_id=c.get('snapshot_id')))
    known={r['canonical_event_id'] for r in rows if r['canonical_event_id']}
    if coverage:
        from app_core.slate_coverage import validate_report
        validate_report(coverage)
        for d in coverage['decisions']:
            if d['canonical_event_id'] not in known:
                rows.append(dict(row_reference=reference+'#/coverage/'+d['canonical_event_id'],canonical_event_id=d['canonical_event_id'],league=d['league'],
                    home_team=d['home_team'],away_team=d['away_team'],market=None,side=None,line=None,quote_clock=None,observation_clock=None,inference_clock=None,
                    first_actual_rejection_code=(d.get('first_observed_failure') or {}).get('code'),first_actual_rejection_stage=(d.get('first_observed_failure') or {}).get('gate'),
                    coverage_decision_state=d['coverage_decision_state'],required_evidence='schedule_only_no_candidate',next_action='Resolve the recorded coverage blocker without dropping this event.',
                    responsible_role='owner / evidence custodian',current_hosted_status='SUPPLIED_CURRENT_RECORD_ONLY' if hosted_current else 'UNKNOWN'))
    return dict(version=VERSION,reference=reference,current_hosted_status='SUPPLIED_CURRENT_RECORD_ONLY' if hosted_current else 'UNKNOWN',rows=rows,
        candidate_rows=len(candidates),inventory_status=coverage['inventory_status'] if coverage else 'UNKNOWN',
        independent_schedule_denominator=coverage['counts']['scheduled_events'] if coverage else None,
        recorded_provider_events=len({(r.get('league'),r.get('provider_event_id')) for r in rows if r.get('provider_event_id')}),
        leagues=dict(Counter(r.get('league') or 'UNKNOWN' for r in rows)),no_probability_reconstruction=True)
