"""Canonical private evidence projection; missing facts stay missing."""
from datetime import date, datetime
import math
import numpy as np
import pandas as pd
from core.exposure_ledger import digest
from core.wager_decisions import aware, decimal_price

FIELDS = '''snapshot_id export_run_id candidate_id game_id matchup_id sport season slate_id event_date game_start_utc home_team_id away_team_id home_team away_team market_type selection line american_odds decimal_odds sportsbook odds_source odds_recorded_at quote_verified prediction_generated_at model_version model_available_at model_trained_through calibration_version calibration_available_at evidence_version evidence_frozen_at selection_policy_version sport_policy_version raw_model_probability calibrated_probability sport_calibrated_probability hierarchical_probability conservative_probability market_probability fair_market_probability edge conservative_edge expected_value conservative_ev calibration_uncertainty effective_evidence_size historical_prior_weight current_season_weight ml_context_probability ml_spread_alignment kalshi_probability consensus_agreement gemini_review_status gemini_reviewed_at gemini_input_hash identity_verified data_quality_status push_semantics_verified candidate_maturity production_eligible production_bet_amount created_process_id payload_hash candidate_rank_before_gate candidate_rank_after_gate selected_as_best_pick best_available_candidate_count decision_bundle_version provider_event_id provider_namespace candidate_pool_complete'''.split()

def missing(value):
    return value is None or (not isinstance(value, (list, dict)) and bool(pd.isna(value))) or value == ''

def evidence_value(value):
    """Normalize known dataframe scalars without inventing missing timestamps.

    Keep an original timestamp's offset (or lack of timezone). Unknown objects
    still fail serialization rather than silently becoming arbitrary strings.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return evidence_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: evidence_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [evidence_value(v) for v in value]
    return value


def project(frame):
    out = frame.copy().astype(object).where(pd.notna(frame), None)
    aliases = {'sport':'league', 'game_id':'matchup_id', 'selection':'best_pick',
        'american_odds':'odds_american', 'sportsbook':'quote_bookmaker',
        'quote_verified':'quote_binding_verified', 'candidate_maturity':'maturity',
        'selected_as_best_pick':'best_available_selected', 'gemini_input_hash':'gemini_review_input_hash',
        'event_date':'game_date'}
    for field in FIELDS:
        if field not in out: out[field] = None
    for idx, row in out.iterrows():
        r = row.to_dict()
        for target, source in aliases.items():
            if missing(r.get(target)): r[target] = r.get(source)
        if missing(r['line']): r['line'] = r.get('total_line' if str(r['market_type']).startswith('total') else 'spread_line')
        if missing(r['decimal_odds']): r['decimal_odds'] = decimal_price(r['american_odds'])
        start = aware(r.get('game_start_utc'))
        if missing(r['slate_id']) and start and r['sport'] not in {'NFL','NCAAF'}:
            from zoneinfo import ZoneInfo
            r['slate_id'] = f"{r['sport']}:{start.astimezone(ZoneInfo('America/New_York')).date()}"
        elif missing(r['slate_id']) and r['sport'] in {'NFL','NCAAF'}:
            # Only provider-supplied season/week; calendar week is not football week.
            season, week = r.get('season'), r.get('schedule_week')
            if not missing(season) and not missing(week):
                r['slate_id'] = f"{r['sport']}:{int(season)}:WEEK_{int(week):02d}"
        r = {k: evidence_value(v) for k, v in r.items()}
        if missing(r['candidate_id']):
            r['candidate_id'] = digest({k:r.get(k) for k in ('snapshot_id','game_id','market_type','selection','line','american_odds','sportsbook')})
        r['payload_hash'] = digest({k:r.get(k) for k in FIELDS if k != 'payload_hash'})
        for k in FIELDS: out.at[idx,k] = r.get(k)
    return out.infer_objects(copy=False)

def pool_status(group):
    """An expected count comes from generation, not the received frame length."""
    from core.wager_decisions import finite
    counts = {finite(v) for v in group.best_available_candidate_count}
    identities = group.candidate_id.dropna()
    return counts == {float(len(group))} and len(identities) == len(group) and not identities.duplicated().any()
