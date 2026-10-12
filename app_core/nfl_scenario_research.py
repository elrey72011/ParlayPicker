"""Explicit sporting scenarios; no quote, acceptance catalog, fetch or authority."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import pandas as pd

from app_core import nfl_scoring_custody as custody, nfl_scoring_research as offer
from app_core import nfl_native_provenance as native, nfl_inference_evidence as evidence
from app_core.producer_provenance import team
from app_core.research_estimate_trace import encode, fact

VERSION = 'nfl-owner-sporting-scenario-v1'
RESULT = 'nfl-owner-sporting-scenario-result-v1'
REVIEW = 'nfl-owner-sporting-scenario-review-v1'
LABELS = ['OWNER_REVIEWED', 'PRIVATE_RESEARCH', 'USER-SPECIFIED SCENARIO']
CONDITION = 'completed_sporting_result'
MAX_PACKET = offer.MAX_PACKET
ROOT = Path(__file__).resolve().parents[1]
require, digest = custody.require, custody.digest
FIELDS = set('version evidence_label scoring event event_verification scenario permission features feature_order feature_available_at model reader owner_review original_inference_time'.split())


def now():
    return datetime.now(timezone.utc).isoformat()


def seal(value):
    return dict(payload=deepcopy(value), sha256=digest(value))


def clock(value, code):
    require(isinstance(value, str) and re.search(r'(Z|[+-]\d\d:\d\d)$', value), code)
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        raise ValueError(code) from None
    require(stamp.tzinfo is not None and not pd.isna(stamp), code)
    return stamp.tz_convert('UTC')


def model_binding():
    # The native predictor/artifacts/configuration are unchanged, including its
    # fixed priors, reliability shrinkage and clipping. No market blend is run.
    return offer.model_binding()


def reader_binding():
    paths = ('app_core/nfl_scenario_research.py', 'app_core/nfl_scoring_custody.py',
             'app_core/nfl_source_projection.py', 'app_core/nfl_native_provenance.py',
             'app_core/market_probability_model.py', 'app_core/research_replay.py')
    return dict(runtime=evidence.runtime(), sources={p: hashlib.sha256(
        (ROOT / p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths})


def load(raw, *, owner_upload=False):
    require(isinstance(raw, bytes) and 0 < len(raw) <= MAX_PACKET, 'NFL_SCENARIO_SCHEMA')
    packet = json.loads(raw, object_pairs_hook=custody.unique)
    require(isinstance(packet, dict) and set(packet) == {'payload', 'sha256'}, 'NFL_SCENARIO_SCHEMA')
    p = packet['payload']
    require(isinstance(p, dict) and digest(p) == packet['sha256'], 'NFL_SCENARIO_INTEGRITY')
    require(set(p) == FIELDS and p['version'] == VERSION and p['evidence_label'] in {'RETAINED', 'SYNTHETIC'}
            and p['original_inference_time'] is None, 'NFL_SCENARIO_SCHEMA')
    require(not owner_upload or p['evidence_label'] == 'RETAINED', 'NFL_SCENARIO_SYNTHETIC_UPLOAD')
    custody.read_score(p['scoring'])
    require(p['scoring']['receipt']['evidence_label'] == p['evidence_label'], 'NFL_SCENARIO_SCHEMA')
    return packet


def subject(p):
    return digest({k: v for k, v in p.items() if k != 'owner_review'})


def checked(packet, at):
    """Validate original facts and actual review before any probability call."""
    p = load(encode(packet).encode())['payload']
    instant = clock(at, 'NFL_SCENARIO_CLOCK_UNKNOWN')
    e, s = p['event'], p['scenario']
    require(isinstance(e, dict) and set(e) == set('canonical_event_id home away start season neutral_site'.split()), 'NFL_SCENARIO_EVENT_MISSING')
    require(type(e['season']) is int and e['neutral_site'] is False
            and all(team(e[k], 'NFL') == e[k] for k in ('home', 'away'))
            and e['home'] != e['away'], 'NFL_SCENARIO_EVENT_CONFLICT')
    start = clock(e['start'], 'NFL_SCENARIO_EVENT_CONFLICT')
    require(instant < start, 'NFL_SCENARIO_STARTED')
    require(isinstance(s, dict) and set(s) == set('selected_team signed_handicap period overtime condition'.split()), 'NFL_SCENARIO_TARGET_MISSING')
    require(s['selected_team'] in {e['home'], e['away']} and s['period'] == 'full_game'
            and s['overtime'] is True and s['condition'] == CONDITION, 'NFL_SCENARIO_TARGET_CONFLICT')
    point = s['signed_handicap']
    require(type(point) in {float, int} and math.isfinite(point) and abs(point % 1) == .5, 'NFL_SCENARIO_INTEGER_PUSH_UNVALIDATED')
    _, original = custody.read_score(p['scoring'])
    r = p['scoring']['receipt']
    observed = clock(r['observed_at'], 'NFL_SCENARIO_CLOCK_UNKNOWN')
    require(observed <= instant, 'NFL_SCENARIO_FUTURE_EVIDENCE')
    permission = p['permission']
    require(isinstance(permission, dict) and set(permission) == set('reviewer reviewed_at effective_from effective_until source_id endpoint private_retention private_research_use credential_exclusion basis'.split())
            and permission['reviewer'] == 'Robert Velarde' and permission['private_retention'] is True
            and permission['private_research_use'] is True and permission['credential_exclusion'] is True
            and permission['source_id'] == r['source_id'] and permission['endpoint'] == r['endpoint']
            and isinstance(permission['basis'], str) and len(permission['basis'].strip()) >= 16, 'NFL_SCENARIO_PERMISSION_MISSING')
    reviewed, begin, end = [clock(permission[k], 'NFL_SCENARIO_PERMISSION_CLOCK') for k in ('reviewed_at', 'effective_from', 'effective_until')]
    request = clock(r['request_started_at'], 'NFL_SCENARIO_CLOCK_UNKNOWN')
    require(reviewed <= request and begin <= request <= observed <= instant < end, 'NFL_SCENARIO_PERMISSION_CLOCK')
    # Preserve complete source bytes. The season/event/history views are separate
    # deterministic projections, never relabelled original responses.
    from app_core.nfl_source_projection import seal as projection_seal, score_projection, VERSION as PROJECTION
    indices = [i for i, row in enumerate(original) if str(row.get('season', '')).isdigit() and int(row['season']) == e['season']]
    view = dict(event=e, objects={'scores': p['scoring']}, projections={'scores': projection_seal(dict(
        version=PROJECTION, body_sha256=p['scoring']['body_sha256'], receipt_sha256=digest(r),
        season=e['season'], selected_row_indices=indices, selection='exact_season_source_order_then_native_completed_before_utc_day',
        home=e['home'], away=e['away'], publisher_available_at=None, publisher_availability='UNKNOWN'))})
    rows = score_projection(view, {'scores': original})
    matches = [row for row in rows if row['game_id'] == e['canonical_event_id']]
    require(len(matches) == 1, 'NFL_SCENARIO_EVENT_MISSING')
    scheduled = matches[0]
    require(native.nfl_stats_identity(scheduled['home_team'], schedule_code=True) == native.nfl_stats_identity(e['home'])
            and native.nfl_stats_identity(scheduled['away_team'], schedule_code=True) == native.nfl_stats_identity(e['away'])
            and scheduled.get('location') == 'Home', 'NFL_SCENARIO_EVENT_CONFLICT')
    v = p['event_verification']
    require(isinstance(v, dict) and set(v) == set('schedule_row_sha256 schedule_timezone observed_at source_references'.split())
            and v['schedule_row_sha256'] == digest(scheduled) and v['schedule_timezone'] == 'America/New_York'
            and isinstance(v['source_references'], list) and 0 < len(v['source_references']) <= 8, 'NFL_SCENARIO_EVENT_VERIFICATION_MISSING')
    for reference in v['source_references']:
        require(isinstance(reference, dict) and set(reference) == {'reference', 'sha256'}
                and custody.known(reference['reference']) and isinstance(reference['sha256'], str)
                and re.fullmatch('[0-9a-f]{64}', reference['sha256']), 'NFL_SCENARIO_EVENT_VERIFICATION_MISSING')
    mapping_observed = clock(v['observed_at'], 'NFL_SCENARIO_CLOCK_UNKNOWN')
    require(mapping_observed <= instant, 'NFL_SCENARIO_FUTURE_EVIDENCE')
    try:
        scheduled_start = pd.Timestamp(scheduled['gameday'] + 'T' + scheduled['gametime']).tz_localize(v['schedule_timezone'], ambiguous='raise', nonexistent='raise').tz_convert('UTC')
    except (KeyError, ValueError, TypeError):
        raise ValueError('NFL_SCENARIO_EVENT_CONFLICT') from None
    require(scheduled_start == start, 'NFL_SCENARIO_EVENT_CONFLICT')
    from core.nfl_teams import NFL_TEAMS
    from app_core.research_replay import cell
    typed = []
    for row in rows:
        require(all(row.get(k) for k in ('game_id', 'season', 'gameday', 'home_team', 'away_team'))
                and all(k in row for k in ('home_score', 'away_score', 'result')), 'NFL_SCENARIO_HISTORY_MISSING')
        day = pd.to_datetime(row['gameday'], errors='coerce', utc=True)
        require(pd.notna(day) and all(native.nfl_stats_identity(row[k], schedule_code=True) in {name.upper() for name in NFL_TEAMS.values()} for k in ('home_team', 'away_team'))
                and row['home_team'] != row['away_team'], 'NFL_SCENARIO_HISTORY_CONFLICT')
        converted = dict(row, season=int(row['season']), **{k: float(row[k]) if row[k] else None for k in ('home_score', 'away_score', 'result')})
        values = [converted[k] for k in ('home_score', 'away_score', 'result')]
        require(all(value is None or math.isfinite(value) for value in values), 'NFL_SCENARIO_HISTORY_CONFLICT')
        if any(value is not None for value in values):
            require(all(value is not None for value in values) and converted['home_score'] >= 0 and converted['away_score'] >= 0
                    and converted['result'] == converted['home_score'] - converted['away_score']
                    and day.date() <= observed.date() and row['game_id'] != e['canonical_event_id'], 'NFL_SCENARIO_HISTORY_CONFLICT')
        typed.append({key: cell(value) for key, value in converted.items()})
    facts = dict(columns=list(rows[0]), rows=typed)
    stats = {}
    for side in ('home', 'away'):
        members = native.selected_rows(facts, e['season'], instant.isoformat(), native.nfl_stats_identity(e[side]))
        require(bool(members), 'NFL_SCENARIO_HISTORY_MISSING')
        stats[side] = native.aggregates(members)
    features = {}
    for name in evidence.FEATURES:
        if name == 'feature_diff_last5':
            features[name] = stats['home']['last5_win_pct'] - stats['away']['last5_win_pct']
        else:
            _, side, key = name.split('_', 2)
            features[name] = stats[side][native.MAPPING[key]]
    require(p['feature_order'] == list(evidence.FEATURES) and isinstance(p['features'], dict)
            and set(p['features']) == set(features), 'NFL_SCENARIO_FEATURE_ORDER')
    require(all(fact(features[k]) == fact(p['features'][k]) for k in features), 'NFL_SCENARIO_FEATURE_DERIVATION')
    available = clock(p['feature_available_at'], 'NFL_SCENARIO_CLOCK_UNKNOWN')
    require(max(observed, mapping_observed) <= available <= instant, 'NFL_SCENARIO_FEATURE_CLOCK')
    require(p['model'] == model_binding(), 'NFL_SCENARIO_MODEL_CONFLICT')
    require(p['reader'] == reader_binding(), 'NFL_SCENARIO_RUNTIME_CONFLICT')
    review = p['owner_review']
    require(isinstance(review, dict) and set(review) == set('version reviewer reviewed_at subject_sha256 status purpose attestation findings'.split())
            and review['version'] == REVIEW and review['reviewer'] == 'Robert Velarde'
            and review['status'] == 'OWNER_REVIEWED' and review['purpose'] == 'PRIVATE_RESEARCH'
            and review['attestation'] == 'I verified the original evidence and sporting assumptions for this exact scenario.', 'NFL_SCENARIO_OWNER_REVIEW_MISSING')
    require(review['subject_sha256'] == subject(p) and set(review['findings']) == set('event_mapping scenario_definition permissions clocks_and_original_custody feature_provenance model_binding'.split())
            and all(isinstance(value, dict) and set(value) == {'conclusion', 'basis'} and value['conclusion'] == 'VERIFIED'
                    and isinstance(value['basis'], str) and len(value['basis'].strip()) >= 16 for value in review['findings'].values()), 'NFL_SCENARIO_OWNER_REVIEW_CONFLICT')
    verification = clock(review['reviewed_at'], 'NFL_SCENARIO_CLOCK_UNKNOWN')
    require(available <= verification < instant, 'NFL_SCENARIO_OWNER_REVIEW_CLOCK')
    return features


def compute(packet, *, at=None):
    """Actual network-free prospective caller; never a saved-inference replay."""
    at = at or now()
    features = checked(packet, at)
    from app_core import nfl_private_routes
    require(not nfl_private_routes.selection_requested(), 'NFL_SCENARIO_ROUTE_CONFLICT')
    p = packet['payload']; s = p['scenario']; e = p['event']
    from app_core.market_probability_model import predict_market_probabilities
    row = dict(features, League='NFL', home_team=e['home'], away_team=e['away'],
               market_type='spread_home' if s['selected_team'] == e['home'] else 'spread_away',
               spread_line=s['signed_handicap'], game_start_utc=e['start'],
               ml_feature_eligible=True, stats_resolution_status='resolved')
    output = predict_market_probabilities(pd.DataFrame([row])).iloc[0]
    probability = fact(output['ml_probability']).get('value')
    require(output['ml_inference_status'] == 'success' and type(probability) is float and math.isfinite(probability)
            and .20 <= probability <= .80, 'NFL_SCENARIO_INFERENCE_FAILED')
    finished = now()
    require(clock(at, 'NFL_SCENARIO_CLOCK_UNKNOWN') <= clock(finished, 'NFL_SCENARIO_CLOCK_UNKNOWN')
            < clock(e['start'], 'NFL_SCENARIO_CLOCK_UNKNOWN'), 'NFL_SCENARIO_STARTED')
    result = seal(dict(version=RESULT, status='COMPLETE', labels=LABELS, input=deepcopy(packet),
        inference_time=at, completed_at=finished, reader=reader_binding(), model=p['model'],
        native_output={k: fact(output[k]).get('value') for k in ('ml_probability', 'ml_probability_source', 'ml_target', 'ml_projection', 'ml_residual_scale')},
        probability=probability, probability_stage='native_model_output', calibrated=False,
        probability_semantics='full_game_including_overtime_cover_conditional_on_completed_sporting_result',
        publisher_availability='UNKNOWN', source_acceptance=False, wagering_authority=False,
        qualification=False, wager_action='PASS', stake=0))
    require(len(encode(result).encode()) <= MAX_PACKET, 'NFL_SCENARIO_PACKET_LIMIT')
    return result


def read_result(result):
    """Integrity/lineage read-back only; never call numerical inference."""
    require(isinstance(result, dict) and set(result) == {'payload', 'sha256'}
            and isinstance(result['payload'], dict) and digest(result['payload']) == result['sha256'], 'NFL_SCENARIO_INTEGRITY')
    p = result['payload']
    require(set(p) == set('version status labels input inference_time completed_at reader model native_output probability probability_stage calibrated probability_semantics publisher_availability source_acceptance wagering_authority qualification wager_action stake'.split())
            and p['version'] == RESULT and p['status'] == 'COMPLETE' and p['labels'] == LABELS
            and p['probability_stage'] == 'native_model_output' and p['calibrated'] is False
            and p['source_acceptance'] is False and p['wagering_authority'] is False
            and p['qualification'] is False and p['wager_action'] == 'PASS' and type(p['stake']) is int and p['stake'] == 0
            and p['publisher_availability'] == 'UNKNOWN'
            and p['probability_semantics'] == 'full_game_including_overtime_cover_conditional_on_completed_sporting_result', 'NFL_SCENARIO_AUTHORITY_FORBIDDEN')
    require(p['reader'] == reader_binding() and p['model'] == model_binding(), 'NFL_SCENARIO_RUNTIME_CONFLICT')
    checked(p['input'], p['inference_time'])
    require(clock(p['inference_time'], 'NFL_SCENARIO_CLOCK_UNKNOWN') <= clock(p['completed_at'], 'NFL_SCENARIO_CLOCK_UNKNOWN')
            < clock(p['input']['payload']['event']['start'], 'NFL_SCENARIO_CLOCK_UNKNOWN'), 'NFL_SCENARIO_STARTED')
    output = p['native_output']
    require(isinstance(output, dict) and set(output) == set('ml_probability ml_probability_source ml_target ml_projection ml_residual_scale'.split())
            and output['ml_probability_source'] == 'score-distribution-v1:nfl' and output['ml_target'] == 'spread_cover'
            and type(p['probability']) is float and math.isfinite(p['probability']) and .20 <= p['probability'] <= .80
            and p['probability'] == output['ml_probability'], 'NFL_SCENARIO_PROBABILITY_CONFLICT')
    return deepcopy(p)


def run(packet, *, path=None):
    """Explicit application action: existing private store, caller, capture."""
    from app_core import research_replay
    research_replay.scenario_store_ready(path=path)
    result = compute(packet)
    receipt = research_replay.retain_scenario(result, path=path)
    return receipt


def reason(exc):
    code = str(exc)
    return code if re.fullmatch(r'(NFL_SCENARIO_|NFL_PRIVATE_)[A-Z_]+', code) else 'NFL_SCENARIO_UNAVAILABLE'
