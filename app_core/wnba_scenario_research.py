"""Explicit WNBA sporting assumptions. No offers, collection or approval."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re

import pandas as pd

from app_core import wnba_scenario_custody as custody

VERSION = 'wnba-owner-sporting-scenario-v1'
REVIEW = 'wnba-owner-sporting-scenario-review-v1'
RESULT = 'wnba-owner-sporting-scenario-result-v1'
LABELS = ['OWNER_REVIEWED', 'PRIVATE_RESEARCH', 'USER-SPECIFIED SCENARIO']
FEATURES = ('feature_home_ppg', 'feature_away_ppg', 'feature_home_oppg', 'feature_away_oppg',
            'feature_home_games_played', 'feature_away_games_played', 'feature_home_win_pct',
            'feature_away_win_pct', 'feature_diff_last5')
FINDINGS = ('event_mapping', 'scenario_definition', 'permissions', 'clocks_and_original_custody',
            'scoring_semantics_and_complete_histories', 'feature_provenance', 'model_binding')
FIELDS = set('version evidence_label scoring event event_document event_verification scenario permissions projection features feature_order feature_available_at model reader owner_review original_inference_time'.split())
ROOT = Path(__file__).resolve().parents[1]
require, digest, clock, encode = custody.require, custody.digest, custody.clock, custody.encode
FRESHNESS = 21600  # Existing WNBA standings cache TTL; never renewed by review.


def now():
    return datetime.now(timezone.utc).isoformat()


def seal(value):
    return dict(payload=deepcopy(value), sha256=digest(value))


def reader_binding():
    from app_core.nfl_inference_evidence import runtime
    paths = ('app_core/wnba_scenario_custody.py', 'app_core/wnba_scenario_research.py',
             'app_core/wnba_scenario_replay.py', 'app_core/market_probability_model.py',
             'app_core/nfl_inference_evidence.py', 'core/team_mapper.py', 'app_core/research_replay.py')
    return dict(runtime=runtime(), sources={p: hashlib.sha256((ROOT/p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths})


def model_binding():
    from app_core import market_probability_model as model
    from app_core.nfl_inference_evidence import predictor_callables
    raw = (ROOT/'app_core/market_probability_model.py').read_bytes().replace(b'\r\n', b'\n')
    return dict(predictor='score-distribution-v1:wnba', artifact_sha256=hashlib.sha256(raw).hexdigest(),
                configuration=dict(model._LEAGUE_PARAMS['WNBA']), feature_order=list(FEATURES),
                callable_sha256=predictor_callables(), fitted_artifact='NOT_APPLICABLE: unchanged native fixed score calculator',
                form_semantics='season_win_percentage_proxy_not_last_five_games')


def load(raw, *, owner_upload=False):
    require(isinstance(raw, bytes) and 0 < len(raw) <= custody.MAX_PACKET, 'PACKET_LIMIT')
    try:
        packet = custody.json.loads(raw, object_pairs_hook=custody.unique)
    except (ValueError, TypeError):
        raise ValueError('WNBA_SCENARIO_SCHEMA') from None
    require(isinstance(packet, dict) and set(packet) == {'payload', 'sha256'}
            and isinstance(packet['payload'], dict), 'SCHEMA')
    p = packet['payload']
    require(digest(p) == packet['sha256'], 'INTEGRITY')
    require(set(p) == FIELDS and p['version'] == VERSION and p['evidence_label'] in {'SYNTHETIC', 'RETAINED'}
            and p['original_inference_time'] is None, 'SCHEMA')
    require(not owner_upload or p['evidence_label'] == 'RETAINED', 'SYNTHETIC_UPLOAD')
    require(isinstance(p['scoring'], list) and 0 < len(p['scoring']) < custody.MAX_OBJECTS, 'CUSTODY_MISSING')
    objects = p['scoring'] + [p['event_document']]
    require(sum(len(custody.read(o)[0]) for o in objects) <= custody.MAX_AGGREGATE, 'AGGREGATE_LIMIT')
    require(all(o['receipt']['evidence_label'] == p['evidence_label'] for o in objects), 'SCHEMA')
    return packet


def subject(p):
    return digest({k: v for k, v in p.items() if k != 'owner_review'})


def _team(value):
    from core.team_mapper import WNBA_EXACT_MAP
    require(isinstance(value, dict) and type(value.get('id')) is int and value['id'] > 0
            and isinstance(value.get('full_name'), str) and value['full_name'].lower() in WNBA_EXACT_MAP,
            'TEAM_IDENTITY')
    return value['id'], value['full_name']


def projection(p):
    """Complete retained page chain; filter in a separate bound projection."""
    e = p['event']; rows = []; locations = []; cursor = None; seen_cursors = set(); identities = {}
    for page, obj in enumerate(p['scoring']):
        _, body = custody.read(obj); r = obj['receipt']
        require(r['source_id'] == custody.SOURCE and r['parameters'] == dict(seasons=[e['season']], per_page=100, cursor=cursor), 'PAGE_CHAIN')
        if page:
            require(clock(p['scoring'][page-1]['receipt']['received_at']) <= clock(r['request_started_at']), 'PAGE_CHAIN_CLOCK')
        require(cursor not in seen_cursors, 'PAGE_CHAIN'); seen_cursors.add(cursor)
        for index, row in enumerate(body['data']):
            require(isinstance(row, dict) and type(row.get('id')) is int and row['id'] > 0
                    and row['id'] not in identities and type(row.get('season')) is int and row['season'] == e['season'], 'HISTORY_IDENTITY')
            identities[row['id']] = row
            home, away = _team(row.get('home_team')), _team(row.get('visitor_team'))
            require(home[0] != away[0] and home[1] != away[1], 'HISTORY_IDENTITY')
            # The same ID/name must not be substituted between original rows.
            for tid, name in (home, away):
                require(all((tid != old_id or name == old_name) and (name != old_name or tid == old_id)
                            for old_id, old_name in locations), 'TEAM_IDENTITY')
                locations.append((tid, name))
            date = clock(row.get('date'))
            require(type(row.get('postseason')) is bool and isinstance(row.get('status'), str), 'HISTORY_STATUS_UNKNOWN')
            state = row.get('status_state')
            require(state in {'scheduled', 'in_progress', 'final', 'postponed', 'canceled', 'delayed', 'suspended', 'abandoned'}, 'HISTORY_STATUS_UNKNOWN')
            if state == 'final':
                require(type(row.get('period')) is int and row['period'] >= 4
                        and all(type(row.get(k)) is int and row[k] >= 0 for k in ('home_score', 'away_score'))
                        and row['home_score'] != row['away_score'] and date < clock(r['observed_at']), 'HISTORY_CONFLICT')
            rows.append((row, page, index))
        cursor = body['meta'].get('next_cursor')
        require((page == len(p['scoring'])-1) == (cursor is None), 'PAGE_CHAIN_INCOMPLETE')
    target = [row for row, _, _ in rows if row['id'] == e['provider_event_id']]
    require(len(target) == 1, 'EVENT_MISSING')
    t = target[0]
    require(t['home_team']['full_name'] == e['home'] and t['visitor_team']['full_name'] == e['away']
            and clock(t['date']) == clock(e['start']) and t['status_state'] == 'scheduled', 'EVENT_CONFLICT')
    require(e['canonical_event_id'] == 'WNBA:' + str(e['season']) + ':' + e['away'] + ':' + e['home'] + ':' + clock(e['start']).isoformat(), 'EVENT_CONFLICT')
    features = {}; selections = {}
    for side in ('home', 'away'):
        members = [(row, page, index) for row, page, index in rows if not row['postseason']
                   and row['status_state'] == 'final' and e[side] in {row['home_team']['full_name'], row['visitor_team']['full_name']}]
        require(bool(members), 'HISTORY_MISSING')  # Native WNBA has no football-style count floor.
        points = []; allowed = []; wins = 0
        for row, _, _ in members:
            home = row['home_team']['full_name'] == e[side]
            pf, pa = (row['home_score'], row['away_score']) if home else (row['away_score'], row['home_score'])
            points.append(pf); allowed.append(pa); wins += pf > pa
        features['feature_' + side + '_ppg'] = sum(points)/len(points)
        features['feature_' + side + '_oppg'] = sum(allowed)/len(allowed)
        features['feature_' + side + '_games_played'] = len(points)
        features['feature_' + side + '_win_pct'] = wins/len(points)
        selections[side] = [dict(body_sha256=p['scoring'][page]['body_sha256'], row_index=index, game_id=row['id'], row_sha256=digest(row)) for row, page, index in members]
    features['feature_diff_last5'] = features['feature_home_win_pct'] - features['feature_away_win_pct']
    view = dict(version='wnba-season-scoring-projection-v1', selection='complete_season_pages_regular_completed_source_order',
                publisher_availability='UNKNOWN', form_semantics='season_win_percentage_proxy_not_last_five_games',
                bodies=[dict(body_sha256=o['body_sha256'], receipt_sha256=digest(o['receipt'])) for o in p['scoring']],
                event_row_sha256=digest(t), selected_rows=selections, features=features)
    return view


def checked(packet, at):
    p = load(encode(packet).encode())['payload']; instant = clock(at); e = p['event']; s = p['scenario']
    require(isinstance(e, dict) and set(e) == set('canonical_event_id provider_event_id home away start season neutral_site venue'.split())
            and type(e['provider_event_id']) is int and type(e['season']) is int and e['neutral_site'] is False
            and isinstance(e['venue'], str) and len(e['venue'].strip()) >= 4 and e['home'] != e['away'], 'EVENT_CONFLICT')
    require(instant < clock(e['start']), 'STARTED')
    require(isinstance(s, dict) and set(s) == set('selected_team signed_handicap period overtime condition'.split())
            and s['selected_team'] in {e['home'], e['away']} and s['period'] == 'full_game' and s['overtime'] is True
            and s['condition'] == 'completed_sporting_result', 'TARGET_CONFLICT')
    line = s['signed_handicap']
    require(type(line) in {int, float} and math.isfinite(line) and abs(line % 1) == .5, 'INTEGER_PUSH_UNVALIDATED')
    objects = p['scoring'] + [p['event_document']]
    require(all(clock(o['receipt']['observed_at']) <= instant for o in objects), 'FUTURE_EVIDENCE')
    require(all(0 <= (instant-clock(o['receipt']['observed_at'])).total_seconds() <= FRESHNESS for o in objects), 'STALE_EVIDENCE')
    permissions = p['permissions']
    require(isinstance(permissions, list) and len(permissions) == 2, 'PERMISSION_MISSING')
    for source, permission in zip((p['scoring'][0], p['event_document']), permissions):
        r = source['receipt']
        require(isinstance(permission, dict) and set(permission) == set('reviewer reviewed_at effective_from effective_until source_id endpoint private_retention private_research_use credential_exclusion basis'.split())
                and permission['reviewer'] == 'Robert Velarde' and permission['source_id'] == r['source_id'] and permission['endpoint'] == r['endpoint']
                and all(permission[k] is True for k in ('private_retention', 'private_research_use', 'credential_exclusion'))
                and isinstance(permission['basis'], str) and len(permission['basis'].strip()) >= 16, 'PERMISSION_MISSING')
        group = p['scoring'] if source is p['scoring'][0] else [p['event_document']]
        for obj in group:
            r = obj['receipt']; request = clock(r['request_started_at'])
            require(clock(permission['reviewed_at']) <= request and clock(permission['effective_from']) <= request
                    <= clock(r['observed_at']) <= instant < clock(permission['effective_until']), 'PERMISSION_CLOCK')
    _, text = custody.read(p['event_document'])
    require(isinstance(text, str), 'EVENT_DOCUMENT_MISSING')
    v = p['event_verification']
    require(isinstance(v, dict) and set(v) == set('version body_sha256 receipt_sha256 excerpts event_sha256 method reasoning prepared_at'.split())
            and v['version'] == 'wnba-owner-event-interpretation-v1' and v['body_sha256'] == p['event_document']['body_sha256']
            and v['receipt_sha256'] == digest(p['event_document']['receipt']) and v['event_sha256'] == digest(e)
            and v['method'] == 'owner_reading_exact_utf8_character_excerpts_v1'
            and isinstance(v['reasoning'], str) and len(v['reasoning'].strip()) >= 32
            and isinstance(v['excerpts'], list) and 0 < len(v['excerpts']) <= 8, 'EVENT_VERIFICATION_MISSING')
    joined = ''
    for excerpt in v['excerpts']:
        require(isinstance(excerpt, dict) and set(excerpt) == {'start', 'end', 'text'}
                and type(excerpt['start']) is int and type(excerpt['end']) is int
                and 0 <= excerpt['start'] < excerpt['end'] <= len(text)
                and 0 < len(excerpt['text']) <= 4096 and text[excerpt['start']:excerpt['end']] == excerpt['text'], 'EVENT_EXCERPT_CONFLICT')
        joined += excerpt['text']
    require(all(value in joined for value in (e['home'], e['away'], e['venue'])), 'EVENT_INTERPRETATION_UNSUPPORTED')
    require(clock(p['event_document']['receipt']['observed_at']) <= clock(v['prepared_at']) <= instant, 'FUTURE_EVIDENCE')
    expected = projection(p)
    require(all(expected['features'][name] > 0 for name in FEATURES[:4]), 'SCORING_INPUT_UNAVAILABLE')
    require(p['projection'] == expected, 'PROJECTION_CONFLICT')
    require(p['feature_order'] == list(FEATURES) and isinstance(p['features'], dict) and p['features'] == expected['features']
            and all(type(v) in {int, float} and math.isfinite(v) for v in p['features'].values()), 'FEATURE_DERIVATION')
    available = clock(p['feature_available_at'])
    require(max([clock(o['receipt']['observed_at']) for o in objects]+[clock(v['prepared_at'])]) <= available <= instant, 'FEATURE_CLOCK')
    require(p['model'] == model_binding(), 'MODEL_CONFLICT')
    require(p['reader'] == reader_binding(), 'RUNTIME_CONFLICT')
    review = p['owner_review']
    require(isinstance(review, dict) and set(review) == set('version reviewer reviewed_at subject_sha256 status purpose attestation findings'.split())
            and review['version'] == REVIEW and review['reviewer'] == 'Robert Velarde' and review['status'] == 'OWNER_REVIEWED'
            and review['purpose'] == 'PRIVATE_RESEARCH'
            and review['attestation'] == 'I verified the original evidence and sporting assumptions for this exact WNBA scenario.', 'OWNER_REVIEW_MISSING')
    require(review['subject_sha256'] == subject(p) and isinstance(review['findings'], dict) and set(review['findings']) == set(FINDINGS)
            and all(isinstance(f, dict) and set(f) == {'conclusion', 'basis'} and f['conclusion'] == 'VERIFIED'
                    and isinstance(f['basis'], str) and len(f['basis'].strip()) >= 16 for f in review['findings'].values()), 'OWNER_REVIEW_CONFLICT')
    require(available <= clock(review['reviewed_at']) < instant, 'OWNER_REVIEW_CLOCK')
    return deepcopy(expected['features'])


def compute(packet, *, at=None):
    at = at or now(); features = checked(packet, at); p = packet['payload']; e = p['event']; s = p['scenario']
    from app_core import nfl_private_routes, ncaaf_pipeline_evidence, nhl_puck_line_evidence
    require(not any(m.selection_requested() for m in (nfl_private_routes, ncaaf_pipeline_evidence, nhl_puck_line_evidence)), 'ROUTE_CONFLICT')
    from app_core.market_probability_model import predict_market_probabilities
    row = dict(features, League='WNBA', home_team=e['home'], away_team=e['away'],
               market_type='spread_home' if s['selected_team'] == e['home'] else 'spread_away',
               spread_line=s['signed_handicap'], game_start_utc=e['start'], ml_feature_eligible=True, stats_resolution_status='resolved')
    output = predict_market_probabilities(pd.DataFrame([row])).iloc[0]
    probability = float(output['ml_probability'])
    require(output['ml_inference_status'] == 'success' and math.isfinite(probability) and .20 <= probability <= .80, 'INFERENCE_FAILED')
    finished = now(); require(clock(at) <= clock(finished) < clock(e['start']), 'STARTED')
    result = seal(dict(version=RESULT, status='COMPLETE', labels=LABELS, input=deepcopy(packet), inference_time=at, completed_at=finished,
        probability=probability, probability_stage='native_model_output', model=p['model'], reader=p['reader'],
        native_output={k: str(output[k]) if k in {'ml_probability_source', 'ml_target'} else float(output[k]) for k in
                       ('ml_probability', 'ml_probability_source', 'ml_target', 'ml_projection', 'ml_residual_scale')},
        probability_semantics='full_game_including_overtime_cover_conditional_on_completed_sporting_result',
        publisher_availability='UNKNOWN', calibrated=False, qualification=False, source_acceptance=False, wagering_authority=False, wager_action='PASS', stake=0))
    require(len(encode(result).encode()) <= custody.MAX_PACKET, 'PACKET_LIMIT')
    return result


def read_result(result):
    require(isinstance(result, dict) and set(result) == {'payload', 'sha256'} and isinstance(result['payload'], dict)
            and digest(result['payload']) == result['sha256'] and len(encode(result).encode()) <= custody.MAX_PACKET, 'INTEGRITY')
    p = result['payload']
    require(set(p) == set('version status labels input inference_time completed_at probability probability_stage model reader native_output probability_semantics publisher_availability calibrated qualification source_acceptance wagering_authority wager_action stake'.split())
            and p['version'] == RESULT and p['status'] == 'COMPLETE' and p['labels'] == LABELS
            and p['probability_stage'] == 'native_model_output' and p['publisher_availability'] == 'UNKNOWN'
            and p['probability_semantics'] == 'full_game_including_overtime_cover_conditional_on_completed_sporting_result'
            and all(p[k] is False for k in ('calibrated', 'qualification', 'source_acceptance', 'wagering_authority'))
            and p['wager_action'] == 'PASS' and type(p['stake']) is int and p['stake'] == 0, 'AUTHORITY_FORBIDDEN')
    checked(p['input'], p['inference_time'])  # Static validation; never numerical replay.
    require(p['model'] == p['input']['payload']['model'] and p['reader'] == p['input']['payload']['reader'], 'RUNTIME_CONFLICT')
    require(clock(p['inference_time']) <= clock(p['completed_at']) < clock(p['input']['payload']['event']['start']), 'STARTED')
    output = p['native_output']
    require(isinstance(output, dict) and set(output) == set('ml_probability ml_probability_source ml_target ml_projection ml_residual_scale'.split())
            and output['ml_probability_source'] == 'score-distribution-v1:wnba' and output['ml_target'] == 'spread_cover'
            and type(p['probability']) is float and math.isfinite(p['probability']) and .20 <= p['probability'] <= .80
            and p['probability'] == output['ml_probability'], 'PROBABILITY_CONFLICT')
    return deepcopy(p)


def run(packet, *, path=None):
    from app_core import wnba_scenario_replay as replay
    replay.ready(path=path)
    return replay.retain(compute(packet), path=path)


def reason(exc):
    code = str(exc)
    return code if re.fullmatch(r'WNBA_SCENARIO_[A-Z_]+', code) else 'WNBA_SCENARIO_UNAVAILABLE'
