"""Labelled synthetic sporting scenarios only; external transports blocked."""
from copy import deepcopy
import csv
import hashlib
import io
import json
import sqlite3
from zipfile import ZipFile

import pandas as pd
import pytest

from app_core import nfl_scenario_research as scenario, research_replay
from app_core import nfl_scoring_custody as custody, nfl_scoring_profile as profile
from app_core import nfl_native_provenance as native, nfl_inference_evidence as evidence
from scripts.benchmark_drive_history_loading import blocked_network

INFERENCE = '2026-10-06T19:25:05Z'
FINISHED = '2026-10-06T19:25:06Z'


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def sign(p):
    p['owner_review'] = dict(version=scenario.REVIEW, reviewer='Robert Velarde', reviewed_at='2026-10-06T19:25:04Z',
        subject_sha256=scenario.subject(p), status='OWNER_REVIEWED', purpose='PRIVATE_RESEARCH',
        attestation='I verified the original evidence and sporting assumptions for this exact scenario.',
        findings={k: dict(conclusion='VERIFIED', basis='SYNTHETIC labelled substantive owner finding for ' + k) for k in
                  ('event_mapping', 'scenario_definition', 'permissions', 'clocks_and_original_custody', 'feature_provenance', 'model_binding')})
    return scenario.seal(p)


def fixture(monkeypatch, *, selected='Dallas', line=-1.5):
    from app_core.nfl_scoring_acquisition import PRIMARY
    from app_core.research_replay import cell
    from app_core.research_estimate_trace import generated_time
    monkeypatch.setattr('app_core.research_estimate_trace.generated_time', lambda: INFERENCE)
    rows = []
    for index in range(4):
        rows.extend([
            dict(game_id='SYNTHETIC-DAL-' + str(index), season='2026', gameday='2026-09-' + str(10 + index),
                 gametime='13:00', home_team='DAL', away_team='HOU', home_score=str(21 + index), away_score='17', result=str(4 + index), location='Home', stadium='SYNTHETIC_PRIVATE_BODY_CANARY'),
            dict(game_id='SYNTHETIC-BUF-' + str(index), season='2026', gameday='2026-09-' + str(10 + index),
                 gametime='13:00', home_team='NE', away_team='BUF', home_score='20', away_score=str(24 + index), result=str(-4 - index), location='Home', stadium='SYNTHETIC venue')])
    target = dict(game_id='SYNTHETIC-2026-BUF-DAL', season='2026', gameday='2026-10-06', gametime='20:15',
                  home_team='DAL', away_team='BUF', home_score='', away_score='', result='', location='Home', stadium='SYNTHETIC venue')
    rows.append(target)
    mixed = dict(rows[0], game_id='SYNTHETIC-2025-DAL', season='2025')
    rows.insert(0, mixed)
    text = io.StringIO(newline=''); writer = csv.DictWriter(text, fieldnames=list(target)); writer.writeheader(); writer.writerows(rows)
    raw = text.getvalue().encode()
    receipt = dict(source_id=profile.SOURCE_ID, endpoint=profile.ENDPOINT, format='csv', evidence_label='SYNTHETIC',
        request_started_at='2026-10-06T19:25:00Z', received_at='2026-10-06T19:25:01Z', observed_at='2026-10-06T19:25:02Z',
        observation_meaning='actual_local_first_observation_of_complete_decoded_body', provider_clock_meaning='publisher_availability_unknown')
    transfer = dict(wire_bytes=len(raw), wire_sha256=hashlib.sha256(raw).hexdigest(), encoding='identity', requests=[dict(
        attempt=1, request=deepcopy(PRIMARY), started_at=receipt['request_started_at'], received_at=receipt['received_at'], status=200, location_sha256=None)])
    scoring = custody.retained_score(raw, receipt=receipt, transfer=transfer)
    converted = [{k: cell(int(v) if k == 'season' else float(v) if k in {'home_score', 'away_score', 'result'} and v else None if k in {'home_score', 'away_score', 'result'} else v) for k, v in row.items()} for row in rows if row['season'] == '2026']
    source = dict(columns=list(target), rows=converted)
    stats = {side: native.aggregates(native.selected_rows(source, 2026, INFERENCE, native.nfl_stats_identity(name))) for side, name in [('home', 'Dallas'), ('away', 'Buffalo')]}
    features = {}
    for name in evidence.FEATURES:
        if name == 'feature_diff_last5': features[name] = stats['home']['last5_win_pct'] - stats['away']['last5_win_pct']
        else:
            _, side, key = name.split('_', 2); features[name] = stats[side][native.MAPPING[key]]
    p = dict(version=scenario.VERSION, evidence_label='SYNTHETIC', scoring=scoring,
        event=dict(canonical_event_id=target['game_id'], home='Dallas', away='Buffalo', start='2026-10-07T00:15:00Z', season=2026, neutral_site=False),
        event_verification=dict(schedule_row_sha256=scenario.digest(target), schedule_timezone='America/New_York', observed_at='2026-10-06T19:25:02.500000Z',
            source_references=[dict(reference='SYNTHETIC official kickoff and named-side evidence', sha256=hashlib.sha256(b'SYNTHETIC kickoff facts').hexdigest())]),
        scenario=dict(selected_team=selected, signed_handicap=line, period='full_game', overtime=True, condition=scenario.CONDITION),
        permission=dict(reviewer='Robert Velarde', reviewed_at='2026-10-06T19:00:00Z', effective_from='2026-10-01T00:00:00Z', effective_until='2026-10-08T00:00:00Z',
            source_id=profile.SOURCE_ID, endpoint=profile.ENDPOINT, private_retention=True, private_research_use=True, credential_exclusion=True, basis='SYNTHETIC advance private-use permission, not authentic acceptance.'),
        features=features, feature_order=list(evidence.FEATURES), feature_available_at='2026-10-06T19:25:03Z', model=scenario.model_binding(), reader=scenario.reader_binding(), owner_review=None, original_inference_time=None)
    monkeypatch.setattr(scenario, 'now', lambda: FINISHED)
    return sign(p)


def store(tmp_path):
    path = tmp_path / 'SYNTHETIC-working.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE snapshots(snapshot_id TEXT PRIMARY KEY)')
        research_replay.setup(db)
    return path


@pytest.mark.parametrize('selected,line', [('Dallas', -1.5), ('Dallas', 1.5), ('Buffalo', -1.5), ('Buffalo', 1.5)])
def test_unchanged_native_probability_actual_caller_capture_export_replay(monkeypatch, tmp_path, selected, line):
    packet = fixture(monkeypatch, selected=selected, line=line); original = deepcopy(packet)
    from app_core.market_probability_model import predict_market_probabilities
    p = packet['payload']
    expected = predict_market_probabilities(pd.DataFrame([dict(p['features'], League='NFL', market_type='spread_home' if selected == 'Dallas' else 'spread_away', spread_line=line)])).iloc[0].ml_probability
    path = store(tmp_path)
    monkeypatch.setattr(scenario, 'now', iter([INFERENCE, FINISHED]).__next__)
    receipt = scenario.run(packet, path=path)
    # Static replay cannot invoke numerical inference, including through the browser.
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf', lambda *a: pytest.fail('saved replay must not infer'))
    monkeypatch.setattr(scenario, 'model_binding', lambda: p['model'])
    result = research_replay.read_scenario(receipt, path=path)
    assert scenario.read_result(result)['probability'] == expected
    assert result['payload']['input'] == original
    assert result['payload']['reader']['runtime']
    assert result['payload']['inference_time'] == INFERENCE
    assert result['payload']['input']['payload']['scoring']['receipt']['observed_at'] == '2026-10-06T19:25:02Z'
    archive, manifest = research_replay.download_scenario(receipt, path=path)
    with ZipFile(io.BytesIO(archive)) as zipped:
        assert 'package.json' not in zipped.namelist()
        assert json.loads(zipped.read('scenario-result.json')) == result
        for name, expected_hash in manifest['files'].items(): assert hashlib.sha256(zipped.read(name)).hexdigest() == expected_hash
    assert not {'quote', 'operator', 'price', 'ev', 'edge', 'original_blend', 'ui_refresh'} & set(result['payload'])
    assert result['payload']['labels'] == scenario.LABELS
    assert result['payload']['calibrated'] is result['payload']['qualification'] is False
    assert result['payload']['source_acceptance'] is result['payload']['wagering_authority'] is False
    assert result['payload']['wager_action'] == 'PASS' and result['payload']['stake'] == 0
    with pytest.raises(ValueError): research_replay.read_export(receipt['export_id'], path=path)
    with sqlite3.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError): db.execute('DELETE FROM research_replay_exports')
    assert packet == original


@pytest.mark.parametrize('attack,code', [
    ('integer', 'INTEGER_PUSH_UNVALIDATED'), ('nonfinite', 'INTEGER_PUSH_UNVALIDATED'),
    ('wrong_team', 'TARGET_CONFLICT'), ('overtime', 'TARGET_CONFLICT'), ('period', 'TARGET_CONFLICT'),
    ('orientation', 'EVENT_CONFLICT'), ('neutral', 'EVENT_CONFLICT'), ('kickoff', 'EVENT_CONFLICT'),
    ('started', 'STARTED'), ('review_missing', 'OWNER_REVIEW_MISSING'), ('review_route', 'OWNER_REVIEW_MISSING'),
    ('review_hash', 'OWNER_REVIEW_CONFLICT'), ('future_review', 'OWNER_REVIEW_CLOCK'),
    ('future_source', 'FUTURE_EVIDENCE'), ('feature_clock', 'FEATURE_CLOCK'),
    ('features', 'FEATURE_DERIVATION'), ('feature_order', 'FEATURE_ORDER'),
    ('permission', 'PERMISSION_CLOCK'), ('mapping_clock', 'FUTURE_EVIDENCE'), ('clock_unknown', 'CLOCK_UNKNOWN'),
    ('model', 'MODEL_CONFLICT'), ('runtime', 'RUNTIME_CONFLICT')])
def test_rejections_precede_numerical_calls(monkeypatch, attack, code):
    packet = fixture(monkeypatch); p = packet['payload']
    if attack == 'integer': p['scenario']['signed_handicap'] = 3
    elif attack == 'nonfinite': p['scenario']['signed_handicap'] = 'nan'
    elif attack == 'wrong_team': p['scenario']['selected_team'] = 'Houston'
    elif attack == 'overtime': p['scenario']['overtime'] = False
    elif attack == 'period': p['scenario']['period'] = 'regulation'
    elif attack == 'orientation': p['event']['home'], p['event']['away'] = p['event']['away'], p['event']['home']
    elif attack == 'neutral': p['event']['neutral_site'] = True
    elif attack == 'kickoff': p['event']['start'] = '2026-10-07T01:15:00Z'
    elif attack == 'started': p['event']['start'] = '2026-10-06T19:25:05Z'
    elif attack == 'future_source': p['scoring']['receipt']['observed_at'] = '2026-10-06T19:26:00Z'
    elif attack == 'feature_clock': p['feature_available_at'] = '2026-10-06T19:25:01Z'
    elif attack == 'features': p['features']['feature_home_ppg'] += 1
    elif attack == 'feature_order': p['feature_order'] = list(reversed(p['feature_order']))
    elif attack == 'permission': p['permission']['reviewed_at'] = '2026-10-06T19:25:03Z'
    elif attack == 'mapping_clock': p['event_verification']['observed_at'] = '2026-10-06T19:26:00Z'
    elif attack == 'clock_unknown': p['feature_available_at'] = '2026-10-06T19:25:03'
    elif attack == 'model': p['model']['predictor_id'] = 'borrowed-other-model'
    elif attack == 'runtime': p['reader']['runtime']['python'] = 'UNREVIEWED runtime'
    packet = sign(p)
    if attack == 'review_missing': p['owner_review'] = None
    elif attack == 'review_route': p['owner_review']['version'] = 'nfl-owner-scoring-verification-v3'
    elif attack == 'review_hash': p['owner_review']['subject_sha256'] = '0' * 64
    elif attack == 'future_review': p['owner_review']['reviewed_at'] = '2026-10-06T19:25:06Z'
    packet = scenario.seal(p)
    expected = deepcopy(p['model']) if attack != 'model' else None
    if expected is not None: monkeypatch.setattr(scenario, 'model_binding', lambda: expected)
    calls = []
    monkeypatch.setattr('app_core.market_probability_model._normal_cdf', lambda *a: calls.append(a) or pytest.fail('rejection must precede numerical inference'))
    with pytest.raises(ValueError, match='NFL_SCENARIO_' + code): scenario.compute(packet, at=INFERENCE)
    assert calls == []


def test_tampered_and_rehashed_bodies_preserve_owner_subject(monkeypatch):
    packet = fixture(monkeypatch); p = packet['payload']
    p['scoring']['body_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='CUSTODY_CORRUPT'): scenario.checked(scenario.seal(p), INFERENCE)
    packet = fixture(monkeypatch); p = packet['payload']; raw, _ = custody.read_score(p['scoring'])
    changed = raw.replace(b'SYNTHETIC-DAL-0', b'SYNTHETIC-DAL-X')
    t = deepcopy(p['scoring']['transfer']); t.update(wire_bytes=len(changed), wire_sha256=hashlib.sha256(changed).hexdigest())
    p['scoring'] = custody.retained_score(changed, receipt=p['scoring']['receipt'], transfer=t)
    with pytest.raises(ValueError, match='OWNER_REVIEW_CONFLICT'): scenario.checked(scenario.seal(p), INFERENCE)


def test_old_offer_routes_reject_scenario_and_scenario_rejects_offer(monkeypatch):
    from app_core import nfl_private_routes, nfl_scoring_research, nfl_owner_research
    packet = fixture(monkeypatch)
    for reader in (nfl_private_routes, nfl_scoring_research, nfl_owner_research):
        with pytest.raises(ValueError): reader.load(scenario.encode(packet).encode())
    changed = deepcopy(packet['payload']); changed['version'] = nfl_scoring_research.VERSION
    with pytest.raises(ValueError, match='SCHEMA'): scenario.load(scenario.encode(scenario.seal(changed)).encode())
    with pytest.raises(ValueError, match='SYNTHETIC_UPLOAD'): scenario.load(scenario.encode(packet).encode(), owner_upload=True)


def test_missing_or_unprotected_store_stops_before_math(monkeypatch, tmp_path):
    packet = fixture(monkeypatch)
    monkeypatch.setattr(scenario, 'compute', lambda *a: pytest.fail('store must be ready before inference'))
    path = tmp_path / 'missing.sqlite3'
    with pytest.raises(ValueError, match='STORE_UNAVAILABLE'): scenario.run(packet, path=path)
    assert not path.exists()
    path = store(tmp_path)
    with sqlite3.connect(path) as db: db.execute('DROP TRIGGER immutable_research_replay_exports_UPDATE')
    with pytest.raises(ValueError, match='STORE_PROTECTION_MISSING'): scenario.run(packet, path=path)


def test_store_without_unique_identity_stops_before_math(monkeypatch, tmp_path):
    packet = fixture(monkeypatch)
    monkeypatch.setattr(scenario, 'compute', lambda *a: pytest.fail('malformed store must stop before inference'))
    path = tmp_path / 'SYNTHETIC-nonunique-store.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE snapshots(snapshot_id TEXT PRIMARY KEY)')
        db.execute('CREATE TABLE research_source_intakes(reference TEXT, payload TEXT NOT NULL, payload_hash TEXT NOT NULL)')
        research_replay.setup(db)
    with pytest.raises(ValueError, match='STORE_UNAVAILABLE'): scenario.run(packet, path=path)


def scenario_app():
    from app.ui.nfl_private_research import render_scenario
    render_scenario()


def test_actual_browser_private_labels_and_no_automatic_inference(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    packet = fixture(monkeypatch); path = store(tmp_path)
    monkeypatch.setattr(scenario, 'now', iter([INFERENCE, FINISHED]).__next__)
    receipt = scenario.run(packet, path=path)
    monkeypatch.setattr(research_replay, '_scenario_path', lambda _: path)
    monkeypatch.setattr(scenario, 'compute', lambda *a, **k: pytest.fail('render/replay must not infer'))
    app = AppTest.from_function(scenario_app).run()
    assert not app.exception and len(app.metric) == 0
    app.session_state['nfl_scenario_enabled'] = True
    app.session_state['nfl_scenario_staged'] = packet
    app.session_state['nfl_scenario_receipt'] = receipt
    app.run()
    assert not app.exception
    assert app.metric[0].label == 'Uncalibrated NFL model estimate'
    text = ' '.join(str(x.value) for x in list(app.markdown) + list(app.caption))
    assert 'OWNER_REVIEWED / PRIVATE_RESEARCH / USER-SPECIFIED SCENARIO' in text
    assert 'PASS' in text and 'zero stake' in text
    assert 'SYNTHETIC_PRIVATE_BODY_CANARY' not in text and 'bytes_base64' not in text


def test_owner_gate_blocks_scenario_panel_without_token(monkeypatch):
    from streamlit.testing.v1 import AppTest
    def app():
        from app.ui.publish_panel import render_publish_panel
        render_publish_panel(None, None)
    monkeypatch.setenv('PARLAYPICKER_PUBLISH_TOKEN', 'SYNTHETIC-token-123456789')
    monkeypatch.setattr('app.ui.nfl_private_research.render_scenario', lambda: pytest.fail('unauthorized panel'))
    tested = AppTest.from_function(app).run()
    assert not tested.exception and len(tested.metric) == 0


def test_browser_explicit_action_uses_actual_caller_capture_and_replay(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    packet = fixture(monkeypatch); path = store(tmp_path)
    monkeypatch.setattr(research_replay, '_scenario_path', lambda _: path)
    app = AppTest.from_function(scenario_app).run()
    assert app.button(key='nfl_scenario_run').disabled
    app.session_state['nfl_scenario_enabled'] = True
    app.session_state['nfl_scenario_staged'] = packet
    app.run()
    assert app.button(key='nfl_scenario_run').disabled
    app.checkbox(key='nfl_scenario_confirmed').check().run()
    monkeypatch.setattr(scenario, 'now', iter([INFERENCE, FINISHED]).__next__)
    app.button(key='nfl_scenario_run').click().run()
    assert not app.exception and app.metric[0].label == 'Uncalibrated NFL model estimate'
    result = research_replay.read_scenario(app.session_state['nfl_scenario_receipt'], path=path)
    assert result['payload']['inference_time'] == INFERENCE
    assert result['payload']['input'] == packet


@pytest.mark.parametrize('attack', ['calibrated', 'qualification', 'source_acceptance', 'wagering_authority', 'stake', 'wager_action', 'price', 'probability'])
def test_rehashed_result_cannot_promote_authority_or_market_fields(monkeypatch, attack):
    packet = fixture(monkeypatch)
    p = scenario.compute(packet, at=INFERENCE)['payload']
    if attack in {'calibrated', 'qualification', 'source_acceptance', 'wagering_authority'}: p[attack] = True
    elif attack == 'stake': p['stake'] = 1
    elif attack == 'wager_action': p['wager_action'] = 'APPROVED'
    elif attack == 'price': p['price'] = -110
    else: p['probability'] += .01
    with pytest.raises(ValueError): scenario.read_result(scenario.seal(p))


def test_active_offer_context_cannot_borrow_scenario_inference(monkeypatch):
    packet = fixture(monkeypatch)
    monkeypatch.setattr('app_core.nfl_private_routes.selection_requested', lambda: True)
    monkeypatch.setattr('app_core.market_probability_model.predict_market_probabilities', lambda *a: pytest.fail('no numerical call'))
    # Freezing the unchanged synthetic binding isolates dispatch rejection.
    monkeypatch.setattr(scenario, 'model_binding', lambda: packet['payload']['model'])
    with pytest.raises(ValueError, match='NFL_SCENARIO_ROUTE_CONFLICT'): scenario.compute(packet, at=INFERENCE)


def test_equivalent_timezone_instant_and_missing_history(monkeypatch):
    packet = fixture(monkeypatch); p = packet['payload']
    p['event']['start'] = '2026-10-06T20:15:00-04:00'
    assert scenario.checked(sign(p), INFERENCE) == p['features']
    raw, _ = custody.read_score(p['scoring'])
    reader = csv.DictReader(io.StringIO(raw.decode())); rows = [r for r in reader if r['game_id'].startswith('SYNTHETIC-2026') or r['season'] == '2025']
    text = io.StringIO(newline=''); writer = csv.DictWriter(text, fieldnames=reader.fieldnames); writer.writeheader(); writer.writerows(rows)
    changed = text.getvalue().encode(); t = deepcopy(p['scoring']['transfer'])
    t.update(wire_bytes=len(changed), wire_sha256=hashlib.sha256(changed).hexdigest())
    p['scoring'] = custody.retained_score(changed, receipt=p['scoring']['receipt'], transfer=t)
    with pytest.raises(ValueError, match='HISTORY_MISSING'): scenario.checked(sign(p), INFERENCE)
