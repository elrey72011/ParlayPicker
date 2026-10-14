"""Labelled source-shaped synthetic evidence; no authentic inference/acquisition."""
import base64
from copy import deepcopy
import hashlib
import io
import json
import math
import sqlite3
from zipfile import ZipFile

import pandas as pd
import pytest

from app_core import wnba_scenario_research as scenario, wnba_scenario_custody as custody
from app_core import wnba_scenario_replay as replay, research_replay
from scripts.benchmark_drive_history_loading import blocked_network

AT = '2026-10-14T12:00:05Z'
FINISHED = '2026-10-14T12:00:06Z'
HOME, AWAY = 'New York Liberty', 'Las Vegas Aces'


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def receipt(source=custody.SOURCE, endpoint=custody.ENDPOINT, format='json'):
    return dict(source_id=source, endpoint=endpoint, parameters=dict(seasons=[2026], per_page=100, cursor=None) if source == custody.SOURCE else {},
        format=format, evidence_label='SYNTHETIC', request_started_at='2026-10-14T12:00:00Z', received_at='2026-10-14T12:00:01Z', observed_at='2026-10-14T12:00:02Z',
        observation_meaning='actual_local_first_observation_of_complete_decoded_body', publisher_availability='UNKNOWN', status=200)


def row(id, home, away, home_score, away_score, date, final=True, postseason=False):
    def team(name):
        ids = {HOME: 1, AWAY: 2, 'Seattle Storm': 3, 'Phoenix Mercury': 4}
        return dict(id=ids[name], full_name=name, city=name.split()[0], name=name.split()[-1], abbreviation='SYN', conference='SYNTHETIC')
    return dict(id=id, date=date, season=2026, postseason=postseason, status='post' if final else 'pre',
        status_state='final' if final else 'scheduled', period=4 if final else 0, time='0.0' if final else '',
        home_team=team(home), visitor_team=team(away), home_score=home_score, away_score=away_score)


def sign(p, at='2026-10-14T12:00:04Z'):
    p['owner_review'] = dict(version=scenario.REVIEW, reviewer='Robert Velarde', reviewed_at=at, subject_sha256=scenario.subject(p),
        status='OWNER_REVIEWED', purpose='PRIVATE_RESEARCH', attestation='I verified the original evidence and sporting assumptions for this exact WNBA scenario.',
        findings={k: dict(conclusion='VERIFIED', basis='SYNTHETIC substantive finding, not authentic owner review: '+k) for k in scenario.FINDINGS})
    return scenario.seal(p)


def fixture(monkeypatch, selected=HOME, line=1.5):
    rows = []
    for i in range(4):
        rows.extend([row(i+1, HOME, 'Seattle Storm', 80+i, 75, f'2026-09-{10+i}T20:00:00Z'),
                     row(i+101, 'Phoenix Mercury', AWAY, 90, 84+i, f'2026-09-{10+i}T22:00:00Z')])
    rows[0]['period'] = 5  # Final overtime points, not a regulation-only result.
    rows.append(row(999, HOME, AWAY, 0, 0, '2026-10-15T00:00:00Z', final=False, postseason=True))
    body = custody.retain(custody.encode(dict(data=rows, meta=dict(per_page=100))).encode(), receipt())
    text = (f'SYNTHETIC official event page: {AWAY} away at {HOME}, at SYNTHETIC home arena, '
            'October 14, 2026, 8:00 p.m. Eastern (2026-10-15T00:00:00Z). This is a non-neutral home venue. '
            'SYNTHETIC_PRIVATE_BODY_CANARY')
    doc = custody.retain(text.encode(), receipt('official-wnba-event-document-v1', 'https://www.wnba.com/game/SYNTHETIC', 'utf8-text'))
    e = dict(canonical_event_id='WNBA:2026:'+AWAY+':'+HOME+':2026-10-15T00:00:00+00:00', provider_event_id=999,
             home=HOME, away=AWAY, start='2026-10-15T00:00:00Z', season=2026, neutral_site=False, venue='SYNTHETIC home arena')
    permissions = [dict(reviewer='Robert Velarde', reviewed_at='2026-10-14T11:00:00Z', effective_from='2026-10-14T00:00:00Z', effective_until='2026-10-16T00:00:00Z',
        source_id=o['receipt']['source_id'], endpoint=o['receipt']['endpoint'], private_retention=True, private_research_use=True, credential_exclusion=True,
        basis='SYNTHETIC advance terms review for limited private retention and research.') for o in (body, doc)]
    p = dict(version=scenario.VERSION, evidence_label='SYNTHETIC', scoring=[body], event=e, event_document=doc,
        event_verification=dict(version='wnba-owner-event-interpretation-v1', body_sha256=doc['body_sha256'], receipt_sha256=scenario.digest(doc['receipt']),
            excerpts=[dict(start=0, end=len(text), text=text)], event_sha256=scenario.digest(e), method='owner_reading_exact_utf8_character_excerpts_v1',
            reasoning='SYNTHETIC owner interpretation of named away/home, exact timezone, venue and non-neutral designation.', prepared_at='2026-10-14T12:00:02.500000Z'),
        scenario=dict(selected_team=selected, signed_handicap=line, period='full_game', overtime=True, condition='completed_sporting_result'), permissions=permissions,
        projection=None, features=None, feature_order=list(scenario.FEATURES), feature_available_at='2026-10-14T12:00:03Z', model=scenario.model_binding(), reader=scenario.reader_binding(),
        owner_review=None, original_inference_time=None)
    p['projection'] = scenario.projection(p); p['features'] = p['projection']['features']
    frozen_model = deepcopy(p['model'])
    monkeypatch.setattr(scenario, 'model_binding', lambda: frozen_model)
    monkeypatch.setattr(scenario, 'now', lambda: FINISHED)
    return sign(p)


def store(tmp_path):
    path = tmp_path/'SYNTHETIC.sqlite3'
    with sqlite3.connect(path) as db:
        research_replay.setup(db)
    return path


@pytest.mark.parametrize('selected,line', [(HOME, 1.5), (HOME, -1.5), (AWAY, 1.5), (AWAY, -1.5)])
def test_actual_native_caller_capture_export_static_replay_display(monkeypatch, tmp_path, selected, line):
    packet = fixture(monkeypatch, selected, line); p = packet['payload']; original = deepcopy(packet)
    from app_core import market_probability_model as native
    expected = native.predict_market_probabilities(pd.DataFrame([dict(p['features'], League='WNBA', market_type='spread_home' if selected == HOME else 'spread_away', spread_line=line)])).iloc[0].ml_probability
    # Independently express the fixed native parameters; no new predictor fit.
    f = p['features']; margin = .5*(f['feature_home_ppg']+f['feature_away_oppg'])-.5*(f['feature_away_ppg']+f['feature_home_oppg'])+2.2+11*f['feature_diff_last5']
    z = (margin+line if selected == HOME else line-margin)/12.5
    assert expected == max(.2, min(.8, .5+.65*(.5*(1+math.erf(z/math.sqrt(2)))-.5)))
    db = store(tmp_path); monkeypatch.setattr(scenario, 'now', iter([AT, FINISHED]).__next__)
    captured = scenario.run(packet, path=db)
    monkeypatch.setattr(scenario, 'model_binding', lambda: p['model'])
    monkeypatch.setattr(native, '_normal_cdf', lambda *a: pytest.fail('static replay must not infer'))
    result = replay.read(captured, path=db); view = replay.display(result)
    assert result['payload']['probability'] == view['probability'] == expected
    assert result['payload']['input'] == original and packet == original
    assert result['payload']['inference_time'] == AT and result['payload']['completed_at'] == FINISHED
    assert view['labels'] == scenario.LABELS and view['evidence_label'] == 'SYNTHETIC'
    assert view['wager_action'] == 'PASS' and view['stake'] == 0
    assert 'SYNTHETIC_PRIVATE_BODY_CANARY' not in scenario.encode(view)
    assert not {'quote','operator','price','ev','edge','blend'} & set(result['payload'])
    archive, manifest = replay.download(captured, path=db)
    with ZipFile(io.BytesIO(archive)) as zipped:
        assert 'package.json' not in zipped.namelist()
        for name, digest in manifest['files'].items(): assert hashlib.sha256(zipped.read(name)).hexdigest() == digest
        assert zipped.read('originals/'+p['scoring'][0]['body_sha256']+'.body') == custody.read(p['scoring'][0])[0]
    with pytest.raises((ValueError, KeyError)): research_replay.read_export(captured['export_id'], path=db)
    with pytest.raises(ValueError): research_replay.read_scenario(captured, path=db)
    with sqlite3.connect(db) as conn:
        for table in ('research_source_intakes', 'research_replay_exports'):
            with pytest.raises(sqlite3.IntegrityError): conn.execute('DELETE FROM '+table)


@pytest.mark.parametrize('attack,code', [
    ('body','CUSTODY_CORRUPT'), ('missing','CUSTODY_MISSING'), ('substitution','OWNER_REVIEW_CONFLICT'),
    ('event','EVENT_CONFLICT'), ('orientation','EVENT_CONFLICT'), ('neutral','EVENT_CONFLICT'), ('integer','INTEGER_PUSH_UNVALIDATED'),
    ('side','TARGET_CONFLICT'), ('period','TARGET_CONFLICT'), ('ot','TARGET_CONFLICT'), ('line','OWNER_REVIEW_CONFLICT'),
    ('feature','FEATURE_DERIVATION'), ('order','FEATURE_DERIVATION'), ('model','MODEL_CONFLICT'), ('runtime','RUNTIME_CONFLICT'),
    ('missing_review','OWNER_REVIEW_MISSING'), ('other_review','OWNER_REVIEW_CONFLICT'), ('late_review','OWNER_REVIEW_CLOCK'),
    ('foreign_review','OWNER_REVIEW_MISSING'), ('future','FUTURE_EVIDENCE'), ('stale','STALE_EVIDENCE'), ('started','STARTED'),
    ('permission','PERMISSION_CLOCK'), ('availability','FEATURE_CLOCK'), ('excerpt','EVENT_EXCERPT_CONFLICT'),
    ('event_subject','EVENT_VERIFICATION_MISSING'), ('publisher','RECEIPT_MISSING'), ('unknown_clock','CLOCK_UNKNOWN'),
    ('page','PAGE_CHAIN_INCOMPLETE'), ('historical','SCHEMA')])
def test_rejections_before_native_inference(monkeypatch, attack, code):
    packet = fixture(monkeypatch); p = packet['payload']; resign = True; at = AT
    if attack == 'body': p['scoring'][0]['body_sha256'] = '0'*64
    elif attack == 'missing': p['scoring'] = []
    elif attack == 'substitution':
        raw, body = custody.read(p['scoring'][0]); body['data'][0]['home_score'] += 1
        p['scoring'][0] = custody.retain(custody.encode(body).encode(), p['scoring'][0]['receipt'])
        p['projection'] = scenario.projection(p); p['features'] = p['projection']['features']; resign = False
    elif attack == 'event': p['event']['start'] = '2026-10-15T01:00:00Z'; p['event_verification']['event_sha256'] = scenario.digest(p['event'])
    elif attack == 'orientation':
        p['event']['home'], p['event']['away'] = p['event']['away'], p['event']['home']; p['event_verification']['event_sha256'] = scenario.digest(p['event'])
    elif attack == 'neutral': p['event']['neutral_site'] = True
    elif attack == 'integer': p['scenario']['signed_handicap'] = 2
    elif attack == 'side': p['scenario']['selected_team'] = 'Seattle Storm'
    elif attack == 'period': p['scenario']['period'] = 'first_half'
    elif attack == 'ot': p['scenario']['overtime'] = False
    elif attack == 'line': p['scenario']['signed_handicap'] = -2.5; resign = False
    elif attack == 'feature': p['features'] = dict(p['features'], feature_diff_last5=0)
    elif attack == 'order': p['feature_order'].reverse()
    elif attack == 'model': p['model']['artifact_sha256'] = '0'*64
    elif attack == 'runtime': p['reader']['runtime']['python'] = 'UNKNOWN'
    elif attack == 'missing_review': p['owner_review'] = None; resign = False
    elif attack == 'other_review': p['owner_review']['subject_sha256'] = '0'*64; resign = False
    elif attack == 'late_review': p['owner_review']['reviewed_at'] = AT; resign = False
    elif attack == 'foreign_review': p['owner_review']['version'] = 'nfl-owner-sporting-scenario-review-v1'; resign = False
    elif attack == 'future': p['scoring'][0]['receipt']['observed_at'] = '2026-10-14T12:00:06Z'
    elif attack == 'stale': at = '2026-10-14T18:00:03Z'
    elif attack == 'started': at = p['event']['start']
    elif attack == 'permission': p['permissions'][0]['reviewed_at'] = '2026-10-14T12:00:03Z'
    elif attack == 'availability': p['feature_available_at'] = '2026-10-14T12:00:01Z'
    elif attack == 'excerpt': p['event_verification']['excerpts'][0]['text'] = 'UNSUPPORTED'
    elif attack == 'event_subject': p['event_verification']['event_sha256'] = '0'*64
    elif attack == 'publisher': p['scoring'][0]['receipt']['publisher_availability'] = 'VERIFIED'
    elif attack == 'unknown_clock': p['scoring'][0]['receipt']['observed_at'] = None
    elif attack == 'page':
        raw, body = custody.read(p['scoring'][0]); body['meta']['next_cursor'] = 999
        p['scoring'][0] = custody.retain(custody.encode(body).encode(), p['scoring'][0]['receipt'])
    elif attack == 'historical': p['original_inference_time'] = '2020-01-01T00:00:00Z'
    packet = sign(p) if resign else scenario.seal(p)
    monkeypatch.setattr('app_core.market_probability_model.predict_market_probabilities', lambda *a: pytest.fail('invalid evidence reached native inference'))
    with pytest.raises(ValueError, match='WNBA_SCENARIO_'+code): scenario.compute(packet, at=at)


@pytest.mark.parametrize('attack,code', [('ties','HISTORY_CONFLICT'),('future_game','HISTORY_CONFLICT'),('unknown_status','HISTORY_STATUS_UNKNOWN'),
    ('duplicate','HISTORY_IDENTITY'),('other_season','HISTORY_IDENTITY'),('identity','TEAM_IDENTITY'),('no_histories','HISTORY_MISSING')])
def test_rehashed_history_attacks(monkeypatch, attack, code):
    p = fixture(monkeypatch)['payload']; _, body = custody.read(p['scoring'][0])
    if attack == 'ties': body['data'][0]['away_score'] = body['data'][0]['home_score']
    elif attack == 'future_game': body['data'][0]['date'] = '2026-10-16T00:00:00Z'
    elif attack == 'unknown_status': body['data'][0]['status_state'] = 'unknown'
    elif attack == 'duplicate': body['data'].insert(0, deepcopy(body['data'][0]))
    elif attack == 'other_season': body['data'][0]['season'] = 2025
    elif attack == 'identity': body['data'][0]['home_team']['id'] = 5
    elif attack == 'no_histories':
        for row in body['data']: row['postseason'] = True
    p['scoring'][0] = custody.retain(custody.encode(body).encode(), p['scoring'][0]['receipt'])
    monkeypatch.setattr('app_core.market_probability_model.predict_market_probabilities', lambda *a: pytest.fail('rehashed invalid history inferred'))
    with pytest.raises(ValueError, match='WNBA_SCENARIO_'+code): scenario.compute(sign(p), at=AT)


def test_staging_and_cli_inspection_cannot_make_review(monkeypatch, tmp_path, capsys):
    from scripts import wnba_private_scenario as cli
    packet = fixture(monkeypatch); packet['payload']['owner_review'] = None
    raw = scenario.encode(scenario.seal(packet['payload'])).encode()
    with pytest.raises(ValueError, match='SYNTHETIC_UPLOAD'): scenario.load(raw, owner_upload=True)
    p = packet['payload']
    packet = scenario.seal(p); path = tmp_path/'SYNTHETIC-staging.json'; path.write_text(scenario.encode(packet))
    assert cli.main(['inspect', '--packet', str(path), '--synthetic-inspection']) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected['owner_review_recorded'] is False and inspected['numerical_calls'] == 0
    assert inspected['subject_sha256'] == scenario.subject(p)
    assert cli.main(['run', '--packet', str(path), '--authorize-subject', '0'*64, '--synthetic-inspection']) == 1
    assert 'SYNTHETIC_UPLOAD' in capsys.readouterr().out


def test_native_and_historical_football_files_remain_exact():
    # Binding-driven assertion populated after the implementation is sealed.
    root = scenario.ROOT; path = root/'docs/paid-launch/wnba-scenario-binding-v1.json'
    binding = json.loads(path.read_text())
    for name, expected in binding['preserved_runtime_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected


def test_no_store_creation_or_migration(monkeypatch, tmp_path):
    packet = fixture(monkeypatch); path = tmp_path/'absent.sqlite3'
    monkeypatch.setattr(scenario, 'compute', lambda *a: pytest.fail('unprovisioned store inferred'))
    with pytest.raises(ValueError, match='STORE_UNAVAILABLE'): scenario.run(packet, path=path)
    assert not path.exists()


@pytest.mark.parametrize('attack', ['body_limit','bad_json','duplicate_field','credential','untrusted_url','bad_base64','unknown_source','receipt_clock'])
def test_original_custody_failures(attack):
    r = receipt(); raw = custody.encode(dict(data=[row(1, HOME, AWAY, 80, 78, '2026-09-01T20:00:00Z')], meta=dict(per_page=100))).encode()
    if attack == 'body_limit': raw = b'X'*(custody.MAX_BODY+1)
    elif attack == 'bad_json': raw = b'{broken'
    elif attack == 'duplicate_field': raw = b'{"data":[],"data":[],"meta":{}}'
    elif attack == 'credential': raw = b'{"api_key":"SYNTHETIC_SECRET_CANARY"}'
    elif attack == 'untrusted_url': r['endpoint'] = 'https://evil.example/wnba/v1/games'
    elif attack == 'unknown_source': r['source_id'] = 'nba-scoring'
    elif attack == 'receipt_clock': r['received_at'] = '2026-10-14T11:59:59Z'
    elif attack == 'bad_base64':
        obj = custody.retain(raw, r); obj['body_base64'] = '!'
        with pytest.raises(ValueError): custody.read(obj)
        return
    with pytest.raises(ValueError): custody.retain(raw, r)


def test_complete_page_chain_has_exact_original_row_locations(monkeypatch):
    p = fixture(monkeypatch)['payload']; _, body = custody.read(p['scoring'][0]); r = deepcopy(p['scoring'][0]['receipt'])
    first = dict(data=body['data'][:4], meta=dict(per_page=100, next_cursor=40))
    second = dict(data=body['data'][4:], meta=dict(per_page=100))
    r2 = deepcopy(r); r2['parameters']['cursor'] = 40; r2['request_started_at'] = r['received_at']
    p['scoring'] = [custody.retain(custody.encode(first).encode(), r), custody.retain(custody.encode(second).encode(), r2)]
    p['projection'] = scenario.projection(p); p['features'] = p['projection']['features']
    assert scenario.checked(sign(p), AT) == p['features']
    assert len({(row['body_sha256'], row['row_index']) for row in p['projection']['selected_rows']['home']}) == 4
    p['scoring'][1]['receipt']['parameters']['cursor'] = 41
    with pytest.raises(ValueError, match='PAGE_CHAIN'): scenario.checked(sign(p), AT)


def test_private_ui_reopens_saved_probability_without_numerical_call(monkeypatch, tmp_path):
    from app.ui import wnba_private_scenario as ui
    packet = fixture(monkeypatch); monkeypatch.setattr(scenario, 'now', iter([AT, FINISHED]).__next__)
    path = store(tmp_path); receipt_value = scenario.run(packet, path=path)
    expected = replay.display(replay.read(receipt_value, path=path))['probability']
    monkeypatch.setattr(scenario, 'compute', lambda *a: pytest.fail('UI replay inferred'))
    real_read, real_download = replay.read, replay.download
    monkeypatch.setattr(replay, 'read', lambda value, **kwargs: real_read(value, path=path))
    monkeypatch.setattr(replay, 'download', lambda value: real_download(value, path=path))
    class FakeStreamlit:
        session_state = {'wnba_scenario_receipt': receipt_value}
        output = []
        def expander(self, *a, **k): return self
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def checkbox(self, label, **k): return k['key'] == 'wnba_scenario_enabled'
        def button(self, *a, **k): return False
        def file_uploader(self, *a, **k): return None
        def caption(self, value): self.output.append(value)
        def write(self, value): self.output.append(value)
        def metric(self, label, value): self.output.extend([label,value])
        def download_button(self, label, **k): self.output.append(label)
        def error(self, value): pytest.fail(value)
    fake = FakeStreamlit(); monkeypatch.setattr(ui, 'st', fake); ui.render()
    assert f'{expected:.6%}' in fake.output
    assert ' / '.join(scenario.LABELS) in fake.output
    assert 'SYNTHETIC_PRIVATE_BODY_CANARY' not in str(fake.output)


def test_nonpositive_scoring_rejects_before_native_caller(monkeypatch):
    p = fixture(monkeypatch)['payload']; _, body = custody.read(p['scoring'][0])
    for row in body['data']:
        if row['status_state'] == 'final' and row['home_team']['full_name'] == HOME:
            row['home_score'] = 0; row['away_score'] = 1
    p['scoring'][0] = custody.retain(custody.encode(body).encode(), p['scoring'][0]['receipt'])
    p['projection'] = scenario.projection(p); p['features'] = p['projection']['features']
    monkeypatch.setattr('app_core.market_probability_model.predict_market_probabilities', lambda *a: pytest.fail('zero scoring reached native caller'))
    with pytest.raises(ValueError, match='SCORING_INPUT_UNAVAILABLE'): scenario.compute(sign(p), at=AT)
