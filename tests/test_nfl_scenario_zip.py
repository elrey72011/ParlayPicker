"""Synthetic exports only; real recorded binding structures, no native calls."""
from copy import deepcopy
from io import BytesIO
import json
import struct
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import pytest

from app_core import nfl_scenario_zip as replay


def seal(payload):
    return dict(payload=deepcopy(payload), sha256=replay.digest(payload))


def archive(result, receipt, *, manifest_change=None, extra=None):
    files = {'scenario-result.json': replay.encode(result).encode(), 'private-receipt.json': replay.encode(receipt).encode()}
    manifest = dict(version='nfl-private-scenario-export-v1', purpose='PRIVATE_RESEARCH', files={k:replay.sha(v) for k,v in files.items()})
    if manifest_change:
        manifest_change(manifest)
    files['manifest.json'] = replay.encode(manifest).encode()
    if extra:
        files.update(extra)
    out = BytesIO()
    with ZipFile(out, 'w') as zipped:
        for name, raw in files.items():
            info = ZipInfo(name, date_time=(1980,1,1,0,0,0)); info.compress_type = ZIP_DEFLATED
            zipped.writestr(info, raw)
    return out.getvalue()


def fixtures():
    """Independent synthetic execution anchors; never an authentic review.

    The producer bindings intentionally simulate the recorded Windows producer,
    even in Linux CI. No runtime, reader or authentication function is patched.
    Production trust cannot admit these unknown SYNTHETIC result identities.
    """
    trust = replay.production_trust(); pid, producer = next(iter(trust['profiles'].items()))
    records, values = {}, []
    for selected, line, probability, projection in [('Dallas', -2.5, .42, -2.0), ('Buffalo', 2.5, .58, 2.0)]:
        q = dict(version='nfl-owner-sporting-scenario-v1', evidence_label='SYNTHETIC',
            scoring={'SYNTHETIC_original': 'SYNTHETIC_PRIVATE_BODY_CANARY'},
            event=dict(canonical_event_id='SYNTHETIC-BUF-DAL', home='Dallas', away='Buffalo', start='2026-10-07T00:15:00Z', season=2026, neutral_site=False),
            event_verification={'SYNTHETIC': True},
            scenario=dict(selected_team=selected, signed_handicap=line, period='full_game', overtime=True, condition='completed_sporting_result'),
            permission={'SYNTHETIC': 'Not authentic source permission or an owner finding.'},
            features={k:1.0 for k in producer['model']['feature_order']}, feature_order=producer['model']['feature_order'],
            feature_available_at='2026-10-06T19:25:03Z', model=deepcopy(producer['model']), reader=deepcopy(producer['reader']),
            owner_review=None, original_inference_time=None)
        q['owner_review'] = dict(version='nfl-owner-sporting-scenario-review-v1', reviewer='Robert Velarde', reviewed_at='2026-10-06T19:25:04Z',
            subject_sha256=replay.digest({k:v for k,v in q.items() if k != 'owner_review'}), status='OWNER_REVIEWED', purpose='PRIVATE_RESEARCH',
            attestation='I verified the original evidence and sporting assumptions for this exact scenario.',
            findings={k:dict(conclusion='VERIFIED', basis='SYNTHETIC finding only; no actual owner review or provider authority.') for k in replay.FINDINGS})
        packet = seal(q)
        result = seal(dict(version='nfl-owner-sporting-scenario-result-v1', status='COMPLETE', labels=replay.LABELS,
            input=packet, inference_time='2026-10-06T19:25:05Z', completed_at='2026-10-06T19:25:06Z',
            reader=deepcopy(producer['reader']), model=deepcopy(producer['model']),
            native_output=dict(ml_probability=probability, ml_probability_source='score-distribution-v1:nfl', ml_target='spread_cover', ml_projection=projection, ml_residual_scale=13.86),
            probability=probability, probability_stage='native_model_output', calibrated=False,
            probability_semantics='full_game_including_overtime_cover_conditional_on_completed_sporting_result', publisher_availability='UNKNOWN',
            source_acceptance=False, wagering_authority=False, qualification=False, wager_action='PASS', stake=0))
        receipt = dict(version='nfl-private-scenario-capture-v1', export_id='nfl-scenario-result:' + result['sha256'], result_sha256=result['sha256'], input_sha256=packet['sha256'])
        records[result['sha256']] = dict(input_sha256=packet['sha256'], receipt_sha256=replay.digest(receipt), profile=pid)
        values.append((archive(result, receipt), result, receipt))
    return values, replay.SyntheticPolicy(replay.encode(records))


def reseal(result, receipt, policy):
    result = seal(result['payload'])
    receipt = dict(receipt, result_sha256=result['sha256'], export_id='nfl-scenario-result:' + result['sha256'])
    anchors = json.loads(policy.records_json)
    old = next(iter(anchors.values()))
    anchors[result['sha256']] = dict(old, input_sha256=receipt['input_sha256'], receipt_sha256=replay.digest(receipt))
    return archive(result, receipt), replay.SyntheticPolicy(replay.encode(anchors))


def test_two_synthetic_exports_preserve_recorded_bindings_outputs_and_original_clocks():
    values, policy = fixtures(); before = deepcopy(values)
    items = replay.read_archives([x[0] for x in values], synthetic_policy=policy)
    assert [x['payload']['probability'] for x in items] == [.42, .58]
    assert all(x['payload']['reader']['runtime']['platform'] == 'Windows-11-10.0.26300-SP0' for x in items)
    assert all(x['payload']['inference_time'] == '2026-10-06T19:25:05Z' for x in items)
    assert [x['result_sha256'] for x in items] == [x[1]['sha256'] for x in values]
    assert values == before


def test_production_cannot_admit_synthetic_or_self_registered_results():
    values, _ = fixtures()
    with pytest.raises(replay.ReplayError, match='EVIDENCE_LABEL'):
        replay.read_archive(values[0][0])
    result, receipt = deepcopy(values[0][1:])
    result['payload']['input']['payload']['evidence_label'] = 'RETAINED'
    result['payload']['input'] = seal(result['payload']['input']['payload'])
    result = seal(result['payload']); receipt = dict(receipt, result_sha256=result['sha256'], export_id='nfl-scenario-result:' + result['sha256'], input_sha256=result['payload']['input']['sha256'])
    with pytest.raises(replay.ReplayError, match='UNTRUSTED_RESULT'):
        replay.read_archive(archive(result, receipt))


@pytest.mark.parametrize('field,value,path', [
    ('numpy', 'UNKNOWN', 'payload.reader.runtime.numpy'),
    ('platform', 'Linux-forged-producer', 'payload.reader.runtime.platform'),
    ('python', '0.0.0', 'payload.reader.runtime.python'),
])
def test_runtime_mismatch_reports_exact_recorded_field(field, value, path):
    values, policy = fixtures(); result, receipt = deepcopy(values[0][1:])
    result['payload']['reader']['runtime'][field] = value
    raw, policy = reseal(result, receipt, policy)
    with pytest.raises(replay.ReplayError) as caught:
        replay.read_archive(raw, synthetic_policy=policy)
    assert caught.value.code == 'NFL_SCENARIO_ZIP_UNSUPPORTED_BINDING' and caught.value.field == path


@pytest.mark.parametrize('attack', ['source', 'model', 'authority', 'integer', 'projection', 'clock', 'review'])
def test_resealed_synthetic_damage_still_requires_binding_schema_and_clock_contracts(attack):
    values, policy = fixtures(); result, receipt = deepcopy(values[0][1:]); p = result['payload']
    if attack == 'source': p['reader']['sources']['app_core/research_replay.py'] = '0' * 64
    if attack == 'model': p['model']['configuration']['score_parameters']['reliability'] = 1.0
    if attack == 'authority': p['stake'] = 1
    if attack == 'projection': p['native_output']['ml_residual_scale'] = -1.0
    if attack == 'clock': p['inference_time'] = '2099-01-01T00:00:00Z'
    if attack in {'integer', 'review'}:
        q = p['input']['payload']
        if attack == 'integer': q['scenario']['signed_handicap'] = 2
        else: q['owner_review']['status'] = 'NOT_REVIEWED'
        p['input'] = seal(q); receipt['input_sha256'] = p['input']['sha256']
    raw, policy = reseal(result, receipt, policy)
    with pytest.raises(replay.ReplayError): replay.read_archive(raw, synthetic_policy=policy)


@pytest.mark.parametrize('attack', ['garbage','extra','unsafe','duplicate','manifest','receipt','hash','schema','directory_count','trailer','prefix','oversize','bomb','duplicate_json'])
def test_archive_rejections(attack):
    values, policy = fixtures(); raw, result, receipt = deepcopy(values[0])
    if attack == 'garbage': raw = b'not a zip'
    elif attack == 'extra': raw = archive(result, receipt, extra={'extra.txt':b'NO'})
    elif attack == 'unsafe': raw = archive(result, receipt, extra={'../scenario-result.json':b'NO'})
    elif attack == 'duplicate':
        out = BytesIO(raw)
        with ZipFile(out, 'a') as z: z.writestr('private-receipt.json', b'{}')
        raw = out.getvalue()
    elif attack == 'manifest': raw = archive(result, receipt, manifest_change=lambda m:m.update(purpose='PUBLIC'))
    elif attack == 'receipt': raw = archive(result, values[1][2])
    elif attack == 'hash': raw = archive(result, receipt, manifest_change=lambda m:m['files'].update({'scenario-result.json':'0'*64}))
    elif attack == 'schema': raw = archive(result, receipt, manifest_change=lambda m:m.update(version='UNKNOWN'))
    elif attack == 'directory_count': raw = raw[:-12] + struct.pack('<H', 30000) + raw[-10:]
    elif attack == 'trailer': raw += b'UNEXPECTED'
    elif attack == 'prefix': raw = b'UNEXPECTED' + raw
    elif attack == 'oversize': raw = b'x' * (replay.MAX_COMPRESSED + 1)
    elif attack == 'bomb': raw = archive(result, receipt, extra={'scenario-result.json':b'x' * (1024*1024)})
    elif attack == 'duplicate_json': raw = archive(result, receipt, extra={'manifest.json':b'{"version":1,"version":2}'})
    with pytest.raises(replay.ReplayError): replay.read_archive(raw, synthetic_policy=policy)


def test_batch_rejects_duplicate_receipts_and_limits():
    values, policy = fixtures()
    with pytest.raises(replay.ReplayError, match='RECEIPT_CONFLICT'):
        replay.read_archives([values[0][0]] * 2, synthetic_policy=policy)
    with pytest.raises(replay.ReplayError, match='SELECTION_LIMIT'):
        replay.read_archives([values[0][0]] * 5, synthetic_policy=policy)


def test_frozen_live_readers_and_public_model_artifacts_are_unchanged():
    import base64
    root = replay.TRUST_PATH.parents[2]
    profile = next(iter(replay.production_trust()['profiles'].values()))
    for path, expected in profile['reader']['sources'].items():
        assert replay.sha((root / path).read_bytes().replace(b'\r\n', b'\n')) == expected
    for path, artifact in profile['model']['artifacts'].items():
        original = base64.b64decode(artifact['bytes_base64'], validate=True)
        assert replay.sha(original) == artifact['sha256']
        assert original == (root / path).read_bytes().replace(b'\r\n', b'\n')
