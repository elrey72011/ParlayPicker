"""Native current-generation costs, inventory and predecessor ownership."""
import base64
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zlib

import pytest
from scripts import ncaaf_filtered_history_scope as history
from scripts import run_ci_tests as runner
from tests import nfl_profile_generation_inputs as inputs

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'scripts/ci_test_file_costs_v1.json'


def unpack(record):
    raw = zlib.decompress(base64.b85decode(record['compressed_b85']))
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    return json.loads(raw)


def test_current_generation_maxima_preserve_history_and_charge_fixture_work():
    binding = history.authenticate(ROOT)
    metadata = binding['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']
    profile = json.loads(history.physical(ROOT / PROFILE))
    revision = profile['first_driver_capacity_context_revision']
    before = json.loads(inputs.record_bytes(metadata['historical_parent_sources'][PROFILE]))
    assert unpack(revision['previous_rows']) == before['files']
    assert all(profile['sources'][key] == value for key, value in before['sources'].items())
    observations = unpack(revision['eligible_completed_file_samples'])
    assert len(observations) == 4428 and revision['variability_multiplier'] == 1.15
    for row in profile['files']:
        basis = row.get('current_context_basis')
        if basis is None: continue
        matches = [sample['seconds'] for sample in observations if sample['path'] == row['path']]
        assert basis['complete_file_phase_max_seconds'] == max(matches)
        total = max(matches) + basis['introduced_shared_fixture_phase_allowance'] + basis.get('introduced_profile_fixture_allowance', 0)
        assert row['seconds'] == round(total * 1.15, 6)
    consumer = next(row for row in profile['files'] if row['path'] == inputs.CONSUMER)
    assert consumer['current_context_basis']['introduced_profile_fixture_allowance'] > 0
    assert consumer['current_context_basis']['reviewed_append_only_fixture'] is True
    for path, blob in revision['fixture_context_blobs'].items():
        raw = history.physical(ROOT / path)
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == blob


def test_current_inventory_exact_once_order_and_all_thirteen_reserves():
    profile = json.loads(history.physical(ROOT / PROFILE))
    revision = profile['first_driver_capacity_context_revision']
    identities = unpack(revision['current_inventory'])
    prior = unpack(revision['preserved_inventory'])
    assert len(prior) == 9235 and len(identities) == 9238
    assert {tuple(row) for row in prior} <= {tuple(row) for row in identities}
    rows, files, loads = runner.assignment([SimpleNamespace(path=p, nodeid=n) for p, n in identities],
        13, runner.load_profile(ROOT / PROFILE), ROOT)
    assert len(rows) == len({row['nodeid'] for row in rows}) == 9238
    assert [row['nodeid'] for row in rows] == [n for p, n in identities]
    assert all(len({row['shard'] for row in rows if row['path'] == file['path']}) == 1 for file in files)
    assert len(loads) == 13 and all(2700 - total - revision['job_overhead_budget_seconds'] >= 120 for total in loads)
    assert revision['job_limit_seconds'] == 2700 and revision['required_reserve_seconds'] == 120


def test_historical_view_rejects_tampering_and_restores_owned_state(tmp_path, monkeypatch):
    original_path = sys.path
    original_values = list(sys.path)
    original_modules = {name: sys.modules.get(name) for name in ('scripts.ncaaf_filtered_history_scope',
        'scripts.ncaaf_reviewed_target_operator', 'scripts.ncaaf_private_refresh_scope')}
    for index, fail in enumerate((False, True)):
        try:
            with inputs.historical_view(ROOT, ROOT / inputs.CONSUMER, tmp_path / str(index)) as view:
                assert view['history'].authenticate(view['ROOT'])
                assert view['ROOT'] != ROOT and view['history'] is not history
                if fail: raise RuntimeError('SYNTHETIC_BODY_FAILURE')
        except RuntimeError as error:
            assert fail and str(error) == 'SYNTHETIC_BODY_FAILURE'
        assert sys.path is original_path and sys.path == original_values
        assert all(sys.modules.get(name) is module for name, module in original_modules.items())
    with pytest.raises(ValueError, match='NFL_PROFILE_GENERATION_CHANGED'):
        with inputs.historical_view(ROOT, ROOT / 'tests/unreviewed.py', tmp_path / 'unknown'): pass
    native = inputs.load_module
    def fail_setup(path, name):
        sys.path.insert(0, str(tmp_path / 'owned'))
        raise RuntimeError('SYNTHETIC_SETUP_FAILURE')
    monkeypatch.setattr(inputs, 'load_module', fail_setup)
    with pytest.raises(RuntimeError, match='SYNTHETIC_SETUP_FAILURE'):
        with inputs.historical_view(ROOT, ROOT / inputs.CONSUMER, tmp_path / 'failed'): pass
    assert sys.path is original_path and sys.path == original_values
    assert all(sys.modules.get(name) is module for name, module in original_modules.items())
    monkeypatch.setattr(inputs, 'load_module', native)
    record = {'compressed_b85': base64.b85encode(zlib.compress(b'UNREVIEWED')).decode(), 'bytes': 10, 'sha256': '0' * 64}
    with pytest.raises(ValueError, match='NFL_PROFILE_GENERATION_CHANGED'): inputs.record_bytes(record)
