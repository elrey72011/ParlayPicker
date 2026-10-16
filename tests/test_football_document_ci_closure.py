"""SYNTHETIC fixture inputs and local Git only; no acquisition or inference."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from scripts import check_launch_change_scope as guard
from scripts.benchmark_drive_history_loading import blocked_network
from tests import test_nfl_pdf_ci_scope as historical
from tests import test_remote_canonical_scope as remote

ROOT = Path(__file__).resolve().parents[1]
PROFILE = 'scripts/ci_test_file_costs_v1.json'


def test_performance_fixture_needs_no_git_history(tmp_path, monkeypatch):
    path = ROOT/'tests/test_football_document_ci_performance.py'
    spec = importlib.util.spec_from_file_location('synthetic_history_free_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = tmp_path/'SYNTHETIC-history-free-source'
    for name in (guard.GUARD_PATH, 'docs/paid-launch/football-document-ci-performance-binding-v1.json'):
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    assert not (snapshot/'.git').exists()
    monkeypatch.setattr(module, 'ROOT', snapshot)
    def forbidden(*args, **kwargs):
        raise AssertionError('SYNTHETIC_GIT_HISTORY_ACCESS_FORBIDDEN')
    monkeypatch.setattr(subprocess, 'check_output', forbidden)
    with blocked_network():
        raw, reference, binding = module.predecessor.__wrapped__()
        assert hashlib.sha256(raw).hexdigest() == binding['previous_guard_sha256']
        assert guard._listing_capacity_base_source(raw) == reference(raw)


def test_new_provenance_records_observed_checkout_without_rewriting_history(historical_ci_closure_profile):
    source = _verified_ci_closure_profile_bytes(historical_ci_closure_profile)
    current = json.loads(source)
    previous = json.loads(guard._docclose_previous_source(PROFILE, source))
    record = current['sources']['document-ci-completed-56-shard6']
    assert record['head'] == '56da7ac68caccbba56d018d22de4dc3f211e019d'
    assert record['checkout_commit'] == 'c2f3a6eb3d583b407f34c3b1f85051b6d5c15a2c'
    assert record['head'] != record['checkout_commit']
    assert len(record['checkout_receipt_log_sha256']) == 64
    for field in ('revision', 'revision_basis', 'nfl_pdf_ci_cost_provenance',
                  'nfl_listing_cost_revision', 'files', 'fallback'):
        assert current[field] == previous[field]


def test_historical_profile_is_exact_and_idempotent():
    source = (ROOT/PROFILE).read_bytes().replace(b'\r\n', b'\n')
    prior = historical._historical_pdf_cost_profile_bytes(PROFILE, source)
    record = json.loads(guard.FOOTBALL_DOCPERF_RECONSTRUCTION_JSON)[PROFILE]
    assert hashlib.sha256(prior).hexdigest() == record['before_sha256']
    assert historical._historical_pdf_cost_profile_bytes(PROFILE, prior) == prior
    assert json.loads(source)['files'] != json.loads(prior)['files']


@pytest.mark.parametrize('attack', ['append', 'changed_cost', 'wrong_path'])
def test_unknown_profile_cannot_borrow_historical_input(attack):
    source = (ROOT/PROFILE).read_bytes().replace(b'\r\n', b'\n')
    if attack == 'wrong_path':
        assert historical._historical_pdf_cost_profile_bytes('scripts/UNREVIEWED.json', source) == source
        return
    if attack == 'append': changed = source+b'\nSYNTHETIC_UNREVIEWED'
    else:
        data = json.loads(source)
        data['files'][0]['seconds'] += 1
        changed = json.dumps(data).encode()
    with pytest.raises((AssertionError, ValueError)):
        historical._historical_pdf_cost_profile_bytes(PROFILE, changed)


def test_historical_assertions_run_on_isolated_copies(tmp_path):
    first = historical.historical_pdf_cost_profile.__wrapped__(tmp_path/'one')
    second = historical.historical_pdf_cost_profile.__wrapped__(tmp_path/'two')
    original = (ROOT/PROFILE).read_bytes()
    historical.test_successor_costs_and_frozen_metadata_are_separately_bound(first)
    data = json.loads((first/PROFILE).read_text())
    wanted = data['nfl_pdf_ci_cost_provenance'][0]['path']
    next(row for row in data['files'] if row['path'] == wanted)['seconds'] += 1
    (first/PROFILE).write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        historical.test_successor_costs_and_frozen_metadata_are_separately_bound(first)
    historical.test_successor_costs_and_frozen_metadata_are_separately_bound(second)
    assert (ROOT/PROFILE).read_bytes() == original


def test_all_original_test_assertions_and_reconstruction_bindings_preserved():
    def assertions(raw):
        return {node.name: [ast.dump(a) for a in ast.walk(node) if isinstance(a, ast.Assert)]
                for node in ast.parse(raw).body if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
    for path in ('tests/test_football_document_ci_performance.py', 'tests/test_nfl_pdf_ci_scope.py',
                 'tests/test_remote_canonical_scope.py'):
        current = (ROOT/path).read_bytes().replace(b'\r\n', b'\n')
        prior = guard._docclose_previous_source(path, current)
        assert assertions(current) == assertions(prior)
        assert guard._docclose_previous_source(path, current+b'\nSYNTHETIC_UNREVIEWED') == current+b'\nSYNTHETIC_UNREVIEWED'


def test_canonical_template_packing_preserves_isolated_history(tmp_path):
    class Factory:
        def mktemp(self, name):
            path = tmp_path/'SYNTHETIC-template'/name
            path.mkdir(parents=True)
            return path
    saved = guard.ROOT
    try:
        with blocked_network():
            prepared = remote.prepared.__wrapped__(Factory())
            template = prepared[0]
            assert subprocess.check_output(['git', 'config', '--local', '--get', 'gc.auto'], cwd=template).strip() == b'0'
            assert subprocess.check_output(['git', 'config', '--local', '--get', 'maintenance.auto'], cwd=template).strip() == b'false'
            assert not [p for p in (template/'.git/objects').glob('*/*') if p.parent.name not in {'pack', 'info'}]
            first, second = tmp_path/'one', tmp_path/'two'
            shutil.copytree(template, first)
            shutil.copytree(template, second)
            for copy in (first, second):
                assert remote.git(copy, 'rev-parse', 'HEAD') == remote.git(template, 'rev-parse', 'HEAD')
                assert remote.git(copy, 'rev-parse', 'HEAD^{tree}') == remote.git(template, 'rev-parse', 'HEAD^{tree}')
                assert subprocess.run(['git', 'fsck', '--full', '--no-reflogs'], cwd=copy,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
            guard.ROOT = first
            assert remote.assess((first, *prepared[1:]))[0] == 0
            (first/'core/protected.py').write_bytes(b'SYNTHETIC_UNREVIEWED')
            assert remote.assess((first, *prepared[1:]))[0] != 0
            guard.ROOT = second
            assert remote.assess((second, *prepared[1:]))[0] == 0
    finally:
        guard.ROOT = saved


# Explicit exact input for this historical provenance-only assertion. The current
# six-shard/profile tests continue reading the actual checkout.
@pytest.fixture
def historical_ci_closure_profile(tmp_path):
    original=(ROOT/PROFILE).read_bytes().replace(b'\r\n',b'\n')
    original=guard._nfl_pdf_timing_previous_source(PROFILE,original)
    binding=json.loads((ROOT/'docs/paid-launch/ncaaf-socket-ci-binding-v1.json').read_bytes())
    record=binding['source_reconstructions'][PROFILE]
    assert hashlib.sha256(original).hexdigest()==record['after_sha256']
    prior=guard._ncaaf_socket_ci_previous_source(PROFILE,original)
    assert hashlib.sha256(prior).hexdigest()==record['before_sha256']
    root=tmp_path/'SYNTHETIC-exact-ci-closure-profile'
    path=root/PROFILE;path.parent.mkdir(parents=True)
    path.write_bytes(prior)
    return root


def _verified_ci_closure_profile_bytes(root):
    source=(root/PROFILE).read_bytes().replace(b'\r\n',b'\n')
    binding=json.loads((ROOT/'docs/paid-launch/ncaaf-socket-ci-binding-v1.json').read_bytes())
    assert hashlib.sha256(source).hexdigest()==binding['source_reconstructions'][PROFILE]['before_sha256']
    return source
