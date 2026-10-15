"""Labelled synthetic ordered CI corrections; frozen assertions stay intact."""
import importlib.util
import json
from pathlib import Path
import pytest
from scripts import nfl_pdf_scope as scope
from scripts.benchmark_drive_history_loading import blocked_network


@pytest.fixture
def chain(tmp_path):
    spec = importlib.util.spec_from_file_location('synthetic_pdf_scope_tests',
                                                Path(__file__).with_name('test_nfl_pdf_scope.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with blocked_network():
        g, binding, original, _, _, write, commit = module.sealed.__wrapped__(tmp_path)
        g.git('checkout', '-q', original)
        path = 'docs/SYNTHETIC_fixture_correction.txt'
        write(path, b'isolated synthetic fixture correction\n')
        middle = commit('synthetic separately reviewed fixture correction')
        binding = dict(binding, implementation_ancestors=[original, middle],
                       paths=binding['paths']+[path],
                       reviewed_blobs=dict(binding['reviewed_blobs'], **{path:g.git('hash-object',path)}))
        write(scope.BINDING, json.dumps(binding).encode())
        implementation = commit('synthetic bound scheduling correction')
        policy = scope.make_policy(g, binding, implementation)
        write(scope.POLICY, json.dumps(policy).encode())
        seal = commit('synthetic policy-only seal')
        yield g, binding, implementation, seal, write, commit


def test_exact_ordered_corrections_and_policy_only_seal(chain):
    g, binding, _, _, _, _ = chain
    assert scope.run(g, g.ROOT/g.MANIFEST_PATH, None, binding)[0] == 0


def test_successor_costs_and_frozen_metadata_are_separately_bound(historical_pdf_cost_profile):
    import hashlib
    from scripts import check_launch_change_scope as guard
    root = historical_pdf_cost_profile
    source = (root/'scripts/ci_test_file_costs_v1.json').read_bytes().replace(b'\r\n', b'\n')
    data = json.loads(source)
    info = json.loads(guard.NFL_PDF_RECONSTRUCTION_JSON)['scripts/ci_test_file_costs_v1.json']
    prior = guard._pdf_previous_source('scripts/ci_test_file_costs_v1.json', source)
    assert hashlib.sha256(prior).hexdigest() == info['before_sha256']
    frozen = json.loads(prior)
    assert data['revision'] == frozen['revision'] == 3
    assert data['revision_basis'] == frozen['revision_basis']
    assert data['nfl_listing_cost_revision'] == frozen['nfl_listing_cost_revision']
    rows = {r['path']:r for r in data['files']}
    assert len(data['nfl_pdf_ci_cost_provenance']) == 22
    for change in data['nfl_pdf_ci_cost_provenance']:
        evidence = change['completed']
        identity = evidence['source_identity']
        assert identity['completed'] is True
        assert evidence['count'] == rows[change['path']]['profiled_tests']
        assert rows[change['path']]['seconds'] == round(evidence['seconds']*1.15, 5)
        assert len(identity['junit_sha256']) == len(identity['assignment_manifest_sha256']) == 64
        assert len(identity['checkout_commit']) == len(identity['api_head']) == 40
        assert identity['attempt'] == 1


@pytest.mark.parametrize('attack', ['reverse', 'missing', 'extra', 'invalid'])
def test_changed_ordered_implementation_binding_rejects(chain, attack):
    g, binding, implementation, _, write, commit = chain
    altered = list(binding['implementation_ancestors'])
    if attack == 'reverse': altered.reverse()
    elif attack == 'missing': altered.pop()
    elif attack == 'extra': altered.append(implementation)
    else: altered[0] = '0'*40
    changed = dict(binding, implementation_ancestors=altered)
    g.git('checkout', '-q', implementation)
    write(scope.BINDING, json.dumps(changed).encode())
    new = commit('synthetic rehashed invalid ancestry')
    write(scope.POLICY, json.dumps(scope.make_policy(g, changed, new)).encode())
    commit('synthetic invalid policy seal')
    assert scope.run(g, g.ROOT/g.MANIFEST_PATH, None, changed)[0] != 0


# Exact immutable input for the historical PDF cost-revision assertion above.
def _historical_pdf_cost_profile_bytes(path, source):
    import hashlib
    from scripts import check_launch_change_scope as guard
    target = 'scripts/ci_test_file_costs_v1.json'
    if path != target:
        return source
    record = json.loads(guard.FOOTBALL_DOCPERF_RECONSTRUCTION_JSON)[target]
    if hashlib.sha256(source).hexdigest() == record['before_sha256']:
        return source
    result = guard._docperf_previous_source(target, source)
    assert hashlib.sha256(result).hexdigest() == record['before_sha256']
    return result


@pytest.fixture
def historical_pdf_cost_profile(tmp_path):
    root = tmp_path/'SYNTHETIC-historical-pdf-cost-profile'
    path = 'scripts/ci_test_file_costs_v1.json'
    source = (Path(__file__).resolve().parents[1]/path).read_bytes().replace(b'\r\n', b'\n')
    target = root/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_historical_pdf_cost_profile_bytes(path, source))
    return root
