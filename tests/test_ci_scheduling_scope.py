"""SYNTHETIC offline Git seals; no model, evidence or external operation."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from scripts import check_launch_change_scope as guard
from scripts import ci_scheduling_scope as successor
from scripts.benchmark_drive_history_loading import blocked_network

SOURCE = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo).decode().strip()


def raw(repo, revision, path):
    return subprocess.check_output(['git', 'show', revision+':'+path], cwd=repo)


def write(repo, path, data):
    target = repo/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def commit(repo, message):
    git(repo, 'add', '--all')
    git(repo, 'commit', '--allow-empty', '-qm', message)
    return git(repo, 'rev-parse', 'HEAD')


def seal(repo, binding):
    implementation = commit(repo, 'SYNTHETIC reviewed CI implementation')
    policy = successor.make_policy(guard, binding, implementation)
    write(repo, guard.CI_SCHEDULING_POLICY_PATH, json.dumps(policy, indent=2).encode()+b'\n')
    return implementation, commit(repo, 'SYNTHETIC policy-only seal'), policy


@pytest.fixture(scope='session')
def prepared(tmp_path_factory):
    repo = tmp_path_factory.mktemp('ci-scope')/'r'
    repo.mkdir()
    original_root = guard.ROOT
    with blocked_network():
        git(repo, 'init', '-q')
        git(repo, 'config', 'user.name', 'Offline Test')
        git(repo, 'config', 'user.email', 'offline@example.invalid')
        git(repo, 'config', 'core.autocrlf', 'false')
        current = (SOURCE/guard.GUARD_PATH).read_bytes().replace(b'\r\n', b'\n')
        previous = guard._ci_scheduling_previous_guard_source(current)
        original = current.split(b'\nPOLICY_PATH =', 1)[0]+b'\n'
        write(repo, guard.GUARD_PATH, original)
        write(repo, 'README.md', b'SYNTHETIC offline CI fixture\n')
        write(repo, 'core/protected.py', b'FROZEN = True\n')
        write(repo, '.github/workflows/paid-launch.yml', b'name: frozen offline fixture\n')
        write(repo, guard.CLOCK_TEST, b'assert future.expired == 1\n')
        recorded = commit(repo, 'SYNTHETIC original frozen baseline')
        manifest = dict(base_sha=recorded, required_ancestry={'pr_2349_merge_commit': recorded},
            protected_files=['core/protected.py'], protected_git_blobs={'core/protected.py': git(repo, 'rev-parse', recorded+':core/protected.py')},
            tooling_sha256={p: hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in
                            [guard.GUARD_PATH, '.github/workflows/paid-launch.yml']})
        manifest_raw = json.dumps(manifest).encode()+b'\n'
        old_policy = json.loads((SOURCE/guard.SLATE_AUDIT_POLICY_PATH).read_text())
        retained = set(old_policy['unchanged_bindings']) | set(old_policy['implementation_changes']) | set(guard.CI_SCHEDULING_FROZEN_PATHS)
        for path in retained-set(guard.CI_SCHEDULING_PATHS)-{guard.GUARD_PATH, guard.MANIFEST_PATH, '.github/workflows/paid-launch.yml'}:
            if (SOURCE/path).exists():
                write(repo, path, (SOURCE/path).read_bytes().replace(b'\r\n', b'\n'))
        # The application CI checkout is shallow. Synthetic predecessor paths need
        # no real history; the original two assertion bodies remain hash-verified.
        for path in ('.github/workflows/ci.yml', 'docs/ci-test-execution.md', 'scripts/run_ci_tests.py'):
            write(repo, path, b'# SYNTHETIC immutable predecessor\n')
        original_tests = (SOURCE/'tests/test_ci_test_shards.py').read_bytes().replace(b'\r\n', b'\n')[:guard.CI_SCHEDULING_BINDINGS['prior_shard_assertions_bytes']]
        assert hashlib.sha256(original_tests).hexdigest() == guard.CI_SCHEDULING_BINDINGS['prior_shard_assertions_sha256']
        write(repo, 'tests/test_ci_test_shards.py', original_tests)
        fixture_policy = dict(old_policy, unchanged_bindings={p: git(repo, 'hash-object', p) for p in retained if (repo/p).exists()})
        write(repo, guard.SLATE_AUDIT_POLICY_PATH, json.dumps(fixture_policy).encode()+b'\n')
        write(repo, guard.MANIFEST_PATH, manifest_raw)
        write(repo, guard.GUARD_PATH, previous)
        base = commit(repo, 'SYNTHETIC verified main')
        guard.ROOT = repo
        binding = dict(guard.CI_SCHEDULING_BINDINGS, base=base, base_tree=git(repo, 'rev-parse', 'HEAD^{tree}'),
            manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), previous_guard_sha256=hashlib.sha256(previous).hexdigest(),
            previous_policy_blob=guard.blob(base, guard.SLATE_AUDIT_POLICY_PATH))
        for path in guard.CI_SCHEDULING_PATHS:
            write(repo, path, (SOURCE/path).read_bytes().replace(b'\r\n', b'\n'))
        binding['reviewed_blobs'] = {p: git(repo, 'hash-object', p) for p in guard.CI_SCHEDULING_PATHS if p != guard.GUARD_PATH}
        implementation, candidate, policy = seal(repo, binding)
        guard.ROOT = original_root
    return repo, binding, implementation, candidate, policy


@pytest.fixture
def fx(tmp_path, monkeypatch, prepared):
    template, binding, implementation, candidate, policy = prepared
    repo = tmp_path/'r'
    shutil.copytree(template, repo)
    monkeypatch.setattr(guard, 'ROOT', repo)
    with blocked_network():
        yield repo, binding, implementation, candidate, policy


def assess(fx):
    try:
        return guard._run_ci_scheduling_integrated(fx[0]/guard.MANIFEST_PATH, fx[1]['base'], fx[1])
    except (OSError, ValueError) as exc:
        # The public guard CLI also rejects missing modules/invalid reviewed bytes.
        return 1, {'status': 'FAIL', 'reason_codes': [str(exc)]}


def test_exact_ci_only_seal_and_ordered_ci_merge_pass(fx):
    repo, binding, implementation, candidate, policy = fx
    assert assess(fx)[0] == 0, assess(fx)[1]
    assert policy['schema_version'] == 26
    assert git(repo, 'diff', '--name-status', implementation, candidate).splitlines() == ['A\t'+guard.CI_SCHEDULING_POLICY_PATH]
    merge = git(repo, 'commit-tree', git(repo, 'rev-parse', 'HEAD^{tree}'), '-p', binding['base'], '-p', candidate, '-m', 'SYNTHETIC CI merge')
    git(repo, 'checkout', '-q', merge)
    assert assess(fx)[0] == 0


@pytest.mark.parametrize('path', ['scripts/run_ci_tests.py', 'scripts/ci_test_file_costs_v1.json', '.github/workflows/ci.yml',
    'scripts/check_launch_change_scope.py', 'docs/paid-launch/launch-baseline-manifest.json',
    '.github/workflows/paid-launch.yml', 'tests/paid_launch/case_isolation_and_scope.py', 'core/protected.py'])
def test_dirty_staged_committed_and_resealed_changes_fail(fx, path):
    repo, binding, _, _, _ = fx
    write(repo, path, (repo/path).read_bytes()+b'\nUNREVIEWED=True\n')
    assert assess(fx)[0] != 0
    git(repo, 'add', path)
    assert assess(fx)[0] != 0
    commit(repo, 'SYNTHETIC hostile change')
    assert assess(fx)[0] != 0
    git(repo, 'reset', '--soft', binding['base'])
    seal(repo, binding)
    assert assess(fx)[0] != 0


@pytest.mark.parametrize('field, value', [('implementation_changes', {}), ('unchanged_bindings', {}),
    ('tooling_sha256', {}), ('base_tree', '0'*40), ('approval_reference', 'blanket approval'), ('extra_allowlist', ['tests/'])])
def test_mutated_policy_cannot_expand_ci_scope(fx, field, value):
    repo, _, implementation, _, policy = fx
    git(repo, 'checkout', '-q', implementation)
    write(repo, guard.CI_SCHEDULING_POLICY_PATH, json.dumps(dict(policy, **{field:value})).encode()+b'\n')
    commit(repo, 'SYNTHETIC mutated seal')
    assert assess(fx)[0] != 0


@pytest.mark.parametrize('attack', ['extra_commit', 'extra_seal_file', 'wrong_parent_order', 'wrong_merge_tree'])
def test_bad_ancestry_or_seal_cannot_pass(fx, attack):
    repo, binding, implementation, candidate, _ = fx
    if attack == 'extra_commit':
        commit(repo, 'SYNTHETIC extra commit')
    elif attack == 'extra_seal_file':
        git(repo, 'checkout', '-q', implementation)
        write(repo, 'extra.txt', b'unreviewed')
        commit(repo, 'SYNTHETIC invalid seal')
    else:
        parents = [candidate,binding['base']] if attack == 'wrong_parent_order' else [binding['base'],candidate]
        tree = git(repo, 'rev-parse', ('HEAD' if attack == 'wrong_parent_order' else binding['base'])+'^{tree}')
        merge = git(repo, 'commit-tree', tree, '-p', parents[0], '-p', parents[1], '-m', 'SYNTHETIC hostile merge')
        git(repo, 'checkout', '-q', merge)
    assert assess(fx)[0] != 0


def test_exact_predecessor_reader_keeps_coverage_and_all_original_assertions():
    current = (SOURCE/guard.GUARD_PATH).read_bytes().replace(b'\r\n', b'\n')
    previous = guard._ci_scheduling_previous_guard_source(current)
    assert hashlib.sha256(previous).hexdigest() == guard.CI_SCHEDULING_BINDINGS['previous_guard_sha256']
    assert guard._slate_audit_previous_guard_source(current) == guard._slate_audit_previous_guard_source(previous)
    assert guard._football_research_previous_guard_source(current) == guard._football_research_previous_guard_source(previous)
    with pytest.raises(ValueError, match='SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED'):
        guard._slate_audit_previous_guard_source(current+b'\nUNREVIEWED=True\n')
    old = (SOURCE/'tests/test_ci_test_shards.py').read_bytes().replace(b'\r\n', b'\n')[:guard.CI_SCHEDULING_BINDINGS['prior_shard_assertions_bytes']]
    assert hashlib.sha256(old).hexdigest() == guard.CI_SCHEDULING_BINDINGS['prior_shard_assertions_sha256']
    assert (SOURCE/'tests/test_ci_test_shards.py').read_bytes().replace(b'\r\n', b'\n').startswith(old)
