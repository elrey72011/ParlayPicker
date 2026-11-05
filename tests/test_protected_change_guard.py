"""Behavior of the re-baselined protected-change guard, in a throwaway git repo."""
import subprocess

import pytest

from scripts import protected_change_guard as guard


def _git(root, *args):
    subprocess.run(['git', *args], cwd=root, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    _git(tmp_path, 'init', '-q', '-b', 'main')
    _git(tmp_path, 'config', 'user.email', 'guard@example.invalid')
    _git(tmp_path, 'config', 'user.name', 'Guard Test')
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'docs/paid-launch').mkdir(parents=True)
    (tmp_path / 'tests/test_x.py').write_text('def test_a():\n    assert 1 + 1 == 2\n')
    (tmp_path / 'docs/paid-launch/evidence.json').write_text('{"original": true}\n')
    _git(tmp_path, 'add', '-A')
    _git(tmp_path, 'commit', '-q', '-m', 'base')
    monkeypatch.setattr(guard, 'ROOT', tmp_path)
    monkeypatch.setattr(guard, 'runtime_binding_problem', lambda: None)
    monkeypatch.setattr(guard, 'sealed_at_rebaseline', lambda path: False)
    return tmp_path


def _commit_and_check(root):
    _git(root, 'add', '-A')
    _git(root, 'commit', '-q', '-m', 'change')
    report = guard.check('HEAD^1')
    return report['status'], {v['rule'] for v in report['violations']}, report


def test_test_only_timeout_change_passes(repo):
    (repo / 'tests/test_x.py').write_text('def test_a():\n    run(timeout=60)\n    assert 1 + 1 == 2\n')
    assert _commit_and_check(repo)[:2] == ('PASS', set())


def test_new_assertions_and_new_evidence_pass(repo):
    (repo / 'tests/test_x.py').write_text('def test_a():\n    assert 1 + 1 == 2\n    assert True\n')
    (repo / 'docs/paid-launch/new-evidence.json').write_text('{}\n')
    assert _commit_and_check(repo)[:2] == ('PASS', set())


def test_changed_assertion_fails(repo):
    (repo / 'tests/test_x.py').write_text('def test_a():\n    assert 1 + 1 == 3\n')
    assert _commit_and_check(repo)[1] == {'TEST_ASSERTION_REMOVED_OR_CHANGED'}


def test_added_skip_fails(repo):
    (repo / 'tests/test_x.py').write_text('import pytest\n@pytest.mark.skip\ndef test_a():\n    assert 1 + 1 == 2\n')
    assert _commit_and_check(repo)[1] == {'TEST_SKIP_ADDED'}


def test_deleted_test_module_fails(repo):
    (repo / 'tests/test_x.py').unlink()
    assert _commit_and_check(repo)[1] == {'TEST_MODULE_REMOVED'}


def test_archiving_a_test_module_is_reported_not_blocked(repo):
    (repo / 'tests_archive').mkdir()
    _git(repo, 'mv', 'tests/test_x.py', 'tests_archive/test_x.py')
    status, rules, report = _commit_and_check(repo)
    assert (status, rules) == ('PASS', set()) and report['retired_test_modules'] == ['tests_archive/test_x.py']


def test_modified_or_deleted_frozen_evidence_fails(repo):
    (repo / 'docs/paid-launch/evidence.json').write_text('{"original": false}\n')
    assert _commit_and_check(repo)[1] == {'FROZEN_EVIDENCE_CHANGED'}
    (repo / 'docs/paid-launch/evidence.json').unlink()
    assert _commit_and_check(repo)[1] == {'FROZEN_EVIDENCE_CHANGED'}


def test_stale_runtime_binding_fails(repo, monkeypatch):
    monkeypatch.setattr(guard, 'runtime_binding_problem', lambda: 'STALE')
    (repo / 'README.md').write_text('x\n')
    assert _commit_and_check(repo)[1] == {'NCAAF_RUNTIME_BINDING_STALE'}


def test_real_runtime_binding_is_current():
    assert guard.runtime_binding_problem() is None


def test_rename_out_of_pytest_discovery_fails(repo):
    _git(repo, 'mv', 'tests/test_x.py', 'tests/helper_x.py')
    assert _commit_and_check(repo)[1] == {'TEST_MODULE_REMOVED'}


def test_rename_between_discoverable_names_passes(repo):
    _git(repo, 'mv', 'tests/test_x.py', 'tests/test_y.py')
    assert _commit_and_check(repo)[:2] == ('PASS', set())


@pytest.mark.parametrize('body', [
    'import pytest\n@pytest.mark.skip\ndef test_b():\n    assert True\n',
    'import pytest\ndef test_b():\n    pytest.xfail("later")\n',
    'import pytest\nyaml = pytest.importorskip("yaml")\n',
])
def test_new_test_module_with_skip_fails(repo, body):
    (repo / 'tests/test_new.py').write_text(body)
    assert _commit_and_check(repo)[1] == {'TEST_SKIP_ADDED'}


def test_skip_text_inside_a_string_is_not_a_skip(repo):
    (repo / 'tests/test_new.py').write_text('def test_b():\n    assert "@pytest.mark.skip" != "pytest.skip("\n')
    assert _commit_and_check(repo)[:2] == ('PASS', set())
