"""Synthetic Git seals; no historical assertion or workflow exception."""
import json
from pathlib import Path
import subprocess

import pytest
from scripts import nfl_saved_scenario_scope as scope


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    root = tmp_path / 'SYNTHETIC-viewer-git'; root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root).decode().strip()
    def write(path, raw):
        p = root / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)
    def commit(message):
        git('add', '--all'); git('commit', '-qm', message); return git('rev-parse', 'HEAD')
    git('init', '-q'); git('config', 'user.name', 'SYNTHETIC test'); git('config', 'user.email', 'synthetic@example.invalid')
    git('config', 'core.autocrlf', 'false')
    manifest = 'docs/SYNTHETIC-baseline.json'
    write(manifest, b'{"base_sha":"SYNTHETIC-old"}\n')
    write('app_core/nfl_scenario_research.py', b'SYNTHETIC frozen reader\n')
    write('.github/workflows/ci.yml', b'SYNTHETIC unchanged checks\n')
    for name in ('scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py'):
        write(name, b'SYNTHETIC unchanged logic ' + scope.OLD_STAMP + b'\n')
    base = commit('synthetic cleared main'); tree = git('rev-parse', base + '^{tree}')
    monkeypatch.setattr(scope, 'BASE', base); monkeypatch.setattr(scope, 'TREE', tree)
    binding_sha = 'a' * 64
    for name in scope.PATHS:
        raw = (b'SYNTHETIC unchanged logic ' + binding_sha.encode() + b'\n') if name in {
            'scripts/check_launch_change_scope.py', 'scripts/ncaaf_private_refresh_scope.py'} else b'SYNTHETIC implementation\n'
        write(name, raw)
    b = dict(baseline_manifest=manifest, components={'docs/paid-launch/nfl-saved-scenarios-v1.md': 'SYNTHETIC-authority'},
        saved_nfl_scenario_viewer=dict(base=base, base_tree=tree, implementation_paths=sorted(scope.PATHS), authorization_sha256='SYNTHETIC-authority'))
    implementation = commit('synthetic implementation')
    policy = scope.make_policy(root, b, implementation, binding_sha)
    write(scope.POLICY, json.dumps(policy).encode()); seal = commit('synthetic policy only')
    return root, b, binding_sha, git, write, commit, seal


def assess(fx):
    root, b, stamp, *_ = fx
    return scope.validate(root, b, None, None, binding_sha=stamp)


def test_exact_viewer_seal_and_ordered_ci_merge(sealed):
    assert assess(sealed)['status'] == 'PASS'
    root, b, stamp, git, write, commit, seal = sealed
    merge = git('commit-tree', git('rev-parse', seal + '^{tree}'), '-p', b['saved_nfl_scenario_viewer']['base'], '-p', seal, '-m', 'synthetic tested merge')
    git('checkout', '-q', merge)
    assert assess(sealed)['status'] == 'PASS'


@pytest.mark.parametrize('attack', ['dirty', 'untracked', 'frozen_reader', 'workflow', 'policy', 'paths', 'base', 'extra_commit', 'wrong_merge'])
def test_unreviewed_changes_are_rejected(sealed, attack):
    root, b, stamp, git, write, commit, seal = sealed
    if attack == 'dirty': write('app/ui/nfl_private_research.py', b'UNREVIEWED\n')
    elif attack == 'untracked': write('UNREVIEWED.py', b'UNREVIEWED\n')
    elif attack in {'frozen_reader', 'workflow'}:
        write('app_core/nfl_scenario_research.py' if attack == 'frozen_reader' else '.github/workflows/ci.yml', b'UNREVIEWED\n'); commit('unreviewed')
    elif attack == 'policy':
        p = json.loads((root / scope.POLICY).read_bytes()); p['unchanged_blobs'] = {}; write(scope.POLICY, json.dumps(p).encode()); git('add', '--all'); git('commit', '--amend', '--no-edit', '-q')
    elif attack == 'paths': b['saved_nfl_scenario_viewer']['implementation_paths'].append('UNREVIEWED.py')
    elif attack == 'base': b['saved_nfl_scenario_viewer']['base'] = 'f' * 40
    elif attack == 'extra_commit': git('commit', '--allow-empty', '-qm', 'unreviewed')
    else:
        bad = git('commit-tree', git('rev-parse', seal + '^{tree}'), '-p', seal, '-p', b['saved_nfl_scenario_viewer']['base'], '-m', 'wrong parents'); git('checkout', '-q', bad)
    with pytest.raises(ValueError): assess(sealed)
