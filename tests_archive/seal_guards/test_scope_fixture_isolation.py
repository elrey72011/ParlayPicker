"""Synthetic scope-repository housekeeping and isolated mutable copies only."""
import importlib.util
from pathlib import Path
import shutil
import subprocess

from scripts.benchmark_drive_history_loading import blocked_network


def fixture_git():
    path = Path(__file__).with_name('test_mlb_nhl_research_scope.py')
    spec = importlib.util.spec_from_file_location('isolated_scope_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.git


def test_fixture_housekeeping_is_synchronous_without_global_configuration(tmp_path, monkeypatch):
    git = fixture_git()
    calls = []
    original = subprocess.check_output

    def capture(command, **kwargs):
        calls.append(command)
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, 'check_output', capture)
    with blocked_network():
        git(tmp_path, 'init', '-q')
        git(tmp_path, 'config', 'user.name', 'Offline synthetic')
        git(tmp_path, 'config', 'user.email', 'offline@example.invalid')
        git(tmp_path, 'config', 'gc.auto', '1')
        for i in range(600):
            original(['git', 'hash-object', '-w', '--stdin'], cwd=tmp_path,
                     input=('synthetic housekeeping object '+str(i)).encode())
        (tmp_path/'item').write_bytes(b'synthetic committed history')
        git(tmp_path, 'add', '--all')
        git(tmp_path, 'commit', '-qm', 'synthetic fixture history')
        git(tmp_path, 'gc', '--auto')
        head = git(tmp_path, 'rev-parse', 'HEAD')
        a = tmp_path.parent/'copy-one'
        b = tmp_path.parent/'copy-two'
        shutil.copytree(tmp_path, a)
        shutil.copytree(tmp_path, b)
        for root in (tmp_path, a, b):
            assert git(root, 'rev-parse', 'HEAD') == head
            git(root, 'fsck', '--full')
        (a/'item').write_bytes(b'isolated dirty mutation')
        assert git(a, 'diff', '--name-only') == 'item'
        assert not git(b, 'diff', '--name-only')
        assert (tmp_path/'item').read_bytes() == b'synthetic committed history'
        assert git(tmp_path, 'config', '--local', '--get', 'gc.auto') == '1'
        assert subprocess.run(['git', 'config', '--local', '--get', 'gc.autoDetach'],
                              cwd=tmp_path, capture_output=True).returncode == 1
    assert calls
    assert all(c[:5] == ['git', '-c', 'gc.autoDetach=false', '-c',
                        'maintenance.auto=false'] for c in calls)


def test_repository_copy_errors_are_not_concealed(tmp_path, monkeypatch):
    source = tmp_path/'source'
    source.mkdir()
    item = source/'original'
    item.write_bytes(b'synthetic disappearing object')
    copy = shutil.copy2

    def remove_then_copy(src, dst):
        item.unlink()
        return copy(src, dst)

    import pytest
    with pytest.raises(shutil.Error):
        shutil.copytree(source, tmp_path/'destination', copy_function=remove_then_copy)
