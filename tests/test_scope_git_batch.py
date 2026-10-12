"""Synthetic Git only. Live mutable checks, isolated fixtures and exact bytes."""
from pathlib import Path
import hashlib
import io
import subprocess
import types
import pytest
from scripts import scope_git_batch as batch
from scripts.benchmark_drive_history_loading import blocked_network

def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root).decode().strip()

def commit(root, content, name='item'):
    (root/name).write_bytes(content)
    git(root, 'add', '--all'); git(root, 'commit', '-qm', 'labelled synthetic object')
    return git(root, 'rev-parse', 'HEAD')

@pytest.fixture
def repositories(tmp_path):
    roots = []
    with blocked_network():
        for name in ('one', 'two'):
            root = tmp_path/name; root.mkdir()
            git(root, 'init', '-q'); git(root, 'config', 'user.name', 'Offline')
            git(root, 'config', 'user.email', 'offline@example.invalid')
            git(root, 'config', 'core.autocrlf', 'false')
            commit(root, name.encode()); roots.append(root)
        yield roots

def adapter(root):
    g = types.SimpleNamespace(ROOT=root)
    def text(*args, check=True):
        r = subprocess.run(['git', *args], cwd=g.ROOT, capture_output=True, text=True)
        if check and r.returncode: raise RuntimeError('ORIGINAL_GIT_FAILURE')
        return r.stdout.strip()
    g.git = text
    g.git_bytes = lambda *args: subprocess.check_output(['git', *args], cwd=g.ROOT)
    g.exists_at = lambda rev, path: subprocess.run(['git', 'cat-file', '-e', rev+':'+path], cwd=g.ROOT, capture_output=True).returncode == 0
    g._run_synthetic_integrated = lambda: g.git_bytes('show', 'HEAD:item')
    batch.install(g); return g

def test_byte_exact_body_and_blob_identity(repositories):
    root, _ = repositories; body = b'original\x00\xff\r\ncomplete\n'
    rev = commit(root, body)
    g = adapter(root)
    with batch.validation_stream(root):
        assert g.exists_at(rev, 'item')
        assert g.git('rev-parse', rev+':item') == git(root, 'rev-parse', rev+':item')
        assert g.git_bytes('show', rev+':item') == body

def test_head_and_new_objects_reread_in_same_stream(repositories):
    root, _ = repositories; g = adapter(root)
    with batch.validation_stream(root):
        before = g.git('rev-parse', 'HEAD:item')
        assert g.git_bytes('show', 'HEAD:item') == b'one'
        assert not g.exists_at('HEAD', 'new')
        commit(root, b'changed'); commit(root, b'new bytes', 'new')
        assert g.git('rev-parse', 'HEAD:item') != before
        assert g.git_bytes('show', 'HEAD:item') == b'changed'
        assert g.exists_at('HEAD', 'new')

def test_no_cross_repository_contamination(repositories):
    one, two = repositories; g = adapter(one)
    with batch.validation_stream(one) as stream:
        assert g._run_synthetic_integrated() == b'one'
        g.ROOT = two
        assert g.git_bytes('show', 'HEAD:item') == b'two'
        with batch.validation_stream(two) as other:
            assert other is not stream
            assert g._run_synthetic_integrated() == b'two'
        g.ROOT = one
        assert g._run_synthetic_integrated() == b'one'

def test_dirty_staged_committed_and_next_run_are_live(repositories):
    root, _ = repositories; g = adapter(root)
    assert g._run_synthetic_integrated() == b'one'
    (root/'item').write_bytes(b'dirty')
    with batch.validation_stream(root):
        assert g.git('diff', '--name-only') == 'item'
        assert g.git_bytes('show', 'HEAD:item') == b'one'
        git(root, 'add', 'item')
        assert g.git('diff', '--cached', '--name-only') == 'item'
        git(root, 'commit', '-qm', 'changed mutable state')
        assert g.git_bytes('show', 'HEAD:item') == b'dirty'
    assert g._run_synthetic_integrated() == b'dirty'

def test_nested_stream_reused_only_in_one_validation_and_closed(repositories):
    root, _ = repositories; g = adapter(root)
    with batch.validation_stream(root) as first:
        assert g._run_synthetic_integrated() == b'one'
        process = first.process
        with batch.validation_stream(root) as inner: assert inner is first
    assert process.poll() == 0 and batch._active.get() is None
    with batch.validation_stream(root) as second: assert second is not first

def test_exception_cleans_worker_and_context(repositories):
    root, _ = repositories
    with pytest.raises(ValueError):
        with batch.validation_stream(root) as stream:
            assert stream.contents('HEAD:item') == b'one'; process = stream.process
            raise ValueError('labelled failure')
    assert process.poll() == 0 and batch._active.get() is None

def test_missing_object_and_unsupported_revision_preserve_fallback(repositories):
    root, _ = repositories; g = adapter(root)
    with batch.validation_stream(root):
        assert not g.exists_at('HEAD', 'absent')
        with pytest.raises(subprocess.CalledProcessError): g.git_bytes('show', 'HEAD:absent')
        assert g.git_bytes('show', 'HEAD~0:item') == b'one'

def test_tree_and_internal_size_bound_fall_back_without_truncation(repositories, monkeypatch):
    root, _ = repositories; g = adapter(root)
    commit(root, b'complete original bytes')
    monkeypatch.setattr(batch, 'MAX_OBJECT_BYTES', 3)
    with batch.validation_stream(root):
        assert g.git_bytes('show', 'HEAD:item') == b'complete original bytes'

@pytest.mark.parametrize('spec', ['HEAD:item\ncontents HEAD:item', 'HEAD:item\r', 'HEAD:item\x00', 'other:item'])
def test_unapproved_batch_expression_not_dispatched(spec):
    assert batch._SPEC.fullmatch(spec) is None

@pytest.mark.parametrize('body,header,reason', [
    (b'changed', b'0'*40+b' blob 7\n', 'CORRUPT_OBJECT'),
    (b'short', b'0'*40+b' blob 7\n', 'INCOMPLETE_OBJECT'),
    (b'', b'invalid\n', 'INVALID_HEADER'),
])
def test_malformed_or_changed_object_fails_closed(tmp_path, body, header, reason):
    stream = batch.ObjectStream(tmp_path)
    info = b'0'*40+b' blob 7\n'
    stream.process = types.SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(info+header+body+b'\n'))
    with pytest.raises(RuntimeError, match='SCOPE_GIT_BATCH_'+reason): stream.contents('HEAD:item')

def test_guard_reconstruction_cache_is_content_bound_and_bounded():
    from scripts import check_launch_change_scope as guard
    source = Path(guard.__file__).read_bytes().replace(b'\r\n', b'\n')
    guard._guard_performance_cached_base_source.cache_clear()
    before = guard._guard_performance_base_source(source)
    assert hashlib.sha256(before).hexdigest() == guard.GUARD_PERFORMANCE_PREVIOUS_GUARD_SHA256
    assert guard._guard_performance_base_source(source) == before
    assert guard._guard_performance_cached_base_source.cache_info().maxsize == 2
    with pytest.raises(ValueError, match='SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED'):
        guard._guard_performance_base_source(source+b'\nUNREVIEWED=True\n')
    with pytest.raises(ValueError, match='SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED'):
        guard._guard_performance_base_source(source+b'x'*(8*1024*1024))

def test_cached_source_reconstruction_carries_no_repository_state(repositories):
    from scripts import check_launch_change_scope as guard
    source = Path(guard.__file__).read_bytes().replace(b'\r\n', b'\n')
    frozen = guard._guard_performance_base_source(source)
    for root in repositories:
        with batch.validation_stream(root):
            assert guard._guard_performance_base_source(source) == frozen
    assert b'GUARD_PERFORMANCE_BOOTSTRAP' not in frozen
