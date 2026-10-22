"""Synthetic immutable preparation, corruption rejection and ownership controls."""
from pathlib import Path
import subprocess
import pytest
from tests import immutable_git_fixture as image


def prepared(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'input').write_bytes(b'authenticated input')
    repo = tmp_path / 'template'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=repo)
    git('init', '-q')
    git('config', 'user.name', 'Synthetic')
    git('config', 'user.email', 'synthetic@example.invalid')
    git('config', 'gc.auto', '0')
    git('config', 'maintenance.auto', 'false')
    (repo / 'retained').write_bytes(b'required object')
    git('add', 'retained')
    git('commit', '-qm', 'synthetic required object')
    git('gc','--prune=never')
    return source, image.freeze(repo, {'binding': {'expected': True}},
                                 image.source_identity(source, ['input']))


def test_materializations_have_independent_git_and_metadata(tmp_path):
    root, snapshot = prepared(tmp_path)
    a, b = tmp_path/'a', tmp_path/'b'
    first = image.materialize(snapshot, a, root)
    first['binding']['expected'] = False
    (a/'retained').write_bytes(b'changed')
    (a/'.git/config').write_bytes(b'changed')
    second = image.materialize(snapshot, b, root)
    assert second == {'binding': {'expected': True}}
    assert (b/'retained').read_bytes() == b'required object'
    assert (b/'.git/config').read_bytes() != b'changed'


def test_warm_preparation_does_not_cache_source_acceptance(tmp_path):
    root, snapshot = prepared(tmp_path)
    image.materialize(snapshot, tmp_path/'first', root)
    (root/'input').write_bytes(b'altered after warm use')
    with pytest.raises(ValueError, match='FIXTURE_PREPARATION_INPUT_CHANGED'):
        image.materialize(snapshot, tmp_path/'second', root)
    assert not (tmp_path/'second').exists()


@pytest.mark.parametrize('corrupt', [False, True])
def test_missing_or_corrupt_required_object_rejects_and_cleans(tmp_path, corrupt):
    root, snapshot = prepared(tmp_path)
    rows, metadata, inputs = snapshot
    name, _ = next(row for row in rows if row[0].startswith('.git/objects/')
                   and row[1] is not None and (len(Path(row[0]).name) == 38 or row[0].endswith('.pack')))
    altered = tuple((p, b'corrupt') if p == name and corrupt else (p, raw)
                    for p, raw in rows if corrupt or p != name)
    destination = tmp_path/'broken'
    with pytest.raises(subprocess.CalledProcessError):
        image.materialize((altered, metadata, inputs), destination, root)
    assert not destination.exists()


def test_interruption_cleans_only_owned_repository(tmp_path, monkeypatch):
    root, snapshot = prepared(tmp_path)
    unrelated = tmp_path/'unrelated'
    unrelated.mkdir()
    (unrelated/'keep').write_bytes(b'keep')
    def interrupt(*a, **kw):
        raise KeyboardInterrupt()
    monkeypatch.setattr(image.subprocess, 'run', interrupt)
    with pytest.raises(KeyboardInterrupt):
        image.materialize(snapshot, tmp_path/'interrupted', root)
    assert not (tmp_path/'interrupted').exists()
    assert (unrelated/'keep').read_bytes() == b'keep'
