"""Immutable preparation images; each materialization owns a fresh Git repository."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

MAX_BYTES = 32 * 1024 * 1024
MAX_FILES = 4096


def physical(path):
    with io.FileIO(path, 'r') as stream:
        return stream.read()


def source_identity(root, paths):
    return tuple((p, hashlib.sha256(physical(root / p)).hexdigest())
                 for p in sorted(set(paths)))


def freeze(repo, metadata, inputs):
    rows = []
    # Historical readers may intentionally project Path.rglob to tracked files.
    # A repository image must enumerate physical Git internals as well.
    for directory, folders, files in os.walk(repo, followlinks=False):
        if Path(directory)!=repo:
            rows.append((Path(directory).relative_to(repo).as_posix(),None))
        for name in folders + files:
            if os.path.islink(os.path.join(directory, name)):
                raise ValueError('FIXTURE_IMAGE_SYMLINK')
        for name in sorted(files):
            path=Path(directory)/name
            rows.append((path.relative_to(repo).as_posix(), physical(path)))
    rows.sort()
    if len(rows) > MAX_FILES or sum(len(value) for _, value in rows if value is not None) > MAX_BYTES:
        raise ValueError('FIXTURE_IMAGE_LIMIT')
    # JSON round-trip also rejects mutable/non-data fixture metadata.
    return tuple(rows), json.dumps(metadata, sort_keys=True), tuple(inputs)


def materialize(image, repo, root):
    rows, metadata, inputs = image
    if source_identity(root, (p for p, _ in inputs)) != inputs:
        raise ValueError('FIXTURE_PREPARATION_INPUT_CHANGED')
    repo.mkdir(exist_ok=False)
    try:
        for relative, value in rows:
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts or not relative:
                raise ValueError('FIXTURE_IMAGE_PATH')
            target = repo / path
            if value is None:
                target.mkdir(parents=True,exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with io.FileIO(target, 'w') as stream:
                stream.write(value)
        # Validate this writable copy, including required reachable objects.
        subprocess.run(['git', 'fsck', '--strict', '--no-reflogs'], cwd=repo,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return json.loads(metadata)
    except BaseException:
        shutil.rmtree(repo)
        raise
