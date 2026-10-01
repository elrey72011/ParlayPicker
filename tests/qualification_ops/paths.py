"""Explicit clean application dependency, separate from the tooling checkout."""
import os
from pathlib import Path
import subprocess
REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / 'tools/qualification/snapshot_acquire_and_assess.py'
LEGACY = Path(__file__).resolve().parent / 'fixtures/legacy_snapshot_acquire.py'
TEMPLATE = REPO / 'docs/operations/snapshot-acquisition.example.json.template'
SOURCE = Path(os.environ.get('PARLAYPICKER_QUALIFICATION_APPLICATION_CHECKOUT', str(REPO))).resolve()
APPLICATION_SHA = '7c4fe71c7b9bd1a7ae73f8ba04b5e9a79d720eea'
APPLICATION_TREE = 'e9d684d0cdf695e3e6207641ab7a1e28eb00e5ea'
def verify_application():
    def git(*args): return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True).strip()
    if git('rev-parse', 'HEAD') != APPLICATION_SHA or git('rev-parse', 'HEAD^{tree}') != APPLICATION_TREE:
        raise RuntimeError('TEST_APPLICATION_DEPENDENCY_CONFLICT: supply a clean separate pinned checkout')
    if git('status', '--porcelain=v1', '--untracked-files=no'):
        raise RuntimeError('TEST_APPLICATION_DEPENDENCY_MODIFIED')
    return SOURCE
