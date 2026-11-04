"""SYNTHETIC stable Git template; original assertions/history stay intact."""
import ast
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from scripts import check_launch_change_scope as guard
from tests import test_remote_continuation_scope as historical

ROOT=Path(__file__).resolve().parents[1]


def test_remote_template_has_local_controls_and_complete_copy(tmp_path, tmp_path_factory):
    prepared=historical.prepared.__wrapped__(tmp_path_factory)
    repo=prepared[0]
    git=lambda cwd,*a:subprocess.check_output(['git',*a],cwd=cwd).strip()
    assert git(repo,'config','--local','--get','gc.auto')==b'0'
    assert git(repo,'config','--local','--get','maintenance.auto')==b'false'
    assert not [p for p in (repo/'.git/objects').glob('*/*') if p.parent.name not in {'pack','info'}]
    before=git(repo,'rev-parse','HEAD')
    tree=git(repo,'rev-parse','HEAD^{tree}')
    first=tmp_path/'SYNTHETIC-first';second=tmp_path/'SYNTHETIC-second'
    shutil.copytree(repo,first);shutil.copytree(repo,second)
    assert git(first,'rev-parse','HEAD')==git(second,'rev-parse','HEAD')==before
    assert git(first,'rev-parse','HEAD^{tree}')==git(second,'rev-parse','HEAD^{tree}')==tree
    assert subprocess.run(['git','fsck','--full','--no-reflogs'],cwd=first,capture_output=True).returncode==0
    protected='core/protected.py'
    (first/protected).write_bytes((first/protected).read_bytes()+b'\nSYNTHETIC_DIRTY\n')
    assert (first/protected).read_bytes()!=(second/protected).read_bytes()==(repo/protected).read_bytes()
    assert git(repo,'status','--porcelain')==b''


def test_remote_original_assertions_and_changed_source_rejection():
    path='tests/test_remote_continuation_scope.py'
    current=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
    previous=guard._docremote_previous_source(path,current)
    assert hashlib.sha256(previous).hexdigest()=='ec11f00e1f75c1d696ba99e651864a4af1d007185f81221954d7070a883f48dd'
    def assertions(raw):return {n.name:[ast.dump(a) for a in ast.walk(n) if isinstance(a,ast.Assert)] for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
    assert assertions(current)==assertions(previous)
    assert guard._docremote_previous_source(path,current+b'\nUNREVIEWED')==current+b'\nUNREVIEWED'
    assert guard._docremote_previous_source(path,previous)==previous
