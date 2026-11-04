"""Proposed seven-capacity fixture isolation; synthetic, no transport/history."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from tests import test_football_document_predecessor_fixture as old
from scripts import check_launch_change_scope as guard

ROOT=Path(__file__).resolve().parents[1]

def test_exact_six_input_is_idempotent_and_live_seven_source_is_unchanged():
    source=(ROOT/old.TARGET).read_bytes()
    prior=old._exact_six_shard_input(old.TARGET,source)
    record=json.loads(guard.NFL_PDF_TIMING_RECONSTRUCTION_JSON)[old.TARGET]
    assert hashlib.sha256(prior).hexdigest()==record['before_sha256']
    assert prior!=source and old._exact_six_shard_input(old.TARGET,prior)==prior
    assert b"['1', '2', '3', '4', '5', '6', '7']" in source
    assert (ROOT/old.TARGET).read_bytes()==source

@pytest.mark.parametrize('attack',['append','changed_cardinality','wrong_path'])
def test_unknown_inputs_cannot_borrow_exact_predecessor(attack):
    source=(ROOT/old.TARGET).read_bytes()
    if attack=='wrong_path':
        assert old._exact_six_shard_input('tests/UNREVIEWED.py',source)==source
        assert old.historical._historical_five_shard_bytes('tests/UNREVIEWED.py',source)==source
    else:
        changed=source+b' ' if attack=='append' else source.replace(b"'6'",b"'99'",1)
        with pytest.raises(AssertionError):old._exact_six_shard_input(old.TARGET,changed)

def test_isolated_fixture_restores_module_and_preserves_source(tmp_path,monkeypatch):
    original=old.ROOT;file=old.historical.__file__;raw=(ROOT/old.TARGET).read_bytes()
    with monkeypatch.context() as m:first=old.historical_six_shard_input.__wrapped__(tmp_path/'one',m)
    assert old.ROOT==original and old.historical.__file__==file
    with monkeypatch.context() as m:second=old.historical_six_shard_input.__wrapped__(tmp_path/'two',m)
    assert old.ROOT==original and old.historical.__file__==file
    expected=(second/old.TARGET).read_bytes();(first/old.TARGET).write_bytes(b'SYNTHETIC_UNREVIEWED')
    assert (second/old.TARGET).read_bytes()==expected and (ROOT/old.TARGET).read_bytes()==raw

def test_fixture_requires_no_git_history(tmp_path,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('SYNTHETIC_HISTORY_ACCESS_FORBIDDEN')
    monkeypatch.setattr(subprocess,'check_output',forbidden)
    with monkeypatch.context() as m:root=old.historical_six_shard_input.__wrapped__(tmp_path,m)
    assert not (root/'.git').exists()
    assert hashlib.sha256((root/old.TARGET).read_bytes()).hexdigest()==json.loads(guard.FOOTBALL_DOCSIX_RECONSTRUCTION_JSON)[old.TARGET]['after_sha256']

def test_original_substantive_assertions_and_function_identities_are_byte_equivalent():
    text=(ROOT/'tests/test_football_document_predecessor_fixture.py').read_text()
    values={n.name:[ast.dump(a) for a in ast.walk(n) if isinstance(a,ast.Assert)] for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    assert hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()=='9a9e86308286e82f2af3bac787aaf056b1643009be94d512825a7074bcd8c7e8'


# Freeze only this historical seven-way fixture contract; the current nine-way
# gates continue to read the live configuration without an input override.
@pytest.fixture(autouse=True)
def exact_seven_way_predecessor_input(tmp_path,monkeypatch):
    from tests.ncaaf_staged_capacity_fixtures import exact_predecessor_bytes
    root=tmp_path/'SYNTHETIC-exact-seven-way-input'
    for path in (old.TARGET,'tests/test_football_document_predecessor_fixture.py'):
        source=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
        destination=root/path;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(exact_predecessor_bytes(path,source))
    monkeypatch.setitem(globals(),'ROOT',root)
    return root
