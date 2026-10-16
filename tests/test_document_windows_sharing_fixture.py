"""Exact approved fixture semantics; no Git-history dependency or external transports."""
import ast
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tested_windows_sharing_scope',ROOT/'scripts/football_document_windows_sharing_scope.py')
scope=importlib.util.module_from_spec(spec);spec.loader.exec_module(scope)

def current():
    return (ROOT/scope.FIXTURE).read_bytes().replace(b'\r\n',b'\n')

def method(raw):
    cls=next(n for n in ast.parse(raw).body if isinstance(n,ast.ClassDef) and n.name=='Tests')
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='test_M03_windows_subprocess_delete_sharing_release')

def test_exact_fixture_parent_and_every_original_assertion_preserved():
    raw=current();parent=scope.fixture_parent(raw)
    assert scope.assertion_nodes(parent)==scope.assertion_nodes(raw)
    assert len(scope.assertion_nodes(raw))>20

def test_bounded_wait_follows_actual_original_oserror_and_reraise():
    node=method(current())
    replacement=next(n for n in ast.walk(node) if isinstance(n,ast.FunctionDef) and n.name=='replacement')
    attempt=next(n for n in replacement.body if isinstance(n,ast.Try))
    assert ast.unparse(attempt.body[0])=='return original(path, target)'
    handler=attempt.handlers[0]
    assert ast.unparse(handler.type)=='OSError'
    assert [ast.unparse(n) for n in handler.body]==["errors.append(getattr(exc, 'winerror', None))",'first_error.set()','raise']
    waits=[n for n in ast.walk(node) if isinstance(n,ast.Call) and ast.unparse(n.func)=='first_error.wait']
    assert len(waits)==1 and len(waits[0].args)==1 and waits[0].args[0].value==5

@pytest.mark.parametrize('before,after',[
    (b'first_error.wait(5)',b'first_error.wait(50)'),
    (b'first_error.set()',b'first_error.clear()'),
    (b'self.assertTrue(errors)',b'self.assertTrue(True)'),
    (b"self.assertEqual(meter.sequence, 4)",b"self.assertEqual(meter.sequence, 3)"),
])
def test_altered_or_rehashed_fixture_cannot_borrow_reviewed_parent(before,after):
    raw=current();assert before in raw
    with pytest.raises(ValueError,match='FROZEN_FIXTURE_REVIEWED_BYTES_CHANGED'):
        scope.fixture_parent(raw.replace(before,after,1))
