"""Synthetic import-order and exact-byte regressions; no mutable decision cache."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from scripts import check_launch_change_scope as guard

ROOT=Path(__file__).resolve().parents[1]
PROOF='docs/paid-launch/football-document-context-projection-v1.json'


@pytest.mark.parametrize('order',['fixture_first','canonical_first'])
def test_actual_assertions_share_reviewed_runtime_in_either_import_order(order):
    # A fresh interpreter makes the previous-test state explicit. The real two
    # test functions and all their original assertions execute, without a shim.
    code="""
import pathlib,runpy,sys
root=pathlib.Path(sys.argv[1]);sys.path[:0]=[str(root),sys.argv[3]]
import pytest
from scripts import check_launch_change_scope as g
from scripts.benchmark_drive_history_loading import blocked_network
with blocked_network():
    first=runpy.run_path(str(root/'tests/test_football_document_html_scope.py'))
    later=runpy.run_path(str(root/'tests/test_nfl_listing_immutable_scope.py'))
    if sys.argv[2]=='canonical_first':g._scope_reconstruction_for_validation()
    patch=pytest.MonkeyPatch()
    try:
        first['test_pristine_fixture_preserves_bound_context_after_shared_memo_install'](patch)
        later['test_actual_hot_path_never_freezes_the_large_profile_declaration'](patch)
        runtime=g._scope_reconstruction_for_validation()
        assert g._nfl_odds_base_source.__globals__ is vars(runtime)
    finally:patch.undo()
print('SYNTHETIC_IMPORT_ORDER_PASS')
"""
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(ROOT),order,str(Path(pytest.__file__).resolve().parents[1])],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.strip()=='SYNTHETIC_IMPORT_ORDER_PASS'


def test_original_assertions_and_identity_are_unchanged():
    path='tests/test_football_document_html_scope.py'
    now=(ROOT/path).read_bytes();prior=guard._document_context_previous_source(path,now)
    def subject(raw):
        return {n.name:[ast.dump(v,include_attributes=False) for v in ast.walk(n) if isinstance(v,ast.Assert)]
                for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_')}
    assert now!=prior and subject(now)==subject(prior)
    assert now==prior.replace(b'    memo.install(guard)\n',b'    guard._scope_reconstruction_for_validation()\n',1)
    assert guard._document_context_previous_source(path,now+b'\nSYNTHETIC_CHANGED')==now+b'\nSYNTHETIC_CHANGED'


@pytest.fixture(scope='module')
def frozen_reference():
    raw=guard._document_context_base_source((ROOT/guard.GUARD_PATH).read_bytes())
    # Compile exactly the two frozen functions. Executing the entire prior
    # guard would make its bootstrap read the *new* __file__, an invalid input.
    # Unchanged delegated functions remain shared; no validation is stubbed.
    tree=ast.parse(raw);environment=dict(guard.__dict__)
    for name in ('_dfs_guard_matches','_listing_capacity_base_source'):
        node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
        exec(compile(ast.get_source_segment(raw.decode(),node),'<SYNTHETIC exact frozen function>','exec'),environment)
    return SimpleNamespace(**{n:environment[n] for n in ('_dfs_guard_matches','_listing_capacity_base_source')}),raw


def test_projection_derivation_and_live_hash_checks_are_exact(frozen_reference):
    reference,raw=frozen_reference
    data=json.loads((ROOT/PROOF).read_bytes())
    assert data['cache_added'] is False
    current_tree=ast.parse((ROOT/guard.GUARD_PATH).read_bytes())
    reference_tree=ast.parse(raw)
    for name,key in [('_dfs_guard_matches','dfs'),('_listing_capacity_base_source','capacity')]:
        old=[n for n in reference_tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
        new=[n for n in current_tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
        source=raw
        for record in data[key]['records']:
            assert len(source)==record['before_bytes']
            assert hashlib.sha256(source).hexdigest()==record['before_sha256']
            value=b''.join(bytes.fromhex(p['literal_hex']) if 'literal_hex' in p else source[p['start']:p['end']] for p in record['parts'])
            assert hashlib.sha256(value).hexdigest()==record['after_sha256'] and len(value)==record['after_bytes']
            test='len(source) == '+str(len(source))
            before=next(n for n in old.body if isinstance(n,ast.If) and ast.unparse(n.test)==test)
            after=next(n for n in new.body if isinstance(n,ast.If) and ast.unparse(n.test)==test)
            assert ast.dump(before.body[0])==ast.dump(after.body[0])
            assert ast.dump(before.body[1])==ast.dump(after.body[1])
            assert ast.dump(before.body[-1])==ast.dump(after.body[-1])
            source=value
        assert hashlib.sha256(source).hexdigest()==data[key]['target_sha256']


@pytest.mark.parametrize('attack',['unchanged','changed_body','changed_length'])
def test_complete_matcher_and_reconstruction_preserve_frozen_outcomes(frozen_reference,attack):
    reference,raw=frozen_reference
    source=raw if attack=='unchanged' else b'X'+raw[1:] if attack=='changed_body' else raw+b'\nSYNTHETIC_UNREVIEWED'
    for identity in [guard.DOCUMENT_SCAN_CI_REVIEWED_GUARD_SHA256,guard.NFL_PDF_REVIEWED_GUARD_SHA256,
                     guard.FOOTBALL_CATALOG_BINDINGS['successor_guard_sha256'],'0'*64]:
        assert guard._dfs_guard_matches(source,identity)==reference._dfs_guard_matches(source,identity)
    def outcome(fn):
        try:return ('bytes',fn(source))
        except (ValueError,RuntimeError) as exc:return (type(exc).__name__,str(exc))
    assert outcome(guard._listing_capacity_base_source)==outcome(reference._listing_capacity_base_source)
