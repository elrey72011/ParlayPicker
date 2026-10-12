"""Synthetic bootstrap/read-only helper boundaries; no provider operations."""
from pathlib import Path
import types
import pytest
from scripts import check_launch_change_scope as g

def test_isolated_bootstrap_needs_no_transport_or_successor_binding(tmp_path):
    p=tmp_path/'scripts/check_launch_change_scope.py';p.parent.mkdir()
    source=Path(g.__file__).read_bytes().replace(b'\r\n',b'\n');p.write_bytes(source)
    isolated=types.ModuleType('SYNTHETIC_isolated_guard');isolated.__file__=str(p)
    exec(compile(source,str(p),'exec'),isolated.__dict__)
    assert isolated._nfl_observed_base_source(source)==g._nfl_observed_base_source(source)
    assert isolated.NFL_OBSERVED_SOURCE_RECONSTRUCTIONS==g.NFL_OBSERVED_SOURCE_RECONSTRUCTIONS
    with pytest.raises(FileNotFoundError):isolated._scope_git_batch_for_validation()

def test_helper_is_reverified_after_install_and_other_repository_cannot_supply_it(tmp_path,monkeypatch):
    original=Path(g.__file__).resolve();p=tmp_path/'scripts/check_launch_change_scope.py';p.parent.mkdir()
    helper=p.parent/'scope_git_batch.py'
    helper.write_bytes((original.parent/'scope_git_batch.py').read_bytes())
    monkeypatch.setattr(g,'__file__',str(p))
    runtime=g._scope_git_batch_for_validation()
    assert runtime is not None
    helper.write_bytes(helper.read_bytes()+b'\nSYNTHETIC_UNREVIEWED=True\n')
    with pytest.raises(ValueError,match='SUCCESSOR_SCOPE_MODULE_CHANGED'):g._scope_git_batch_for_validation()
    other=tmp_path/'other';other.mkdir();monkeypatch.setattr(g,'ROOT',other)
    with pytest.raises(ValueError,match='SUCCESSOR_SCOPE_MODULE_CHANGED'):g._scope_git_batch_for_validation()
