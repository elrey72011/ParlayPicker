"""Exact original bytes and live rejection behavior; no history-dependent inputs."""
from pathlib import Path
import hashlib,json,inspect
import pytest
from scripts import check_launch_change_scope as guard
from scripts import nfl_reconcile_performance_scope as scope

def original():return (Path(__file__).resolve().parents[1]/guard.GUARD_PATH).read_bytes().replace(b'\r\n',b'\n')

@pytest.mark.parametrize('target',['prior','common'])
def test_exact_predecessor_bytes_without_missing_git_history(target):
 source=original();b=json.loads((Path(__file__).resolve().parents[1]/scope.BINDING).read_bytes())
 result=guard._nfl_recon_perf_project(source,target)
 assert hashlib.sha256(result).hexdigest()==b['projection_subjects'][target]['sha256']
 assert len(result)==b['projection_subjects'][target]['bytes']
 assert guard._nfl_recon_perf_project(source,target)==result

@pytest.mark.parametrize('attack',['append','same_length','truncate','identity'])
def test_changed_bytes_cannot_reuse_projection(attack):
 source=original()
 if attack=='append':source+=b'\nUNREVIEWED=True\n'
 elif attack=='same_length':source=source.replace(b'NFL_RECON_PERF_BOOTSTRAP = True',b'NFL_RECON_PERF_BOOTSTRAP = None',1)
 elif attack=='truncate':source=source[:-1]
 else:source=source.replace(guard.NFL_RECON_PERF_REVIEWED_GUARD_SHA256.encode(),b'1'*64,1)
 with pytest.raises(ValueError):guard._nfl_recon_perf_project(source,'prior')

def test_unknown_projection_target_is_not_a_fallback():
 with pytest.raises(ValueError):guard._nfl_recon_perf_project(original(),'UNREVIEWED')

def test_all_reviewed_identities_and_unknown_identity_preserve_rejection():
 source=original()
 identities=sorted({v for k,v in vars(guard).items() if k.endswith('REVIEWED_GUARD_SHA256') and isinstance(v,str)})
 assert identities and all(guard._dfs_guard_matches(source,i) for i in identities)
 assert guard._dfs_guard_matches(source,'f'*64) is False
 assert all(guard._dfs_guard_matches(source+b'\nUNREVIEWED=True\n',i) is False for i in identities)

def test_context_limit_and_shared_runtime_identity_remain_enforced():
 runtime=guard._scope_reconstruction_for_validation()
 assert runtime.MAX_CONTEXT_FIELDS==512
 for name in runtime.NAMES:
  fn=getattr(guard,name)
  assert hasattr(fn,'reconstruction_memo') and fn.reconstruction_context_fields<=512
 assert runtime is guard._scope_reconstruction_for_validation()
 source=original();assert guard._nfl_recon_perf_project(source,'prior')
 assert not hasattr(guard._nfl_recon_perf_project,'cache_info')
