"""Shallow-safe exact predecessor checks; full-history live scope is checked separately."""
from pathlib import Path
from copy import deepcopy
import hashlib
import pytest
from scripts import ncaaf_private_refresh_scope as boundary,ncaaf_sidebar_startup_scope as successor
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PREDECESSORS={'app/ui/sidebar_controls.py': (18035, '76b0e6647f7572e0c87be2037d17f569c53ac8f0d0aacd6f68e4d069e099cf82'), 'app/ui/results_overview.py': (8096, '27c0376e09ecf392c156bac57df2866a3458b4158d3cfd3b08cd6b16b9ad4c62'), 'scripts/ncaaf_saved_refresh_scope.py': (10179, 'cbc3cf1e50fbd591633c61545baef6f0fd4d20974b015c051d18c8fd8d2f570f'), 'tests/nfl_profile_generation_inputs.py': (5854, 'd7ffe52dddd73244083c393cb7d151b70f7e6242b3cb86fc9c1960edcee24ffd'), 'scripts/ci_test_file_costs_v1.json': (3397960, '948d2213647c839470c7b0ed0f323315f3113b3c5741bb4dd85917b3ac4fb6de')}
EXPECTED_PARENT_BYTES=4111078
EXPECTED_LEGACY_TEST=(10320,'8f0159b8786cebe2709a001d9f76e745e54baf49732d52ac37a1a219043124ed')

def test_exact_sidebar_successor_preserves_independently_pinned_base_bytes():
    b=boundary.authenticate(ROOT);m=successor.metadata(b)
    assert m['base']=='5dfbb74cd252140512acb0fd9f31d40306408888'
    assert set(m['historical_sources'])==set(EXPECTED_PREDECESSORS)|{'tests/test_ncaaf_prospective.py'}
    for path,value in m['historical_sources'].items():
        raw=successor.record_bytes(value)
        size,digest=EXPECTED_LEGACY_TEST if path=='tests/test_ncaaf_prospective.py' else EXPECTED_PREDECESSORS[path]
        assert len(raw)==size and hashlib.sha256(raw).hexdigest()==digest
    parent=successor.parent_binding(b)
    assert len(parent)==EXPECTED_PARENT_BYTES and hashlib.sha256(parent).hexdigest()==successor.OLD_STAMP
    from scripts import ncaaf_filtered_history_scope as history
    import json
    assert hashlib.sha256(history.capacity_parent_binding(b)).hexdigest()=='24d9bc417f9cb323f35395191d394fd6423b72ac924b7040b4bbf45f92a813db'
    for path in successor.UI_PREDECESSORS:
        current=boundary.physical(ROOT/path)
        prior=history.predecessor_source(ROOT,path,current)
        assert prior==successor.record_bytes(m['historical_sources'][path])
        assert history.predecessor_source(ROOT,path,prior)==prior
        assert history.predecessor_source(ROOT,path,current+b'\nUNAPPROVED_SIDEBAR_EDIT\n')==current+b'\nUNAPPROVED_SIDEBAR_EDIT\n'
    profile=json.loads(boundary.physical(ROOT/successor.PROFILE))
    successor.verify_estimates(profile)
    row=next(r for r in profile['files'] if r['path']=='tests/test_legacy_sidebar_startup.py')
    row['current_context_basis']=row.pop('sidebar_startup_estimate_basis')
    with pytest.raises(ValueError,match='SIDEBAR_ESTIMATE_CLASSIFICATION_CHANGED'):successor.verify_estimates(profile)

@pytest.mark.parametrize('warm',[False,True])
def test_corrupted_predecessor_identity_rejects_cold_and_after_prior_read(warm):
    b=boundary.authenticate(ROOT);value=deepcopy(next(iter(successor.metadata(b)['historical_sources'].values())))
    if warm:successor.record_bytes(value)
    value['sha256']='0'*64
    with pytest.raises(ValueError,match='SIDEBAR_PREDECESSOR_CHANGED'):successor.record_bytes(value)

def test_resealed_parent_component_substitution_rejects():
    b=deepcopy(boundary.authenticate(ROOT));m=successor.metadata(b)
    key=next(iter(m['parent_components']));m['parent_components'][key]='0'*64
    with pytest.raises(ValueError,match='SIDEBAR_PARENT_BINDING_CHANGED'):successor.parent_binding(b)

@pytest.mark.parametrize('path',['app/ui/sidebar_controls.py','scripts/ncaaf_sidebar_startup_scope.py',boundary.BINDING])
def test_physical_source_or_binding_tampering_rejects(tmp_path,path):
    b=boundary.authenticate(ROOT)
    for relative in [boundary.BINDING,*b['components']]:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(boundary.physical(ROOT/relative))
    assert boundary.authenticate(tmp_path)==b
    p=tmp_path/path;p.write_bytes(p.read_bytes()+b'\nSYNTHETIC_UNAUTHORIZED_EDIT\n')
    with pytest.raises(ValueError):boundary.authenticate(tmp_path)
