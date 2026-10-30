"""Exact successor authentication, not permission to operate."""
from pathlib import Path
import hashlib,json
import pytest
from scripts import ncaaf_private_refresh_scope as scope
from scripts import ncaaf_saved_refresh_scope as successor
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_PREDECESSORS={'app/ui/publish_panel.py': (17802, '83b4dc26cf76f2f18156d448855c73c5074853e4608d0223ffb6681792e74172'), 'scripts/nfl_tls_clock_scope.py': (9171, '4d80f1de3f2612159dda88a08fa24f5fb1784c3a659cd9bca6a164ca58cd5440'), 'tests/nfl_profile_generation_inputs.py': (5525, '6cacf422682be99b8a53e8fc7ee5a19220c6d4631a086a2d6f6e832b844d9428'), 'scripts/ci_test_file_costs_v1.json': (3396845, '599043bc08bce3f668613ef984c94bf485d1a92cc5eb4449bc20d413117d559e')}

def test_exact_source_successor_preserves_independent_predecessors():
    b=scope.authenticate(ROOT)
    m=successor.metadata(b)['ncaaf_saved_refresh_successor']
    assert m['base']=='e5d3b4d5e79115d42093adcd2708c0413df75a66'
    from tests.nfl_profile_generation_inputs import record_bytes
    assert set(m['historical_sources']) == set(EXPECTED_PREDECESSORS)
    for path,record in m['historical_sources'].items():
        raw=record_bytes(record)
        size,digest=EXPECTED_PREDECESSORS[path]
        assert len(raw)==size and hashlib.sha256(raw).hexdigest()==digest

@pytest.mark.parametrize('path',['app/ui/ncaaf_saved_refresh.py','app/ui/publish_panel.py',scope.BINDING])
def test_tampered_or_resealed_source_rejects(tmp_path,path):
    b=scope.authenticate(ROOT)
    for relative in [scope.BINDING,*b['components']]:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(scope.physical(ROOT/relative))
    assert scope.authenticate(tmp_path)==b
    target=tmp_path/path;target.write_bytes(target.read_bytes()+b'\nTAMPERED\n')
    with pytest.raises(ValueError):scope.authenticate(tmp_path)
