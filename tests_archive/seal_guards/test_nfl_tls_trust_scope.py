"""Predecessor expectations were independently read from cleared c548ddb."""
import ast
import copy
import hashlib
from pathlib import Path

import pytest
from scripts import ncaaf_filtered_history_scope as history
from tests.nfl_profile_generation_inputs import record_bytes

ROOT = Path(__file__).resolve().parents[1]
PREDECESSORS = {
    'app_core/nfl_build_preflight.py': (12444, '2e8120a76062e16efb7270e304f54f6d7d08aff3e50ba2d4ccc821e1047cfd92'),
    'scripts/nfl_first_driver_capacity_scope.py': (10150, '6ee2301fc64c6db682f204dd478036173373be8f2918f820db57c8a19b6079f7'),
    'tests/nfl_profile_generation_inputs.py': (4915, '787441c9794893e2df413a874bdb1d9064b101c0ec442d529df4403e14124c34')}
UNCHANGED = {
    'app_core/nfl_storage_diagnostic.py': (18301, 'cd6d72b2775d5d57279d88bdd71751c56c05d56c26808fb4e2ef4add4c12929b'),
    'app_core/nfl_diagnostic_receipts.py': (7352, 'a6236b6e7ad13dd90d23e31e55ea15c8e92dc406111d1c9fbe20f5ec8b8ad312'),
    'app_core/nfl_workflow_store.py': (15950, '545a7257b5117a43bd6d55c88e69cfa42a5671392cdbdf5f8a7a8e4c4fb90c3e'),
    'app_core/market_probability_model.py': (14263, '5cd25936548870df4bca01b929d4c2df72c7c905b4f32f6f7d7efe07c1c3cd63')}


def metadata(binding):
    return binding['qualification_release_repair']['build_preflight']['receipt_successor']['first_driver_successor']['capacity_context_successor']['tls_trust_successor']


def test_pre_tls_beforeimages_match_independently_pinned_bytes():
    binding = history.authenticate(ROOT)
    record = metadata(binding)
    assert record['base'] == 'c548ddb2693b64c94942ff315eed93e3fcb2f56e'
    assert record['tree'] == 'e9c9ab85b66576cb9a04daa83ed956af69955b26'
    assert record['prior_binding_sha256'] == 'ac2abfe85d86f20c9cbbb10be72fdba9a4a61c6c54ab588e4a12bed45fb0167b'
    assert set(record['historical_sources']) == set(PREDECESSORS)
    for path, (size, digest) in PREDECESSORS.items():
        raw = record_bytes(record['historical_sources'][path])
        assert len(raw) == size and hashlib.sha256(raw).hexdigest() == digest


@pytest.mark.parametrize('path', list(UNCHANGED))
def test_connection_diagnostic_store_model_and_receipts_are_unchanged(path):
    raw = (ROOT/path).read_bytes().replace(b'\r\n', b'\n')
    size, digest = UNCHANGED[path]
    assert len(raw) == size and hashlib.sha256(raw).hexdigest() == digest


def test_cold_and_warm_predecessor_tampering_rejects():
    record = metadata(history.authenticate(ROOT))
    row = record['historical_sources']['app_core/nfl_build_preflight.py']
    for warm in (False, True):
        if warm:
            assert record_bytes(row)
        bad = copy.deepcopy(row)
        bad['sha256'] = '0'*64
        with pytest.raises(ValueError, match='NFL_PROFILE_GENERATION_CHANGED'):
            record_bytes(bad)


def test_historical_approval_constants_are_not_rewritten():
    record = metadata(history.authenticate(ROOT))
    path = 'scripts/nfl_first_driver_capacity_scope.py'
    before = record_bytes(record['historical_sources'][path])
    def approvals(raw):
        return next(ast.literal_eval(n.value) for n in ast.parse(raw).body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'APPROVED' for t in n.targets))
    assert approvals(before) == approvals((ROOT/path).read_bytes())
