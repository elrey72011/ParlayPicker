"""Exact local cloud preparation scope; no fixture changes to predecessors."""
import hashlib,json
from pathlib import Path
from scripts import check_launch_change_scope as guard
from scripts import ncaaf_cloud_backup_scope as scope

ROOT=Path(__file__).resolve().parents[1]
BASE='d4b73dcc05b45abb1d57dc995571ce892d4cd5cf'


def test_exact_parent_guard_and_baseline_preserved():
    current=(ROOT/'scripts/check_launch_change_scope.py').read_bytes().replace(b'\r\n',b'\n')
    assert guard._cloud_guard_matches(current)
    original=guard._cloud_peel_exact_guard(current)
    b=json.loads((ROOT/scope.BINDING).read_text())
    assert b['base']==BASE and b['previous_guard_sha256']==hashlib.sha256(original).hexdigest()
    baseline=(ROOT/'docs/paid-launch/launch-baseline-manifest.json').read_bytes().replace(b'\r\n',b'\n')
    assert b['manifest_sha256']==hashlib.sha256(baseline).hexdigest()


def test_frozen_setup_and_protected_files_unmodified():
    # Full-history protected-scope verifies these exact base bindings. The
    # application shard needs no network/history fetch: a Git blob digest
    # verifies the same bytes, including length and normalized line endings.
    policy=json.loads((ROOT/scope.POLICY).read_text())
    assert policy['base_sha']==BASE
    for path in ['app_core/ncaaf_pilot_setup.py','app_core/evidence_drive.py',
        'app_core/evidence_remote.py','app_core/prediction_evidence.py','app/ui/lock_picks.py',
        'app_core/public_history.py','app_core/stage_timing.py','app_core/performance_spans.py',
        'tests/test_ncaaf_pilot_setup.py','.github/workflows/ci.yml']:
        current=(ROOT/path).read_bytes().replace(b'\r\n',b'\n')
        blob=hashlib.sha1(b'blob '+str(len(current)).encode()+b'\0'+current).hexdigest()
        assert blob==policy['unchanged_bindings'][path]


def test_report_distinguishes_approved_fixture_from_unapproved_changes(monkeypatch):
    # Classification unit test only. Exact authorization/ancestry validation
    # remains the separate full-history protected-scope check; do not fetch
    # history or weaken its validator from an offline shallow application job.
    from types import SimpleNamespace
    binding=json.loads((ROOT/scope.BINDING).read_text())
    policy=json.loads((ROOT/scope.POLICY).read_text())
    monkeypatch.setattr(scope,'validate',lambda *args:(policy,[]))
    original={'label':'SYNTHETIC baseline report'}
    fake=SimpleNamespace(run=lambda *args:(0,original),git=lambda *args:'SYNTHETIC-head')
    code,report=scope.run(fake,ROOT/guard.MANIFEST_PATH,BASE,binding)
    assert code==0 and report['status']=='PASS'
    assert report['original_guard_report']==original
    assert report['protected_changes']==report['existing_test_changes']==[]
    assert report['new_existing_test_exceptions']==[]
    changes=report['approved_fixture_clock_changes']
    assert len(changes)==1 and changes[0]['path']=='tests/test_ncaaf_pilot.py'
    exact=report['approved_integration_changes'][changes[0]['path']]
    assert changes[0]['before_blob']==exact['before_blob']
    assert changes[0]['after_blob']==exact['after_blob']

