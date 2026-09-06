import json
import os
import subprocess
import sys

from app_core.evidence_health import evidence_health
from app_core import prediction_evidence as evidence
from test_prediction_evidence import frozen, fixture_frames


def test_missing_probe_does_not_create_store(tmp_path):
    path = tmp_path / 'missing' / 'evidence.sqlite3'
    assert evidence_health(path)['status'] == 'missing'
    assert not path.parent.exists()


def test_health_and_fresh_process_access(frozen):
    context, path, _ = frozen
    audit, final = fixture_frames()
    evidence.capture_run(context, audit, final, audit, path=path)
    result = evidence_health(path)
    assert result['status'] == 'healthy'
    assert result['snapshots'] == 1
    assert result['latest_candidates'] == result['latest_exact_quote_times'] == 2
    assert result['prior_process_snapshots_accessible'] is False
    script = ('import json,sys; from app_core.evidence_health import evidence_health; '
              'print(json.dumps(evidence_health(sys.argv[1])))')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(evidence.ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    child = subprocess.run([sys.executable, '-c', script, str(path)], env=env,
                           capture_output=True, text=True, check=True)
    restarted = json.loads(child.stdout)
    assert restarted['latest_snapshot_id'] == result['latest_snapshot_id']
    assert restarted['prior_process_snapshots_accessible'] is True
    assert restarted['durability_across_redeployment_verified'] is False


def test_corrupt_store_is_reported(tmp_path):
    path = tmp_path / 'evidence.sqlite3'
    path.write_bytes(b'not sqlite')
    assert evidence_health(path)['status'] == 'error'


def test_empty_store_and_legacy_runtime_metadata(frozen):
    context, path, _ = frozen
    assert evidence_health(path)['status'] == 'empty'
    audit, final = fixture_frames()
    evidence.capture_run(context, audit, final, audit, path=path)
    with evidence.connect(path) as db:
        db.execute('DROP TABLE snapshot_runtime')
    result = evidence_health(path)
    assert result['status'] == 'healthy'
    assert result['prior_process_snapshots_accessible'] is None
