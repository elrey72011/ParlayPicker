"""Static guard for the GitHub Actions receipt job that previously never started."""
from pathlib import Path

import yaml


def test_receipt_workflow_has_valid_runner_context_and_audited_order():
    path = Path(__file__).resolve().parents[1] / '.github/workflows/mlb-receipt-reconciliation.yml'
    workflow = yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    assert 'workflow_dispatch' in workflow['on']
    assert workflow['on']['schedule'][0]['cron'] == '30 12,18 * * *'
    job = workflow['jobs']['reconcile']
    assert job['if'] == "vars.RESEARCH_SCHEDULER_ENABLED == 'true' && inputs.readiness_audit != true"
    assert 'ODDS_API_KEY' in job['env']
    # `runner` is only available in step scope, not in jobs.<job>.env.
    assert 'runner.' not in str(job.get('env', {}))
    steps = job['steps']
    command = next(i for i, step in enumerate(steps) if step.get('id') == 'reconcile')
    script = steps[command]['run']
    assert '$RUNNER_TEMP/mlb-receipt-evidence' in script
    assert 'reconcile-remote --max-feeds 100 --output' in script
    assert '--capture-live --max-capture-feeds 20' in script
    artifact = next(i for i, step in enumerate(steps) if step.get('uses') == 'actions/upload-artifact@v4')
    assert command < artifact
    assert steps[artifact]['with']['if-no-files-found'] == 'error'
    assert any('DEPENDENCY_INSTALL_ERROR' in step.get('run', '') for step in steps)
    assert any('ARTIFACT_FAILURE' in step.get('run', '') for step in steps)
    readiness = workflow['jobs']['readiness-audit']
    assert readiness['if'] == 'inputs.readiness_audit == true'
    assert 'ODDS_API_KEY' not in readiness['env']
    assert any('run_mlb_production_readiness.py' in step.get('run', '') for step in readiness['steps'])
    assert any(step.get('with', {}).get('name') == 'mlb-production-readiness-audit'
               for step in readiness['steps'])
