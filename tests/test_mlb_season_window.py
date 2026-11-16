"""MLB season gate: the window itself, and how the receipt workflow applies it. SYNTHETIC dates."""
from datetime import date
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.mlb_season_window import in_season

ROOT = Path(__file__).resolve().parents[1]
GATE = "github.event_name != 'schedule' || steps.season.outputs.active == 'true'"


def reconcile_steps():
    path = ROOT / '.github/workflows/mlb-receipt-reconciliation.yml'
    workflow = yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
    return workflow['jobs']['reconcile']['steps']


def test_window_opens_before_the_earliest_opener_and_closes_after_the_last_result():
    assert not in_season(date(2027, 1, 10))
    assert not in_season(date(2027, 3, 14))
    assert in_season(date(2027, 3, 15))
    assert in_season(date(2027, 3, 18))    # earliest overseas opening series so far
    assert in_season(date(2027, 7, 4))
    assert in_season(date(2027, 11, 5))    # latest World Series finish so far
    assert in_season(date(2027, 11, 15))
    assert not in_season(date(2027, 11, 16))
    assert not in_season(date(2027, 12, 31))


def test_gate_script_prints_one_word_without_the_application_dependencies():
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/mlb_season_window.py')],
                            capture_output=True, text=True, check=False, timeout=30)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in ('true', 'false')


def test_only_scheduled_runs_skip_the_install_and_the_store_restore_outside_the_season():
    steps = reconcile_steps()
    ids = [step.get('id') for step in steps]
    season = ids.index('season')

    assert 'scripts/mlb_season_window.py' in steps[season]['run']
    assert 'active=' in steps[season]['run'] and '$GITHUB_OUTPUT' in steps[season]['run']
    # The condition names the schedule event, so a manual run does the work in any month.
    for gated in ('dependencies', 'reconcile'):
        assert season < ids.index(gated)
        assert steps[ids.index(gated)]['if'] == GATE


def test_an_unreadable_gate_counts_as_in_season():
    season = next(step for step in reconcile_steps() if step.get('id') == 'season')

    # Capture is never lost to the gate: if the script cannot run, the work goes ahead.
    assert '|| echo true' in season['run']
    assert 'if' not in season


def test_a_skipped_run_uploads_no_audit_and_raises_no_classified_failure():
    steps = reconcile_steps()
    artifact = next(step for step in steps if step.get('uses') == 'actions/upload-artifact@v4')

    # The audit upload requires a file. It must stay tied to the reconcile step having run,
    # or an off-season run would fail on the missing audit instead of finishing cleanly.
    assert "steps.reconcile.outcome != 'skipped'" in artifact['if']
    assert artifact['with']['if-no-files-found'] == 'error'
