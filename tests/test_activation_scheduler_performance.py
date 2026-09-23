"""Scheduled activation work retains immutable evidence before slow stages."""

import json
import sys

import pytest
import yaml


def test_unique_drive_restored_plan_is_used_without_local_file(tmp_path):
    from app_core.prediction_evidence import connect
    from scripts.bootstrap_evidence import _plans

    db_path = tmp_path / 'evidence.sqlite3'
    plan = {'sport': 'MLB', 'plan_hash': 'frozen-plan'}
    with connect(db_path) as db:
        db.execute('INSERT INTO validation_plans VALUES (?,?,?)',
                   (plan['plan_hash'], 'MLB', json.dumps(plan)))
    assert _plans(db_path, tmp_path / 'missing-plans')['MLB'] == plan


def test_ambiguous_drive_plans_require_explicit_choice(tmp_path):
    from app_core.prediction_evidence import connect
    from scripts.bootstrap_evidence import _plans

    db_path = tmp_path / 'evidence.sqlite3'
    with connect(db_path) as db:
        for name in ('one', 'two'):
            db.execute('INSERT INTO validation_plans VALUES (?,?,?)',
                       (name, 'MLB', json.dumps({'sport': 'MLB', 'plan_hash': name})))
    with pytest.raises(ValueError, match='Multiple frozen plans for MLB'):
        _plans(db_path, tmp_path / 'missing-plans')


def test_close_only_job_skips_validation_and_verifies_backup(monkeypatch, tmp_path):
    from scripts import bootstrap_evidence
    from app_core import activation_closing, evidence_remote
    from core import activation_validation

    calls = []
    monkeypatch.setattr(sys, 'argv', ['bootstrap_evidence.py', '--database', str(tmp_path / 'evidence.db'),
                                   '--restore', '--capture-closes', '--skip-validation', '--backup'])
    monkeypatch.setattr(evidence_remote, 'restore', lambda _db: calls.append('restore') or 1)
    monkeypatch.setattr(activation_closing, 'capture_live', lambda _db: calls.append('close') or {'verified': 0})
    monkeypatch.setattr(evidence_remote, 'sync', lambda *_args, **kwargs: calls.append(('backup', kwargs)) or True)
    monkeypatch.setattr(evidence_remote, 'remote_status', lambda: {'status': 'synced'})
    monkeypatch.setattr(activation_validation, 'validate', lambda *_args: pytest.fail('validation must be skipped'))

    assert bootstrap_evidence.main() == 0
    assert calls == ['restore', 'close', ('backup', {'incremental': True})]


def test_close_backup_precedes_slow_grading(monkeypatch, tmp_path):
    from scripts import bootstrap_evidence
    from app_core import activation_closing, evidence_remote, prediction_evidence

    calls = []
    monkeypatch.setattr(sys, 'argv', ['bootstrap_evidence.py', '--database', str(tmp_path / 'evidence.db'),
                                   '--capture-closes', '--grade', '--backup'])
    monkeypatch.setattr(activation_closing, 'capture_live', lambda _db: calls.append('close') or {'verified': 1})
    monkeypatch.setattr(evidence_remote, 'sync', lambda *_args, **_kwargs: calls.append('backup') or True)
    monkeypatch.setattr(evidence_remote, 'remote_status', lambda: {'status': 'synced'})

    def grade(*_args):
        calls.append('grade')
        raise TimeoutError('provider stalled')

    monkeypatch.setattr(prediction_evidence, 'refresh_outcomes', grade)
    with pytest.raises(TimeoutError):
        bootstrap_evidence.main()
    assert calls == ['close', 'backup', 'grade']


def test_score_backup_precedes_slow_ranking_rebuild(monkeypatch, tmp_path):
    from scripts import bootstrap_evidence
    from app_core import evidence_remote, prediction_evidence
    from core import ranking_evidence_rebuild

    calls = []
    monkeypatch.setattr(sys, 'argv', ['bootstrap_evidence.py', '--database', str(tmp_path / 'evidence.db'),
                                   '--grade', '--backup'])
    monkeypatch.setattr(prediction_evidence, 'refresh_outcomes',
                        lambda _db: calls.append('grade') or {'revisions': 1})
    monkeypatch.setattr(evidence_remote, 'sync', lambda *_args, **_kwargs: calls.append('backup') or True)
    monkeypatch.setattr(evidence_remote, 'remote_status', lambda: {'status': 'synced'})

    def rebuild(*_args):
        calls.append('rebuild')
        raise TimeoutError('rebuild stalled')

    monkeypatch.setattr(ranking_evidence_rebuild, 'rebuild', rebuild)
    with pytest.raises(TimeoutError):
        bootstrap_evidence.main()
    assert calls == ['grade', 'backup', 'rebuild']


def test_closing_schedule_is_independent_of_daily_grading():
    from pathlib import Path

    frequent = yaml.safe_load(Path('.github/workflows/research-scheduler.yml').read_text())
    daily = yaml.safe_load(Path('.github/workflows/activation-grading.yml').read_text())
    close_steps = frequent['jobs']['activation-evidence']['steps']
    close = next(step['run'] for step in close_steps if 'scripts/bootstrap_evidence.py' in step.get('run', ''))
    grade_steps = daily['jobs']['grade-and-validate']['steps']
    grade = next(step['run'] for step in grade_steps if 'scripts/bootstrap_evidence.py' in step.get('run', ''))
    assert '--capture-closes --skip-validation --backup' in close
    assert '--grade' not in close
    assert '--grade --backup' in grade
    assert '--capture-closes' not in grade
    assert daily.get('on', daily.get(True))['schedule'] == [{'cron': '0 14 * * *'}]
