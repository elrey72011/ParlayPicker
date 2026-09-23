import json
from app_core.evidence_config import runtime_configuration_status


def test_runtime_configuration_does_not_expose_secrets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', 'secret-invalid-json')
    monkeypatch.setenv('ODDS_API_KEY', 'secret-provider-token')
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'secret-folder')
    result = runtime_configuration_status()
    assert result['PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT'] == 'invalid'
    assert result['THE_ODDS_API_KEY'] == 'configured'
    assert result['PARLAYPICKER_DRIVE_FOLDER_ID'] == 'configured'
    assert 'secret' not in json.dumps(result)
    assert set(result.values()) <= {'configured', 'missing', 'invalid'}


def test_runtime_odds_status_uses_live_pipeline_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('THE_ODDS_API_KEY', raising=False)
    monkeypatch.setenv('ODDS_API_KEY', 'private-live-key')
    result = runtime_configuration_status()
    assert result['THE_ODDS_API_KEY'] == 'configured'
    assert 'private-live-key' not in json.dumps(result)


def test_runtime_odds_status_does_not_count_unused_alias(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('ODDS_API_KEY', raising=False)
    monkeypatch.setenv('THE_ODDS_API_KEY', 'unused-key')
    assert runtime_configuration_status()['THE_ODDS_API_KEY'] == 'missing'


def test_health_counts_immutable_operational_tables(tmp_path):
    from app_core.prediction_evidence import connect
    from app_core.evidence_health import evidence_health
    path = tmp_path / 'evidence.sqlite3'
    connect(path).close()
    state = evidence_health(path)
    assert state['closing_observations'] == 0
    assert state['validation_plans'] == 0
    assert state['snapshots'] == 0
