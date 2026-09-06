import json
import tomllib

import pytest

from app_core.evidence_config import EvidenceConfigurationError, safe_error, service_account_info
from app_core import evidence_remote as remote


def test_literal_toml_preserves_json_key_escapes(monkeypatch):
    info = {'type': 'service_account', 'client_email': 'test@example.invalid',
            'private_key': 'BEGIN\nSYNTHETIC_TEST_ONLY\nEND\n', 'token_uri': 'https://oauth2.googleapis.com/token'}
    document = "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT = '''\n" + json.dumps(info, indent=2) + "\n'''"
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', tomllib.loads(document)['PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT'])
    assert service_account_info() == info


def test_double_quote_toml_reports_fix_without_exposing_secret(monkeypatch):
    raw = json.dumps({'type': 'service_account', 'private_key': 'PRIVATE_MARKER\nSECOND_LINE'})
    document = 'PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT = """\n' + raw + '\n"""'
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', tomllib.loads(document)['PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT'])
    with pytest.raises(EvidenceConfigurationError) as caught:
        service_account_info()
    assert 'triple SINGLE quotes' in str(caught.value)
    assert 'line ' in str(caught.value)
    assert 'PRIVATE_MARKER' not in str(caught.value)


@pytest.mark.parametrize('raw', ['', '{}', '[]', '{"type":"authorized_user"}', 'path/to/key.json'])
def test_missing_or_wrong_credentials_are_actionable(monkeypatch, raw):
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', raw)
    with pytest.raises(EvidenceConfigurationError):
        service_account_info()


def test_foreign_error_details_are_not_shown():
    assert 'PRIVATE_MARKER' not in safe_error(ValueError('PRIVATE_MARKER'), 'Restore')


def test_restore_status_preserves_safe_configuration_hint(monkeypatch, tmp_path):
    monkeypatch.setenv('PARLAYPICKER_DRIVE_FOLDER_ID', 'folder')
    monkeypatch.setenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', 'not-json-PRIVATE_MARKER')
    monkeypatch.setattr(remote, '_status', {})
    monkeypatch.setattr(remote, '_restored', set())
    monkeypatch.setattr(remote, '_client', service_account_info)
    with pytest.raises(RuntimeError, match='triple SINGLE quotes'):
        remote.restore_once(tmp_path / 'empty.sqlite3')
    assert 'PRIVATE_MARKER' not in remote.remote_status()['error']
