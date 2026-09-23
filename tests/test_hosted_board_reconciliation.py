"""Hosted reads must match the exact reviewed publication before success."""
import hashlib
import json

import pytest

from app_core import hosted_board_reconciliation as hosted
from app_core import netlify_publishing as netlify
from scripts.publish_board import assets_from_html, render, build_public_assets
from test_sftp_publishing import package


def publication():
    assets = assets_from_html(render(package(), live=True))
    return {name: assets[name].encode('utf-8') for name in hosted.ASSETS}


def expected(assets):
    return json.loads(assets['version.json']), hashlib.sha256(assets['index.html']).hexdigest()


def test_hosted_html_and_sidecars_match_intended_build():
    assets = publication()
    version, html_hash = expected(assets)
    report = hosted.verify_assets(assets, expected_version=version, expected_html_hash=html_hash)
    assert report['status'] == 'MATCH'
    assert report['board_hash'] == version['board_hash']
    assert report['analysis_built_at'] == package()['built_at']
    assert report['source_git_sha'] == version['source_git_sha']


@pytest.mark.parametrize('change,reason', [
    ('board', 'HOSTED_BOARD_HASH_MISMATCH'),
    ('sidecar', 'HOSTED_SIDECAR_MISMATCH'),
    ('build', 'HOSTED_BUILD_MISMATCH'),
    ('html', 'HOSTED_HTML_HASH_MISMATCH'),
])
def test_mismatch_is_surfaced(change, reason):
    assets = publication()
    version, html_hash = expected(assets)
    if change == 'board':
        assets['board-data.json'] += b' '
    elif change == 'sidecar':
        revised = dict(version, published_at='2026-09-24T00:00:00Z')
        assets['version.json'] = json.dumps(revised, sort_keys=True).encode()
    elif change == 'build':
        version = dict(version, source_fingerprint='0' * 64)
    else:
        assets['index.html'] += b'<!-- changed -->'
    with pytest.raises(hosted.HostedMismatch, match=reason):
        hosted.verify_assets(assets, expected_version=version, expected_html_hash=html_hash)


def test_repackaging_changes_publication_time_only():
    original = package()
    first_board, first_version = build_public_assets(original, '2026-09-08T20:01:00Z')
    later_board, later_version = build_public_assets(original, '2026-09-09T20:01:00Z')
    assert later_board == first_board
    assert json.loads(later_version)['board_hash'] == json.loads(first_version)['board_hash']
    assert json.loads(later_board)['built_at'] == original['built_at']
    assert json.loads(later_version)['published_at'] != json.loads(first_version)['published_at']


def test_netlify_ready_requires_public_build_reconciliation(monkeypatch):
    assets = publication()
    version, html_hash = expected(assets)
    def api_call(method, path, token):
        if path.startswith('/deploys/'):
            return {'id': 'deploy-123', 'site_id': 'site-1234', 'state': 'ready'}
        return {'id': 'site-1234', 'published_deploy': {'id': 'deploy-123'},
                'ssl_url': 'https://example.netlify.app'}
    monkeypatch.setattr(netlify, 'api_call', api_call)
    monkeypatch.setattr(hosted, 'fetch_assets', lambda url: assets)
    result = netlify.deployment_status('deploy-123', 'site-1234', 'token', version, html_hash)
    assert result['state'] == 'ready' and result['reconciliation']['status'] == 'MATCH'
    assets['board-data.json'] += b' '
    result = netlify.deployment_status('deploy-123', 'site-1234', 'token', version, html_hash)
    assert result['state'] == 'content_mismatch'
    assert result['reconciliation_reason'] == 'HOSTED_BOARD_HASH_MISMATCH'
    assert netlify.deployment_status('deploy-123', 'site-1234', 'token')['state'] == 'verification_missing'
