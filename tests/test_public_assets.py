import hashlib
import json
from copy import deepcopy
import pytest
from scripts.publish_board import assets_from_html, build_public_assets, render, publish_package
from test_sftp_publishing import package


def test_payload_equivalence_hash_stability_and_no_secrets():
    original = package()
    before = deepcopy(original)
    board, version = build_public_assets(original)
    manifest = json.loads(version)
    assert json.loads(board) == original == before
    assert manifest['board_hash'] == hashlib.sha256(board.encode()).hexdigest()
    assert manifest['build_id'] and manifest['published_at']
    assert json.loads(build_public_assets(dict(reversed(list(original.items()))))[1])['board_hash'] == manifest['board_hash']
    original['built_at'] = '2026-09-09T20:00:00Z'
    assert json.loads(build_public_assets(original)[1])['board_hash'] != manifest['board_hash']
    for key in ('api_key', 'password', 'private_key', 'service_account'):
        with pytest.raises(ValueError):
            build_public_assets({**original, key: 'SECRET'})


def test_exact_reviewed_payload_and_version_last(tmp_path, monkeypatch):
    from scripts import publish_board
    writes = []
    original_write = publish_board.atomic_write
    def record(path, content):
        writes.append(path.name)
        original_write(path, content)
    monkeypatch.setattr(publish_board, 'atomic_write', record)
    target = publish_package(package(), tmp_path/'site')
    assert writes[-2:] == ['board-data.json', 'version.json']
    html = target.read_text(encoding='utf-8')
    assets = assets_from_html(html)
    assert json.loads(assets['board-data.json']) == package()
    assert '<meta name="pp-live-publication"' in html
    assert '<meta name="pp-live-publication"' not in render(package())
    with pytest.raises(ValueError, match='hash mismatch'):
        assets_from_html(html.replace('2026-09-08T20:00:00Z', '2026-09-09T20:00:00Z'))




def test_rollback_restores_matching_payload_with_new_publication_time(tmp_path):
    from scripts.publish_board import rollback_publication
    from datetime import datetime
    first = package()
    publish_package(first, tmp_path/'site')
    second = {**first, 'built_at': '2026-09-09T20:00:00Z'}
    publish_package(second, tmp_path/'site')
    old_version = json.loads((tmp_path/'site/version.json').read_text())
    rollback_publication(tmp_path/'site')
    restored = json.loads((tmp_path/'site/board-data.json').read_text())
    new_version = json.loads((tmp_path/'site/version.json').read_text())
    assert restored == first
    assert datetime.fromisoformat(new_version['published_at']) > datetime.fromisoformat(old_version['published_at'])
    assert json.loads(assets_from_html((tmp_path/'site/index.html').read_text(encoding='utf-8'))['board-data.json']) == first


def test_interrupted_publication_keeps_old_version_commit_marker(tmp_path, monkeypatch):
    from scripts import publish_board
    publish_package(package(), tmp_path/'site')
    old = (tmp_path/'site/version.json').read_bytes()
    write = publish_board.atomic_write
    def fail(path, content):
        if path.name == 'board-data.json':
            raise OSError('interrupted upload')
        write(path, content)
    monkeypatch.setattr(publish_board, 'atomic_write', fail)
    with pytest.raises(OSError):
        publish_package({**package(), 'built_at': '2026-09-09T20:00:00Z'}, tmp_path/'site')
    assert (tmp_path/'site/version.json').read_bytes() == old


@pytest.mark.parametrize('changed', ['publishing/board.html', 'publishing/site.css',
                                    'publishing/site.js', 'app_core/public_site_shell.py',
                                    'scripts/publish_board.py'])
def test_preview_fingerprint_tracks_production_sources(tmp_path, monkeypatch, changed):
    from scripts import publish_board
    from app.ui.publish_panel import source_fingerprint
    sources = ('publishing/board.html', 'publishing/site.css', 'publishing/site.js',
               'app_core/public_site_shell.py', 'scripts/publish_board.py')
    for name in sources:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((publish_board.ROOT / name).read_bytes())
    monkeypatch.setattr(publish_board, 'ROOT', tmp_path)
    before = source_fingerprint(None, None, None, None, {})
    assert source_fingerprint(None, None, None, None, {}) == before
    with (tmp_path / changed).open('ab') as target:
        target.write(b'\n/* changed source */\n')
    assert source_fingerprint(None, None, None, None, {}) != before
