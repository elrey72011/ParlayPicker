"""Read-only verification of a hosted board against the reviewed publication."""
from __future__ import annotations

import hashlib
import json
from urllib.parse import urlsplit

import requests

from scripts.publish_board import assets_from_html


LIMIT = 10_000_000
ASSETS = ('index.html', 'board-data.json', 'version.json')


class HostedMismatch(ValueError):
    """The public assets do not agree with each other or the intended build."""


def _json(raw: bytes, name: str):
    try:
        return json.loads(raw.decode('utf-8'))
    except (UnicodeError, ValueError) as exc:
        raise HostedMismatch(f'{name.upper()}_INVALID') from exc


def verify_assets(assets: dict[str, bytes], *, expected_version: dict | None = None,
                  expected_html_hash: str | None = None) -> dict:
    """Require exact sidecar, embedded HTML, build, and source parity."""
    if set(assets) != set(ASSETS) or any(not isinstance(assets[name], bytes) or
                                          len(assets[name]) > LIMIT for name in ASSETS):
        raise HostedMismatch('HOSTED_ASSETS_INCOMPLETE')
    board, version = _json(assets['board-data.json'], 'board'), _json(assets['version.json'], 'version')
    if not isinstance(board, dict) or not isinstance(version, dict):
        raise HostedMismatch('HOSTED_SCHEMA_INVALID')
    board_hash = hashlib.sha256(assets['board-data.json']).hexdigest()
    if version.get('board_hash') != board_hash or version.get('build_id') != board_hash:
        raise HostedMismatch('HOSTED_BOARD_HASH_MISMATCH')
    try:
        embedded = assets_from_html(assets['index.html'].decode('utf-8'))
    except (UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise HostedMismatch('HOSTED_HTML_INVALID') from exc
    if (embedded['board-data.json'].encode('utf-8') != assets['board-data.json'] or
            embedded['version.json'].encode('utf-8') != assets['version.json']):
        raise HostedMismatch('HOSTED_SIDECAR_MISMATCH')
    if expected_version is not None and version != expected_version:
        raise HostedMismatch('HOSTED_BUILD_MISMATCH')
    if (expected_html_hash is not None and
            hashlib.sha256(assets['index.html']).hexdigest() != expected_html_hash):
        raise HostedMismatch('HOSTED_HTML_HASH_MISMATCH')
    times = sorted({str(value) for section in board.get('games', {}).values()
                    for row in section if isinstance(row, dict)
                    for key in ('as_of', 'quote_time') if (value := row.get(key))})
    return {'status': 'MATCH', 'board_hash': board_hash,
            'source_git_sha': version.get('source_git_sha'),
            'source_git_dirty': version.get('source_git_dirty'),
            'source_fingerprint': version.get('source_fingerprint'),
            'published_at': version.get('published_at'),
            'analysis_built_at': board.get('built_at'),
            'evidence_timestamps': times}


def fetch_assets(site_url: str) -> dict[str, bytes]:
    parsed = urlsplit(site_url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('A public HTTPS site URL is required')
    base = site_url.rstrip('/') + '/'
    assets = {}
    for name in ASSETS:
        url = base + name
        try:
            with requests.get(url, timeout=(10, 30), allow_redirects=False, stream=True,
                              headers={'Cache-Control': 'no-cache'}) as response:
                if response.status_code != 200:
                    raise HostedMismatch('HOSTED_FETCH_FAILED')
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > LIMIT:
                        raise HostedMismatch('HOSTED_ASSET_TOO_LARGE')
                    chunks.append(chunk)
                assets[name] = b''.join(chunks)
        except requests.RequestException as exc:
            raise HostedMismatch('HOSTED_FETCH_FAILED') from exc
    return assets


def reconcile(site_url: str, *, expected_version: dict,
              expected_html_hash: str | None = None) -> dict:
    if not isinstance(expected_version, dict) or not expected_version.get('board_hash'):
        raise ValueError('An intended publication version is required')
    return verify_assets(fetch_assets(site_url), expected_version=expected_version,
                         expected_html_hash=expected_html_hash)
