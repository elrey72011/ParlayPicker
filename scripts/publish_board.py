"""Build, preview, publish, or roll back a local static daily board."""
import argparse
import hashlib
import re
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app_core.public_board import build_package, validate_package


def production_source_fingerprint():
    """Invalidate owner previews when any production rendering source changes."""
    digest = hashlib.sha256()
    for name in ('publishing/board.html', 'publishing/site.css', 'publishing/site.js',
                 'app_core/public_site_shell.py', 'scripts/publish_board.py'):
        digest.update(name.encode('utf-8'))
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def public_package(package):
    validate_package(package)
    from app_core.public_record import current_records
    if 'results' in package:
        package = {**package, 'results': current_records(package['results'])}
    return package


def build_public_assets(package, published_at=None):
    package = public_package(package)
    board = json.dumps(package, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(board.encode('utf-8')).hexdigest()
    version = {'build_id': digest, 'board_hash': digest,
               'published_at': published_at or datetime.now(timezone.utc).isoformat()}
    return board, json.dumps(version, sort_keys=True)


def render(package, *, live=False):
    package = public_package(package)
    board, version = build_public_assets(package)
    # JSON cannot terminate the data script; all displayed strings use textContent.
    encoded = json.dumps(package, allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    from app_core.public_site_shell import STYLES, header
    template = (ROOT/'publishing/board.html').read_text(encoding='utf-8-sig')
    html = template.replace('__SITE_STYLES__', STYLES).replace('__SITE_HEADER__', header(board=True)).replace('__PUBLIC_DATA__', encoded)

    metadata = '<script id="board-version" type="application/json">' + version + '</script>'
    if live:
        metadata += '<meta name="pp-live-publication" content="true">'
    html = html.replace('<script id="board-data"', metadata + '<script id="board-data"')
    script = (ROOT/'publishing/site.js').read_text(encoding='utf-8')
    return html.replace('</body>', '<script id="publication-client">'+script+'</script></body>')


def assets_from_html(html):
    """Derive sidecars from the exact reviewed HTML, never from newer analysis."""
    def embedded(name):
        match = re.search(r'<script id="' + name + r'" type="application/json">(.*?)</script>', html, re.S)
        if not match:
            raise ValueError('Publication metadata is missing')
        return json.loads(match.group(1))
    if 'id="board-version"' not in html:
        return assets_from_html(render(embedded('board-data'), live=True))
    version = embedded('board-version')
    board, expected = build_public_assets(embedded('board-data'), version['published_at'])
    if version != json.loads(expected):
        raise ValueError('Publication payload hash mismatch')
    return {'index.html': html, 'site.css': (ROOT/'publishing/site.css').read_text(encoding='utf-8'),
            'site.js': (ROOT/'publishing/site.js').read_text(encoding='utf-8'),
            'board-data.json': board, 'version.json': expected}


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix='.publish-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def publish(draft, destination):
    package = json.loads((Path(draft)/'public-board.json').read_text(encoding='utf-8'))
    return publish_package(package, destination)


def publish_package(package, destination):
    html = render(package, live=True)
    dest = Path(destination).resolve()
    if dest == ROOT or dest in ROOT.parents:
        raise ValueError('Publish into a dedicated output directory, not the repository root or its parents')
    target = dest/'index.html'
    if target.exists():
        atomic_write(dest/'previous.html', target.read_text(encoding='utf-8'))
    for name, content in assets_from_html(html).items():
        atomic_write(dest/name, content)
    return target


def rollback_publication(destination):
    destination = Path(destination)
    previous = (destination/'previous.html').read_text(encoding='utf-8')
    # A rollback is a new publication of the saved data, never a history rewrite.
    payload = re.search(r'<script id="board-data" type="application/json">(.*?)</script>', previous, re.S)
    if not payload:
        raise ValueError('Previous publication has no embedded fallback')
    html = render(json.loads(payload.group(1)), live=True)
    for name, content in assets_from_html(html).items():
        atomic_write(destination/name, content)
    return destination/'index.html'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    for name in ('overall','sides','totals'):
        build.add_argument('--'+name, required=True)
    for name in ('props','props-as-of','dfs','dfs-sport','dfs-slate','dfs-start'):
        build.add_argument('--'+name)
    build.add_argument('--output', default='outputs/public-board-draft')
    pub = sub.add_parser('publish');pub.add_argument('--draft', required=True);pub.add_argument('--destination', required=True)
    rollback = sub.add_parser('rollback');rollback.add_argument('--destination', required=True)
    args = parser.parse_args()
    if args.command == 'build':
        load = lambda name: pd.read_csv(name) if name else None
        package = build_package(load(args.overall), load(args.sides), load(args.totals), props=load(args.props), props_as_of=args.props_as_of,
                                dfs=load(args.dfs), dfs_sport=args.dfs_sport, dfs_slate=args.dfs_slate, dfs_start=args.dfs_start)
        html = render(package)
        output = Path(args.output)
        atomic_write(output/'public-board.json', json.dumps(package, indent=2, allow_nan=False))
        atomic_write(output/'preview.html', html)
        for name, content in assets_from_html(html).items():
            if name != 'index.html':
                atomic_write(output/name, content)
        print((output/'preview.html').resolve())
    elif args.command == 'publish':
        print(publish(args.draft, args.destination))
    else:
        rollback_publication(args.destination)
        print('Restored previous publication')

if __name__ == '__main__':
    main()
