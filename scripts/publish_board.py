"""Build, preview, publish, or roll back a local static daily board."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from app_core.public_board import build_package, validate_package


def render(package):
    validate_package(package)
    # JSON cannot terminate the data script; all displayed strings use textContent.
    encoded = json.dumps(package, allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    return (ROOT/'publishing/board.html').read_text(encoding='utf-8-sig').replace('__PUBLIC_DATA__', encoded)


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
    html = render(package)
    dest = Path(destination).resolve()
    if dest == ROOT or dest in ROOT.parents:
        raise ValueError('Publish into a dedicated output directory, not the repository root or its parents')
    target = dest/'index.html'
    if target.exists():
        atomic_write(dest/'previous.html', target.read_text(encoding='utf-8'))
    atomic_write(target, html)
    return target


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
        print((output/'preview.html').resolve())
    elif args.command == 'publish':
        print(publish(args.draft, args.destination))
    else:
        destination = Path(args.destination)
        previous = (destination/'previous.html').read_text(encoding='utf-8')
        atomic_write(destination/'index.html', previous)
        print('Restored previous publication')

if __name__ == '__main__':
    main()
