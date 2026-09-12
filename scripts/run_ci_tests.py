"""Run one deterministic CI shard, keeping tests from each file together."""
import argparse
from pathlib import Path
import sys

# Match `python -m pytest` import behavior when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest


class FileShard:
    def __init__(self, index, count):
        self.index, self.count = index, count

    def pytest_collection_modifyitems(self, config, items):
        files = sorted({str(item.path) for item in items})
        assigned = set(files[self.index::self.count])
        selected, deselected = [], []
        for item in items:
            (selected if str(item.path) in assigned else deselected).append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, required=True, help='One-based shard number')
    parser.add_argument('--shards', type=int, default=2)
    args, pytest_args = parser.parse_known_args(argv)
    if not 1 <= args.shard <= args.shards:
        parser.error('Require 1 <= shard <= shards')
    return pytest.main(pytest_args or ['tests'], plugins=[FileShard(args.shard - 1, args.shards)])


if __name__ == '__main__':
    raise SystemExit(main())
