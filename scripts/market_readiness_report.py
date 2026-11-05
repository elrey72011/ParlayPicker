"""Read-only 12-market readiness report from the Drive evidence (roadmap C0).

Usage (GitHub Actions supplies the credentials):
    python scripts/market_readiness_report.py --directory /tmp/readiness --output readiness.json \
        [--cache-dir /tmp/drive-cache] [--with-sources]

It restores the canonical prospective-evidence store into an empty local
directory and reports
``prospective_evidence.all_market_readiness`` for all 12 sport × market
families, with the next blocker for each. ``--with-sources`` also restores each
sport's research store (descriptive source counts only; much slower).

The canonical store is tens of thousands of small Drive objects. With
``--cache-dir`` they are read through ``DriveStore.read_cached_objects``: bytes
are reused only when their SHA-256 matches Drive's fresh checksum, so a cache
kept between runs (actions/cache) makes later runs fast without trusting it.

It never writes to Drive. The Drive client is wrapped so that any upload raises
``READ_ONLY_VIOLATION`` and fails the run. Unlike ``load_readiness(authenticate=True)``,
it skips the local reconciliation and the backup that follows it. It makes no
Odds API or CFBD calls.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ('sport', 'market_family', 'unique_events', 'settled_events', 'verified_entry_prices',
           'verified_closes', 'model_status', 'calibration_status', 'validation_plan_status',
           'validation_count', 'holdout_count', 'close_clv_status', 'deployment_state', 'next_blocker')


class ReadOnlyViolation(RuntimeError):
    pass


class ReadOnlyDrive:
    """Delegates reads to the real client; refuses every write."""

    def __init__(self, client, cache_dir=None):
        self._client = client
        self._cache_dir = cache_dir
        self.writes_refused = 0

    def read_objects(self, *, Prefix):
        if self._cache_dir is not None and callable(getattr(self._client, 'read_cached_objects', None)):
            return self._client.read_cached_objects(Prefix=Prefix, cache_dir=self._cache_dir)
        return self._client.read_objects(Prefix=Prefix)

    def put_object(self, **kwargs):
        self.writes_refused += 1
        raise ReadOnlyViolation('READ_ONLY_VIOLATION:put_object:' + str(kwargs.get('Key', ''))[:80])

    def run_parallel(self, operation, items, progress=None):
        # Run through this wrapper so no worker client can bypass put_object.
        out = []
        for i, item in enumerate(items, 1):
            out.append(operation(self, item))
            if progress:
                progress(i, len(items))
        return out

    def __getattr__(self, name):
        return getattr(self._client, name)


def restore(root: Path, client, folder, *, with_sources=False):
    """Restore the canonical store (and, optionally, every sport's research store). Returns counts."""
    import time
    from app_core.prospective_remote import sync
    started = time.monotonic()
    counts = dict(canonical=sync(root / 'prospective-evidence.sqlite3', client, folder, {}))
    print(f'canonical restore: {time.monotonic() - started:.0f}s', flush=True)
    if not with_sources:
        return counts
    from app_core.mlb_receipt_remote import recover
    from app_core.prospective_readiness_report import SOURCE_FILENAMES
    from app_core.prospective_sport_adapters import ADAPTERS
    for sport, adapter in ADAPTERS.items():
        counts[sport] = adapter.restore(root / adapter.path_name, client, folder, {})
    counts['MLB_RECEIPTS'] = dict(records_restored=recover(client, root / SOURCE_FILENAMES['MLB']))
    return counts


def report(root: Path, *, client=None, folder=None, cache_dir=None, with_sources=False):
    from app_core.prospective_readiness_report import load_readiness
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise SystemExit('the restore directory must start empty, so no local row can be uploaded')
    if client is None:
        from app_core.evidence_drive import DriveStore
        from app_core.evidence_remote import settings
        folder = folder or settings()[0]
        if not folder or not os.getenv('PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT', '').strip():
            raise SystemExit('MISSING_CREDENTIALS: PARLAYPICKER_DRIVE_FOLDER_ID / PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT')
        client = DriveStore(folder)
    guarded = ReadOnlyDrive(client, cache_dir)
    counts = restore(root, guarded, folder, with_sources=with_sources)
    readiness = load_readiness(root, authenticate=False)  # local read of the restored copy
    markets = [dict(row, remote_evidence_status='RESTORED_READ_ONLY', remote_evidence_verified=True)
               for row in readiness['markets']]
    return dict(remote=dict(status='RESTORED_READ_ONLY', writes_refused=guarded.writes_refused, counts=counts),
                markets=markets, sources=readiness['sources'])


def markdown(result):
    lines = ['| ' + ' | '.join(COLUMNS) + ' |', '|' + '---|' * len(COLUMNS)]
    for row in result['markets']:
        lines.append('| ' + ' | '.join('' if row.get(c) is None else str(row.get(c)) for c in COLUMNS) + ' |')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--directory', type=Path, required=True, help='empty directory to restore into')
    parser.add_argument('--output', type=Path, required=True, help='JSON report path')
    parser.add_argument('--cache-dir', type=Path, help='checksum-verified Drive object cache kept between runs')
    parser.add_argument('--with-sources', action='store_true', help="also restore each sport's research store")
    args = parser.parse_args(argv)
    result = report(args.directory, cache_dir=args.cache_dir, with_sources=args.with_sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + '\n', encoding='utf-8')
    table = markdown(result)
    args.output.with_suffix('.md').write_text(table, encoding='utf-8')
    summary = os.getenv('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a', encoding='utf-8') as fh:
            fh.write('## Prospective market readiness (read-only restore)\n\n' + table)
    print(table)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
