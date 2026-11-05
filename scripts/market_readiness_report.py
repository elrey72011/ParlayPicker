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
                markets=markets, sources=readiness['sources'],
                tables=table_counts(root / 'prospective-evidence.sqlite3'),
                plans=plan_evaluations(root / 'prospective-evidence.sqlite3'),
                research_bindings=research_bindings(root / 'prospective-evidence.sqlite3'),
                price_edge=price_edge(root / 'prospective-evidence.sqlite3'))


def price_edge(path):
    """Cross-book price screen per research market (app_core.price_edge_screen), read-only."""
    from app_core import price_edge_screen
    from app_core.prospective_research_models import SCOPES
    if not Path(path).is_file():
        return []
    out = []
    for sport, markets in SCOPES.items():
        for market in markets:
            try:
                out.append(price_edge_screen.from_store(path, sport, market))
            except Exception as exc:  # report, never guess
                out.append(dict(sport=sport, market_family=market, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


def research_bindings(path, now=None):
    """Each research market's place on the C2 chain (prospective_calibration.binding_status), read-only."""
    from app_core import prospective_calibration as calibration
    from app_core.prospective_research_models import SCOPES
    if not Path(path).is_file():
        return []
    out = []
    for sport, markets in SCOPES.items():
        for market in markets:
            try:
                out.append(calibration.binding_status(path, sport, market, now=now))
            except Exception as exc:  # report, never guess
                out.append(dict(sport=sport, market_family=market, evaluation_error=f'{type(exc).__name__}: {exc}'[:300]))
    return out


def _scalars(value):
    """The scalar fields of a nested report section (counts, metrics, statuses)."""
    plain = (str, int, float, bool, type(None))
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, plain)] if all(isinstance(v, plain) for v in value) else len(value)
    if not isinstance(value, dict):
        return value if isinstance(value, plain) else type(value).__name__
    return {k: v for k, v in value.items() if isinstance(v, plain)}


def plan_evaluations(path):
    """Each frozen validation plan's own evaluation, read-only (prospective_evidence.evaluate_validation_plan)."""
    from app_core import prospective_evidence as evidence
    if not Path(path).is_file():
        return []
    out = []
    for plan in evidence.read_records(path, 'prospective_validation_plan'):
        row = dict(validation_plan_id=plan['validation_plan_id'], sport=plan['sport'],
                   market_family=plan['market_family'], version=plan.get('version'),
                   model_id=plan.get('model_id'), calibration_id=plan.get('calibration_id'))
        try:
            result = evidence.evaluate_validation_plan(path, plan['validation_plan_id'])
            row.update({k: _scalars(v) for k, v in result.items()
                        if k not in row and k not in ('plan_artifact_hash',)})
        except Exception as exc:  # report, never guess
            row['evaluation_error'] = f'{type(exc).__name__}: {exc}'[:300]
        out.append(row)
    return out


def table_counts(path):
    """Row counts per canonical table, split by sport where the table has one (read-only)."""
    import sqlite3
    if not Path(path).is_file():
        return {}
    out = {}
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for name in names:
            columns = {r[1] for r in db.execute(f'PRAGMA table_info("{name}")')}
            if 'sport' in columns:
                out[name] = {str(sport): n for sport, n in db.execute(
                    f'SELECT sport, COUNT(*) FROM "{name}" GROUP BY sport ORDER BY sport')}
            else:
                out[name] = db.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
    return out


def markdown(result):
    lines = ['| ' + ' | '.join(COLUMNS) + ' |', '|' + '---|' * len(COLUMNS)]
    for row in result['markets']:
        lines.append('| ' + ' | '.join('' if row.get(c) is None else str(row.get(c)) for c in COLUMNS) + ' |')
    tables = result.get('tables') or {}
    if tables:
        lines += ['', '| canonical table | rows |', '|---|---|']
        for name, count in tables.items():
            shown = ', '.join(f'{k}: {v}' for k, v in count.items()) if isinstance(count, dict) else str(count)
            lines.append(f'| {name} | {shown or 0} |')
    plans = result.get('plans') or []
    if plans:
        lines += ['', '| validation plan | version | model | status | blockers |', '|---|---|---|---|---|']
        for plan in plans:
            blockers = plan.get('blockers') or plan.get('reasons') or plan.get('evaluation_error') or ''
            lines.append(f"| {plan['validation_plan_id']} | {plan.get('version')} | {plan.get('model_id') or ''} | "
                         f"{plan.get('status', '')} | {blockers} |")
    bindings = result.get('research_bindings') or []
    if bindings:
        lines += ['', '| research market | training events | candidate decided | calibration | games/day | '
                  'window days | bound plan | next blocker |', '|---|---|---|---|---|---|---|---|']
        for row in bindings:
            lines.append(f"| {row['sport']} {row['market_family']} | {row.get('most_training_events', '')} | "
                         f"{row.get('candidate_decided', '')} | {row.get('calibration_id') or ''} | "
                         f"{row.get('games_per_day_14d', '')} | {row.get('proposed_window_days') or ''} | "
                         f"{row.get('bound_plan_id') or ''} | "
                         f"{row.get('next_blocker') or row.get('evaluation_error', '')} |")
    edges = result.get('price_edge') or []
    if edges:
        lines += ['', '| price screen (best price, all candidates) | candidates | EV >= 2% picks | mean EV | closes | mean CLV | beat close | settled | ROI |',
                  '|---|---|---|---|---|---|---|---|---|']
        def pct(v):
            return '' if v is None else f'{v:+.1%}'
        for row in edges:
            t = (row.get('by_threshold') or {}).get('all') or {}
            flagged = ((row.get('by_threshold') or {}).get('0.02') or {}).get('picks', '')
            lines.append(f"| {row['sport']} {row['market_family']} | {row.get('candidates', row.get('evaluation_error', ''))} | "
                         f"{flagged} | {pct(t.get('mean_ev'))} | {t.get('with_comparable_close', '')} | "
                         f"{pct(t.get('mean_price_clv'))} | {pct(t.get('beat_close_share'))} | {t.get('settled', '')} | "
                         f"{pct(t.get('roi_per_pick'))} |")
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
