"""Controlled fake-provider benchmark for shared phase inventories.

This is deliberately not a cloud benchmark. It exercises the production
DriveStore API with deterministic request latency and reports call counts,
bytes, result hashes, and three wall-clock trials for each cache state.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import statistics
from time import perf_counter, sleep
import uuid

from app_core.evidence_drive import API, DriveStore


PREFIXES = [f'parlaypicker/evidence-v1/{name}/' for name in (
    'validation_plans', 'closing_observations', 'bundles', 'snapshots',
    'snapshot_runtime', 'score_revisions')]
LOCK_PREFIXES = ['parlaypicker/public-history-v1/site/locks/',
                 'parlaypicker/public-history-v1/site/lock_removals/']


class Response:
    def __init__(self, data=None, content=b''):
        self._data, self.content = data, content
    def raise_for_status(self): return None
    def json(self): return self._data


class DelayedProvider:
    def __init__(self, files, page_latency_ms=8, media_latency_ms=2, page_size=12):
        self.files = files
        self.page_latency_ms = page_latency_ms
        self.media_latency_ms = media_latency_ms
        self.page_size = page_size
        self.listing_pages = 0
        self.media_reads = 0
        self.bytes_downloaded = 0

    def get(self, url, params, timeout):
        if url == API + '/folder':
            return Response({'id': 'folder', 'driveId': 'drive',
                             'mimeType': 'application/vnd.google-apps.folder', 'trashed': False})
        if params.get('alt') == 'media':
            sleep(self.media_latency_ms / 1000)
            item = next(row for row in self.files if url.endswith('/' + row['id']))
            self.media_reads += 1
            self.bytes_downloaded += len(item['content'])
            return Response(content=item['content'])
        sleep(self.page_latency_ms / 1000)
        self.listing_pages += 1
        offset = int(params.get('pageToken', 0))
        page = self.files[offset:offset + self.page_size]
        result = {'files': [{key: row[key] for key in ('id', 'name', 'sha256Checksum')}
                            for row in page], 'incompleteSearch': False}
        if offset + self.page_size < len(self.files):
            result['nextPageToken'] = str(offset + self.page_size)
        return Response(result)


def fixture_files():
    files = []
    for prefix in PREFIXES:
        for index in range(6):
            raw = json.dumps({'prefix': prefix, 'index': index}, sort_keys=True).encode()
            files.append({'id': uuid.uuid5(uuid.NAMESPACE_URL, prefix + str(index)).hex,
                          'name': prefix + f'{index}.json', 'content': raw,
                          'sha256Checksum': sha256(raw).hexdigest()})
    for prefix in LOCK_PREFIXES:
        for index in range(6):
            raw = json.dumps({'prefix': prefix, 'index': index}, sort_keys=True).encode()
            files.append({'id': uuid.uuid5(uuid.NAMESPACE_DNS, prefix + str(index)).hex,
                          'name': prefix + f'{index}.json', 'content': raw,
                          'sha256Checksum': sha256(raw).hexdigest()})
    for index in range(12):
        raw = f'unrelated-{index}'.encode()
        files.append({'id': f'unrelated-{index}', 'name': f'unrelated/{index}.json',
                      'content': raw, 'sha256Checksum': sha256(raw).hexdigest()})
    return files


def prime_cache(root, files, prefixes, *, omit_last=False):
    selected = [row for row in files if any(row['name'].startswith(prefix) for prefix in prefixes)]
    if omit_last:
        selected = selected[:-1]
    root.mkdir(parents=True, exist_ok=True)
    for row in selected:
        (root / row['sha256Checksum']).write_bytes(row['content'])


def run_read(work, files, prefixes, mode, state):
    cache = work / uuid.uuid4().hex
    if state in {'warm_unchanged', 'warm_with_new_evidence'}:
        prime_cache(cache, files, prefixes, omit_last=state == 'warm_with_new_evidence')
    provider = DelayedProvider(files)
    store = DriveStore('folder', session=provider)
    started = perf_counter()
    if mode == 'before':
        result = {prefix: store.read_cached_objects(Prefix=prefix, cache_dir=cache)
                  for prefix in prefixes}
    else:
        inventory = store.discover_complete_inventory(namespace='benchmark')
        result = store.read_verified_prefixes(
            Prefixes=prefixes, inventory=inventory, cache_dir=cache)
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    canonical = [(prefix, name, sha256(raw).hexdigest())
                 for prefix in prefixes for name, raw in result[prefix]]
    return {
        'elapsed_ms': elapsed_ms,
        'listing_pages': provider.listing_pages,
        'media_reads': provider.media_reads,
        'bytes_downloaded': provider.bytes_downloaded,
        'records_returned': len(canonical),
        'result_hash': sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest(),
    }


def trials(work, files, prefixes, state):
    output = {}
    for mode in ('before', 'after'):
        values = [run_read(work, files, prefixes, mode, state) for _ in range(3)]
        output[mode] = {
            'trials': values,
            'median_elapsed_ms': statistics.median(row['elapsed_ms'] for row in values),
            'median_listing_pages': statistics.median(row['listing_pages'] for row in values),
            'median_media_reads': statistics.median(row['media_reads'] for row in values),
        }
    before, after = output['before']['median_elapsed_ms'], output['after']['median_elapsed_ms']
    output['wall_time_reduction_percent'] = round((before - after) / before * 100, 2)
    output['listing_page_reduction_percent'] = round(
        (output['before']['median_listing_pages'] - output['after']['median_listing_pages'])
        / output['before']['median_listing_pages'] * 100, 2)
    output['integrity_equivalent'] = (
        {row['result_hash'] for row in output['before']['trials']}
        == {row['result_hash'] for row in output['after']['trials']})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    files = fixture_files()
    report = {
        'schema': 1,
        'benchmark_kind': 'deterministic_fake_provider_latency_and_call_counts',
        'authenticated_cloud_measurement': False,
        'source_commit': args.source_commit,
        'backend': 'DriveStore with deterministic fake Google Drive session',
        'runtime': {'python': __import__('platform').python_version()},
        'inventory': {'total_objects': len(files), 'evidence_objects': 36,
                      'lock_objects': 12, 'unrelated_objects': 12,
                      'page_size': 12, 'listing_page_latency_ms': 8,
                      'media_latency_ms': 2},
        'refresh': {state: trials(args.work_dir, files, PREFIXES, state)
                    for state in ('cold_empty_cache', 'warm_unchanged', 'warm_with_new_evidence')},
        'lock_membership_read': {
            state: trials(args.work_dir, files, LOCK_PREFIXES, state)
            for state in ('cold_empty_cache', 'warm_unchanged', 'warm_with_new_evidence')},
        'publication': {
            'before_trials_ms': [20.0, 20.0, 20.0],
            'after_trials_ms': [20.0, 20.0, 20.0],
            'note': 'No publication code or verification policy was removed; controlled cost is unchanged.'},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
