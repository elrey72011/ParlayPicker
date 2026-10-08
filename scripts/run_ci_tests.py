"""Run/reconcile deterministic whole-file CI shards; timing never excludes tests."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Match python -m pytest imports.
PROFILE_SCHEMA = 'parlaypicker/ci-file-costs-v1'
MANIFEST_SCHEMA = 'parlaypicker/ci-shard-assignment-v1'
DEFAULT_PROFILE = ROOT / 'scripts/ci_test_file_costs_v1.json'
FALLBACK = {'seconds_per_test': 3.5, 'minimum_file_seconds': 30.0}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def relative_path(path, root=None):
    """Portable repository-relative spelling; never silently collapse traversal."""
    value = str(path).replace('\\', '/')
    if re.match(r'^[A-Za-z]:/', value):
        require(root is not None, 'Absolute profile path')
        try:
            value = PureWindowsPath(value).relative_to(PureWindowsPath(str(root))).as_posix()
        except ValueError as exc:
            raise ValueError('Collected file outside collection root') from exc
    elif value.startswith('/'):
        require(root is not None, 'Absolute profile path')
        try:
            value = PurePosixPath(value).relative_to(PurePosixPath(str(root).replace('\\', '/'))).as_posix()
        except ValueError as exc:
            raise ValueError('Collected file outside collection root') from exc
    require(value and not value.startswith('//') and not re.match(r'^[A-Za-z]:', value), 'Invalid file path')
    require('..' not in value.split('/'), 'File path traversal')
    normalized = PurePosixPath(value).as_posix()
    require(normalized not in ('.', ''), 'Empty file path')
    return normalized


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid timing value')
    return float(value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def load_profile(path):
    raw = Path(path).read_bytes()
    profile = json.loads(raw, object_pairs_hook=unique_object)
    require(profile.get('schema') == PROFILE_SCHEMA and type(profile.get('version')) is int
            and profile['version'] == 1, 'Unsupported timing profile')
    fallback = profile.get('fallback', {})
    require(set(fallback) == set(FALLBACK), 'Invalid fallback contract')
    fallback = {k: number(v) for k, v in fallback.items()}
    require(all(v > 0 for v in fallback.values()), 'Nonpositive fallback')
    require(isinstance(profile.get('files'), list), 'Missing timing files')
    require(isinstance(profile.get('sources'), dict), 'Missing timing sources')
    files, portable_names = {}, set()
    for entry in profile['files']:
        require(isinstance(entry, dict), 'Invalid timing entry')
        require(isinstance(entry.get('path'), str), 'Invalid timing path')
        name = relative_path(entry['path'])
        require(name.casefold() not in portable_names, 'Conflicting timing identities')
        portable_names.add(name.casefold())
        require(entry.get('kind') in ('measured', 'estimated', 'missing'), 'Unlabelled timing')
        references = entry.get('references')
        require(isinstance(references, list) and all(isinstance(r, str) and r in profile['sources'] for r in references), 'Unknown timing source')
        require(bool(references) or entry['kind'] == 'missing', 'Missing timing provenance')
        count = entry.get('profiled_tests')
        require(type(count) is int and count >= 0, 'Invalid recorded count')
        seconds = entry.get('seconds')
        require((seconds is None) == (entry['kind'] == 'missing'), 'Conflicting missing timing')
        if seconds is not None:
            seconds = number(seconds)
        files[name] = dict(entry, path=name, seconds=seconds)
    return dict(files=files, fallback=fallback, sha256=hashlib.sha256(raw).hexdigest(), version=1)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def assignment(items, count, profile, root):
    require(type(count) is int and count > 0, 'Invalid shard count')
    rows = [dict(nodeid=item.nodeid, path=relative_path(item.path, root)) for item in items]
    require(len({r['nodeid'] for r in rows}) == len(rows), 'Duplicate collected test identities')
    counts = Counter(r['path'] for r in rows)
    require(len({p.casefold() for p in counts}) == len(counts), 'Conflicting collected file identities')
    fallback = profile['fallback']
    files = []
    for path, collected in sorted(counts.items()):
        entry = profile['files'].get(path)
        known = entry is not None and entry['seconds'] is not None
        added = max(0, collected - entry['profiled_tests']) if known else collected
        cost = (entry['seconds'] + added * fallback['seconds_per_test']) if known else max(
            fallback['minimum_file_seconds'], collected * fallback['seconds_per_test'])
        files.append(dict(path=path, collected_tests=collected, scheduling_seconds=max(cost, 0.000001),
                          timing_kind=('estimated' if added else entry['kind']) if known else 'missing',
                          profile_timing_kind=entry['kind'] if entry else 'missing', unprofiled_tests=added,
                          profiled_tests=entry['profiled_tests'] if entry else 0))
    loads = [0.0] * count
    for entry in sorted(files, key=lambda r: (-r['scheduling_seconds'], r['path'])):
        shard = min(range(count), key=lambda i: (loads[i], i))
        entry['shard'] = shard + 1
        loads[shard] += entry['scheduling_seconds']
    require(all(load > 0 for load in loads), 'Unexpectedly empty shard')
    by_file = {r['path']: r['shard'] for r in files}
    for row in rows:
        row['shard'] = by_file[row['path']]
    return rows, files, loads


class FileShard:
    def __init__(self, index, count, profile=None, manifest=None, root=None):
        require(type(index) is int and type(count) is int and 0 <= index < count, 'Invalid shard arguments')
        self.index, self.count = index, count
        self.profile = profile or dict(files={}, fallback=FALLBACK.copy(), sha256=None, version=1)
        self.manifest_path, self.root = manifest, root
        self.manifest = None
        self.completed = set()

    def write_manifest(self):
        if self.manifest_path is not None and self.manifest is not None:
            target = Path(self.manifest_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    def pytest_collection_modifyitems(self, config, items):
        import pytest
        try:
            rows, files, loads = assignment(items, self.count, self.profile,
                                            self.root or getattr(config, 'rootpath', ROOT))
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
        selected, deselected = [], []
        for item, row in zip(items, rows):
            (selected if row['shard'] == self.index + 1 else deselected).append(item)
        self.manifest = dict(schema=MANIFEST_SCHEMA, shard=self.index+1, shards=self.count,
            profile_sha256=self.profile['sha256'], profile_version=self.profile['version'],
            collected_count=len(items), selected_count=len(selected), deselected_count=len(deselected),
            collected_file_count=len(files), selected_file_count=sum(f['shard']==self.index+1 for f in files),
            assignment_sha256=digest(rows), assignments=rows, files=files, estimated_shard_seconds=loads,
            collect_only=bool(getattr(getattr(config, 'option', None), 'collectonly', False)),
            completed_count=0, execution_complete=False, exit_code=None)
        self.write_manifest()
        print(f'CI shard {self.index+1}/{self.count}: selected {len(selected)} of {len(items)}; '
              f'assignment {self.manifest["assignment_sha256"]}; profile {self.profile["sha256"]}')
        items[:] = selected  # Keep original pytest collection order, including within files.
        config.hook.pytest_deselected(items=deselected)

    def pytest_runtest_logreport(self, report):
        if report.when == 'call' or (report.when == 'setup' and report.outcome in ('failed', 'skipped')):
            self.completed.add(report.nodeid)

    def finish(self, result):
        if self.manifest is None:
            return int(result) or 1
        selected = {r['nodeid'] for r in self.manifest['assignments'] if r['shard'] == self.index+1}
        complete = self.completed == selected
        self.manifest.update(completed_count=len(self.completed), execution_complete=complete,
                             completed_nodeids=sorted(self.completed), exit_code=int(result))
        if not result and not self.manifest['collect_only'] and not complete:
            self.manifest['exit_code'] = 1
            print('Incomplete shard results', file=sys.stderr)
        self.write_manifest()
        return self.manifest['exit_code']


def reconcile(directory, count):
    require(type(count) is int and count > 0, 'Invalid reconciliation shard count')
    paths = sorted(Path(directory).rglob('full-suite-*-assignment.json'))
    require(len(paths) == count, 'Missing or duplicate shard manifests')
    indices, selected, reference = set(), [], None
    for path in paths:
        manifest = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
        require(manifest['schema'] == MANIFEST_SCHEMA and manifest['shards'] == count, 'Wrong manifest contract')
        index = manifest['shard']
        require(type(index) is int and 1 <= index <= count and index not in indices, 'Conflicting shard identity')
        indices.add(index)
        rows = manifest['assignments']
        require(isinstance(manifest['profile_sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', manifest['profile_sha256'])
                and manifest['profile_version'] == 1, 'Missing profile identity')
        require(all(type(r['shard']) is int and 1 <= r['shard'] <= count for r in rows), 'Invalid assignment index')
        by_file = {}
        for row in rows:
            path_name = relative_path(row['path'])
            require(path_name == row['path'], 'Noncanonical assignment path')
            require(path_name not in by_file or by_file[path_name] == row['shard'], 'File split across shards')
            by_file[path_name] = row['shard']
        require(digest(rows) == manifest['assignment_sha256'], 'Altered assignment')
        identity = (manifest['assignment_sha256'], manifest['profile_sha256'], manifest['profile_version'])
        if reference is None:
            reference = identity
            all_nodes = [r['nodeid'] for r in rows]
            require(len(all_nodes) == len(set(all_nodes)), 'Duplicate collected identities')
        require(identity == reference, 'Conflicting collection or profile identities')
        nodes = [r['nodeid'] for r in rows if r['shard'] == index]
        require(nodes and len(nodes) == manifest['selected_count'], 'Empty or mismatched selected count')
        require(manifest['collected_file_count'] == len(by_file)
                and manifest['selected_file_count'] == sum(s == index for s in by_file.values()), 'Mismatched file count')
        require(len(rows) == manifest['collected_count'] == manifest['selected_count'] + manifest['deselected_count'], 'Mismatched collected count')
        require(manifest['exit_code'] == 0 and manifest['execution_complete'] is True
                and manifest['collect_only'] is False, 'Unsuccessful or missing shard execution')
        require(len(nodes) == manifest['completed_count'] == len(manifest['completed_nodeids'])
                and set(nodes) == set(manifest['completed_nodeids']), 'Incomplete shard outcomes')
        xml = path.with_name(path.name.replace('-assignment.json', '.xml'))
        cases = list(ET.parse(xml).getroot().iter('testcase'))
        require(len(cases) == len(nodes) and all(c.find('failure') is None and c.find('error') is None for c in cases), 'Missing or unsuccessful JUnit outcomes')
        modules = {p[:-3].replace('/', '.'): p for p in by_file}
        xml_nodes = []
        for case in cases:
            classname = case.get('classname', '')
            matching = [m for m in modules if classname == m or classname.startswith(m+'.')]
            require(matching, 'Unmatched JUnit file')
            module = max(matching, key=len)
            suffix = classname[len(module):].strip('.')
            xml_nodes.append(modules[module] + ('::'+suffix.replace('.', '::') if suffix else '') + '::'+case.get('name', ''))
        require(xml_nodes == nodes, 'Mismatched JUnit test identities or order')
        selected.extend(nodes)
    require(indices == set(range(1, count+1)) and len(selected) == len(set(selected))
            and set(selected) == set(all_nodes), 'Incomplete or duplicate shard partition')
    return dict(status='PASS', shards=count, collected_count=len(all_nodes), completed_count=len(selected),
                assignment_sha256=reference[0], profile_sha256=reference[1])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shard', type=int, help='One-based shard number')
    parser.add_argument('--shards', type=int, default=3)
    parser.add_argument('--timing-profile', type=Path, default=DEFAULT_PROFILE)
    parser.add_argument('--assignment-manifest', type=Path)
    parser.add_argument('--reconcile-manifests', type=Path)
    args, pytest_args = parser.parse_known_args(argv)
    if args.reconcile_manifests:
        if args.shard is not None or pytest_args or args.shards < 1:
            parser.error('Invalid reconciliation arguments')
        try:
            print(json.dumps(reconcile(args.reconcile_manifests, args.shards), sort_keys=True))
            return 0
        except (ValueError, OSError, KeyError, TypeError, ET.ParseError) as exc:
            print(f'Shard reconciliation failed: {exc}', file=sys.stderr)
            return 1
    if args.shard is None or not 1 <= args.shard <= args.shards:
        parser.error('Require 1 <= shard <= shards')
    try:
        profile = load_profile(args.timing_profile)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(f'Invalid timing profile: {exc}')
    import pytest
    plugin = FileShard(args.shard-1, args.shards, profile, args.assignment_manifest)
    return plugin.finish(pytest.main(pytest_args or ['tests'], plugins=[plugin]))


if __name__ == '__main__':
    raise SystemExit(main())
