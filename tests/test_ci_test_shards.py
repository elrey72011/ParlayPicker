from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

from scripts.run_ci_tests import FileShard


def test_shards_partition_every_collected_item_and_preserve_file_order():
    original = [SimpleNamespace(path=Path(name), nodeid=f'{name}::{i}')
                for name in ('tests/test_z.py', 'tests/test_a.py', 'tests/test_m.py')
                for i in range(3)]
    groups = []
    for shard in range(2):
        deselected = []
        hook = SimpleNamespace(pytest_deselected=lambda items: deselected.extend(items))
        items = original.copy()
        FileShard(shard, 2).pytest_collection_modifyitems(SimpleNamespace(hook=hook), items)
        assert len(items) + len(deselected) == len(original)
        assert items == [item for item in original if item in items]
        groups.append(items)
    ids = [item.nodeid for group in groups for item in group]
    assert sorted(ids) == sorted(item.nodeid for item in original)
    assert len(ids) == len(set(ids))
    assert not {item.path for item in groups[0]} & {item.path for item in groups[1]}


def test_runner_propagates_failure_and_rejects_invalid_shards(tmp_path):
    runner = Path(__file__).resolve().parents[1] / 'scripts/run_ci_tests.py'
    (tmp_path / 'test_a.py').write_text('def test_fails(): assert False\n')
    (tmp_path / 'test_b.py').write_text('def test_passes(): assert True\n')
    def run(shard):
        return subprocess.run([sys.executable, str(runner), '--shard', str(shard), '--shards', '2',
                               '-q', str(tmp_path)], capture_output=True, text=True)
    failed, passed, invalid = run(1), run(2), run(0)
    assert failed.returncode == 1 and '1 failed, 1 deselected' in failed.stdout
    assert passed.returncode == 0 and '1 passed, 1 deselected' in passed.stdout
    assert invalid.returncode == 2 and 'Require 1 <= shard <= shards' in invalid.stderr


# New scheduling cases are SYNTHETIC CI fixtures; no historical sports execution.
import json
import math
import xml.etree.ElementTree as ET
import pytest
from scripts import run_ci_tests as runner


def synthetic_profile(rows=()):
    return dict(schema=runner.PROFILE_SCHEMA, version=1, fallback=runner.FALLBACK.copy(),
                sources={'SYNTHETIC': {'kind': 'synthetic-fixture'}},
                files=[dict(path=path, seconds=cost, profiled_tests=count, kind='estimated',
                            references=['SYNTHETIC']) for path, cost, count in rows])


def write_profile(tmp_path, profile):
    path = tmp_path/'SYNTHETIC-profile.json'
    path.write_text(json.dumps(profile), encoding='utf-8')
    return path


def synthetic_items(paths):
    return [SimpleNamespace(path=Path(path), nodeid=f'{path}::test_SYNTHETIC_{i}')
            for path, count in paths for i in range(count)]


def select(items, profile, count=3, root=Path.cwd()):
    groups, manifests = [], []
    for index in range(count):
        selected = items.copy()
        deselected = []
        config = SimpleNamespace(rootpath=root, hook=SimpleNamespace(
            pytest_deselected=lambda items: deselected.extend(items)))
        plugin = runner.FileShard(index, count, profile)
        plugin.pytest_collection_modifyitems(config, selected)
        assert len(selected) + len(deselected) == len(items)
        assert selected == [item for item in items if item in selected]
        groups.append(selected)
        manifests.append(plugin.manifest)
    ids = [item.nodeid for group in groups for item in group]
    assert len(ids) == len(set(ids)) == len(items)
    assert set(ids) == {item.nodeid for item in items}
    assert len({m['assignment_sha256'] for m in manifests}) == 1
    return groups, manifests


@pytest.mark.parametrize('count', [2, 3])
def test_duration_assignment_is_deterministic_complete_and_balances_unequal_files(tmp_path, count):
    rows = [(f'tests/test_{letter}.py', cost, 2) for letter, cost in
            zip('abcdefg', [300, 120, 100, 60, 40, 30, 20])]
    profile = runner.load_profile(write_profile(tmp_path, synthetic_profile(rows)))
    items = synthetic_items([(path, 2) for path, _, _ in rows])
    groups, manifests = select(items, profile, count)
    assert [[i.nodeid for i in g] for g in groups] == [[i.nodeid for i in g] for g in select(items, profile, count)[0]]
    for path, _, _ in rows:
        assert sum(any(str(i.path).replace('\\', '/') == path for i in group) for group in groups) == 1
    if count == 3:
        assert max(manifests[0]['estimated_shard_seconds']) == 300
        assert max(manifests[0]['estimated_shard_seconds']) < max(sum(row[1] for row in rows[i::3]) for i in range(3))


def test_new_missing_and_additional_tests_use_explicit_fallback_without_exclusion(tmp_path):
    profile_data = synthetic_profile([('tests/test_known.py', 10, 1)])
    profile_data['files'].append(dict(path='tests/test_missing.py', kind='missing', seconds=None,
                                     profiled_tests=0, references=[]))
    profile = runner.load_profile(write_profile(tmp_path, profile_data))
    _, manifests = select(synthetic_items([('tests/test_known.py', 4), ('tests/test_missing.py', 1),
                                          ('tests/test_new.py', 10)]), profile)
    files = {f['path']: f for f in manifests[0]['files']}
    assert files['tests/test_known.py']['scheduling_seconds'] == 10 + 3*3.5
    assert files['tests/test_known.py']['timing_kind'] == 'estimated'
    assert files['tests/test_known.py']['unprofiled_tests'] == 3
    assert files['tests/test_missing.py']['scheduling_seconds'] == 30
    assert files['tests/test_new.py']['scheduling_seconds'] == 35
    assert files['tests/test_new.py']['timing_kind'] == 'missing'
    assert sum(m['selected_count'] for m in manifests) == 15


def test_windows_and_posix_absolute_paths_have_identical_repository_relative_assignments(tmp_path):
    profile = runner.load_profile(write_profile(tmp_path, synthetic_profile()))
    paths = ['tests/test_a.py', 'tests/test_b.py', 'tests/test_c.py']
    def items(prefix):
        return [SimpleNamespace(path=prefix+path, nodeid=path+'::test_SYNTHETIC') for path in paths]
    windows = select(items('c:\\repo\\'), profile, root='C:\\Repo')[1]
    posix = select(items('/repo/'), profile, root='/repo')[1]
    assert windows[0]['assignments'] == posix[0]['assignments']
    assert windows[0]['assignment_sha256'] == posix[0]['assignment_sha256']
    assert runner.relative_path('.\\tests\\test_a.py') == 'tests/test_a.py'


@pytest.mark.parametrize('change', ['version', 'negative', 'nan', 'infinity', 'boolean', 'bad_count',
                                  'absolute', 'traversal', 'duplicate', 'case_collision', 'unknown_source',
                                  'missing_seconds', 'fallback', 'unlabelled'])
def test_invalid_advisory_profiles_reject_without_running_tests(tmp_path, change):
    data = synthetic_profile([('tests/test_a.py', 10, 2)])
    row = data['files'][0]
    if change == 'version': data['version'] = 2
    elif change in ('negative', 'nan', 'infinity', 'boolean'):
        row['seconds'] = {'negative': -1, 'nan': math.nan, 'infinity': math.inf, 'boolean': True}[change]
    elif change == 'bad_count': row['profiled_tests'] = 2.5
    elif change == 'absolute': row['path'] = 'C:/repo/tests/test_a.py'
    elif change == 'traversal': row['path'] = 'tests/../test_a.py'
    elif change == 'duplicate': data['files'].append(dict(row, path='tests\\test_a.py'))
    elif change == 'case_collision': data['files'].append(dict(row, path='tests/test_A.py'))
    elif change == 'unknown_source': row['references'] = ['UNKNOWN']
    elif change == 'missing_seconds': row['seconds'] = None
    elif change == 'fallback': data['fallback']['seconds_per_test'] = 0
    elif change == 'unlabelled': row['kind'] = 'guessed'
    with pytest.raises(ValueError): runner.load_profile(write_profile(tmp_path, data))


def test_duplicate_json_keys_reject(tmp_path):
    path = tmp_path/'duplicate.json'
    path.write_text('{"schema": "a", "schema": "b"}')
    with pytest.raises(ValueError, match='Duplicate JSON key'): runner.load_profile(path)


@pytest.mark.parametrize('paths', [[('tests/test_a.py', 1)], [('tests/test_a.py', 1), ('tests/test_A.py', 1), ('tests/test_c.py', 1)]])
def test_empty_shards_and_portable_file_collisions_fail(tmp_path, paths):
    profile = runner.load_profile(write_profile(tmp_path, synthetic_profile()))
    with pytest.raises(pytest.UsageError): select(synthetic_items(paths), profile)


def test_duplicate_collection_identities_and_external_paths_fail(tmp_path):
    profile = runner.load_profile(write_profile(tmp_path, synthetic_profile()))
    items = synthetic_items([('tests/test_a.py', 1), ('tests/test_b.py', 1), ('tests/test_c.py', 1)])
    with pytest.raises(pytest.UsageError, match='Duplicate collected'): select(items+items[:1], profile)
    with pytest.raises(ValueError, match='outside'): runner.relative_path('/elsewhere/test.py', '/repo')


@pytest.mark.parametrize('args', [[], ['--shard', '0'], ['--shard', '4', '--shards', '3'],
                                  ['--shard', '1', '--shards', '0'], ['--shard', '-1']])
def test_invalid_shard_cli_arguments_keep_exit_two(args):
    with pytest.raises(SystemExit) as result: runner.main(args)
    assert result.value.code == 2


def test_runner_missing_successful_results_cannot_pass(tmp_path):
    plugin = runner.FileShard(0, 3, manifest=tmp_path/'manifest.json')
    assert plugin.finish(0) != 0
    items = synthetic_items([('test_a.py', 1), ('test_b.py', 1), ('test_c.py', 1)])
    plugin.pytest_collection_modifyitems(SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None)), items)
    assert plugin.finish(0) == 1
    assert json.loads((tmp_path/'manifest.json').read_text())['execution_complete'] is False


def synthetic_reports(tmp_path):
    profile = dict(files={}, fallback=runner.FALLBACK.copy(), sha256='a'*64, version=1)
    original = synthetic_items([('test_a.py', 2), ('test_b.py', 2), ('test_c.py', 2)])
    for index in range(3):
        path = tmp_path/f'full-suite-{index+1}-assignment.json'
        plugin = runner.FileShard(index, 3, profile, manifest=path)
        selected = original.copy()
        plugin.pytest_collection_modifyitems(SimpleNamespace(hook=SimpleNamespace(pytest_deselected=lambda items: None)), selected)
        suite = ET.Element('testsuite')
        for item in selected:
            plugin.pytest_runtest_logreport(SimpleNamespace(nodeid=item.nodeid, when='call', outcome='passed'))
            file, name = item.nodeid.split('::')
            ET.SubElement(suite, 'testcase', classname=file[:-3], name=name)
        assert plugin.finish(0) == 0
        ET.ElementTree(suite).write(tmp_path/f'full-suite-{index+1}.xml')


def test_complete_synthetic_manifests_reconcile_exactly(tmp_path):
    synthetic_reports(tmp_path)
    assert runner.reconcile(tmp_path, 3)['completed_count'] == 6


@pytest.mark.parametrize('attack', ['missing_manifest', 'duplicate_manifest', 'cancelled', 'empty', 'missing_xml',
                                  'failed_xml', 'duplicate_xml', 'conflicting_profile', 'altered_assignment',
                                  'missing_outcome', 'collect_only', 'wrong_shard'])
def test_failed_cancelled_missing_conflicting_or_empty_shards_cannot_reconcile(tmp_path, attack):
    synthetic_reports(tmp_path)
    path = tmp_path/'full-suite-1-assignment.json'
    data = json.loads(path.read_text())
    if attack == 'missing_manifest': path.unlink()
    elif attack == 'duplicate_manifest': (tmp_path/'full-suite-9-assignment.json').write_text(json.dumps(data))
    elif attack == 'missing_xml': (tmp_path/'full-suite-1.xml').unlink()
    elif attack in ('failed_xml', 'duplicate_xml'):
        xml = tmp_path/'full-suite-1.xml'
        tree = ET.parse(xml)
        if attack == 'failed_xml': ET.SubElement(tree.getroot()[0], 'failure')
        else: tree.getroot()[1].set('name', tree.getroot()[0].get('name'))
        tree.write(xml)
    else:
        if attack == 'cancelled': data['exit_code'] = 2
        elif attack == 'empty': data['selected_count'] = 0
        elif attack == 'conflicting_profile': data['profile_sha256'] = 'b'*64
        elif attack == 'altered_assignment': data['assignments'][0]['shard'] = 2
        elif attack == 'missing_outcome': data['completed_nodeids'].pop()
        elif attack == 'collect_only': data['collect_only'] = True
        elif attack == 'wrong_shard': data['shard'] = 4
        path.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)): runner.reconcile(tmp_path, 3)


def test_actual_runner_capture_and_junit_reconcile_three_synthetic_shards(tmp_path):
    for letter in 'abc':
        (tmp_path/f'test_{letter}.py').write_text('def test_SYNTHETIC(): assert True\n')
    for index in range(1, 4):
        result = subprocess.run([sys.executable, str(runner.ROOT/'scripts/run_ci_tests.py'),
            '--shard', str(index), '--shards', '3', '--assignment-manifest', str(tmp_path/f'full-suite-{index}-assignment.json'),
            '-q', str(tmp_path), '--junitxml='+str(tmp_path/f'full-suite-{index}.xml')], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout+result.stderr
    assert runner.reconcile(tmp_path, 3)['completed_count'] == 3


def test_reviewed_workflow_preserves_limits_production_safety_and_three_shard_gate():
    import yaml
    workflow = yaml.load((runner.ROOT/'.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    jobs = workflow['jobs']
    assert jobs['full-suite-tests']['strategy']['matrix']['shard'] == ['1', '2', '3']
    assert jobs['full-suite-tests']['timeout-minutes'] == '30'
    assert jobs['production-safety']['timeout-minutes'] == '20'
    assert jobs['full-suite']['needs'] == 'full-suite-tests'
    assert jobs['full-suite']['steps'][0]['run'] == 'test "$SHARD_RESULT" = success'
    assert '--reconcile-manifests' in jobs['full-suite']['steps'][-1]['run']
    assert '--assignment-manifest' in jobs['full-suite-tests']['steps'][-2]['run']


def test_retained_profile_distinguishes_branch_head_tested_checkout_and_estimates():
    data = json.loads(runner.DEFAULT_PROFILE.read_text())
    assert data['sources']['main-full2']['checkout_commit'] == '8493961f82a7b953dcbd16bdf5b4e44fc8011afa'
    assert data['sources']['pr-full2']['head'] == 'c5fd78845c94b7d77cb928297665657b9d5b8500'
    assert data['sources']['pr-full2']['checkout_commit'] == 'c99b663ed641f736fdeb596f12147c449155d8cb'
    assert data['sources']['old-success-full1']['checkout_commit'] == 'b81751f2c4ad70190b62cafdf4546d605a4f466c'
    for row in data['files']:
        assert row['kind'] in ('measured', 'estimated', 'missing')
        if row['kind'] == 'measured':
            assert row['references'][0] in ('main-full2', 'pr-full2')
        for ref in row['references']:
            assert len(data['sources'][ref]['checkout_commit']) == 40
