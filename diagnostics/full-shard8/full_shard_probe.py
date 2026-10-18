"""Observer plugin for the unchanged native run_ci_tests.py shard selection."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
import pytest
from failure_capture import FailureCapture

TARGET = 'tests/test_drive_history_loading.py::test_actual_completed_analysis_publish_caller_defers_history'
HEAD = 'cf8b62b5048fc4f70ced131edcaa148175ca2532'
MERGE = '4df0975694226f23e0951ad5f24bf6b89bcb990d'
TREE = '88e0f3936f0f7e3519326bf760bafc52e5e7b476'
SOURCE = '6a9068743f14abd04bf6244bf5fef2c26a530ddf5eaee8f536c149f0a1753486'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def require(condition, code):
    if not condition:
        raise pytest.UsageError(code)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b'\n')


class Observer:
    def __init__(self, config):
        self.config = config
        self.root = Path.cwd().resolve()
        self.materials = Path(__file__).parent
        self.out = Path(os.environ['NCAAF_DIAGNOSTIC_OUTPUT']).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.selection = json.loads((self.materials / 'selection.json').read_bytes())
        self.preview = json.loads((self.materials / 'preview.json').read_bytes())
        self.observed = []
        self.reports = []
        self.errors = []
        self.capture = FailureCapture(self.root, self.out)
        self.started = time.perf_counter()
        self.collection_started = self.started
        self.collection_seconds = None
        self.capture_started = False
        self.xml_seconds = 0.0
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
        local = os.environ.get('NCAAF_DIAGNOSTIC_LOCAL_VALIDATION') == '1'
        require(head in ({HEAD, MERGE} if local else {MERGE}) and tree == TREE, 'CHECKOUT_CONFLICT')
        require(not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip(), 'DIRTY_CHECKOUT')
        require(hashlib.sha256((self.root / 'tests/test_drive_history_loading.py').read_bytes()).hexdigest() == SOURCE, 'SOURCE_CONFLICT')
        write(self.out / 'helper-context.json', dict(head=head, tree=tree,
              shallow=subprocess.check_output(['git', 'rev-parse', '--is-shallow-repository'], text=True).strip(),
              source_sha256=SOURCE, plugin_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              capture_sha256=hashlib.sha256((self.materials / 'failure_capture.py').read_bytes()).hexdigest(),
              native_runner_sha256=hashlib.sha256((self.root / 'scripts/run_ci_tests.py').read_bytes()).hexdigest(),
              default_temp=not bool(os.environ.get('PYTEST_DEBUG_TEMPROOT')), original_timeout_seconds=3))

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_collection_modifyitems(self, config, items):
        universe = list(items)
        yield
        from run_ci_tests import assignment, load_profile, digest
        rows, _, _ = assignment(universe, 9, load_profile(self.root / 'scripts/ci_test_file_costs_v1.json'), self.root)
        require(rows == self.preview['assignments'], 'COLLECTION_OR_ASSIGNMENT_CONFLICT')
        require(digest(rows) == self.preview['assignment_sha256'], 'ASSIGNMENT_HASH_CONFLICT')
        expected = self.selection['selected']
        require(len(expected) == len(set(expected)) == 937, 'SELECTION_COUNT_CONFLICT')
        require(hashlib.sha256(canonical(expected)).hexdigest() == self.selection['selection_sha256'], 'SELECTION_HASH_CONFLICT')
        require(expected == [r['nodeid'] for r in rows if r['shard'] == 8], 'FULL_SHARD_CONFLICT')
        require([item.nodeid for item in items] == expected, 'NATIVE_SELECTION_ORDER_CONFLICT')
        require(TARGET in expected, 'TARGET_MISSING')
        self.collection_seconds = time.perf_counter() - self.collection_started
        write(self.out / 'selection-verified.json', dict(collected=len(rows), selected=len(items),
              selected_nodeids=expected, selection_sha256=self.selection['selection_sha256'],
              assignment_sha256=digest(rows), profile_sha256=self.preview['profile_sha256'],
              collection_seconds=self.collection_seconds, native_selection_unchanged=True))

    def pytest_runtest_logstart(self, nodeid, location):
        require(nodeid == self.selection['selected'][len(self.observed)], 'EXECUTION_ORDER_CONFLICT')
        self.observed.append(nodeid)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        if item.nodeid == TARGET:
            self.capture.start()
            self.capture_started = True

    @pytest.hookimpl(hookwrapper=True, trylast=True)
    def pytest_runtest_teardown(self, item):
        yield
        if item.nodeid == TARGET and self.capture_started:
            self.capture.stop()

    def pytest_runtest_logreport(self, report):
        row = dict(nodeid=report.nodeid, phase=report.when, outcome=report.outcome, seconds=report.duration)
        self.reports.append(row)
        with (self.out / 'phase-progress.jsonl').open('ab') as f:
            f.write(canonical(row) + b'\n')
        if report.when == 'teardown':
            self.safe_xml()

    def pytest_runtest_makereport(self, item, call):
        if call.excinfo is None:
            return
        allowed = {'AssertionError', 'RuntimeError', 'ValueError', 'TypeError', 'KeyError', 'FileNotFoundError',
                   'PermissionError', 'OSError', 'TimeoutError', 'ConnectionError', 'ImportError', 'ModuleNotFoundError'}
        name = call.excinfo.type.__name__
        locations = []
        tb = call.excinfo.tb
        while tb is not None and len(locations) < 12:
            location = self.capture.location(tb.tb_frame)
            if location is not None:
                locations.append(location)
            tb = tb.tb_next
        self.errors.append(dict(nodeid=item.nodeid, phase=call.when,
              code=name if name in allowed else 'OTHER_TEST_EXCEPTION', locations=locations))
        write(self.out / 'test-error-codes.json', self.errors)

    def safe_xml(self):
        started = time.perf_counter()
        suite = ET.Element('testsuite', name='sanitized_native_shard8_diagnostic')
        by_node = {}
        for row in self.reports:
            by_node.setdefault(row['nodeid'], []).append(row)
        for node in self.observed:
            phases = by_node.get(node, [])
            if not any(r['phase'] == 'teardown' for r in phases):
                continue
            prefix, bracket, parameter = node.partition('[')
            parts = prefix.split('::')
            case = ET.SubElement(suite, 'testcase', classname=parts[0][:-3].replace('/', '.') +
                      ('.' + '.'.join(parts[1:-1]) if len(parts) > 2 else ''), name=parts[-1] + bracket + parameter,
                      time=str(sum(r['seconds'] for r in phases)))
            if any(r['outcome'] == 'failed' and r['phase'] in ('setup', 'teardown') for r in phases):
                ET.SubElement(case, 'error', type='ACTUAL_SETUP_OR_TEARDOWN_ERROR', message='ACTUAL_TEST_ERROR')
            elif any(r['outcome'] == 'failed' for r in phases):
                ET.SubElement(case, 'failure', type='ACTUAL_TEST_FAILURE', message='ACTUAL_TEST_FAILED')
            elif any(r['outcome'] == 'skipped' for r in phases):
                ET.SubElement(case, 'skipped', message='ACTUAL_TEST_SKIPPED')
        suite.set('tests', str(len(suite)))
        ET.ElementTree(suite).write(self.out / 'full-suite-8.xml', encoding='utf-8', xml_declaration=True)
        self.xml_seconds += time.perf_counter() - started

    def pytest_sessionfinish(self, session, exitstatus):
        if self.capture_started and not self.capture.restored:
            self.capture.stop()
        self.safe_xml()
        phases = Counter()
        for r in self.reports:
            phases[r['phase']] += r['seconds']
        completed = [n for n in self.observed if any(r['nodeid'] == n and r['phase'] == 'teardown' for r in self.reports)]
        write(self.out / 'diagnostic.json', dict(exit_code=int(exitstatus), expected=937, observed=self.observed,
              completed=completed, execution_complete=completed == self.selection['selected'],
              collection_seconds=self.collection_seconds, pytest_seconds=time.perf_counter() - self.started,
              phase_seconds=dict(phases), target_reports=[r for r in self.reports if r['nodeid'] == TARGET],
              capture=self.capture.result(), xml_reporting_seconds=self.xml_seconds,
              collect_only=session.config.option.collectonly,
              limitation='Actual native shard selector and original order. One isolated job, failure-only watchdog wrapper and sanitized reports differ from original release. Timings overlap. Passing does not establish cause.'))


def pytest_configure(config):
    config.pluginmanager.register(Observer(config), 'bounded-full-shard-observer')
