"""Bounded observational probe. Never reads frames' locals or call/return values."""
from collections import defaultdict
from pathlib import Path
import json
import os
import sys
import threading
import time
import types


class PhaseProbe:
    TOOL = 5
    LIBRARY_FILES = {'threading.py', 'queue.py', 'selectors.py', '_bootstrap.py', '_bootstrap_external.py'}
    def __init__(self, root, output):
        self.root = str(Path(root).resolve()).replace('\\', '/').rstrip('/')+'/'
        self.output = Path(output)
        self.begin = time.perf_counter()
        self.codes = {}
        self.spans = defaultdict(list)
        self.aggregates = defaultdict(lambda: {'calls': 0, 'wall_seconds': 0.0, 'thread_cpu_seconds': 0.0})
        self.threads = {}
        self.main = threading.get_ident()
        self.sampler_id = None
        self.stop_event = threading.Event()
        self.worker = None
        self.active = False
        self.errors = 0
        self.samples = 0
        self.events = 0
        self.callback_seconds = 0.0
        self.io_seconds = 0.0
        self.lock = threading.Lock()
        self.handle = None
        self.reruns = 0

    def emit(self, code, **safe):
        start = time.perf_counter()
        row = {'code': code, 'at_seconds': start-self.begin, **safe}
        with self.lock:
            if self.events < 5000 and self.handle is not None:
                self.handle.write(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n')
                self.handle.flush()
                self.events += 1
        self.io_seconds += time.perf_counter()-start

    def ordinal(self, ident):
        if ident not in self.threads:
            self.threads[ident] = len(self.threads)+1
        return self.threads[ident]

    def add(self, function, label):
        if isinstance(function, (staticmethod, classmethod)):
            function = function.__func__
        code = getattr(function, '__code__', None)
        if isinstance(code, types.CodeType):
            self.codes[code] = label

    def discover(self):
        # Use only already-loaded modules: no application import/prewarming.
        choices = [
            ('streamlit.testing.v1.app_test', 'AppTest', ('run', '_run')),
            ('streamlit.testing.v1.local_script_runner', 'LocalScriptRunner', ('run', 'join', '_on_script_finished')),
            ('streamlit.runtime.scriptrunner.script_runner', 'ScriptRunner', ('start', '_run_script_thread', '_run_script', '_on_script_finished', 'request_stop')),
            ('threading', 'Thread', ('start', 'join')),
            ('threading', 'Condition', ('wait',)),
            ('threading', 'Event', ('wait',)),
        ]
        for module, cls, methods in choices:
            loaded = sys.modules.get(module)
            owner = loaded.__dict__.get(cls) if loaded else None
            if owner is not None:
                for name in methods:
                    self.add(owner.__dict__.get(name), cls+'.'+name)
        for module, names in [
            ('streamlit.testing.v1.local_script_runner', ('require_widgets_deltas',)),
            ('importlib._bootstrap', ('_find_and_load', '_find_and_load_unlocked')),
            ('subprocess', ('run',)),
        ]:
            loaded = sys.modules.get(module)
            if loaded:
                for name in names:
                    self.add(loaded.__dict__.get(name), module+'.'+name)

    def enter(self, code, offset):
        if code not in self.codes:
            return sys.monitoring.DISABLE
        started = time.perf_counter()
        ident = threading.get_ident()
        if ident == self.sampler_id:
            return
        label = self.codes[code]
        self.spans[(ident, code)].append((started, time.thread_time()))
        if label == 'AppTest._run':
            self.reruns += 1
        if label.startswith(('AppTest.', 'LocalScriptRunner.', 'ScriptRunner.')) or label.endswith('require_widgets_deltas'):
            self.emit('CALL_START', function=label, thread=self.ordinal(ident), rerun=self.reruns)
        self.callback_seconds += time.perf_counter()-started

    def leave(self, code, offset, ignored_value):
        self.finish(code, 'CALL_END')

    def unwind(self, code, offset, ignored_value):
        if code in self.codes:
            self.finish(code, 'CALL_UNWIND')

    def finish(self, code, event_code):
        started = time.perf_counter()
        ident = threading.get_ident()
        if ident == self.sampler_id:
            return
        stack = self.spans[(ident, code)]
        if stack:
            at, cpu = stack.pop()
            label = self.codes[code]
            row = self.aggregates[(self.ordinal(ident), label)]
            row['calls'] += 1
            row['wall_seconds'] += started-at
            row['thread_cpu_seconds'] += time.thread_time()-cpu
            if label.startswith(('AppTest.', 'LocalScriptRunner.', 'ScriptRunner.')) or label.endswith('require_widgets_deltas'):
                self.emit(event_code, function=label, thread=self.ordinal(ident), rerun=self.reruns, seconds=started-at)
        else:
            self.errors += 1
        self.callback_seconds += time.perf_counter()-started

    def location(self, frame):
        name = frame.f_code.co_filename.replace('\\', '/')
        function = frame.f_code.co_name
        if name.startswith(self.root):
            relative = name[len(self.root):]
            if relative.startswith(('app/', 'app_core/', 'core/', 'scripts/', 'tests/')) or relative == 'streamlit_app.py':
                return {'module': relative, 'function': function, 'line': frame.f_lineno}
        if '/streamlit/' in name:
            return {'module': 'streamlit/'+name.split('/streamlit/', 1)[1], 'function': function, 'line': frame.f_lineno}
        leaf = name.rsplit('/', 1)[-1]
        if leaf in self.LIBRARY_FILES or name.startswith('<frozen importlib.'):
            return {'module': leaf, 'function': function, 'line': frame.f_lineno}
        if function in ('main', 'app', '<module>') and (name.startswith('<') or '/tmp' in name or '/Temp/' in name):
            return {'module': 'SYNTHETIC_TEMP_SCRIPT', 'function': function, 'line': frame.f_lineno}
        return None

    def snapshot(self, code):
        rows = []
        for ident, frame in list(sys._current_frames().items())[:16]:
            if ident == self.sampler_id:
                continue
            locations = []
            while frame is not None and len(locations) < 12:
                location = self.location(frame)
                if location is not None:
                    locations.append(location)
                frame = frame.f_back
            if locations:
                rows.append({'thread': self.ordinal(ident), 'role': 'MAIN' if ident == self.main else 'WORKER', 'stack': locations})
        self.emit(code, threads=rows, active_thread_count=threading.active_count(), process_cpu_seconds=time.process_time())

    def sample(self):
        self.sampler_id = threading.get_ident()
        for _ in range(100):
            if self.stop_event.wait(.2):
                break
            self.snapshot('THREAD_SAMPLE')
            self.samples += 1

    def start(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.handle = (self.output/'apptest-events.jsonl').open('x', encoding='utf-8')
        self.discover()
        m = sys.monitoring
        if m.get_tool(self.TOOL) is not None:
            self.handle.close();self.handle=None
            raise RuntimeError('OBSERVER_TOOL_IN_USE')
        m.use_tool_id(self.TOOL, 'bounded-apptest-probe')
        self.active = True
        try:
            m.register_callback(self.TOOL, m.events.PY_START, self.enter)
            m.register_callback(self.TOOL, m.events.PY_RETURN, self.leave)
            m.register_callback(self.TOOL, m.events.PY_UNWIND, self.unwind)
            self.worker = threading.Thread(target=self.sample, daemon=True)
            self.worker.start()
            for code in self.codes:
                m.set_local_events(self.TOOL, code, m.events.PY_RETURN)
            m.set_events(self.TOOL,m.events.PY_START|m.events.PY_UNWIND)
            self.snapshot('OBSERVER_STARTED')
        except BaseException:
            self.stop()
            raise

    def stop(self):
        if not self.active:
            return
        m = sys.monitoring
        m.set_events(self.TOOL,0)
        for code in self.codes:
            m.set_local_events(self.TOOL, code, 0)
        for event in (m.events.PY_START, m.events.PY_RETURN, m.events.PY_UNWIND):
            m.register_callback(self.TOOL, event, None)
        m.free_tool_id(self.TOOL)
        self.active = False
        self.stop_event.set()
        if self.worker is not None and self.worker.ident is not None:
            self.worker.join(1)
        self.snapshot('OBSERVER_STOPPED')
        self.handle.close()
        self.handle = None

    def result(self):
        return {'reruns': self.reruns, 'samples': self.samples, 'events': self.events,
                'observer_errors': self.errors, 'callback_seconds': self.callback_seconds,
                'output_io_seconds': self.io_seconds,
                'sampler_stopped': self.worker is None or not self.worker.is_alive(),
                'monitor_restored': sys.monitoring.get_tool(self.TOOL) is None,
                'open_spans': [{'thread': self.ordinal(ident), 'function': self.codes[code], 'count': len(spans)} for (ident,code),spans in self.spans.items() if spans],
                'inclusive_overlapping_timings': [{'thread': thread, 'function': label, **row} for (thread,label),row in self.aggregates.items()],
                'limitation': 'Instrumentation can affect behavior. Inclusive durations overlap and must not be added. No values/locals/exception text are read or retained.'}
