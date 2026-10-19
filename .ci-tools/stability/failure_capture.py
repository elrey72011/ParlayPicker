"""Failure-only bounded thread locations; never reads arguments or locals."""
from pathlib import Path
import json
import re
import sys
import threading
import time


class FailureCapture:
    def __init__(self, root, out, near_deadline=2.75):
        self.root = Path(root).resolve()
        self.out = Path(out)
        self.near_deadline = near_deadline
        self.owner = self.original = self.wrapper = None
        self.workers = []
        self.reruns = []
        self.snapshots = []
        self.errors = 0
        self.restored = False

    def location(self, frame):
        # Code metadata only. No frame locals/globals/source lines/values.
        code = frame.f_code
        name = code.co_name
        if not re.fullmatch(r'[A-Za-z_<>][A-Za-z_0-9<>]*', name):
            return None
        filename = code.co_filename.replace('\\', '/')
        root = str(self.root).replace('\\', '/').rstrip('/') + '/'
        if filename.startswith(root):
            module = filename[len(root):]
            if not (module in {'streamlit_app.py', 'tests/test_drive_history_loading.py', 'tests/test_publish_panel.py'}
                    or module.startswith(('app/', 'app_core/'))):
                return None
        elif '/streamlit/' in filename:
            module = 'streamlit/' + filename.rsplit('/streamlit/', 1)[1]
            if not module.startswith(('streamlit/testing/', 'streamlit/runtime/', 'streamlit/components/')):
                return None
        elif filename.rsplit('/', 1)[-1] in {'threading.py', 'queue.py', 'selectors.py', 'ast.py', 'pathlib.py', 'glob.py'}:
            module = 'stdlib/' + filename.rsplit('/', 1)[-1]
        elif filename in {'<frozen importlib._bootstrap>', '<frozen importlib._bootstrap_external>'}:
            module = filename[8:-1]
        else:
            return None
        return dict(module=module, function=name, line=frame.f_lineno)

    def snapshot(self, code, elapsed):
        if len(self.snapshots) >= 2:
            return
        started = time.perf_counter()
        threads = threading.enumerate()[:16]
        frames = sys._current_frames()
        rows = []
        try:
            for ordinal, thread in enumerate(threads, 1):
                frame = frames.get(thread.ident)
                locations = []
                traversed = 0
                while frame is not None and traversed < 64 and len(locations) < 12:
                    location = self.location(frame)
                    if location is not None:
                        locations.append(location)
                    frame = frame.f_back
                    traversed += 1
                rows.append(dict(thread=ordinal, alive=thread.is_alive(), daemon=thread.daemon, locations=locations))
                if time.perf_counter() - started > 0.1:
                    break
        finally:
            frames.clear()
        self.snapshots.append(dict(code=code, token_entry_seconds=elapsed, threads=rows,
                                   capture_seconds=time.perf_counter() - started))
        self.flush()

    def flush(self):
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'failure-capture.json').write_text(json.dumps(self.result(), sort_keys=True, indent=2) + '\n')

    def start(self, owner=None):
        if owner is None:
            module = sys.modules.get('streamlit.testing.v1.app_test')
            if module is None:
                raise ValueError('APPTEST_NOT_ALREADY_LOADED')
            owner = module.AppTest
        self.owner = owner
        self.original = owner._run
        capture = self

        def observe(instance, *args, **kwargs):
            ordinal = len(capture.reruns) + 1
            started = time.perf_counter()
            done = threading.Event()
            worker = None
            failed = False
            if ordinal in (1, 2):
                def watch():
                    if not done.wait(capture.near_deadline):
                        try:
                            capture.snapshot('APPTEST_APPROACHING_EXISTING_DEADLINE', time.perf_counter() - started)
                        except BaseException:
                            capture.errors += 1  # Fixed count only; never exception text.
                worker = threading.Thread(target=watch, name='bounded-diagnostic-watch', daemon=True)
                capture.workers.append((worker, done))
                worker.start()
            try:
                return capture.original(instance, *args, **kwargs)
            except BaseException:
                failed = True
                raise
            finally:
                elapsed = time.perf_counter() - started
                done.set()
                if worker is not None:
                    worker.join(timeout=1.0)
                capture.reruns.append(dict(ordinal=ordinal, seconds=elapsed, raised=failed))
        self.wrapper = observe
        owner._run = observe

    def stop(self):
        for worker, done in self.workers:
            done.set()
            worker.join(timeout=1.0)
        if self.owner is not None:
            if self.owner._run is self.wrapper:
                self.owner._run = self.original
                self.restored = self.owner._run is self.original
            else:
                self.errors += 1
        self.flush()

    def result(self):
        return dict(reruns=self.reruns, snapshots=self.snapshots, observer_errors=self.errors,
                    original_restored=self.restored, workers_stopped=all(not w.is_alive() for w, _ in self.workers),
                    near_deadline_seconds=self.near_deadline,
                    limitation='One diagnostic execution; wrapper/watchdog can affect scheduling. Snapshots only if token entry approaches unchanged timeout, not continuous tracing. No argument/locals/exception text.')
