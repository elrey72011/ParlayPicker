"""Push-only, bounded offline scope profiler. Never replaces a validation decision."""
import argparse
import collections
import cProfile
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pstats
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time
import types
import functools

FILES = ["tests/test_slate_audit_scope.py", "tests/test_canonical_download_scope.py",
         "tests/test_football_research_scope.py", "tests/test_nfl_inputs_scope.py"]
SELECTION = ("exact_seal_and_ci_merge_preserve_predecessor or "
             "(dirty_staged_committed_resealed_changes_reject and core/protected.py) or "
             "(mutated_policy_rejects and implementation_changes) or "
             "exact_predecessor_source_reconstruction or test_nfl_inputs_scope")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not 30 <= args.timeout_seconds <= 1200:
        parser.error("diagnostic deadline must be 30..1200 seconds")
    repo = args.repo.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(repo); sys.path.insert(0, str(repo))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    if revision != args.revision:
        raise SystemExit("DIAGNOSTIC_REVISION_MISMATCH")
    import pytest
    state = {"phase": "session", "node": "session"}
    git_records, copies, fixtures, phases, outcomes, selected = [], [], [], [], [], []
    events = (output / "progress.jsonl").open("x", encoding="utf-8", buffering=1)
    started = time.perf_counter(); interrupted = False
    real_popen, real_copy = subprocess.Popen, shutil.copytree
    profile = cProfile.Profile()
    lock = threading.Lock(); stopping = threading.Event()

    def emit(kind, **fields):
        with lock:
            events.write(json.dumps(dict(kind=kind, elapsed_seconds=time.perf_counter()-started,
                                         **fields), sort_keys=True)+"\n")
            events.flush()

    def cache_snapshot():
        found = {}
        for name, module in list(sys.modules.items()):
            if module is None or name != "scripts.check_launch_change_scope": continue
            for key, value in list(vars(module).items()):
                if not isinstance(value, (types.FunctionType, functools._lru_cache_wrapper)): continue
                memo=vars(value).get("reconstruction_memo")
                if memo is not None:
                    found[name+":"+key+":pure"] = dict(hits=memo.hits,misses=memo.misses,entries=len(memo.entries),retained_bytes=memo.retained_bytes,fields=value.reconstruction_context_fields)
                if callable(value) and callable(getattr(value, "cache_info", None)):
                    try:
                        info = value.cache_info()
                        if info.maxsize is not None:
                            found[name+":"+key] = info._asdict()
                    except (AttributeError, TypeError): pass
        return found

    def memory():
        return dict(max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    semantics="Linux process high-water RSS; cache occupancy separately recorded")

    def heartbeat():
        while not stopping.wait(10): emit("heartbeat", state=dict(state), memory=memory())

    class TimedProcess(real_popen):
        def __init__(self, *a, **kw):
            self._t = time.perf_counter(); self._done = False; self._state = dict(state)
            command = a[0] if a else kw.get("args")
            self._git = isinstance(command, (tuple, list)) and Path(str(command[0])).stem == "git"
            self._operation = str(command[1]) if self._git and len(command) > 1 else "other"
            super().__init__(*a, **kw)
        def finish(self):
            if self._git and not self._done and self.returncode is not None:
                self._done = True
                row = dict(operation=self._operation, seconds=time.perf_counter()-self._t,
                           returncode=self.returncode, **self._state)
                git_records.append(row); emit("git", **row)
        def communicate(self, *a, **kw):
            result = super().communicate(*a, **kw); self.finish(); return result
        def wait(self, *a, **kw):
            result = super().wait(*a, **kw); self.finish(); return result

    copy_depth = 0
    def copytree(*a, **kw):
        nonlocal copy_depth
        top = copy_depth == 0; copy_depth += 1; t = time.perf_counter()
        try: return real_copy(*a, **kw)
        finally:
            copy_depth -= 1
            if top:
                row = dict(seconds=time.perf_counter()-t, **state)
                copies.append(row); emit("repository_copy", **row)

    class Monitor:
        @pytest.hookimpl(hookwrapper=True)
        def pytest_collection(self, session):
            state.update(phase="collection"); t = time.perf_counter(); yield
            phases.append(dict(phase="collection", seconds=time.perf_counter()-t))
        def pytest_collection_finish(self, session):
            selected.extend(item.nodeid for item in session.items)
            emit("selected", identities=selected)
        @pytest.hookimpl(hookwrapper=True)
        def pytest_fixture_setup(self, fixturedef, request):
            t = time.perf_counter(); yield
            row = dict(name=fixturedef.argname, scope=fixturedef.scope,
                       node=request.node.nodeid, seconds=time.perf_counter()-t)
            fixtures.append(row); emit("fixture", **row)
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_setup(self, item):
            state.update(phase="setup", node=item.nodeid); t=time.perf_counter(); yield
            phases.append(dict(seconds=time.perf_counter()-t, **state))
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            state.update(phase="call", node=item.nodeid); t=time.perf_counter(); yield
            phases.append(dict(seconds=time.perf_counter()-t, **state))
            emit("call_completed", node=item.nodeid, caches=cache_snapshot(), memory=memory())
        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_teardown(self, item):
            state.update(phase="teardown", node=item.nodeid); t=time.perf_counter(); yield
            phases.append(dict(seconds=time.perf_counter()-t, **state))
        def pytest_runtest_logreport(self, report):
            if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
                row = dict(nodeid=report.nodeid, outcome=report.outcome)
                outcomes.append(row); emit("outcome", **row)

    class Deadline(BaseException): pass
    def expire(signum, frame): raise Deadline()
    signal.signal(signal.SIGALRM, expire); signal.alarm(args.timeout_seconds)
    subprocess.Popen, shutil.copytree = TimedProcess, copytree
    emit("start", revision=revision, files=FILES, selection=SELECTION, deadline=args.timeout_seconds)
    thread=threading.Thread(target=heartbeat, daemon=True); thread.start()
    code=124; profile.enable()
    try:
        code=int(pytest.main(["-q", *FILES, "-k", SELECTION, "--durations=20",
                             "--basetemp", str(output/"fixtures"),
                             "--junitxml", str(output/"results.xml")], plugins=[Monitor()]))
    except Deadline:
        interrupted=True; emit("deadline", state=dict(state))
    finally:
        profile.disable(); signal.alarm(0); stopping.set(); thread.join(timeout=2)
        subprocess.Popen, shutil.copytree = real_popen, real_copy
        t=time.perf_counter(); profile.dump_stats(output/"profile.pstats")
        functions=[]
        for (file,line,name),(primitive,calls,own,total,callers) in pstats.Stats(profile).stats.items():
            functions.append(dict(file=file,line=line,function=name,calls=calls,
                                  self_seconds=own,cumulative_seconds=total))
        totals=collections.defaultdict(float)
        for row in phases: totals[row["phase"]]+=row["seconds"]
        data=dict(schema="scope-performance-diagnostic-v1", label=args.label, revision=revision,
                  complete=not interrupted, exit_code=code, elapsed_seconds=time.perf_counter()-started,
                  selected=selected, outcomes=outcomes, phase_totals=dict(totals), fixtures=fixtures,
                  copies=copies, git=git_records, caches=cache_snapshot(), memory=memory(),
                  functions=sorted(functions,key=lambda x:x["self_seconds"],reverse=True),
                  source_hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in FILES+["tests/conftest.py", "scripts/check_launch_change_scope.py"]},
                  environment=dict(python=sys.version, python_sha256=hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
                                   git=subprocess.check_output(["git", "--version"]).decode().strip(),
                                   packages={k:importlib.metadata.version(k) for k in ("pytest","numpy","pandas","requests")},
                                   pip_freeze_sha256=os.environ.get("DIAGNOSTIC_LOCK_SHA256", "UNKNOWN")),
                  semantics="Fixture/copy/Git/reconstruction times are inclusive subsets; never sum overlapping categories. Cat-file process lifetimes include Python time; stream operation cProfile times are separate. All revisions use identical instrumentation; cProfile overhead is included. Interrupted observations are not completed measurements.")
        (output/"receipt.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
        emit("reporting", seconds=time.perf_counter()-t, complete=not interrupted)
        events.close()
    return code


if __name__ == "__main__": raise SystemExit(main())
