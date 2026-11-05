"""Closing capture loop. SYNTHETIC capture, restore and backup; no network or sleep."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("closing_capture_loop", ROOT / "scripts" / "closing_capture_loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


def test_loop_restores_once_captures_each_cycle_and_backs_up_only_after_saves():
    now = [0.0]
    calls = {"restore": 0, "backup": 0}
    results = iter([{"verified": 0, "reason": "no_candidates_in_closing_window"},
                    {"verified": 2, "unavailable": 1},
                    RuntimeError("provider down"),
                    {"verified": 1}])

    def capture(path):
        value = next(results)
        if isinstance(value, Exception):
            raise value
        return value

    def restore(path):
        calls["restore"] += 1

    def backup(path):
        calls["backup"] += 1

    def sleep(seconds):
        now[0] += seconds

    totals = loop.run("db", minutes=16, interval=5, capture=capture, restore=restore, backup=backup,
                      clock=lambda: now[0], sleep=sleep)
    assert totals == {"cycles": 4, "verified": 3, "unavailable": 1, "backups": 2, "errors": 1}
    assert calls == {"restore": 1, "backup": 2}
