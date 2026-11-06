"""Research loop (owner 2026-11-06): many research cycles from one scheduled start."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import research_loop as rl  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _cycle(clock, results, minutes=20):
    def run(number):
        clock.now += minutes * 60
        code, state = results[(number - 1) % len(results)]
        return code, {"execution_state": state}
    return run


def test_cycles_every_interval_until_too_little_time_remains(tmp_path):
    clock = Clock()
    totals = rl.loop(340, 30, out_dir=tmp_path, cycle=_cycle(clock, [(0, "COMPLETE")]),
                     waiting=lambda: False, clock=clock, sleep=clock.sleep)
    # Starts at 0, 30, ..., 270 minutes (10 cycles); 300 + 45 > 340, so no 11th.
    assert totals == {"cycles": 10, "succeeded": 10, "skipped": 0, "failed": 0, "ended": "deadline"}


def test_failed_and_skipped_cycles_do_not_end_the_loop(tmp_path):
    clock = Clock()
    totals = rl.loop(120, 30, out_dir=tmp_path,
                     cycle=_cycle(clock, [(1, "FAILED"), (0, "SKIPPED"), (0, "COMPLETE")]),
                     waiting=lambda: False, clock=clock, sleep=clock.sleep)
    assert totals["cycles"] == 3 and totals["failed"] == 1 and totals["skipped"] == 1 and totals["succeeded"] == 1


def test_long_cycles_start_the_next_one_immediately(tmp_path):
    clock = Clock()
    totals = rl.loop(180, 30, out_dir=tmp_path, cycle=_cycle(clock, [(0, "COMPLETE")], minutes=40),
                     waiting=lambda: False, clock=clock, sleep=clock.sleep)
    assert totals["cycles"] == 4          # 0, 40, 80, 120 minutes; 160 + 45 > 180


def test_an_owner_run_waiting_ends_the_loop_before_the_next_cycle(tmp_path):
    clock = Clock()
    asks = iter([False, False, True])
    totals = rl.loop(340, 30, out_dir=tmp_path, cycle=_cycle(clock, [(0, "COMPLETE")]),
                     waiting=lambda: next(asks), clock=clock, sleep=clock.sleep)
    assert totals["cycles"] == 2 and totals["ended"] == "owner_run_waiting"


def test_owner_run_check_reads_queued_runs_and_never_raises():
    class Reply:
        status_code = 200

        def __init__(self, n):
            self.n = n

        def json(self):
            return {"total_count": self.n}
    env = {"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r"}.get
    seen = []

    def get(url, **kwargs):
        seen.append((url.rsplit("/", 2)[-2], kwargs["params"]["status"]))
        return Reply(1 if "bound-plan" in url and kwargs["params"]["status"] == "waiting" else 0)
    assert rl.owner_run_waiting(get, env) is True
    assert ("bound-plan.yml", "queued") in seen
    assert rl.owner_run_waiting(lambda *a, **k: Reply(0), env) is False
    assert rl.owner_run_waiting(lambda *a, **k: (_ for _ in ()).throw(OSError()), env) is False
    assert rl.owner_run_waiting(get, {}.get) is False                       # no token: never blocks


def test_cycle_runs_the_unchanged_scheduler_with_its_own_audit_path(tmp_path):
    calls = []

    class Done:
        returncode = 0

    def run(cmd, env, cwd):
        calls.append((cmd[-1], env["RESEARCH_AUDIT_PATH"]))
        Path(env["RESEARCH_AUDIT_PATH"]).write_text('{"execution_state": "COMPLETE"}')
        return Done()
    code, audit = rl.run_cycle(3, out_dir=tmp_path, run=run)
    assert code == 0 and audit == {"execution_state": "COMPLETE"}
    assert calls[0][0].endswith("run_research_scheduler.py") and calls[0][1].endswith("research-cycle-audit-03.json")


def test_a_clock_that_moves_between_reads_never_sleeps_negative(tmp_path):
    """2026-11-06: a 58-minute cycle ended the loop with 'sleep length must be non-negative'."""
    class Moving(Clock):
        def __call__(self):
            self.now += 0.001                 # real time passes between every read
            return self.now
    clock, slept = Moving(), []

    def sleep(seconds):
        assert seconds >= 0
        slept.append(seconds)
        clock.sleep(seconds)
    totals = rl.loop(180, 30, out_dir=tmp_path, cycle=_cycle(clock, [(0, "COMPLETE")], minutes=58),
                     waiting=lambda: False, clock=clock, sleep=sleep)
    assert totals["cycles"] == 3 and totals["failed"] == 0 and slept and min(slept) == 0.0
