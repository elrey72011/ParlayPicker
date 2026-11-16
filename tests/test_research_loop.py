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


# ---- a cycle is not started unless it has room to finish (owner 2026-11-16) ----

def _timed(clock, cycles):
    """Cycles that take the given minutes and end in the given state, in order."""
    plan = iter(cycles)

    def run(number):
        minutes, code, state = next(plan)
        clock.now += minutes * 60
        return code, {"execution_state": state}
    return run


def test_a_cycle_is_not_started_without_room_for_one_as_long_as_the_longest_so_far(tmp_path, capsys):
    """Sunday 15 November: cycles of 80, 68, 80 and 66 minutes. The fifth started with 46 minutes left,
    ran past the job's limit and was killed part-way through. It is no longer started."""
    import json
    clock = Clock()
    cycles = [(80.5, 0, "COMPLETE"), (68.1, 0, "COMPLETE"), (79.5, 0, "COMPLETE"), (65.5, 0, "COMPLETE"),
              (60, 0, "COMPLETE")]
    totals = rl.loop(340, 30, out_dir=tmp_path, cycle=_timed(clock, cycles), waiting=lambda: False,
                     clock=clock, sleep=clock.sleep)
    assert totals == {"cycles": 4, "succeeded": 4, "skipped": 0, "failed": 0, "ended": "no_room_for_a_full_cycle"}
    assert abs(clock.now - 293.6 * 60) < 1e-6           # it stops when the fourth ends; no time is slept away
    said = [json.loads(line) for line in capsys.readouterr().out.splitlines() if "stopped" in line]
    assert said == [{"stopped": "no_room_for_a_full_cycle", "longest_cycle_minutes": 80.5, "minutes_left": 46.4}]


def test_a_cycle_that_has_room_still_runs(tmp_path):
    clock = Clock()
    totals = rl.loop(340, 30, out_dir=tmp_path, cycle=_timed(clock, [(85, 0, "COMPLETE")] * 5), waiting=lambda: False,
                     clock=clock, sleep=clock.sleep)
    # 85 minutes each: the fourth starts at 255 with exactly 85 left and ends on the deadline.
    assert totals == {"cycles": 4, "succeeded": 4, "skipped": 0, "failed": 0, "ended": "deadline"}
    clock = Clock()
    totals = rl.loop(340, 30, out_dir=tmp_path, cycle=_timed(clock, [(70, 0, "COMPLETE")] * 6), waiting=lambda: False,
                     clock=clock, sleep=clock.sleep)
    # 70 minutes each: four fit (ending at 280); 60 minutes are left, more than the old 45 but not a full cycle.
    assert totals["cycles"] == 4 and totals["ended"] == "no_room_for_a_full_cycle"


def test_a_failed_cycle_counts_as_the_longest_and_a_skipped_one_does_not(tmp_path):
    clock = Clock()
    totals = rl.loop(340, 30, out_dir=tmp_path, cycle=_timed(clock, [(200, 1, "FAILED"), (100, 0, "COMPLETE")]),
                     waiting=lambda: False, clock=clock, sleep=clock.sleep)
    assert totals == {"cycles": 1, "succeeded": 0, "skipped": 0, "failed": 1, "ended": "no_room_for_a_full_cycle"}
    clock = Clock()
    cycles = [(100, 0, "SKIPPED")] + [(50, 0, "COMPLETE")] * 6
    totals = rl.loop(340, 30, out_dir=tmp_path, cycle=_timed(clock, cycles), waiting=lambda: False,
                     clock=clock, sleep=clock.sleep)
    # The slow skipped cycle is not a measure of a working one: 50-minute cycles keep starting to 300 minutes.
    assert totals == {"cycles": 5, "succeeded": 4, "skipped": 1, "failed": 0, "ended": "deadline"}


def test_the_job_limit_leaves_room_after_the_loops_own_deadline():
    """The rule keeps a cycle inside the loop's deadline; the job's limit must sit after that deadline."""
    import yaml
    workflow = yaml.safe_load((Path(__file__).resolve().parents[1] / ".github" / "workflows" / "research-loop.yml").read_text())
    job = workflow["jobs"]["research-loop"]
    step = next(s for s in job["steps"] if "research_loop.py" in str(s.get("run", "")))
    assert "'340'" in step["env"]["MINUTES"] and job["timeout-minutes"] == 355       # 340 minutes of cycles, killed at 355
