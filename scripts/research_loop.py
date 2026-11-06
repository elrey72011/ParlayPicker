"""Run the research cycle repeatedly inside one long job (owner request 2026-11-06).

    python scripts/research_loop.py --minutes 340 --interval 30

GitHub starts the research scheduler's 30-minute cron only a few times a day, so
capture, grading, Pinnacle calibration and certified closes all ran 2-3 times a day.
This runs ``scripts/run_research_scheduler.py`` (unchanged; its own operating window,
budgets and audit apply) as a fresh process every ``--interval`` minutes until
``--minutes`` have passed, so one scheduled start yields about a dozen cycles.

* A cycle that fails or is outside the operating window does not end the loop.
* A new cycle starts only if at least ``MIN_CYCLE_MINUTES`` remain.
* Before each cycle it checks for a queued or waiting Bound plan / Deployment review
  run (owner steps that share the evidence lock) and, if one is waiting, ends so it can run.
* Exit status: 0 when at least one cycle succeeded or skipped, else 1.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MIN_CYCLE_MINUTES = 45
YIELD_TO = ("bound-plan.yml", "deployment-review.yml")


def owner_run_waiting(get=None, setting=os.environ.get):
    """True when a Bound plan or Deployment review run is queued or waiting (GitHub API)."""
    token, repo = setting("GITHUB_TOKEN", ""), setting("GITHUB_REPOSITORY", "")
    if not token or not repo:
        return False
    if get is None:
        import requests
        get = requests.get
    for workflow in YIELD_TO:
        for status in ("queued", "waiting", "pending"):
            try:
                response = get(f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/runs",
                               params={"status": status, "per_page": 1}, timeout=15,
                               headers={"Authorization": f"Bearer {token}",
                                        "Accept": "application/vnd.github+json"})
                if response.status_code == 200 and response.json().get("total_count", 0) > 0:
                    return True
            except Exception:  # an API hiccup never stops research
                continue
    return False


def run_cycle(number, *, out_dir, run=subprocess.run):
    """One research cycle as a fresh process. Returns (exit code, audit or None)."""
    audit = out_dir / f"research-cycle-audit-{number:02d}.json"
    env = dict(os.environ, RESEARCH_AUDIT_PATH=str(audit),
               GITHUB_STEP_SUMMARY=str(out_dir / f"summary-{number:02d}.md"))
    code = run([sys.executable, "-u", str(ROOT / "scripts" / "run_research_scheduler.py")], env=env, cwd=ROOT).returncode
    try:
        return code, json.loads(audit.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return code, None


def loop(minutes, interval, *, out_dir, cycle=None, waiting=owner_run_waiting, clock=time.monotonic, sleep=time.sleep):
    out_dir.mkdir(parents=True, exist_ok=True)
    cycle = cycle or (lambda n: run_cycle(n, out_dir=out_dir))
    deadline = clock() + minutes * 60
    totals = {"cycles": 0, "succeeded": 0, "skipped": 0, "failed": 0, "ended": "deadline"}
    while clock() + MIN_CYCLE_MINUTES * 60 <= deadline:
        if waiting():
            totals["ended"] = "owner_run_waiting"
            break
        started = clock()
        totals["cycles"] += 1
        code, audit = cycle(totals["cycles"])
        state = (audit or {}).get("execution_state")
        kind = "skipped" if state == "SKIPPED" else "succeeded" if code == 0 else "failed"
        totals[kind] += 1
        print(json.dumps({"cycle": totals["cycles"], "exit": code, "execution_state": state,
                          "minutes": round((clock() - started) / 60, 1)}), flush=True)
        next_start = max(clock(), started + interval * 60)
        if next_start + MIN_CYCLE_MINUTES * 60 > deadline:
            break
        sleep(max(0.0, next_start - clock()))  # the clock moves on; never a negative sleep
    return totals


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=340)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("RUNNER_TEMP", "output")) / "research-loop")
    args = parser.parse_args(argv)
    totals = loop(args.minutes, args.interval, out_dir=args.out)
    print(json.dumps(totals), flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"# Research loop\n\n{totals['cycles']} cycles: {totals['succeeded']} succeeded, "
                     f"{totals['skipped']} skipped (outside the operating window), {totals['failed']} failed. "
                     f"Ended: {totals['ended']}.\n")
    return 0 if totals["succeeded"] or totals["skipped"] or not totals["cycles"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
