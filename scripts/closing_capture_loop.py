"""Capture verified closes for saved picks continuously through a slate (roadmap E1.4 / G8).

    python scripts/closing_capture_loop.py --minutes 340 --interval 5

A close is recorded only within 30 minutes of kickoff (``activation_closing.capture_live``),
but GitHub starts scheduled runs hours apart, so the 30-minute workflow almost never
landed in that window and no close was ever saved. This restores the evidence store once,
then calls the same ``capture_live`` every ``--interval`` minutes until ``--minutes`` have
passed, backing up after each capture that saved something. The Odds API is called only
when a saved pick is inside its closing window.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(database, minutes, interval, *, capture=None, restore=None, backup=None,
        clock=time.monotonic, sleep=time.sleep):
    if capture is None:
        from app_core.activation_closing import capture_live as capture
    if restore is None:
        from app_core.evidence_remote import restore
    if backup is None:
        from app_core.evidence_remote import sync

        def backup(path):
            if not sync(path, incremental=True):
                raise RuntimeError("Evidence backup was not verified")
    restore(database)
    deadline = clock() + minutes * 60
    totals = {"cycles": 0, "verified": 0, "unavailable": 0, "backups": 0, "errors": 0}
    pending = False

    def back_up():
        # A failed backup is retried at the next cycle; it never ends the slate (audit 2026-11-09).
        nonlocal pending
        try:
            backup(database)
        except Exception as exc:
            totals["backup_errors"] = totals.get("backup_errors", 0) + 1
            print(json.dumps({"cycle": totals["cycles"], "backup_error": type(exc).__name__}), flush=True)
            return
        pending = False
        totals["backups"] += 1
    while True:
        totals["cycles"] += 1
        unmatched = {}
        try:
            try:
                result = capture(database, unmatched=unmatched)
            except TypeError:  # a capture without the diagnostic parameter
                result = capture(database)
        except Exception as exc:  # one failed cycle must not end the slate
            totals["errors"] += 1
            result = {"verified": 0, "error": type(exc).__name__}
        totals["verified"] += result.get("verified", 0)
        totals["unavailable"] += result.get("unavailable", 0)
        print(json.dumps({"cycle": totals["cycles"], **{k: v for k, v in result.items() if k != "reasons"},
                          "reasons": result.get("reasons"),
                          **({"unmatched_detail": unmatched} if unmatched else {})}), flush=True)
        pending = pending or bool(result.get("verified"))
        if pending:
            back_up()
        if clock() + interval * 60 > deadline:
            return totals
        sleep(interval * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="data/prediction_evidence/evidence.sqlite3")
    parser.add_argument("--minutes", type=float, default=340)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.database, args.minutes, args.interval)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
