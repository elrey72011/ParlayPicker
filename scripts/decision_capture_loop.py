"""C6 step 1: fresh decision quotes and bound predictions through a slate (roadmap C6).

    python scripts/decision_capture_loop.py --minutes 340 --interval 15

Off unless ``PARLAYPICKER_DECISION_MARKETS`` names a market (for example ``NCAAB:SPREAD``);
a named market is used only while it is qualified. Restores the canonical evidence store
once, then every ``--interval`` minutes runs ``app_core.decision_capture.run_once`` and
backs the store up after any cycle that stored a quote or a prediction. Credits come from
the loop's own allowance (``DECISION_LIMITS``), in its own ledger on the evidence remote.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def run(minutes, interval, *, root, client, folder, key, request_get, setting=None, cycle=None, sync=None,
        clock=time.monotonic, sleep=time.sleep):
    from app_core import decision_capture as decision
    if not decision.markets(setting):
        return {"status": "OFF", "cycles": 0}
    if sync is None:
        from app_core.prospective_remote import sync
    cycle = cycle or decision.run_once
    root.mkdir(parents=True, exist_ok=True)
    path = root / "prospective-evidence.sqlite3"
    session = {}
    sync(path, client, folder, session)  # restore before anything is read or written
    state = decision.load_state(client, folder)
    budget = decision.budget_for(state, lambda: decision.save_state(client, folder, state))
    deadline = clock() + minutes * 60
    totals = {"status": "RAN", "cycles": 0, "stored_quotes": 0, "predictions": 0, "backups": 0, "errors": 0}
    while True:
        totals["cycles"] += 1
        try:
            report = cycle(path, key=key, request_get=request_get, budget=budget, setting=setting)
        except Exception as exc:  # one failed cycle must not end the slate
            totals["errors"] += 1
            report = {"error": type(exc).__name__}
        changed = 0
        for entry in (report.get("markets") or {}).values():
            stored = (entry.get("capture") or {}).get("stored_quotes", 0)
            made = (entry.get("predictions") or {}).get("predictions", 0)
            totals["stored_quotes"] += stored
            totals["predictions"] += made
            changed += stored + made
        print(json.dumps({"cycle": totals["cycles"], **report}, default=str), flush=True)
        if changed:
            sync(path, client, folder, session)
            totals["backups"] += 1
        if clock() + interval * 60 > deadline:
            return totals
        sleep(interval * 60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=340)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--root", default="output/decision-capture")
    args = parser.parse_args(argv)
    from app_core import decision_capture as decision
    if not decision.markets():
        print(json.dumps({"status": "OFF", "detail": f"{decision.DECISION_ENV} names no market"}))
        return 0
    import requests
    from app_core.evidence_drive import DriveStore
    from app_core.evidence_remote import settings
    folder, _ = settings()
    print(json.dumps(run(args.minutes, args.interval, root=Path(args.root), client=DriveStore(folder), folder=folder,
                         key=os.getenv("ODDS_API_KEY"), request_get=requests.get)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
