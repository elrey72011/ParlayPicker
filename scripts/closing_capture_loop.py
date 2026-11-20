"""Capture verified closes for saved picks continuously through a slate (roadmap E1.4 / G8).

    python scripts/closing_capture_loop.py --minutes 340 --interval 10 --window 12 --max-credits 300

A close is recorded only within 30 minutes of kickoff (``activation_closing.capture_live``),
but GitHub starts scheduled runs hours apart, so the 30-minute workflow almost never
landed in that window and no close was ever saved. This restores the evidence store once,
then calls the same ``capture_live`` every ``--interval`` minutes until ``--minutes`` have
passed, backing up after each capture that saved something. The Odds API is called only
when a saved pick is inside its closing window. Before each cycle it asks Drive for the
evidence files created since it began and merges any newly saved run, so the picks of a
Refresh made while it runs get their closes too.

``--window`` narrows that window (minutes before the start). With a window just longer than the
interval each pick is read once, in the last cycle before its game, not in every cycle of its last
half hour. ``--max-credits`` ends the run once the Odds API credits it may have spent reach that
number: this job has no allowance in ``app_core.research_api_budget``, so without it the account's
monthly quota is the only limit.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The most one fetch of a sport can cost. A request is 3 credits (three markets, one group of books).
# NFL asks two sport keys and one 2-credit Novig recovery; NCAAF up to five 2-credit Novig recoveries;
# MLB two sport keys. The provider's own count is printed beside the estimate when it answers.
CREDITS = {"NFL": 8, "NCAAF": 13, "MLB": 6}
DEFAULT_CREDITS = 3


# How far before the loop's own start the "saved since" listing reaches: a file can take a while to show
# in Drive's listings, and the first restore may have passed its page before the file was written.
REFRESH_MARGIN_SECONDS = 15 * 60


def run(database, minutes, interval, *, capture=None, restore=None, backup=None,
        clock=time.monotonic, sleep=time.sleep, window=None, max_credits=0, fetch=None, refresh=None):
    if capture is None:
        from app_core.activation_closing import capture_live as capture
    if restore is None:
        from app_core.evidence_remote import restore
        if refresh is None:
            # The store is restored once; a Refresh made while the loop runs is picked up before each cycle
            # (2026-11-19: without it the loop read closes for an earlier run's picks, never the last one's).
            from app_core.evidence_remote import restore_newer
            began = time.time() - REFRESH_MARGIN_SECONDS

            def refresh(path):
                return restore_newer(path, began)
    if backup is None:
        from app_core.evidence_remote import sync

        def backup(path):
            if not sync(path, incremental=True):
                raise RuntimeError("Evidence backup was not verified")
    restore(database)
    start = clock()
    deadline = start + minutes * 60
    totals = {"cycles": 0, "verified": 0, "unavailable": 0, "backups": 0, "errors": 0}
    pending = False
    # A capture is given only the options it takes (the tests pass plain stand-ins).
    accepted = inspect.signature(capture).parameters
    options, fetched = {}, {}
    if window is not None and "within_seconds" in accepted:
        options["within_seconds"] = window * 60
    if "fetch" in accepted:
        provider_usage = provider_refusal = None
        if fetch is None:
            from app_core import odds_api
            from core.streamlit_pipeline import fetch_live_odds_dataframe as fetch
            provider_usage = odds_api.last_usage
            # A refused request (HTTP 401 or 403, e.g. the month's credits used up) is printed as such:
            # on 2026-11-20 seven picks read "game_not_in_live_odds" beside the last good count.
            provider_refusal = odds_api.last_refusal

        def counted(sports):
            cost = sum(CREDITS.get(str(sport).upper(), DEFAULT_CREDITS) for sport in sports)
            totals["credits_estimate"] = totals.get("credits_estimate", 0) + cost
            fetched.update(sports=list(sports), credits_estimate=totals["credits_estimate"])
            try:
                return fetch(sports)
            finally:
                if provider_usage:
                    fetched["provider_usage"] = dict(provider_usage)
                if provider_refusal:
                    fetched["provider_refusal"] = dict(provider_refusal)
        options["fetch"] = counted

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
        fetched.clear()
        if refresh is not None:
            try:
                added = refresh(database)
            except Exception as exc:  # the picks already held are still read
                totals["refresh_errors"] = totals.get("refresh_errors", 0) + 1
                print(json.dumps({"cycle": totals["cycles"], "refresh_error": type(exc).__name__}), flush=True)
            else:
                if added:
                    totals["runs_added"] = totals.get("runs_added", 0) + added
                    fetched["runs_added"] = added
        try:
            result = capture(database, **({"unmatched": unmatched} if "unmatched" in accepted else {}), **options)
        except Exception as exc:  # one failed cycle must not end the slate
            totals["errors"] += 1
            result = {"verified": 0, "error": type(exc).__name__}
        totals["verified"] += result.get("verified", 0)
        totals["unavailable"] += result.get("unavailable", 0)
        print(json.dumps({"cycle": totals["cycles"], **{k: v for k, v in result.items() if k != "reasons"},
                          "reasons": result.get("reasons"),
                          **({"unmatched_detail": unmatched} if unmatched else {}), **fetched}), flush=True)
        pending = pending or bool(result.get("verified"))
        if pending:
            back_up()
        if max_credits and totals.get("credits_estimate", 0) >= max_credits:
            print(json.dumps({"stopped": "credit_cap", "credits_estimate": totals["credits_estimate"],
                              "max_credits": max_credits}), flush=True)
            return totals
        now = clock()
        if now + interval * 60 > deadline:
            return totals
        # Cycles keep to the grid set at the start, so a slow cycle does not push the next one later and
        # a stalled one is not made up for: with a window barely longer than the interval, drift would
        # let a game fall between two cycles, and catching up would fetch twice for nothing.
        sleep(interval * 60 - (now - start) % (interval * 60))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database", default="data/prediction_evidence/evidence.sqlite3")
    parser.add_argument("--minutes", type=float, default=340)
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--window", type=float, help="minutes before its start a game is read (30 at most, the default)")
    parser.add_argument("--max-credits", type=int, default=0, help="end the run at this many estimated Odds API credits (0: no cap)")
    args = parser.parse_args(argv)
    if args.window is not None and args.window <= args.interval:
        parser.error("--window must be longer than --interval, or a game can fall between two cycles")
    print(json.dumps(run(args.database, args.minutes, args.interval, window=args.window, max_credits=args.max_credits)),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
