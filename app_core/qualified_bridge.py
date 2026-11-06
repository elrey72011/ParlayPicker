"""C6 bridge: a qualified market's own bound predictions as app best-pick rows (roadmap C6).

``docs/roadmap/C6-PLAN-2026-11-06.md``. The research models that can qualify a market live
outside the app pipeline, and the app's ML model is blocked for spreads and totals, so a
qualified market would otherwise have no rows for the wager contract or Gemini. This turns
the latest bound prediction per upcoming game, for **qualified markets only**, into a row
carrying every field ``core.sport_market_gate.market_gate`` checks, copied from the stored
prediction, quote, event, model and calibration records (never recomputed or guessed).

The rows then go through the existing chain unchanged: live wager contract, market gate
(deployment, owner activation, exposure), maturity, Gemini secondary review, stake. This
module grants no authority: a row that is stale, mismatched or unactivated is blocked by
the gate exactly as any other row would be.
"""
from __future__ import annotations

from datetime import datetime, timezone

from app_core import prospective_evidence as evidence
from app_core.research_api_budget import REFERENCE_BOOKS

MARKET_TYPES = {("SPREAD", True): "spread_home", ("SPREAD", False): "spread_away",
                ("PUCK_LINE", True): "spread_home", ("PUCK_LINE", False): "spread_away",
                ("RUN_LINE", True): "spread_home", ("RUN_LINE", False): "spread_away",
                ("TOTAL", "Over"): "total_over", ("TOTAL", "Under"): "total_under"}


def _market_type(family, event, selection):
    if family == "TOTAL":
        return MARKET_TYPES.get((family, selection))
    return MARKET_TYPES.get((family, selection == event["home_team"]))


def _label(selection, line, family):
    return f"{selection} {line:g}" if family == "TOTAL" else f"{selection} {line:+g}"


def row(prediction, quote, event, policy_version=None):
    """One best-pick row from a stored bound prediction and its quote and event."""
    family = prediction["market_family"]
    market_type = _market_type(family, event, quote["selection"])
    return {
        # pipeline columns the board and the contract's aliases read
        "league": event["sport"], "sport": event["sport"], "home_team": event["home_team"],
        "away_team": event["away_team"], "matchup_id": event["game_id"], "game_id": event["game_id"],
        "market_type": market_type, "best_pick": _label(quote["selection"], quote["line"], family),
        "selection": quote["selection"], "line": quote["line"], "odds_american": quote["american_odds"],
        "decimal_odds": quote["decimal_odds"], "book": quote["sportsbook"], "quote_time": quote["quote_timestamp"],
        "start": event["scheduled_start"],
        # exact model and calibration scope
        "model_id": prediction["model_id"], "model_version": prediction["model_version"],
        "calibration_id": prediction["calibration_id"], "calibration_version": prediction["calibration_version"],
        "model_trained_through": prediction["model_trained_through"],
        "model_available_at": prediction["model_available_at"],
        "calibration_available_at": prediction["calibration_available_at"],
        "evidence_frozen_at": prediction["feature_frozen_at"],
        "prediction_generated_at": prediction["prediction_timestamp"],
        "mean_probability": prediction["mean_probability"],
        "conservative_probability": prediction["conservative_probability"],
        "push_probability": prediction["push_probability"], "loss_probability": prediction["loss_probability"],
        # identity and canonical quote binding
        "provider_namespace": event["provider_namespace"], "provider_event_id": event["provider_event_id"],
        "identity_verified": bool(event.get("home_team_id") and event.get("away_team_id")),
        "team_ids": [event["home_team_id"], event["away_team_id"]],
        "exact_quote_verified": quote.get("quote_verified") == 1,
        "prospective_quote_id": quote["quote_id"], "prospective_event_id": event["event_id"],
        "quote_source_id": quote["source_id"], "quote_source_hash": quote.get("source_hash"),
        "sport_market_family": family, "sport_policy_version": policy_version,
        "selected_as_best_pick": True, "candidate_source": "qualified_bridge",
    }


def expected_value(prediction, quote):
    """Per-unit EV at the quoted price: win x (decimal - 1) - loss; a push returns the stake."""
    return prediction["mean_probability"] * (quote["decimal_odds"] - 1) - prediction["loss_probability"]


def _rank(prediction, quote):
    """Newest prediction first; at the same time (both sides of a C6 decision capture), the better EV."""
    return (prediction["prediction_timestamp"], expected_value(prediction, quote), prediction["observation_id"])


def rows(path, qualified, *, now=None, policy_versions=None):
    """Best-pick rows for upcoming games in the ``qualified`` {(sport, family)} markets.

    One row per game: its newest bound prediction, and when that time has both sides
    (``app_core.decision_capture``), the side with the better expected value.
    """
    now = now or datetime.now(timezone.utc)
    policy_versions = policy_versions or {}
    t = evidence._time
    out = []
    for sport, family in sorted(qualified):
        events = {e["event_id"]: e for e in evidence.read_records(path, "prospective_event", sport=sport)}
        quotes = {q["quote_id"]: q for q in evidence.read_records(path, "prospective_quote", sport=sport,
                                                                     market_family=family)}
        latest = {}
        for p in evidence.read_records(path, "prospective_prediction", sport=sport, market_family=family):
            event, quote = events.get(p["event_id"]), quotes.get(p["quote_id"])
            if (event is None or quote is None or quote["sportsbook"] in REFERENCE_BOOKS
                    or not t(p["prediction_timestamp"], "prediction") <= now < t(event["scheduled_start"], "start")):
                continue
            best = latest.get(p["event_id"])
            if best is None or _rank(p, quote) > _rank(*best[:2]):
                latest[p["event_id"]] = (p, quote, event)
        for p, quote, event in latest.values():
            out.append(row(p, quote, event, policy_versions.get(f"{sport}:{family}")))
    return sorted(out, key=lambda r: (r["start"], r["sport"], r["game_id"]))


BRIDGE_ENV = "PARLAYPICKER_QUALIFIED_BRIDGE"
FAMILY_OF = {"spread_home": "SPREAD", "spread_away": "SPREAD", "total_over": "TOTAL", "total_under": "TOTAL"}


def enabled(setting=None):
    import os
    return str((setting or os.environ.get)(BRIDGE_ENV, "") or "").strip().lower() == "true"


def _family(sport, market_type):
    from core.market_policy import sport_market_family
    return sport_market_family(sport, market_type)


REFRESH_ENV = "PARLAYPICKER_QUALIFIED_BRIDGE_REFRESH"
REFRESH_EVERY = 600  # seconds; a full read of the evidence remote takes minutes
STAMP = ".qualified-bridge-refreshed"


def refresh_store(path=None, *, setting=None, clock=None, client=None, folder=None, sync=None):
    """Bring the app's canonical store up to date from the evidence remote, read-only.

    The decision-capture job (``app_core.decision_capture``) writes fresh quotes and
    predictions to the remote every 15 minutes during a slate; the app's market gate
    and this bridge read the local store (``prospective_evidence._path``). This restores
    every remote row the local store lacks. The client refuses every write
    (``ReadOnlyDrive``), cached object bytes are reused only when they match Drive's
    checksum, and it runs at most once per ``REFRESH_EVERY`` seconds. Off with
    ``PARLAYPICKER_QUALIFIED_BRIDGE_REFRESH=false``. Returns a diagnostic; never raises.
    """
    import os
    import time
    from pathlib import Path
    clock = clock or time.time
    if str((setting or os.environ.get)(REFRESH_ENV, "true") or "true").strip().lower() == "false":
        return {"status": "OFF"}
    target = Path(evidence._path(path))
    stamp = target.with_name(STAMP)
    try:
        last = float(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        last = None
    started = clock()
    if last is not None and 0 <= started - last < REFRESH_EVERY:
        return {"status": "RECENT", "seconds_since_refresh": round(started - last)}
    lock = target.with_name(STAMP + ".lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            held = started - lock.stat().st_mtime
        except OSError:
            held = 0
        if held < 2 * REFRESH_EVERY:
            return {"status": "IN_PROGRESS"}  # another session is refreshing
        lock.unlink(missing_ok=True)          # a crashed refresh left it behind
        return {"status": "STALE_LOCK_CLEARED"}
    os.close(lock_fd)
    try:
        return _refresh(target, stamp, started, clock, client, folder, sync)
    finally:
        lock.unlink(missing_ok=True)


def _refresh(target, stamp, started, clock, client, folder, sync):
    import os
    try:
        if client is None:
            from app_core.evidence_drive import DriveStore
            from app_core.evidence_remote import settings
            folder = folder or settings()[0]
            if not folder or not os.getenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "").strip():
                return {"status": "MISSING_CREDENTIALS"}
            client = DriveStore(folder)
        if sync is None:
            from app_core.prospective_remote import sync
        from app_core.read_only_drive import ReadOnlyDrive, ReadOnlyViolation
        cache = target.parent / "drive-cache"
        cache.mkdir(exist_ok=True)
        guarded = ReadOnlyDrive(client, cache)
        try:
            counts = sync(target, guarded, folder, {})
            status = "REFRESHED"
        except ReadOnlyViolation:
            # The restore finished; the local store also holds rows the remote lacks.
            counts, status = None, "REFRESHED_LOCAL_ROWS_NOT_UPLOADED"
        stamp.write_text(str(started), encoding="utf-8")
        return {"status": status, "seconds": round(clock() - started, 1), "counts": counts}
    except Exception as exc:  # never break Refresh; the gate then reads what is local
        return {"status": "REFRESH_FAILED", "error": type(exc).__name__}


def refresh_if_enabled(setting=None, **kwargs):
    """``refresh_store`` when the bridge is on (``PARLAYPICKER_QUALIFIED_BRIDGE=true``), else OFF."""
    return refresh_store(setting=setting, **kwargs) if enabled(setting) else {"status": "OFF"}


def inject(analysis_df, best_picks_df, *, now=None, setting=None, configuration=None, assess=None, make_rows=None):
    """Replace each qualified market's app rows with its bridge rows, before eligibility and Gemini.

    Off unless ``PARLAYPICKER_QUALIFIED_BRIDGE=true``. Returns (analysis_df, best_picks_df, diagnostic).
    Unqualified markets and every other row are untouched.
    """
    import pandas as pd
    if not enabled(setting):
        return analysis_df, best_picks_df, {"status": "OFF"}
    now = now or datetime.now(timezone.utc)
    if configuration is None:
        from core.live_wager_contract import load_configuration
        _, configuration, _ = load_configuration(now)
    path = (configuration or {}).get("prospective_evidence_path")
    if assess is None:
        from app_core.selection import qualification
        qualified = qualification.qualified_set(qualification.assess(path))
    else:
        qualified = frozenset(assess(path))
    if not qualified:
        return analysis_df, best_picks_df, {"status": "NO_QUALIFIED_MARKET"}
    versions = {key: ((record or {}).get("policy") or {}).get("version")
                for key, record in ((configuration or {}).get("market_activations") or {}).items()}
    bridged = (make_rows or rows)(path, qualified, now=now, policy_versions=versions)

    def replace(frame):
        frame = frame if frame is not None else pd.DataFrame()
        if not frame.empty and {"league", "market_type"} <= set(frame.columns):
            keep = [(str(lg).upper(), _family(str(lg).upper(), mt)) not in qualified
                    for lg, mt in zip(frame["league"], frame["market_type"])]
            frame = frame.loc[keep]
        return pd.concat([frame, pd.DataFrame(bridged)], ignore_index=True) if bridged else frame.reset_index(drop=True)

    return replace(analysis_df), replace(best_picks_df), {
        "status": "INJECTED", "qualified_markets": sorted(" ".join(m) for m in qualified), "rows": len(bridged)}
