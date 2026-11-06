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
