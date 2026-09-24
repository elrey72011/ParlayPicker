"""Exact sport/market authority checks before a straight wager receives stake.

The prospective store supplies a reviewed deployment; an independent owner
activation and current exposure ledger supply bankroll authority. Candidate
fields cannot create either authority. Missing facts produce explicit blockers.
"""
from __future__ import annotations

from datetime import datetime
import sqlite3
from typing import Mapping

from app_core.public_quote_policy import FALLBACK_BOOKS, canonical_book_label, supported_quote
from core.market_policy import sport_market_family
from core.sport_market_activation import verify_market_activation
from core.sport_policy import SportPolicy
from core.wager_decisions import aware, decimal_price, finite


IDENTIFIERS = ("model_id", "model_version", "calibration_id", "calibration_version")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _source_replays_quote(event: Mapping, quote: Mapping, sport: str) -> bool:
    """Reparse retained provider bytes; normalized DB columns alone are claims."""
    try:
        from app_core.prospective_evidence import verify_provider_offer
        return event.get("sport") == sport and verify_provider_offer(event, quote)
    except (KeyError, TypeError, ValueError, sqlite3.Error):
        return False


def _canonical_quote_matches(row: Mapping, family: str, path: object) -> bool:
    """Prove a new sport's fallback-book offer against retained source bytes."""
    quote_id = row.get("prospective_quote_id") or row.get("quote_id")
    event_id = row.get("prospective_event_id") or row.get("event_id")
    if not all(_text(value) for value in (quote_id, event_id, row.get("quote_source_id"),
                                           row.get("quote_source_hash"))):
        return False
    try:
        from app_core.prospective_evidence import load_record
        quote = load_record(path, "prospective_quote", quote_id)
        event = load_record(path, "prospective_event", event_id)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        return False
    if not quote or not event or quote.get("quote_verified") != 1 or not _source_replays_quote(event, quote, row.get("sport")):
        return False
    if quote.get("sport") != row.get("sport") or event.get("sport") != row.get("sport"):
        return False
    teams = row.get("team_ids")
    return (quote.get("event_id") == event_id == event.get("event_id")
            and event.get("game_id") == row.get("game_id")
            and quote.get("market_family") == family
            and event.get("provider_namespace") == row.get("provider_namespace")
            and event.get("provider_event_id") == row.get("provider_event_id")
            and _text(event.get("home_team_id")) and _text(event.get("away_team_id"))
            and list(teams or []) == [event["home_team_id"], event["away_team_id"]]
            and aware(event.get("scheduled_start")) == aware(row.get("start"))
            and quote.get("selection") == row.get("selection")
            and finite(quote.get("line")) == finite(row.get("line"))
            and finite(quote.get("american_odds")) == finite(row.get("odds_american"))
            and canonical_book_label(quote.get("sportsbook")) == canonical_book_label(row.get("book"))
            and aware(quote.get("quote_timestamp")) == aware(row.get("quote_time"))
            and quote.get("source_id") == row.get("quote_source_id")
            and quote.get("source_hash") == row.get("quote_source_hash"))


def market_gate(row: Mapping, deployment: Mapping | None, activation: Mapping | None,
                exposure: Mapping | None, now: datetime, *,
                allow_test_only: bool = False,
                evidence_path: object = None) -> tuple[SportPolicy | None, list[str]]:
    """Return owner policy only when every exact market authority is present."""
    family = sport_market_family(row.get("sport"), row.get("market_type"))
    blockers = []
    if family is None:
        return None, ["unsupported_sport_market_family"]
    if not isinstance(deployment, Mapping) or (
            deployment.get("sport") != row.get("sport") or
            deployment.get("market_family") != family):
        return None, ["exact_market_deployment_missing"]
    if deployment.get("deployment_state") not in {
            "PROVISIONAL_VALIDATED", "STANDARD_VALIDATED", "PREMIUM_VALIDATED"} or (
            deployment.get("validation_state") != deployment.get("deployment_state")):
        blockers.append("exact_market_not_validated")
    if not _text(deployment.get("validation_id")) or not _text(deployment.get("artifact_id")):
        blockers.append("exact_market_validation_artifact_missing")
    for field in IDENTIFIERS:
        if not _text(row.get(field)) or row.get(field) != deployment.get(field):
            blockers.append("exact_market_" + field + "_mismatch")
    # Existing UI rows use ``market_family='side'`` as a display grouping.
    # Only the explicit prospective scope field may claim this authority.
    if row.get("sport_market_family") not in (None, family):
        blockers.append("candidate_market_family_mismatch")
    if row.get("validation_id") not in (None, deployment.get("validation_id")):
        blockers.append("candidate_validation_id_mismatch")
    start, prediction, quote = (aware(row.get(field)) for field in
                                ("start", "prediction_generated_at", "quote_time"))
    trained, model_at, calibration_at, frozen = (aware(row.get(field)) for field in
                                                 ("model_trained_through", "model_available_at",
                                                  "calibration_available_at", "evidence_frozen_at"))
    if (None in (start, prediction, quote, trained, model_at, calibration_at, frozen)
            or not trained < prediction <= now < start
            or not model_at <= prediction or not calibration_at <= prediction
            or not frozen <= prediction or not quote <= prediction):
        blockers.append("market_model_calibration_chronology_invalid")
    if (not _text(row.get("provider_namespace")) or not _text(row.get("provider_event_id"))
            or row.get("identity_verified") is not True):
        blockers.append("exact_provider_event_identity_missing")
    teams = row.get("team_ids")
    if (not isinstance(teams, (tuple, list)) or len(teams) != 2 or
            not all(_text(team) for team in teams) or len(set(teams)) != 2):
        blockers.append("stable_team_identity_missing")
    odds = finite(row.get("odds_american"))
    ordinary_book = supported_quote({"sport": row.get("sport"), "quote_source": row.get("book")})
    new_sport_book = (row.get("sport") in {"NBA", "NCAAB", "NHL"}
                      and canonical_book_label(row.get("book")) in FALLBACK_BOOKS)
    canonical_quote = _canonical_quote_matches(row, family, evidence_path) if new_sport_book else False
    if new_sport_book and not canonical_quote:
        blockers.append("canonical_quote_source_binding_missing")
    if (not _text(row.get("selection")) or finite(row.get("line")) is None
            or odds is None or abs(odds) < 100 or decimal_price(odds) is None
            or not (ordinary_book or (new_sport_book and canonical_quote))
            or row.get("exact_quote_verified") is not True
            or quote is None or not 0 <= (now - quote).total_seconds() <= 1800):
        blockers.append("exact_pregame_quote_missing_or_stale")
    if not isinstance(activation, Mapping) or not isinstance(exposure, Mapping):
        blockers.append("owner_market_activation_or_exposure_missing")
        return None, blockers
    try:
        policy = verify_market_activation(activation, deployment, exposure, now=now,
                                          allow_test_only=allow_test_only)
    except (ValueError, KeyError, TypeError):
        blockers.append("owner_market_activation_or_exposure_invalid")
        return None, blockers
    if row.get("sport_policy_version") != policy.version:
        blockers.append("market_policy_version_mismatch")
    return (policy if not blockers else None), blockers
