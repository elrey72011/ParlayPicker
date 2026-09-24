"""Exact sportsbook ticket quotes. No provider is connected by default.

An integrating provider must return a ticket-level response with the exact
selections and component identities it priced. Leg-odds multiplication is never
accepted as a quote. This module neither validates models nor places orders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Mapping, Protocol

from app_core.public_quote_policy import canonical_book_label
from app_core.quote_freshness import QUOTE_MAX_AGE_SECONDS
from core.true_parlay_engine import component_hashes, leg_hash, ticket_hash
from core.wager_decisions import aware, decimal_price, finite


PRODUCTS = frozenset({"STANDARD_PARLAY", "SAME_GAME_PARLAY", "CROSS_GAME_PARLAY"})
MANUAL_SOURCES = frozenset({"SPORTSBOOK_DISPLAY", "SPORTSBOOK_SCREENSHOT"})


@dataclass(frozen=True)
class ProviderCapabilities:
    supports_standard_parlay: bool = False
    supports_sgp: bool = False
    supports_cross_game: bool = False
    supports_exact_ticket_quote: bool = False
    supports_quote_expiration: bool = False
    supports_provider_ticket_id: bool = False
    supports_settlement_rules: bool = False


@dataclass(frozen=True)
class ParlayQuoteEvidence:
    quote_id: str
    provider: str
    sportsbook: str
    provider_ticket_id: str
    product_type: str
    ticket_hash: str
    leg_hashes: list[str]
    american_odds: float
    decimal_odds: float
    quoted_at: str
    expires_at: str
    verification_state: str
    raw_evidence_hash: str
    provider_response_id: str | None
    selection_bindings: list[dict]
    component_hashes: list[str]
    sgp_components: list[dict]
    source: str
    source_type: str
    settlement_rules_id: str
    owner_id: str | None = None
    owner_authorization_id: str | None = None
    owner_confirmed_at: str | None = None
    artifact_reference: str | None = None

    def as_engine_quote(self) -> dict:
        """Pass exact evidence to the existing fail-closed decision engine."""
        return asdict(self)


@dataclass(frozen=True)
class QuoteAttempt:
    evidence: ParlayQuoteEvidence | None
    blockers: tuple[str, ...]
    raw_evidence_bytes: bytes | None = field(default=None, repr=False)

    @property
    def available(self) -> bool:
        return self.evidence is not None and not self.blockers


class TicketQuoteProvider(Protocol):
    provider_name: str
    sportsbook: str
    capabilities: ProviderCapabilities

    def fetch_exact_ticket_quote(self, ticket_request: Mapping[str, Any]) -> Mapping[str, Any]: ...


class NoConnectedTicketProvider:
    """Explicit blocker until a real, authorized sportsbook adapter is wired."""

    provider_name = "UNCONNECTED"
    sportsbook = ""
    capabilities = ProviderCapabilities()

    def fetch_exact_ticket_quote(self, ticket_request: Mapping[str, Any]) -> Mapping[str, Any]:
        raise RuntimeError("No sportsbook ticket quote provider is connected")


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _now(value: datetime | None) -> datetime:
    result = value or datetime.now(timezone.utc)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    return result.astimezone(timezone.utc)


def _component_legs(component: dict) -> list[dict]:
    return component["legs"] if component.get("product_type") == "SAME_GAME_PARLAY" else [component]


def bind_ticket_request(ticket_request: Mapping[str, Any]) -> dict:
    """Build the exact identity expected back from a ticket quote provider."""
    if not isinstance(ticket_request, Mapping):
        raise ValueError("ticket request must be a mapping")
    product = ticket_request.get("product_type")
    components = ticket_request.get("components")
    book = canonical_book_label(ticket_request.get("sportsbook"))
    if product not in PRODUCTS or not _text(book) or not isinstance(components, list) or not components:
        raise ValueError("ticket request identity missing")
    if any(not isinstance(component, dict) or
           (component.get("product_type") == "SAME_GAME_PARLAY" and
            (not isinstance(component.get("legs"), list) or not component["legs"]))
           for component in components):
        raise ValueError("ticket components malformed")
    if product != "CROSS_GAME_PARLAY" and any(
            component.get("product_type") == "SAME_GAME_PARLAY" for component in components):
        raise ValueError("SGP blocks require a cross-game ticket")
    legs = [leg for component in components for leg in _component_legs(component)]
    if any(not isinstance(leg, dict) for leg in legs):
        raise ValueError("ticket legs malformed")
    selections = []
    for component in components:
        sgp_id = component.get("parlay_id") if component.get("product_type") == "SAME_GAME_PARLAY" else None
        if sgp_id is not None and not _text(sgp_id):
            raise ValueError("SGP component identity missing")
        for leg in _component_legs(component):
            required = ("candidate_id", "game_id", "sport", "provider_namespace",
                        "provider_event_id", "market_type", "selection", "sportsbook")
            if any(not _text(leg.get(key)) for key in required):
                raise ValueError("provider leg identity missing")
            if canonical_book_label(leg["sportsbook"]) != book or finite(leg.get("line")) is None:
                raise ValueError("provider leg book or line invalid")
            selections.append({
                "leg_hash": leg_hash(leg), "provider_namespace": leg["provider_namespace"],
                "provider_event_id": leg["provider_event_id"],
                "provider_market_id": leg.get("provider_market_id"),
                "provider_selection_id": leg.get("provider_selection_id"),
                "market_type": leg["market_type"], "selection": leg["selection"],
                "line": float(leg["line"]), "sportsbook": book,
                "sgp_component_id": sgp_id,
            })
    if len({item["leg_hash"] for item in selections}) != len(selections):
        raise ValueError("duplicate exact legs")
    if (product == "STANDARD_PARLAY" and len({leg["game_id"] for leg in legs}) != len(legs)) or (
            product == "SAME_GAME_PARLAY" and len({leg["game_id"] for leg in legs}) != 1):
        raise ValueError("ticket product game identity invalid")
    sgp_components = []
    if product == "CROSS_GAME_PARLAY":
        for component in components:
            if component.get("product_type") == "SAME_GAME_PARLAY":
                if any(not _text(component.get(key)) for key in
                       ("parlay_id", "ticket_hash", "provider_component_id")):
                    raise ValueError("cross-game SGP provider component identity missing")
                expected = ticket_hash("SAME_GAME_PARLAY", component["legs"], component.get("sportsbook"))
                if component["ticket_hash"] != expected:
                    raise ValueError("cross-game SGP ticket hash mismatch")
                sgp_components.append({
                    "parlay_id": component["parlay_id"],
                    "ticket_hash": component["ticket_hash"],
                    "provider_component_id": component["provider_component_id"],
                    "component_hash": component_hashes([component])[0],
                })
    selections.sort(key=lambda item: item["leg_hash"])
    sgp_components.sort(key=lambda item: item["component_hash"])
    return {
        "product_type": product, "sportsbook": book,
        "ticket_hash": ticket_hash(product, components, book),
        "leg_hashes": sorted(item["leg_hash"] for item in selections),
        "selection_bindings": selections,
        "component_hashes": component_hashes(components),
        "sgp_components": sgp_components,
    }


def _validate_common(binding: dict, raw: Mapping[str, Any], now: datetime) -> tuple[str, ...]:
    blockers = []
    if raw.get("product_type") != binding["product_type"] or canonical_book_label(raw.get("sportsbook")) != binding["sportsbook"]:
        blockers.append("TICKET_BINDING_MISMATCH")
    if (raw.get("selection_bindings") != binding["selection_bindings"] or
            raw.get("component_hashes") != binding["component_hashes"] or
            raw.get("sgp_components") != binding["sgp_components"]):
        blockers.append("TICKET_BINDING_MISMATCH")
    if raw.get("price_origin") != "EXECUTABLE_TICKET":
        blockers.append("TICKET_PRICE_INFERRED")
    if any(not _text(raw.get(key)) for key in ("quote_id", "provider_ticket_id", "settlement_rules_id")):
        blockers.append("TICKET_QUOTE_MALFORMED")
    american = finite(raw.get("american_odds"))
    decimal = finite(raw.get("decimal_odds"))
    converted = decimal_price(american)
    if (american is None or decimal is None or decimal <= 1 or converted is None or
            abs(converted - decimal) > 0.002):
        blockers.append("TICKET_PRICE_MALFORMED")
    quoted = aware(raw.get("quoted_at"))
    expires = aware(raw.get("expires_at"))
    if quoted is None or expires is None or expires <= quoted:
        blockers.append("TICKET_TIME_MALFORMED")
    else:
        if quoted > now:
            blockers.append("TICKET_QUOTE_FUTURE")
        elif now - quoted > timedelta(seconds=QUOTE_MAX_AGE_SECONDS):
            blockers.append("TICKET_QUOTE_STALE")
        if expires <= now:
            blockers.append("TICKET_QUOTE_EXPIRED")
        if expires - quoted > timedelta(seconds=QUOTE_MAX_AGE_SECONDS):
            blockers.append("TICKET_EXPIRATION_UNBOUNDED")
    return tuple(sorted(set(blockers)))


def _evidence(binding: dict, raw: Mapping[str, Any], *, provider: str,
              source: str, source_type: str, raw_hash: str,
              owner: Mapping[str, Any] | None = None) -> ParlayQuoteEvidence:
    owner = owner or {}
    return ParlayQuoteEvidence(
        quote_id=raw["quote_id"], provider=provider, sportsbook=binding["sportsbook"],
        provider_ticket_id=raw["provider_ticket_id"], product_type=binding["product_type"],
        ticket_hash=binding["ticket_hash"], leg_hashes=binding["leg_hashes"],
        american_odds=float(raw["american_odds"]), decimal_odds=float(raw["decimal_odds"]),
        quoted_at=raw["quoted_at"], expires_at=raw["expires_at"],
        verification_state="VERIFIED", raw_evidence_hash=raw_hash,
        provider_response_id=raw.get("provider_response_id"),
        selection_bindings=binding["selection_bindings"],
        component_hashes=binding["component_hashes"], sgp_components=binding["sgp_components"],
        source=source, source_type=source_type,
        settlement_rules_id=raw["settlement_rules_id"],
        owner_id=owner.get("owner_id"), owner_authorization_id=owner.get("authorization_id"),
        owner_confirmed_at=owner.get("confirmed_at"), artifact_reference=owner.get("artifact_reference"),
    )


def get_parlay_ticket_quote(ticket_request: Mapping[str, Any], provider: TicketQuoteProvider,
                            *, now: datetime | None = None) -> QuoteAttempt:
    """Request one actual ticket price; unsupported capability returns a blocker."""
    now = _now(now)
    try:
        binding = bind_ticket_request(ticket_request)
    except (ValueError, TypeError, OverflowError):
        return QuoteAttempt(None, ("TICKET_REQUEST_INVALID",))
    caps = getattr(provider, "capabilities", None)
    required_product = {
        "STANDARD_PARLAY": "supports_standard_parlay",
        "SAME_GAME_PARLAY": "supports_sgp",
        "CROSS_GAME_PARLAY": "supports_cross_game",
    }[binding["product_type"]]
    if not isinstance(caps, ProviderCapabilities) or any(type(getattr(caps, key)) is not bool for key in
            ProviderCapabilities.__dataclass_fields__):
        return QuoteAttempt(None, ("PROVIDER_CAPABILITY_UNKNOWN",))
    if not all((getattr(caps, required_product), caps.supports_exact_ticket_quote,
                caps.supports_quote_expiration, caps.supports_provider_ticket_id,
                caps.supports_settlement_rules)):
        return QuoteAttempt(None, ("PROVIDER_CAPABILITY_UNSUPPORTED",))
    provider_name = getattr(provider, "provider_name", None)
    if not _text(provider_name) or canonical_book_label(getattr(provider, "sportsbook", None)) != binding["sportsbook"]:
        return QuoteAttempt(None, ("PROVIDER_BOOK_MISMATCH",))
    try:
        raw = provider.fetch_exact_ticket_quote(ticket_request)
    except TimeoutError:
        return QuoteAttempt(None, ("PROVIDER_TIMEOUT",))
    except PermissionError:
        return QuoteAttempt(None, ("PROVIDER_AUTH_FAILURE",))
    except ConnectionError:
        return QuoteAttempt(None, ("PROVIDER_NETWORK_FAILURE",))
    except Exception:
        return QuoteAttempt(None, ("PROVIDER_FAILURE",))
    if not isinstance(raw, Mapping):
        return QuoteAttempt(None, ("TICKET_QUOTE_MALFORMED",))
    if raw.get("provider") != provider_name or not _text(raw.get("provider_response_id")):
        return QuoteAttempt(None, ("TICKET_QUOTE_MALFORMED",))
    try:
        raw_bytes = _json(dict(raw))
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    except (TypeError, ValueError, OverflowError):
        return QuoteAttempt(None, ("TICKET_QUOTE_MALFORMED",))
    blockers = _validate_common(binding, raw, now)
    if blockers:
        return QuoteAttempt(None, blockers)
    return QuoteAttempt(_evidence(binding, raw, provider=provider_name, source="SPORTSBOOK",
                                  source_type="PROVIDER_API", raw_hash=raw_hash), (), raw_bytes)


def capture_owner_confirmed_quote(ticket_request: Mapping[str, Any], confirmation: Mapping[str, Any],
                                  *, artifact_bytes: bytes, now: datetime | None = None) -> QuoteAttempt:
    """Capture an authenticated owner's exact displayed ticket without creating validation.

    The caller must authenticate the owner and retain the referenced artifact.
    The SHA-256 here binds the supplied artifact bytes; this function cannot
    prove that an arbitrary artifact depicts a real sportsbook screen.
    """
    now = _now(now)
    try:
        binding = bind_ticket_request(ticket_request)
    except (ValueError, TypeError, OverflowError):
        return QuoteAttempt(None, ("TICKET_REQUEST_INVALID",))
    if not isinstance(confirmation, Mapping) or not isinstance(artifact_bytes, bytes) or not artifact_bytes:
        return QuoteAttempt(None, ("OWNER_CONFIRMATION_MISSING",))
    if (confirmation.get("owner_confirmed") is not True or
            confirmation.get("source_type") not in MANUAL_SOURCES or
            any(not _text(confirmation.get(key)) for key in
                ("owner_id", "authorization_id", "artifact_reference"))):
        return QuoteAttempt(None, ("OWNER_CONFIRMATION_MISSING",))
    confirmed = aware(confirmation.get("confirmed_at"))
    quoted = aware(confirmation.get("quoted_at"))
    if confirmed is None or quoted is None or not quoted <= confirmed <= now:
        return QuoteAttempt(None, ("OWNER_CONFIRMATION_TIME_INVALID",))
    try:
        _json(dict(confirmation))
    except (TypeError, ValueError, OverflowError):
        return QuoteAttempt(None, ("TICKET_QUOTE_MALFORMED",))
    blockers = _validate_common(binding, confirmation, now)
    if blockers:
        return QuoteAttempt(None, blockers)
    raw_hash = hashlib.sha256(artifact_bytes).hexdigest()
    return QuoteAttempt(_evidence(binding, confirmation, provider="OWNER_CONFIRMED",
                                  source="OWNER_CONFIRMED", source_type=confirmation["source_type"],
                                  raw_hash=raw_hash, owner=confirmation), (), artifact_bytes)
