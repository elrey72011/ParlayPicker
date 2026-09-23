"""Synthetic adapter fixtures exercise quote mechanics, never live validation."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import sqlite3
import unittest

from app_core import parlay_persistence as store
from app_core.parlay_ticket_quotes import (
    NoConnectedTicketProvider, ProviderCapabilities, bind_ticket_request,
    capture_owner_confirmed_quote, get_parlay_ticket_quote,
)
from core.true_parlay_engine import evaluate_ticket, leg_hash, ticket_hash


NOW = datetime(2026, 9, 23, 18, tzinfo=timezone.utc)


def at(minutes):
    return (NOW + timedelta(minutes=minutes)).isoformat()


def leg(number, *, game=None):
    return {
        "candidate_id": f"synthetic-c{number}", "game_id": game or f"g{number}",
        "sport": "MLB", "provider_namespace": "synthetic-feed",
        "provider_event_id": f"synthetic-event-{game or number}",
        "provider_market_id": f"synthetic-market-{number}",
        "provider_selection_id": f"synthetic-selection-{number}",
        "market_type": "spread_away", "selection": f"A{number} +1.5",
        "line": 1.5, "sportsbook": "Novig", "team_ids": [f"A{number}", f"B{number}"],
        "american_odds": -110, "quote_timestamp": at(-1),
        "analysis_timestamp": at(-2), "start": at(120),
        "identity_verified": True, "quote_verified": True, "production_eligible": True,
        "model_version": "synthetic-m1", "model_trained_through": at(-10000),
        "model_available_at": at(-100), "calibration_version": "synthetic-c1",
        "calibration_available_at": at(-100), "policy_version": "synthetic-policy",
        "evidence_snapshot_id": "synthetic-e1", "evidence_frozen_at": at(-3),
        "critical_input_state": "CLEAR", "material_news_status": "CLEAR",
        "probability_semantics": "UNCONDITIONAL", "probability_mean": .60,
        "probability_conservative": .55, "probability_push": 0., "probability_loss": .40,
    }


def request(product="STANDARD_PARLAY"):
    if product == "SAME_GAME_PARLAY":
        components = [leg(1, game="g1"), leg(2, game="g1")]
    elif product == "CROSS_GAME_PARLAY":
        sgp_legs = [leg(1, game="g1"), leg(2, game="g1")]
        components = [{"product_type": "SAME_GAME_PARLAY", "parlay_id": "synthetic-sgp",
                       "ticket_hash": ticket_hash("SAME_GAME_PARLAY", sgp_legs, "Novig"),
                       "provider_component_id": "synthetic-book-sgp-id",
                       "sportsbook": "Novig", "legs": sgp_legs}, leg(3)]
    else:
        components = [leg(1), leg(2)]
    return {"product_type": product, "sportsbook": "Novig", "components": components}


def provider_response(ticket_request, *, price=3.5, quote_id="synthetic-quote-1"):
    binding = bind_ticket_request(ticket_request)
    return {
        "provider": "synthetic-adapter", "product_type": binding["product_type"],
        "sportsbook": "Novig", "quote_id": quote_id,
        "provider_response_id": "synthetic-response-" + quote_id,
        "provider_ticket_id": "synthetic-book-ticket-" + quote_id,
        "selection_bindings": deepcopy(binding["selection_bindings"]),
        "component_hashes": binding["component_hashes"],
        "sgp_components": deepcopy(binding["sgp_components"]),
        "price_origin": "EXECUTABLE_TICKET", "american_odds": round((price - 1) * 100),
        "decimal_odds": price, "quoted_at": at(-1), "expires_at": at(10),
        "settlement_rules_id": "synthetic-rules",
    }


class FixtureProvider:
    provider_name = "synthetic-adapter"
    sportsbook = "Novig"
    capabilities = ProviderCapabilities(True, True, True, True, True, True, True)

    def __init__(self, response):
        self.response = response
        self.calls = 0

    def fetch_exact_ticket_quote(self, ticket_request):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return deepcopy(self.response)


class TicketQuoteTests(unittest.TestCase):
    def test_exact_standard_sgp_and_cross_game_bindings(self):
        for product in ("STANDARD_PARLAY", "SAME_GAME_PARLAY", "CROSS_GAME_PARLAY"):
            with self.subTest(product=product):
                ticket = request(product)
                result = get_parlay_ticket_quote(ticket, FixtureProvider(provider_response(ticket)), now=NOW)
                self.assertTrue(result.available)
                evidence = result.evidence.as_engine_quote()
                binding = bind_ticket_request(ticket)
                self.assertEqual(evidence["ticket_hash"], binding["ticket_hash"])
                self.assertEqual(evidence["leg_hashes"], binding["leg_hashes"])
                self.assertEqual(evidence["selection_bindings"], binding["selection_bindings"])
                self.assertEqual(evidence["sgp_components"], binding["sgp_components"])
                self.assertEqual(evidence["source"], "SPORTSBOOK")
                self.assertEqual(evidence["verification_state"], "VERIFIED")
                self.assertEqual(len(evidence["raw_evidence_hash"]), 64)
                self.assertEqual(hashlib.sha256(result.raw_evidence_bytes).hexdigest(),
                                 evidence["raw_evidence_hash"])
                self.assertNotIn("validation_id", evidence)

    def test_unsupported_provider_never_fetches_or_synthesizes_quote(self):
        ticket = request("SAME_GAME_PARLAY")
        provider = FixtureProvider(provider_response(ticket))
        provider.capabilities = ProviderCapabilities(supports_standard_parlay=True)
        result = get_parlay_ticket_quote(ticket, provider, now=NOW)
        self.assertEqual(result.blockers, ("PROVIDER_CAPABILITY_UNSUPPORTED",))
        self.assertIsNone(result.evidence)
        self.assertEqual(provider.calls, 0)
        disconnected = get_parlay_ticket_quote(ticket, NoConnectedTicketProvider(), now=NOW)
        self.assertEqual(disconnected.blockers, ("PROVIDER_CAPABILITY_UNSUPPORTED",))

    def test_leg_odds_product_cannot_become_ticket_price(self):
        ticket = request()
        raw = provider_response(ticket)
        raw["price_origin"] = "MULTIPLIED_LEG_ODDS"
        result = get_parlay_ticket_quote(ticket, FixtureProvider(raw), now=NOW)
        self.assertEqual(result.blockers, ("TICKET_PRICE_INFERRED",))
        self.assertIsNone(result.evidence)

    def test_time_price_and_identifier_rejections(self):
        ticket = request()
        for mutation, blocker in (
            ({"quoted_at": at(-31)}, "TICKET_QUOTE_STALE"),
            ({"quoted_at": at(1)}, "TICKET_QUOTE_FUTURE"),
            ({"expires_at": at(-0.5)}, "TICKET_QUOTE_EXPIRED"),
            ({"expires_at": at(40)}, "TICKET_EXPIRATION_UNBOUNDED"),
            ({"quoted_at": "not-a-time"}, "TICKET_TIME_MALFORMED"),
            ({"decimal_odds": float("nan")}, "TICKET_QUOTE_MALFORMED"),
            ({"decimal_odds": float("inf")}, "TICKET_QUOTE_MALFORMED"),
            ({"american_odds": 100}, "TICKET_PRICE_MALFORMED"),
            ({"provider_ticket_id": None}, "TICKET_QUOTE_MALFORMED"),
            ({"provider_response_id": None}, "TICKET_QUOTE_MALFORMED"),
        ):
            with self.subTest(mutation=mutation):
                raw = provider_response(ticket)
                raw.update(mutation)
                result = get_parlay_ticket_quote(ticket, FixtureProvider(raw), now=NOW)
                self.assertIn(blocker, result.blockers)
                self.assertIsNone(result.evidence)

    def test_event_line_selection_market_book_and_sgp_component_mutations(self):
        original = request("CROSS_GAME_PARLAY")
        raw = provider_response(original)
        for key, value in (("provider_event_id", "other-event"), ("line", 2.5),
                           ("selection", "Other +1.5"), ("market_type", "total_over"),
                           ("sportsbook", "FanDuel")):
            with self.subTest(key=key):
                ticket = deepcopy(original)
                ticket["components"][1][key] = value
                result = get_parlay_ticket_quote(ticket, FixtureProvider(raw), now=NOW)
                self.assertIsNone(result.evidence)
                self.assertIn("TICKET_BINDING_MISMATCH" if key != "sportsbook" else "TICKET_REQUEST_INVALID",
                              result.blockers)
        ticket = deepcopy(original)
        ticket["components"][0]["provider_component_id"] = "another-component"
        result = get_parlay_ticket_quote(ticket, FixtureProvider(raw), now=NOW)
        self.assertIn("TICKET_BINDING_MISMATCH", result.blockers)

    def test_timeout_and_auth_failure_are_explicit(self):
        ticket = request()
        for exc, blocker in ((TimeoutError(), "PROVIDER_TIMEOUT"),
                             (PermissionError(), "PROVIDER_AUTH_FAILURE"),
                             (ConnectionError(), "PROVIDER_NETWORK_FAILURE")):
            result = get_parlay_ticket_quote(ticket, FixtureProvider(exc), now=NOW)
            self.assertEqual(result.blockers, (blocker,))

    def test_manual_quote_requires_owner_attestation_and_artifact(self):
        ticket = request()
        confirmation = provider_response(ticket)
        confirmation.update(owner_confirmed=True, owner_id="synthetic-owner",
                            authorization_id="synthetic-auth-record",
                            confirmed_at=at(0), source_type="SPORTSBOOK_SCREENSHOT",
                            artifact_reference="synthetic-private-capture")
        confirmation.pop("provider_response_id")
        missing = capture_owner_confirmed_quote(ticket, dict(confirmation, owner_confirmed=False),
                                                artifact_bytes=b"synthetic capture", now=NOW)
        self.assertEqual(missing.blockers, ("OWNER_CONFIRMATION_MISSING",))
        result = capture_owner_confirmed_quote(ticket, confirmation,
                                               artifact_bytes=b"synthetic capture", now=NOW)
        self.assertTrue(result.available)
        self.assertEqual(result.evidence.source, "OWNER_CONFIRMED")
        self.assertEqual(result.evidence.raw_evidence_hash,
                         hashlib.sha256(b"synthetic capture").hexdigest())
        self.assertEqual(result.raw_evidence_bytes, b"synthetic capture")
        self.assertIsNone(result.evidence.provider_response_id)
        self.assertNotIn("validation_id", result.evidence.as_engine_quote())

    def test_repricing_appends_and_persists_provider_evidence(self):
        import tempfile
        from pathlib import Path

        ticket = request()
        first_quote = get_parlay_ticket_quote(ticket, FixtureProvider(provider_response(ticket)), now=NOW).evidence
        second_quote = get_parlay_ticket_quote(
            ticket, FixtureProvider(provider_response(ticket, price=2.0, quote_id="synthetic-quote-2")), now=NOW,
        ).evidence
        self.assertIsNotNone(first_quote)
        self.assertIsNotNone(second_quote)
        binding = bind_ticket_request(ticket)
        legs = [dict(row, leg_hash=leg_hash(row)) for row in ticket["components"]]
        base = {"parlay_id": binding["ticket_hash"], "product_type": "STANDARD_PARLAY",
                "status": "UNVALIDATED", "sportsbook": "Novig", "ticket_hash": binding["ticket_hash"],
                "production_eligible": False, "recommended_stake_dollars": 0,
                "blockers": ["PRODUCT_UNVALIDATED"], "decision_at": at(0)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parlay.sqlite3"
            first = store.save_decision(path, base, legs, quote=first_quote.as_engine_quote())
            second = store.save_decision(path, dict(base, decision_at=at(1)), legs,
                                         quote=second_quote.as_engine_quote())
            self.assertEqual(store.decision_history(path, binding["ticket_hash"]), [first, second])
            self.assertEqual(store.load_decision(path, first)["quote"]["decimal_odds"], 3.5)
            self.assertEqual(store.load_decision(path, second)["quote"]["decimal_odds"], 2.0)
            with store.connect(path) as db:
                rows = db.execute("SELECT provider,source_type,provider_response_id,raw_evidence_hash "
                                  "FROM parlay_quote ORDER BY quote_id").fetchall()
                self.assertEqual(len(rows), 2)
                self.assertEqual(rows[0][0:3], ("synthetic-adapter", "PROVIDER_API",
                                                "synthetic-response-synthetic-quote-1"))
                self.assertEqual(len(rows[0][3]), 64)
                with self.assertRaises(sqlite3.IntegrityError):
                    db.execute("UPDATE parlay_quote SET odds_decimal=1 WHERE decision_id=?", (first,))

    def test_better_ticket_price_does_not_validate_product(self):
        ticket = request()
        raw = provider_response(ticket, price=4.0)
        evidence = get_parlay_ticket_quote(ticket, FixtureProvider(raw), now=NOW).evidence
        self.assertIsNotNone(evidence)
        result = evaluate_ticket("STANDARD_PARLAY", ticket["components"], sportsbook="Novig",
                                 quote=evidence.as_engine_quote(),
                                 policy={"product_type": "STANDARD_PARLAY",
                                         "validation_state": "UNVALIDATED"}, now=NOW)
        self.assertIn("PRODUCT_UNVALIDATED", result["blockers"])
        self.assertEqual(result["recommended_stake"], 0)


if __name__ == "__main__":
    unittest.main()
