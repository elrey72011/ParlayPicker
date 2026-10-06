"""Offline research adapter; document identity never proves listing applicability.

The accepted listing catalog is intentionally empty. Adding a real receipt is a
separate source/reader approval, not part of this software change. Tests supply
synthetic reviewed receipts at this boundary, never through provider JSON.
"""
from __future__ import annotations
import hashlib
import json
from copy import deepcopy

VERSION = "odds-api-novig-nfl-spread-source-v1"
RULES = "novig:nfl-001:half-point:full-game-ot:fvs:v1"
DOCUMENTS = {
    "provider_markets": "91866afcaf430477d8891ae0c87e819f8bab68af91a23dde2c73e27da79d0cdc",
    "provider_v4": "0f733b857e2de1bf4d71a182f12cd74a8c170a9ab6164fdd36ce63d36ce1733e",
    "provider_books": "a584fa89af44cc65c584111364cbd879ede6796b5fb1d7e49f92ac6f362fb763",
    "book_rulebook": "8b2be1a5bd292058c40e22bec08d85c569303ccd369204706a41b2e3095f605b",
    "book_nfl_001": "e15d5e0237c3496c8486b4878e71dba86c209780fbb1a2f8bf15d0e8d91fdd73",
}
# id -> legacy receipt or independently accepted exact packet/review hashes.
# Intake never adds entries; real-listing registration remains separate.
ACCEPTED_LISTINGS = {}

# Public templates prove document content, not exact listing applicability.
# These negative assessments reuse the existing private retention carrier and
# cannot accept a receipt, supply period/rules or authorize numeric value.
UNVERIFIED_MARKETS = {
    ("baseball_mlb", "spreads"): ("odds-api-novig-mlb-spread-unverified-v1",
        "book_mlb_001", "fbc1d024c6aff0f63678eb5a3ab519bf9e1fdd6a70a6f81cd74dcf87a9a63ffe"),
    ("americanfootball_nfl", "totals"): ("odds-api-novig-nfl-total-unverified-v1",
        "book_nfl_003", "93b92ee90e07b50ce5fff0ea2f7520e9eaa6c6509bb1489e320556192dda769f"),
}


def unverified_assessment(scope, offer, reference):
    version, document, sha = UNVERIFIED_MARKETS[scope]
    documents = {k:v for k,v in DOCUMENTS.items() if k != "book_nfl_001"}
    documents[document] = sha
    errors = ["SOURCE_MARKET_LISTING_BINDING_NOT_VERIFIED"]
    rejected = reference in ACCEPTED_LISTINGS if isinstance(reference, str) else False
    if rejected or (offer.get("sport"), offer.get("market")) != scope:
        errors.append("SOURCE_SCOPE_UNSUPPORTED")
        rejected = True
    return dict(version=version, reference=reference if isinstance(reference,str) else "",
        documents=documents, status="REJECTED" if rejected else "UNKNOWN",
        diagnostics=sorted(errors), receipt=None)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def identity(game, book, market, outcome):
    from app_core.producer_provenance import clock, number, team
    return dict(provider_namespace=game.get("odds_feed_source", "the_odds_api"),
        provider_event_id=game.get("id"), sport=game.get("sport_key"),
        home=team(game.get("home_team"), "NFL"), away=team(game.get("away_team"), "NFL"),
        start=clock(game.get("commence_time")), bookmaker=book.get("key"),
        market=market.get("key"), selection=team(outcome.get("name"), "NFL"),
        line=number(outcome.get("point")), price=number(outcome.get("price")),
        provider_quote_id=outcome.get("quote_id"),
        source_time=clock(market.get("last_update") or book.get("last_update")))


def verify(ref, offer, *, inference_time=None):
    """Bind an independently accepted receipt, not a feed's claim of verification."""
    from app_core.producer_provenance import clock, text
    result = dict(version=VERSION, reference=ref if isinstance(ref, str) else "",
                  documents=DOCUMENTS.copy(), status="UNKNOWN", diagnostics=[], receipt=None)
    errors = result["diagnostics"]
    accepted = ACCEPTED_LISTINGS.get(result["reference"])
    if not accepted:
        errors.append("SOURCE_LISTING_BINDING_NOT_VERIFIED")
        return result
    if not isinstance(accepted, dict):
        result.update(status="REJECTED", diagnostics=["SOURCE_RECEIPT_SCHEMA_UNSUPPORTED"])
        return result
    receipt = accepted.get("receipt")
    try:
        if not isinstance(receipt, dict) or digest(receipt) != accepted.get("sha256"):
            raise ValueError("SOURCE_RECEIPT_INTEGRITY_FAILURE")
        result["receipt"] = deepcopy(receipt)
        if set(receipt) != {"version", "identity", "documents", "listing_id", "rule_version",
                "effective_from", "effective_until", "verified_at", "verification_expires",
                "period", "overtime", "count", "reference_team", "comparison", "bridge_source",
                "listing_source", "effective_source", "review_receipt"}:
            raise ValueError("SOURCE_RECEIPT_SCHEMA_UNSUPPORTED")
        if receipt["version"] != VERSION or receipt["documents"] != DOCUMENTS:
            errors.append("SOURCE_DOCUMENT_VERSION_UNVERIFIED")
        if receipt["rule_version"] != RULES:
            errors.append("SOURCE_RULE_VERSION_UNSUPPORTED")
        for field in ("listing_id", "bridge_source", "listing_source", "effective_source", "review_receipt"):
            if not text(receipt[field]): errors.append("SOURCE_MISSING_" + field.upper())
        for field, value in offer.items():
            if receipt["identity"].get(field) != value: errors.append("SOURCE_CONFLICT_" + field.upper())
        if set(receipt["identity"]) != set(offer): errors.append("SOURCE_IDENTITY_SCHEMA_UNSUPPORTED")
        if any(v is None or v == "" for k,v in offer.items() if k != "provider_quote_id"): errors.append("SOURCE_OFFER_FACTS_MISSING")
        if offer["provider_quote_id"] is not None and not text(offer["provider_quote_id"]):
            errors.append("SOURCE_PROVIDER_QUOTE_ID_INVALID")
        line = offer["line"]
        if (offer["provider_namespace"] != "the_odds_api" or offer["sport"] != "americanfootball_nfl"
                or offer["bookmaker"] != "novig" or offer["market"] != "spreads"):
            errors.append("SOURCE_SCOPE_UNSUPPORTED")
        if line is None or abs(line*2-round(line*2)) > 1e-9 or abs(line-round(line)) <= 1e-9:
            errors.append("SOURCE_HALF_POINT_REQUIRED")
        if offer["home"] == offer["away"] or offer["selection"] not in {offer["home"], offer["away"]}:
            errors.append("SOURCE_NAMED_ORIENTATION_CONFLICT")
        # YES above the negative selected-side handicap; never use sorted keys.
        if receipt["reference_team"] != offer["selection"] or receipt["comparison"] != "above" or receipt["count"] != (-line if line is not None else None):
            errors.append("SOURCE_SELECTED_SIDE_CONFLICT")
        if receipt["period"] != "full_game" or receipt["overtime"] is not True:
            errors.append("SOURCE_PERIOD_CONFLICT")
        clocks = [clock(receipt[f]) for f in ("effective_from", "effective_until", "verified_at", "verification_expires")]
        source = clock(offer["source_time"])
        inferred = clock(inference_time) if inference_time is not None else clocks[2]
        if not all(clocks) or not source or not inferred:
            errors.append("SOURCE_EFFECTIVE_CLOCKS_UNKNOWN")
        elif not (clocks[0] <= source < clocks[1] and clocks[2] <= inferred < clocks[3] and (inference_time is None or source <= inferred < clocks[1])):
            errors.append("SOURCE_CONTRACT_STALE_OR_NOT_YET_EFFECTIVE")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(str(exc) if str(exc).startswith("SOURCE_") else "SOURCE_RECEIPT_SCHEMA_UNSUPPORTED")
    result["diagnostics"] = sorted(set(errors))
    result["status"] = "VERIFIED" if not errors else "REJECTED"
    return result


def adapt(game, book, market, outcome):
    """Retain prospective assessments; only the accepted NFL spread scope can verify."""
    from app_core.source_evidence_intake import for_offer
    offer = identity(game, book, market, outcome)
    intake = for_offer(offer, reference=outcome.get("source_evidence_ref", market.get("source_evidence_ref")),
                       quote_clock_field="market.last_update" if market.get("last_update") else "bookmaker.last_update")
    if intake is not None:
        facts = dict(source_contract=dict(intake, identity=offer))
        if intake["status"] == "VERIFIED":
            if market.get("period") not in (None, "", "full_game") or market.get("settlement_rules") not in (None, "", RULES):
                facts["source_contract"].update(status="REJECTED", diagnostics=["SOURCE_TRANSPORT_RULE_PERIOD_CONFLICT"])
            else:
                facts.update(period="full_game", period_source=intake["version"], rules=RULES, rules_source=intake["version"])
        return facts
    scope = (game.get("sport_key"), market.get("key"))
    if scope in UNVERIFIED_MARKETS and str(book.get("key", "")).startswith("novig"):
        offer = identity(game, book, market, outcome)
        assessment = unverified_assessment(scope, offer,
            outcome.get("source_contract_ref", market.get("source_contract_ref")))
        return dict(source_contract=dict(assessment, identity=offer))
    requested = "source_contract_ref" in market or "source_contract_ref" in outcome
    eligible = (game.get("sport_key") == "americanfootball_nfl" and str(book.get("key", "")).startswith("novig")
                and market.get("key") == "spreads")
    if not requested and not eligible:
        return {}
    offer = identity(game, book, market, outcome)
    contract = verify(outcome.get("source_contract_ref", market.get("source_contract_ref")), offer)
    facts = dict(source_contract=dict(contract, identity=offer))
    if contract["status"] == "VERIFIED":
        if market.get("period") not in (None, "", "full_game") or market.get("settlement_rules") not in (None, "", RULES):
            facts["source_contract"]["status"] = "REJECTED"
            facts["source_contract"]["diagnostics"] = ["SOURCE_TRANSPORT_RULE_PERIOD_CONFLICT"]
        else:
            facts.update(period="full_game", period_source=VERSION, rules=RULES, rules_source=VERSION)
    return facts


def replay(contract, inference_time):
    from app_core.source_evidence_intake import VERSION as intake_version, replay as replay_intake
    if contract.get("version") == intake_version:
        return replay_intake(contract, inference_time)
    unverified = next((scope for scope, values in UNVERIFIED_MARKETS.items()
                       if values[0] == contract.get("version")), None)
    if unverified is not None:
        expected = unverified_assessment(unverified, contract.get("identity", {}), contract.get("reference"))
    else:
        expected = verify(contract.get("reference"), contract.get("identity", {}), inference_time=inference_time)
    retained = {k:v for k,v in contract.items() if k != "identity"}
    if retained != expected:
        return sorted(set(["SOURCE_CAPTURE_BINDING_CONFLICT"] + expected["diagnostics"]))
    return expected["diagnostics"]
