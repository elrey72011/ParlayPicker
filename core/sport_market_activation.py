"""Explicit owner authority for one reviewed sport/market deployment.

Validation evidence cannot call this function on its own: an owner must
confirm a policy and a configured bankroll/exposure ledger independently.
The returned record is hash-bound, expires, and never submits an order.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Mapping

from core.exposure_ledger import LIMITS, digest, verify_snapshot
from core.market_policy import SPORT_MARKET_FAMILIES
from core.sport_policy import SportPolicy
from core.wager_decisions import aware, finite


SCHEMA = "sport-market-owner-activation-v1"
VALIDATED = {"PROVISIONAL_VALIDATED", "STANDARD_VALIDATED", "PREMIUM_VALIDATED"}
BOUND = ("sport", "market_family", "validation_id", "artifact_id", "model_id",
         "model_version", "calibration_id", "calibration_version", "deployment_state")


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _scope(state: Mapping) -> bool:
    sport, family = state.get("sport"), state.get("market_family")
    return sport in SPORT_MARKET_FAMILIES and family in SPORT_MARKET_FAMILIES[sport]


def _valid_state(state: Mapping) -> bool:
    return (_scope(state) and state.get("deployment_state") in VALIDATED
            and state.get("validation_state") == state.get("deployment_state")
            and all(_text(state.get(key)) for key in BOUND if key != "deployment_state"))


def _policy_matches_frozen_plan(state: Mapping, policy: SportPolicy) -> bool:
    approved = state.get("validated_policy")
    if not isinstance(approved, Mapping):
        return False
    expected = dict(approved)
    # A frozen prospective plan predates its eventual validation ID. Only the
    # reviewed artifact may supply that ID after the holdout passes.
    if expected.pop("validation_id", None) not in (None, "", state.get("validation_id")):
        return False
    actual = asdict(policy)
    actual.pop("validation_id")
    # The frozen plan is JSON, so tuple-valued policy fields round-trip as
    # arrays. Compare their canonical JSON forms rather than Python containers.
    try:
        return digest(expected) == digest(actual)
    except (TypeError, ValueError):
        return False


def activate_market(state: Mapping, policy: SportPolicy, exposure: Mapping, *,
                    owner_id: str, expires_at: str, confirm: bool = False,
                    now: datetime | None = None) -> dict:
    """Build an exact owner-confirmed activation after independent review."""
    now = now or datetime.now(timezone.utc)
    if not confirm or not _text(owner_id):
        raise ValueError("Explicit owner confirmation and identity required")
    if not _valid_state(state):
        raise ValueError("Exact reviewed sport/market deployment required")
    if (policy.sport != state["sport"] or policy.validation_id != state["validation_id"]
            or policy.deployment_state != state["deployment_state"]
            or not _policy_matches_frozen_plan(state, policy)):
        raise ValueError("Policy does not match exact reviewed deployment")
    until = aware(expires_at)
    if until is None or until <= now:
        raise ValueError("Owner activation requires a future expiry")
    verified = verify_snapshot(dict(exposure), now=now)
    cap = {key: verified[key] for key in LIMITS}
    record = {"schema": SCHEMA, "owner_confirmed": True, "owner_id": owner_id,
              "activated_at": now.isoformat(), "expires_at": until.isoformat(),
              "policy": asdict(policy), "bankroll": verified["bankroll"],
              "unit_value": verified["unit_value"], "currency": verified["currency"],
              "exposure_limits": cap,
              "source_exposure_snapshot_hash": verified["snapshot_hash"],
              **{key: state[key] for key in BOUND}}
    record["activation_hash"] = digest(record)
    return record


def verify_market_activation(record: Mapping, state: Mapping, exposure: Mapping, *,
                             now: datetime | None = None, allow_test_only: bool = False) -> SportPolicy:
    """Verify exact immutable validation binding and current owner exposure."""
    now = now or datetime.now(timezone.utc)
    if not isinstance(record, Mapping) or record.get("schema") != SCHEMA or record.get("owner_confirmed") is not True:
        raise ValueError("Owner market activation missing")
    if digest({key: value for key, value in record.items() if key != "activation_hash"}) != record.get("activation_hash"):
        raise ValueError("Owner market activation hash mismatch")
    if not _valid_state(state) or any(record.get(key) != state.get(key) for key in BOUND):
        raise ValueError("Owner activation is for another sport/market deployment")
    activated, expires = aware(record.get("activated_at")), aware(record.get("expires_at"))
    if not _text(record.get("owner_id")) or activated is None or expires is None or not activated <= now < expires:
        raise ValueError("Owner market activation expired or invalid")
    if not allow_test_only and any(token in str(record.get(key, "")).upper() for key in
                                   ("validation_id", "artifact_id", "model_id", "calibration_id")
                                   for token in ("TEST", "SYNTHETIC")):
        raise ValueError("Test market activation prohibited in production")
    try:
        policy = SportPolicy(**record["policy"])
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("Invalid market policy") from exc
    if (policy.sport != state["sport"] or policy.validation_id != state["validation_id"]
            or policy.deployment_state != state["deployment_state"]
            or not _policy_matches_frozen_plan(state, policy)):
        raise ValueError("Owner policy differs from reviewed deployment")
    verified = verify_snapshot(dict(exposure), now=now)
    if (verified["currency"] != record.get("currency")
            or verified["bankroll"] != finite(record.get("bankroll"))
            or verified["unit_value"] != finite(record.get("unit_value"))):
        raise ValueError("Owner bankroll authority changed")
    limits = record.get("exposure_limits")
    if not isinstance(limits, Mapping) or any(
            finite(limits.get(key)) is None or verified[key] > finite(limits[key]) for key in LIMITS):
        raise ValueError("Current exposure limit exceeds owner activation")
    return policy
