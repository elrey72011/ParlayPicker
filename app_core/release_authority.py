"""Read-only current-authority adapter for public release boundaries.

The saved wager/trial contract remains the immutable decision snapshot.  This
module only re-reads the existing owner authority and exposure ledgers at the
moment a publisher is asked to act.  It never fetches a new quote, changes a
price, refreshes an analysis timestamp, creates consent, or reserves exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import sqlite3
from pathlib import Path
from typing import Callable, Mapping

from core.exposure_ledger import LIMITS, digest, snapshot, verify_snapshot
from core.market_policy import sport_market_family
from core.sport_market_activation import BOUND, SCHEMA, verify_market_activation
from core.wager_decisions import aware, finite


Setting = Callable[[str], object]


@dataclass(frozen=True)
class AuthorityResolution:
    """Immutable facts read from authority sources for one publication call."""

    authorities: Mapping[str, Mapping]
    required: bool
    resolved_at: str
    source_status: str
    provider_status: str
    reason_codes: tuple[str, ...]


def _setting(setting: Setting | None, name: str, default: str) -> str:
    if setting is None:
        import os
        value = os.environ.get(name, default)
    else:
        value = setting(name)
        if value is None or not str(value).strip():
            value = default
    return str(value).strip()


def _clock(value: datetime | None) -> datetime:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ValueError("Current-authority time must include a timezone")
    return current.astimezone(timezone.utc)


def _binding(contract: Mapping) -> dict:
    return {
        "sport": contract.get("sport"),
        "market": contract.get("market_type"),
        "selection": contract.get("selection"),
        "line": contract.get("line"),
        "odds": contract.get("odds"),
        "sportsbook": contract.get("sportsbook"),
        "quote_timestamp": contract.get("quote_timestamp"),
        "event_start": contract.get("start"),
    }


def _deadlines(row: Mapping, minutes: int) -> dict:
    quote = aware(row.get("quote_time"))
    review = aware(row.get("as_of"))
    return {
        "provider_expires_at": (
            quote + timedelta(minutes=minutes)
        ).isoformat() if quote else None,
        "review_expires_at": (
            review + timedelta(minutes=minutes)
        ).isoformat() if review else None,
    }


def _invalid_authority(reason: str, contract: Mapping, deadlines: Mapping) -> dict:
    return {
        "status": "UNAVAILABLE",
        "revoked": False,
        "authority_reason": reason,
        "binding": _binding(contract),
        **deadlines,
    }


def _valid_exposure(path: Path, now: datetime) -> Mapping | None:
    try:
        return verify_snapshot(snapshot(path, now=now), now=now)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error):
        return None


def _strict_authority(
    contract: Mapping,
    row: Mapping,
    *,
    now: datetime,
    minutes: int,
    activation_dir: Path,
    exposure_path: Path,
) -> tuple[dict, str | None]:
    deadlines = _deadlines(row, minutes)
    family = sport_market_family(contract.get("sport"), contract.get("market_type"))
    if family is None or contract.get("market_family") != family:
        reason = "CURRENT_MARKET_SCOPE_INVALID"
        return _invalid_authority(reason, contract, deadlines), reason
    target = activation_dir / f"{contract.get('sport')}-{family}.json"
    try:
        activation = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        reason = "CURRENT_MARKET_ACTIVATION_UNAVAILABLE"
        return _invalid_authority(reason, contract, deadlines), reason
    if not isinstance(activation, dict):
        reason = "CURRENT_MARKET_ACTIVATION_INVALID"
        return _invalid_authority(reason, contract, deadlines), reason
    unsigned = {key: value for key, value in activation.items() if key != "activation_hash"}
    if (activation.get("schema") != SCHEMA
            or activation.get("owner_confirmed") is not True
            or digest(unsigned) != activation.get("activation_hash")):
        reason = "CURRENT_MARKET_ACTIVATION_INVALID"
        return _invalid_authority(reason, contract, deadlines), reason
    exposure = _valid_exposure(exposure_path, now)
    if exposure is None:
        reason = "CURRENT_EXPOSURE_UNAVAILABLE"
        return _invalid_authority(reason, contract, deadlines), reason
    activated, expires = aware(activation.get("activated_at")), aware(activation.get("expires_at"))
    if activated is None or activated > now or expires is None:
        reason = "CURRENT_MARKET_ACTIVATION_INVALID"
        return _invalid_authority(reason, contract, deadlines), reason
    if expires <= now:
        reason = "CURRENT_MARKET_ACTIVATION_EXPIRED"
        return {
            "status": "EXPIRED",
            "revoked": False,
            "effective_at": activated.isoformat(),
            "expires_at": expires.isoformat(),
            "authority_id": activation.get("activation_hash"),
            "authority_reason": reason,
            "binding": _binding(contract),
            **deadlines,
        }, reason
    policy = activation.get("policy")
    identifiers = {
        "sport": contract.get("sport"),
        "market_family": contract.get("market_family"),
        "validation_id": contract.get("validation_id"),
        "artifact_id": contract.get("validation_artifact_id"),
        "model_id": contract.get("model_id"),
        "model_version": contract.get("model_version"),
        "calibration_id": contract.get("calibration_id"),
        "calibration_version": contract.get("calibration_version"),
        "deployment_state": contract.get("deployment_state"),
    }
    if (not isinstance(policy, dict)
            or policy.get("version") != contract.get("sport_policy_version")
            or any(activation.get(key) != identifiers[key] for key in BOUND)):
        reason = "CURRENT_MARKET_ACTIVATION_BINDING_MISMATCH"
        return _invalid_authority(reason, contract, deadlines), reason
    limits = activation.get("exposure_limits")
    exposure_ok = (
        exposure.get("currency") == activation.get("currency")
        and finite(exposure.get("bankroll")) == finite(activation.get("bankroll"))
        and finite(exposure.get("unit_value")) == finite(activation.get("unit_value"))
        and isinstance(limits, Mapping)
        and all(finite(limits.get(key)) is not None
                and finite(exposure.get(key)) is not None
                and finite(exposure[key]) <= finite(limits[key]) for key in LIMITS)
    )
    if not exposure_ok:
        reason = "CURRENT_EXPOSURE_BINDING_MISMATCH"
        return _invalid_authority(reason, contract, deadlines), reason
    # Reuse the production gate's canonical verifier as the final authority
    # boundary.  Besides the checks above (which retain precise diagnostics),
    # it also enforces the frozen policy contract and prohibits synthetic/test
    # model identities on a real publication route.
    state = {
        "sport": identifiers["sport"],
        "market_family": identifiers["market_family"],
        "validation_state": identifiers["deployment_state"],
        "validated_policy": activation.get("policy"),
        **{key: identifiers[key] for key in BOUND},
    }
    try:
        verify_market_activation(activation, state, exposure, now=now)
    except (ValueError, TypeError, KeyError):
        reason = "CURRENT_MARKET_AUTHORITY_VERIFICATION_FAILED"
        return _invalid_authority(reason, contract, deadlines), reason
    return {
        "status": "AUTHORIZED",
        "allocation_policy": {
            "sport_cap": policy["sport_exposure_cap"],
            "configuration": {key: exposure[key] for key in
                              ("bankroll", "unit_value", "currency", *LIMITS)},
        },
        "revoked": False,
        "effective_at": activated.isoformat(),
        "expires_at": expires.isoformat(),
        "authority_id": activation.get("activation_hash"),
        "authority_reason": None,
        "binding": _binding(contract),
        **deadlines,
    }, None


def _trial_authority(
    contract: Mapping,
    row: Mapping,
    *,
    now: datetime,
    minutes: int,
    setting: Setting | None,
    exposure_path: Path,
) -> tuple[dict, str | None]:
    from app_core.trial_authority import consent_status, exposure_status

    deadlines = _deadlines(row, minutes)
    consent_path = _setting(
        setting,
        "PARLAYPICKER_CONTROLLED_TRIAL_CONSENT_LEDGER",
        "data/exposure/controlled-trial-consent.sqlite3",
    )
    consent, consent_reason = consent_status(consent_path, now=now)
    if consent is None:
        reason = "CURRENT_TRIAL_" + consent_reason
        return _invalid_authority(reason, contract, deadlines), reason
    amount = finite(contract.get("recommended_bet_amount"))
    fraction = finite(contract.get("bankroll_fraction"))
    bankroll = amount / fraction if amount and fraction and fraction > 0 else None
    if bankroll is None or not math.isfinite(bankroll):
        reason = "CURRENT_TRIAL_BANKROLL_BINDING_MISSING"
        return _invalid_authority(reason, contract, deadlines), reason
    exposure, _, exposure_reason = exposure_status(bankroll, now=now, path=exposure_path)
    if exposure is None:
        reason = "CURRENT_TRIAL_" + exposure_reason
        return _invalid_authority(reason, contract, deadlines), reason
    return {
        "status": "AUTHORIZED",
        "revoked": False,
        "allocation_policy": {
            "configuration": {key: exposure[key] for key in
                              ("bankroll", "unit_value", "currency", *LIMITS)},
        },
        "effective_at": consent.get("recorded_at"),
        "expires_at": consent.get("expires_at"),
        "authority_id": consent.get("event_id"),
        "authority_reason": None,
        "binding": _binding(contract),
        **deadlines,
    }, None


def resolve_publication_authority(
    package: Mapping,
    *,
    at: datetime | None = None,
    setting: Setting | None = None,
) -> AuthorityResolution:
    """Resolve every modern actionable row against existing current ledgers.

    A valid frozen quote is retained until its existing expiry.  This adapter
    does not make a provider call; the provider status says so explicitly.
    """
    from app_core.quote_freshness import package_age_minutes
    from app_core.release_preflight import ACTIONABLE, row_identity

    now = _clock(at)
    minutes = package_age_minutes(package)
    activation_dir = Path(_setting(
        setting, "PARLAYPICKER_MARKET_ACTIVATIONS_DIR", "data/policies/active_markets"
    ))
    exposure_path = Path(_setting(
        setting, "PARLAYPICKER_EXPOSURE_LEDGER", "data/exposure/exposure.sqlite3"
    ))
    authorities: dict[str, Mapping] = {}
    reasons: list[str] = []
    actionable = 0
    seen: set[str] = set()
    for values in package.get("games", {}).values():
        for row in values:
            if not isinstance(row, Mapping) or row.get("status") not in ACTIONABLE:
                continue
            row_id = row_identity(row)
            if row_id in seen:
                continue
            seen.add(row_id)
            actionable += 1
            contract = (row.get("wager_contract") if row.get("status") == "APPROVED"
                        else row.get("controlled_trial_contract"))
            if not isinstance(contract, Mapping):
                reasons.append("CURRENT_AUTHORITY_CONTRACT_MISSING")
                continue
            if row.get("status") == "APPROVED":
                current, reason = _strict_authority(
                    contract, row, now=now, minutes=minutes,
                    activation_dir=activation_dir, exposure_path=exposure_path,
                )
            else:
                current, reason = _trial_authority(
                    contract, row, now=now, minutes=minutes,
                    setting=setting, exposure_path=exposure_path,
                )
            authorities[row_id] = current
            if reason:
                reasons.append(reason)
    if not actionable:
        source_status = "NOT_APPLICABLE"
    elif not reasons and len(authorities) == actionable:
        source_status = "VERIFIED"
    else:
        source_status = "BLOCKED"
    return AuthorityResolution(
        authorities=authorities,
        required=bool(actionable),
        resolved_at=now.isoformat(),
        source_status=source_status,
        provider_status=(
            "FROZEN_QUOTE_POLICY" if actionable else "NOT_APPLICABLE"
        ),
        reason_codes=tuple(sorted(set(reasons))),
    )


def authorize_publication(
    package: dict,
    *,
    at: datetime | None = None,
    setting: Setting | None = None,
    resolution: AuthorityResolution | None = None,
) -> dict:
    """Run the public release gate with a freshly resolved authority snapshot."""
    from app_core.release_preflight import ReleasePreflightError, require_actionable_release

    current = resolution or resolve_publication_authority(package, at=at, setting=setting)
    try:
        report = require_actionable_release(
            package,
            at=at,
            current_authority=current.authorities,
            require_current_authority=current.required,
        )
    except ReleasePreflightError as exc:
        exc.report["authority_resolution"] = {
            "required": current.required,
            "resolved_at": current.resolved_at,
            "source_status": current.source_status,
            "provider_status": current.provider_status,
            "reason_codes": list(current.reason_codes),
        }
        raise ReleasePreflightError(exc.report) from None
    report["authority_resolution"] = {
        "required": current.required,
        "resolved_at": current.resolved_at,
        "source_status": current.source_status,
        "provider_status": current.provider_status,
        "reason_codes": list(current.reason_codes),
    }
    from app_core.release_capacity import check_capacity
    if report["preflight_enforced"]:
        capacity = check_capacity(
            package, current.authorities, now=_clock(at),
            exposure_path=Path(_setting(setting, "PARLAYPICKER_EXPOSURE_LEDGER", "data/exposure/exposure.sqlite3")),
            evidence_path=Path(_setting(setting, "PARLAYPICKER_EVIDENCE_DIR", "data/prediction_evidence")) / "evidence.sqlite3",
            reservation_path=Path(_setting(setting, "PARLAYPICKER_CONTROLLED_TRIAL_RESERVATION_LEDGER",
                                           "data/exposure/controlled-trial-reservations.sqlite3")),
        )
    else:
        capacity = {"status": "NOT_APPLICABLE_LEGACY_HISTORY", "reason_codes": [], "tickets": []}
    report["capacity_check"] = capacity
    if capacity["status"] == "BLOCKED":
        report["actionable_release_allowed"] = False
        for code in capacity["reason_codes"]:
            report["blocker_counts"][code] = sum(code in ticket["reason_codes"] for ticket in capacity["tickets"]) or 1
        tickets = {ticket["row_id"]: ticket for ticket in capacity["tickets"]}
        for row in report["rows"]:
            reasons = tickets.get(row["row_id"], {}).get("reason_codes", [])
            if reasons:
                row["current_blockers"] = list(dict.fromkeys(row["current_blockers"] + reasons))
                row["release_blockers"] = list(dict.fromkeys(row["release_blockers"] + reasons))
                row["current_status"] = "BLOCKED_ACTIONABLE"
                row["current_primary_reason"] = row["release_blockers"][0]
        raise ReleasePreflightError(report)
    return report
