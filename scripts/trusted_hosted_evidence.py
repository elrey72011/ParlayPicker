"""Authenticated out-of-band resolver for paid-launch hosted evidence.

The evidence documents under review cannot attest to themselves.  This module
loads a separately configured, HMAC-authenticated registry maintained by the
authorized execution surface.  It is read-only and never dispatches a job,
uploads an artifact, enables billing, or creates an approval.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Any, Callable

try:
    from scripts.paid_launch_evidence import IndependentVerification
except ModuleNotFoundError:  # Direct ``python scripts/verify_paid_launch.py``.
    from paid_launch_evidence import IndependentVerification


SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_RECORD_KEYS = frozenset({
    "attestation_id", "evidence_kind", "environment", "source_revision",
    "reference_provider", "execution_id", "artifact_sha256", "scenario_ids",
    "verified_at",
})


def _canonical(payload: dict[str, Any]) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("signature_hmac_sha256", None)
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sign_registry(payload: dict[str, Any], secret: str) -> str:
    """Return the detached registry signature (used by authorized tooling/tests)."""
    return hmac.new(secret.encode("utf-8"), _canonical(payload), hashlib.sha256).hexdigest()


def _time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def configured_resolver(
    *,
    environment: str,
    source_revision: str,
    evidence_root: Path,
) -> tuple[Callable[[str, dict[str, Any]], IndependentVerification | None], dict[str, Any]]:
    """Build a resolver from secure process configuration, or fail closed."""
    registry_value = os.environ.get("PAID_TRUSTED_ATTESTATION_REGISTRY", "").strip()
    secret = os.environ.get("PAID_TRUSTED_ATTESTATION_HMAC_SECRET", "")

    def unavailable(_kind: str, _payload: dict[str, Any]) -> None:
        return None

    if not registry_value or len(secret) < 32:
        return unavailable, {
            "status": "BLOCKED",
            "reason_codes": ["TRUSTED_ATTESTATION_CONFIGURATION_MISSING"],
        }
    registry_path = Path(registry_value).expanduser().resolve()
    evidence_path = evidence_root.resolve()
    if _under(registry_path, evidence_path):
        return unavailable, {
            "status": "BLOCKED",
            "reason_codes": ["ATTESTATION_REGISTRY_NOT_OUT_OF_BAND"],
        }
    try:
        raw = registry_path.read_text(encoding="utf-8")
        registry = json.loads(raw)
    except (OSError, ValueError, TypeError):
        return unavailable, {
            "status": "BLOCKED",
            "reason_codes": ["ATTESTATION_REGISTRY_UNAVAILABLE"],
        }
    if not isinstance(registry, dict) or registry.get("schema_version") != SCHEMA_VERSION:
        return unavailable, {
            "status": "BLOCKED",
            "reason_codes": ["ATTESTATION_REGISTRY_SCHEMA_INVALID"],
        }
    supplied = registry.get("signature_hmac_sha256")
    expected = sign_registry(registry, secret)
    if not isinstance(supplied, str) or not hmac.compare_digest(supplied, expected):
        return unavailable, {
            "status": "BLOCKED",
            "reason_codes": ["ATTESTATION_REGISTRY_SIGNATURE_INVALID"],
        }
    records = registry.get("attestations")
    if not isinstance(records, list):
        return unavailable, {
            "status": "BLOCKED",
            "reason_codes": ["ATTESTATION_RECORDS_INVALID"],
        }
    index: dict[tuple[str, str, str, str, str], IndependentVerification] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != REQUIRED_RECORD_KEYS:
            return unavailable, {
                "status": "BLOCKED",
                "reason_codes": ["ATTESTATION_RECORD_INVALID"],
            }
        verified = _time(record.get("verified_at"))
        hashes, scenarios = record.get("artifact_sha256"), record.get("scenario_ids")
        if (verified is None
                or not REVISION_RE.fullmatch(str(record.get("source_revision") or ""))
                or not isinstance(hashes, list) or not hashes
                or any(not isinstance(value, str) or not SHA256_RE.fullmatch(value) for value in hashes)
                or len(set(hashes)) != len(hashes)
                or not isinstance(scenarios, list) or not scenarios
                or any(not isinstance(value, str) or not value.strip() for value in scenarios)
                or len(set(scenarios)) != len(scenarios)
                or not all(isinstance(record.get(name), str) and record[name].strip()
                           for name in ("attestation_id", "evidence_kind", "environment",
                                        "reference_provider", "execution_id"))):
            return unavailable, {
                "status": "BLOCKED",
                "reason_codes": ["ATTESTATION_RECORD_INVALID"],
            }
        key = (
            record["evidence_kind"], record["environment"],
            record["source_revision"], record["reference_provider"],
            record["execution_id"],
        )
        if key in index:
            return unavailable, {
                "status": "BLOCKED",
                "reason_codes": ["ATTESTATION_IDENTITY_CONFLICT"],
            }
        index[key] = IndependentVerification(
            evidence_kind=record["evidence_kind"],
            environment=record["environment"],
            source_revision=record["source_revision"],
            reference_provider=record["reference_provider"],
            execution_id=record["execution_id"],
            artifact_sha256=frozenset(hashes),
            scenario_ids=frozenset(scenarios),
            verified_at=verified,
            attestation_id=record["attestation_id"],
        )

    def resolve(kind: str, payload: dict[str, Any]) -> IndependentVerification | None:
        execution = payload.get("execution") if isinstance(payload, dict) else None
        reference = execution.get("reference") if isinstance(execution, dict) else None
        if not isinstance(reference, dict):
            return None
        provider = reference.get("provider")
        execution_id = reference.get("run_id") or reference.get("execution_id")
        key = (kind, environment, source_revision, str(provider or ""), str(execution_id or ""))
        return index.get(key)

    return resolve, {
        "status": "READY",
        "registry_id": registry.get("registry_id"),
        "record_count": len(index),
        "reason_codes": [],
    }
