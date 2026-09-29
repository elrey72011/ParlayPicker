"""Versioned, read-only validation for paid-launch staging evidence."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = 2
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
TRUSTED_GENERATORS = {
    "github_actions",
    "owner_authorized_staging",
    "deployment_platform",
}

# Freshness is evidence-class specific.  A short-lived load observation should
# not silently inherit the same lifetime as a recovery exercise or pilot.
EVIDENCE_POLICY: dict[str, dict[str, Any]] = {
    "billing_sandbox": {
        "max_age": timedelta(days=7),
        "execution_kinds": {"hosted_integration"},
        "scenarios": {
            "checkout",
            "webhook_signature",
            "repeated_event",
            "out_of_order_event",
            "billing_portal",
            "cancellation",
            "sales_disabled",
        },
    },
    "publication_recovery": {
        "max_age": timedelta(days=7),
        "execution_kinds": {"hosted_recovery"},
        "scenarios": {
            "publication_retry",
            "worker_restart",
            "withdrawal",
            "open_page_expiry",
            "rollback",
        },
    },
    "results_reconciliation": {
        "max_age": timedelta(days=7),
        "execution_kinds": {"hosted_integration"},
        "scenarios": {"original_result", "correction", "unknown_outcome"},
    },
    "alert_faults": {
        "max_age": timedelta(days=7),
        "execution_kinds": {"hosted_fault_test"},
        "scenarios": {
            "notification_suppression",
            "bounce_complaint",
            "unknown_outcome",
            "worker_restart",
        },
    },
    "load": {
        "max_age": timedelta(days=3),
        "execution_kinds": {"hosted_load_test"},
        "scenarios": {"subscriber_api_load", "mobile_lcp", "release_to_email"},
    },
    "backup_restore": {
        "max_age": timedelta(days=14),
        "execution_kinds": {"hosted_recovery"},
        "scenarios": {"isolated_restore", "provider_reconciliation", "rpo_rto"},
    },
    "pilot": {
        "max_age": timedelta(days=3),
        "execution_kinds": {"observed_pilot"},
        "scenarios": {"observed_interval", "incident_review"},
    },
}


def canonical_payload(payload: Mapping[str, Any]) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("payload_sha256", None)
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(payload)).hexdigest()


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_evidence(
    payload: object,
    *,
    expected_kind: str,
    expected_environment: str,
    expected_revision: str,
    evidence_root: Path,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate evidence structure and whether it can prove hosted execution.

    Structural validity and hosted sufficiency are separate.  A fixture can
    exercise this schema successfully but can never satisfy hosted sign-off.
    No network, database, billing, activation, or filesystem write is made.
    """

    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    reasons: list[str] = []
    structural: list[str] = []
    if not isinstance(payload, dict):
        return {
            "status": "FAIL",
            "structural_status": "FAIL",
            "hosted_status": "BLOCKED",
            "reason_codes": ["EVIDENCE_NOT_AN_OBJECT"],
        }

    if payload.get("schema_version") != SCHEMA_VERSION:
        structural.append("UNSUPPORTED_EVIDENCE_SCHEMA")
    if payload.get("evidence_kind") != expected_kind:
        structural.append("EVIDENCE_KIND_MISMATCH")
    if payload.get("status") != "PASS":
        reasons.append("EVIDENCE_STATUS_NOT_PASS")
    if payload.get("environment") != expected_environment:
        reasons.append("EVIDENCE_ENVIRONMENT_MISMATCH")
    if expected_environment not in {"staging", "production"}:
        reasons.append("HOSTED_ENVIRONMENT_REQUIRED")

    policy = EVIDENCE_POLICY.get(expected_kind)
    if policy is None:
        structural.append("UNSUPPORTED_EVIDENCE_KIND")
        policy = {"max_age": timedelta(0), "execution_kinds": set(), "scenarios": set()}

    release = payload.get("release_candidate")
    if not isinstance(release, dict):
        structural.append("RELEASE_CANDIDATE_MISSING")
        release = {}
    if not REVISION_RE.fullmatch(expected_revision or ""):
        reasons.append("EXPECTED_RELEASE_REVISION_MISSING")
    for field in ("source_revision", "deployed_revision", "served_revision"):
        revision = release.get(field)
        if not isinstance(revision, str) or not REVISION_RE.fullmatch(revision):
            structural.append(f"{field.upper()}_INVALID")
        elif expected_revision and revision != expected_revision:
            reasons.append(f"{field.upper()}_MISMATCH")

    execution = payload.get("execution")
    if not isinstance(execution, dict):
        structural.append("EXECUTION_MISSING")
        execution = {}
    execution_kind = execution.get("kind")
    if execution_kind not in policy["execution_kinds"]:
        structural.append("EXECUTION_KIND_UNSUPPORTED")
    started_at = _timestamp(execution.get("started_at"))
    completed_at = _timestamp(execution.get("completed_at"))
    if started_at is None or completed_at is None or completed_at < started_at:
        structural.append("EXECUTION_INTERVAL_INVALID")
    elif completed_at > checked_at + timedelta(minutes=5):
        reasons.append("EXECUTION_COMPLETED_IN_FUTURE")
    elif checked_at - completed_at > policy["max_age"]:
        reasons.append("EVIDENCE_EXPIRED")
    reference = execution.get("reference")
    if not isinstance(reference, dict):
        structural.append("EXECUTION_REFERENCE_MISSING")
    else:
        provider = reference.get("provider")
        identity = reference.get("run_id") or reference.get("execution_id")
        if provider not in TRUSTED_GENERATORS or not str(identity or "").strip():
            structural.append("EXECUTION_REFERENCE_UNTRUSTED")
    fixture = execution.get("fixture") is True
    synthetic = execution.get("synthetic") is True
    if fixture:
        reasons.append("FIXTURE_ONLY_EVIDENCE")
    if synthetic:
        reasons.append("SYNTHETIC_EXECUTION_NOT_HOSTED_PROOF")

    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        structural.append("PROVENANCE_MISSING")
        provenance = {}
    if provenance.get("generator") not in TRUSTED_GENERATORS:
        structural.append("UNTRUSTED_EVIDENCE_GENERATOR")
    if not str(provenance.get("reviewer") or "").strip():
        structural.append("EVIDENCE_REVIEWER_MISSING")
    review_reference = provenance.get("review_reference")
    if not isinstance(review_reference, dict) or not str(
        review_reference.get("id") or ""
    ).strip():
        structural.append("EVIDENCE_REVIEW_REFERENCE_MISSING")
    elif review_reference.get("provider") not in {
        "github",
        "deployment_change",
        "owner_approval",
    }:
        structural.append("EVIDENCE_REVIEW_REFERENCE_UNTRUSTED")
    reviewed_at = _timestamp(provenance.get("reviewed_at"))
    if reviewed_at is None:
        structural.append("REVIEW_TIMESTAMP_INVALID")
    elif completed_at is not None and reviewed_at < completed_at:
        structural.append("REVIEW_PRECEDES_EXECUTION")
    elif reviewed_at > checked_at + timedelta(minutes=5):
        reasons.append("REVIEW_TIMESTAMP_IN_FUTURE")

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        structural.append("SCENARIOS_MISSING")
        scenarios = []
    scenario_by_id = {
        item.get("id"): item
        for item in scenarios
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    missing_scenarios = sorted(policy["scenarios"] - set(scenario_by_id))
    if missing_scenarios:
        structural.append("REQUIRED_SCENARIOS_MISSING")
    for scenario_id in policy["scenarios"] & set(scenario_by_id):
        scenario = scenario_by_id[scenario_id]
        if scenario.get("executed") is not True or scenario.get("status") != "PASS":
            reasons.append("REQUIRED_SCENARIO_NOT_EXECUTED")
            break

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        structural.append("ARTIFACT_REFERENCES_MISSING")
        artifacts = []
    root = evidence_root.resolve()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            structural.append("ARTIFACT_REFERENCE_INVALID")
            continue
        relative = artifact.get("path")
        expected_hash = artifact.get("sha256")
        if not isinstance(relative, str) or not relative.strip():
            structural.append("ARTIFACT_PATH_INVALID")
            continue
        candidate = (root / relative).resolve()
        if not _is_under(candidate, root):
            structural.append("ARTIFACT_PATH_OUTSIDE_EVIDENCE_ROOT")
            continue
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            structural.append("ARTIFACT_HASH_INVALID")
        elif not candidate.is_file():
            reasons.append("REFERENCED_ARTIFACT_MISSING")
        elif hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_hash:
            reasons.append("REFERENCED_ARTIFACT_HASH_MISMATCH")

    supplied_digest = payload.get("payload_sha256")
    if not isinstance(supplied_digest, str) or not SHA256_RE.fullmatch(supplied_digest):
        structural.append("PAYLOAD_DIGEST_INVALID")
    elif supplied_digest != payload_digest(payload):
        reasons.append("PAYLOAD_DIGEST_MISMATCH")

    if expected_kind == "pilot":
        observation = payload.get("observation")
        if not isinstance(observation, dict):
            structural.append("PILOT_OBSERVATION_MISSING")
        else:
            interval_start = _timestamp(observation.get("interval_start"))
            interval_end = _timestamp(observation.get("interval_end"))
            days = observation.get("observed_consecutive_days")
            slates = observation.get("observed_active_slates")
            if (
                interval_start is None
                or interval_end is None
                or interval_end < interval_start
                or interval_end > checked_at
            ):
                reasons.append("PILOT_INTERVAL_NOT_OBSERVED")
            if not isinstance(days, int) or days < 14:
                reasons.append("PILOT_DURATION_INCOMPLETE")
            elif interval_start and interval_end and days > (interval_end - interval_start).days + 1:
                reasons.append("PILOT_DURATION_EXCEEDS_INTERVAL")
            if not isinstance(slates, int) or slates < 2:
                reasons.append("PILOT_SLATES_INCOMPLETE")

    structural = sorted(set(structural))
    reasons = sorted(set(reasons))
    structural_status = "PASS" if not structural else "FAIL"
    hosted_status = "PASS" if not structural and not reasons else "BLOCKED"
    return {
        "status": "PASS" if hosted_status == "PASS" else "FAIL",
        "structural_status": structural_status,
        "hosted_status": hosted_status,
        "schema_version": payload.get("schema_version"),
        "evidence_kind": payload.get("evidence_kind"),
        "max_age_seconds": int(policy["max_age"].total_seconds()),
        "reason_codes": sorted(set(structural + reasons)),
    }
