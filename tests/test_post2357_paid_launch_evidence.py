from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import partial
import hashlib
import json
from pathlib import Path

import pytest

from scripts.paid_launch_evidence import (
    EVIDENCE_POLICY,
    IndependentVerification,
    payload_digest,
    validate_evidence,
)
from scripts import verify_paid_launch


NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
REVISION = "a" * 40


def valid_payload(root: Path, kind: str = "billing_sandbox") -> dict:
    artifact = root / f"{kind}.txt"
    artifact.write_text("observed hosted evidence\n", encoding="utf-8")
    payload = {
        "schema_version": 2,
        "evidence_kind": kind,
        "status": "PASS",
        "environment": "staging",
        "release_candidate": {
            "source_revision": REVISION,
            "deployed_revision": REVISION,
            "served_revision": REVISION,
        },
        "execution": {
            "kind": next(iter(EVIDENCE_POLICY[kind]["execution_kinds"])),
            "started_at": (NOW - timedelta(hours=2)).isoformat(),
            "completed_at": (NOW - timedelta(hours=1)).isoformat(),
            "fixture": False,
            "synthetic": False,
            "reference": {"provider": "github_actions", "run_id": "12345"},
        },
        "provenance": {
            "generator": "github_actions",
            "reviewer": "release-owner@example.invalid",
            "reviewed_at": (NOW - timedelta(minutes=30)).isoformat(),
            "review_reference": {"provider": "github", "id": "review-12345"},
        },
        "scenarios": [
            {"id": item, "executed": True, "status": "PASS"}
            for item in sorted(EVIDENCE_POLICY[kind]["scenarios"])
        ],
        "artifacts": [
            {
                "path": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }
        ],
    }
    if kind == "pilot":
        payload["observation"] = {
            "interval_start": (NOW - timedelta(days=14)).isoformat(),
            "interval_end": (NOW - timedelta(hours=1)).isoformat(),
            "observed_consecutive_days": 14,
            "observed_active_slates": 2,
        }
    payload["payload_sha256"] = payload_digest(payload)
    return payload


def independent(payload: dict, kind: str = "billing_sandbox") -> IndependentVerification:
    reference = payload["execution"]["reference"]
    return IndependentVerification(
        evidence_kind=kind,
        environment=payload["environment"],
        source_revision=payload["release_candidate"]["source_revision"],
        reference_provider=reference["provider"],
        execution_id=str(reference.get("run_id") or reference.get("execution_id")),
        artifact_sha256=frozenset(item["sha256"] for item in payload["artifacts"]),
        scenario_ids=frozenset(item["id"] for item in payload["scenarios"]),
        verified_at=NOW - timedelta(minutes=15),
        attestation_id="trusted-resolver:12345",
    )


def checked(payload: object, root: Path, kind: str = "billing_sandbox",
            proof: IndependentVerification | None = None) -> dict:
    return validate_evidence(
        payload,
        expected_kind=kind,
        expected_environment="staging",
        expected_revision=REVISION,
        evidence_root=root,
        now=NOW,
        independent_verification=proof,
    )


def test_v01_status_only_and_unsupported_schema_are_rejected(tmp_path):
    status_only = checked({"status": "PASS"}, tmp_path)
    unsupported = valid_payload(tmp_path)
    unsupported["schema_version"] = 1
    unsupported["payload_sha256"] = payload_digest(unsupported)

    assert status_only["structural_status"] == "FAIL"
    assert "UNSUPPORTED_EVIDENCE_SCHEMA" in status_only["reason_codes"]
    assert checked(unsupported, tmp_path)["hosted_status"] == "BLOCKED"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda item: item.update(environment="production"), "EVIDENCE_ENVIRONMENT_MISMATCH"),
        (lambda item: item["release_candidate"].update(served_revision="b" * 40), "SERVED_REVISION_MISMATCH"),
        (
            lambda item: item["execution"].update(
                started_at=(NOW - timedelta(days=9)).isoformat(),
                completed_at=(NOW - timedelta(days=8)).isoformat(),
            ),
            "EVIDENCE_EXPIRED",
        ),
    ],
)
def test_v02_wrong_environment_build_and_expiry_fail(tmp_path, mutation, reason):
    payload = valid_payload(tmp_path)
    mutation(payload)
    payload["payload_sha256"] = payload_digest(payload)
    result = checked(payload, tmp_path)

    assert result["hosted_status"] == "BLOCKED"
    assert reason in result["reason_codes"]


def test_v03_fixture_and_skipped_scenario_cannot_count_as_hosted(tmp_path):
    fixture = valid_payload(tmp_path)
    fixture["execution"]["fixture"] = True
    fixture["payload_sha256"] = payload_digest(fixture)
    skipped = valid_payload(tmp_path)
    skipped["scenarios"][0].update(executed=False, status="SKIPPED")
    skipped["payload_sha256"] = payload_digest(skipped)

    fixture_result = checked(fixture, tmp_path)
    assert fixture_result["structural_status"] == "PASS"
    assert fixture_result["hosted_status"] == "BLOCKED"
    assert "FIXTURE_ONLY_EVIDENCE" in fixture_result["reason_codes"]
    assert "REQUIRED_SCENARIO_NOT_EXECUTED" in checked(skipped, tmp_path)["reason_codes"]


def test_v04_future_pilot_interval_and_synthetic_load_are_blocked(tmp_path):
    pilot = valid_payload(tmp_path, "pilot")
    pilot["observation"]["interval_end"] = (NOW + timedelta(days=7)).isoformat()
    pilot["payload_sha256"] = payload_digest(pilot)
    load = valid_payload(tmp_path, "load")
    load["execution"]["synthetic"] = True
    load["payload_sha256"] = payload_digest(load)

    assert "PILOT_INTERVAL_NOT_OBSERVED" in checked(pilot, tmp_path, "pilot")["reason_codes"]
    assert "SYNTHETIC_EXECUTION_NOT_HOSTED_PROOF" in checked(load, tmp_path, "load")["reason_codes"]


def test_v05_known_valid_fixture_is_structurally_valid_without_hosted_claim(tmp_path):
    payload = valid_payload(tmp_path)
    payload["execution"]["fixture"] = True
    payload["payload_sha256"] = payload_digest(payload)

    result = checked(payload, tmp_path)

    assert result["structural_status"] == "PASS"
    assert result["hosted_status"] == "BLOCKED"


def test_v02_tampered_payload_and_referenced_content_fail(tmp_path):
    payload = valid_payload(tmp_path)
    payload["provenance"]["reviewer"] = "tampered@example.invalid"
    assert "PAYLOAD_DIGEST_MISMATCH" in checked(payload, tmp_path)["reason_codes"]

    payload = valid_payload(tmp_path)
    (tmp_path / "billing_sandbox.txt").write_text("tampered\n", encoding="utf-8")
    payload["payload_sha256"] = payload_digest(payload)
    assert "REFERENCED_ARTIFACT_HASH_MISMATCH" in checked(payload, tmp_path)["reason_codes"]


def test_v05_valid_document_is_not_independent_hosted_proof(tmp_path):
    result = checked(valid_payload(tmp_path), tmp_path)

    assert result["structural_status"] == "PASS"
    assert result["document_status"] == "PASS"
    assert result["execution_provenance_status"] == "EXECUTION_PROVENANCE_UNVERIFIED"
    assert result["hosted_scenarios_status"] == "HOSTED_SCENARIOS_UNVERIFIED"
    assert result["hosted_status"] == "BLOCKED"
    assert "EXECUTION_PROVENANCE_NOT_INDEPENDENTLY_VERIFIED" in result["reason_codes"]


def test_v05_independently_resolved_execution_and_artifacts_can_pass(tmp_path):
    payload = valid_payload(tmp_path)
    result = checked(payload, tmp_path, proof=independent(payload))

    assert result["evidence_structure_status"] == "EVIDENCE_STRUCTURE_VALID"
    assert result["execution_provenance_status"] == "EXECUTION_PROVENANCE_VERIFIED"
    assert result["hosted_scenarios_status"] == "HOSTED_SCENARIOS_VERIFIED"
    assert result["hosted_status"] == "PASS"
    assert result["reason_codes"] == []


def test_v05_independent_proof_must_match_the_declared_execution(tmp_path):
    payload = valid_payload(tmp_path)
    proof = independent(payload)
    mismatched = IndependentVerification(
        **{**proof.__dict__, "execution_id": "different-run"})

    result = checked(payload, tmp_path, proof=mismatched)

    assert result["document_status"] == "PASS"
    assert result["execution_provenance_status"] == "EXECUTION_PROVENANCE_MISMATCH"
    assert result["hosted_status"] == "BLOCKED"
    assert "INDEPENDENT_EXECUTION_ID_MISMATCH" in result["reason_codes"]


class _StatusResponse:
    status_code = 200
    headers = {"content-type": "application/json"}

    @staticmethod
    def json():
        return {"schema_version": 1, "source_revision": REVISION}


def _configure_verifier(monkeypatch, root: Path) -> Path:
    evidence_root = root / "docs" / "paid-launch" / "evidence"
    evidence_root.mkdir(parents=True)
    monkeypatch.setattr(verify_paid_launch, "ROOT", root)
    monkeypatch.setattr(
        verify_paid_launch, "validate_evidence",
        partial(verify_paid_launch.validate_evidence, now=NOW),
    )
    monkeypatch.setattr(
        verify_paid_launch, "check_scope", lambda: {"status": "PASS", "reason_codes": []}
    )
    monkeypatch.setattr(verify_paid_launch.httpx, "get", lambda *args, **kwargs: _StatusResponse())
    for name in verify_paid_launch.REQUIRED_CONFIGURATION:
        monkeypatch.setenv(name, "configured-test-value")
    monkeypatch.setenv("PAID_PUBLIC_BASE_URL", "https://staging.example.invalid")
    monkeypatch.setenv("PAID_SOURCE_REVISION", REVISION)
    monkeypatch.setenv("PAID_LIVE_BILLING_ENABLED", "false")
    return evidence_root


def _write_required_evidence(root: Path, *, status_only: bool) -> None:
    names = {
        "billing_sandbox": "billing-sandbox-evidence.json",
        "publication_recovery": "publication-recovery-evidence.json",
        "results_reconciliation": "results-reconciliation-evidence.json",
        "alert_faults": "alert-fault-evidence.json",
        "load": "performance-load-evidence.json",
        "backup_restore": "backup-restore-evidence.json",
        "pilot": "pilot-ledger.json",
    }
    for kind, filename in names.items():
        payload = {"status": "PASS"} if status_only else valid_payload(root, kind)
        (root / filename).write_text(json.dumps(payload), encoding="utf-8")


def test_v01_v06_real_verifier_rejects_status_only_and_is_read_only(tmp_path, monkeypatch):
    evidence_root = _configure_verifier(monkeypatch, tmp_path)
    _write_required_evidence(evidence_root, status_only=True)
    before = {path: path.read_bytes() for path in evidence_root.iterdir()}

    code, report = verify_paid_launch.verify("staging")

    after = {path: path.read_bytes() for path in evidence_root.iterdir()}
    assert code == 2
    assert report["status"] == "BLOCKED"
    assert report["read_only"] is True
    assert before == after
    assert all(
        item["structural_status"] == "FAIL"
        for item in report["checks"]["evidence"].values()
    )


def test_v05_real_verifier_blocks_complete_shapes_without_independent_proof(tmp_path, monkeypatch):
    evidence_root = _configure_verifier(monkeypatch, tmp_path)
    _write_required_evidence(evidence_root, status_only=False)

    code, report = verify_paid_launch.verify("staging")

    assert code == 2
    assert report["status"] == "BLOCKED"
    assert all(
        item["document_status"] == "PASS" and item["hosted_status"] == "BLOCKED"
        for item in report["checks"]["evidence"].values()
    )


def test_v05_real_verifier_accepts_trusted_resolver_results(tmp_path, monkeypatch):
    evidence_root = _configure_verifier(monkeypatch, tmp_path)
    _write_required_evidence(evidence_root, status_only=False)

    code, report = verify_paid_launch.verify(
        "staging", independent_resolver=lambda kind, payload: independent(payload, kind))

    assert report["blockers"] == []
    assert code == 0
    assert report["status"] == "PASS"
    assert report["completion_label"] == "STAGING_VERIFIED_LIVE_BLOCKED"
    assert all(
        item["execution_provenance_status"] == "EXECUTION_PROVENANCE_VERIFIED"
        and item["hosted_scenarios_status"] == "HOSTED_SCENARIOS_VERIFIED"
        for item in report["checks"]["evidence"].values()
    )
