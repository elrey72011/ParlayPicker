"""PR C fault tests. Synthetic authority proves plumbing, never qualification."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json

import pytest

from app_core.release_authority import (
    AuthorityResolution,
    resolve_publication_authority,
)
from app_core.release_preflight import ReleasePreflightError, evaluate_release, row_identity
from core.exposure_ledger import append, snapshot
from core.sport_market_activation import activate_market
from core.sport_policy import SportPolicy
from scripts.trusted_hosted_evidence import sign_registry
from scripts import verify_paid_launch
from test_current_wagers_trace_and_release import (
    NOW,
    _approved_source,
    _package,
)
from test_live_wager_contract import leg
from test_post2357_paid_launch_evidence import (
    _configure_verifier,
    _write_required_evidence,
)


class _Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        value = NOW + timedelta(minutes=1)
        return value.astimezone(tz or timezone.utc)


def _current_source(tmp_path, monkeypatch):
    exposure_path = tmp_path / "exposure.sqlite3"
    append(
        exposure_path,
        {
            "status": "CONFIGURED", "bankroll": 1000.0, "unit_value": 1.0,
            "currency": "USD", "total_cap": .05, "daily_cap": .05,
            "weekly_cap": .1, "game_cap": .01, "team_cap": .01,
        },
        confirmed=True,
        now=NOW,
    )
    exposure = snapshot(exposure_path, now=NOW)
    policy = SportPolicy(
        "NFL", "owner-policy-v1", validation_id="validation-current-1",
        deployment_state="PROVISIONAL_VALIDATED", provisional_allowed=True,
        provisional_stake_cap=.0025, kelly_fraction=.1, sport_exposure_cap=.03,
    )
    contract = deepcopy(leg()["wager_contract"])
    contract.update({
        "sport_policy_version": policy.version,
        "validation_id": policy.validation_id,
        "validation_artifact_id": "artifact-current-1",
        "model_id": "model-current-1",
        "model_version": "model-version-current-1",
        "calibration_id": "calibration-current-1",
        "calibration_version": "calibration-version-current-1",
        "deployment_state": policy.deployment_state,
    })
    state = {
        "sport": contract["sport"],
        "market_family": contract["market_family"],
        "validation_state": policy.deployment_state,
        "deployment_state": policy.deployment_state,
        "validation_id": contract["validation_id"],
        "artifact_id": contract["validation_artifact_id"],
        "model_id": contract["model_id"],
        "model_version": contract["model_version"],
        "calibration_id": contract["calibration_id"],
        "calibration_version": contract["calibration_version"],
        "validated_policy": dict(asdict(policy), validation_id=""),
    }
    activation = activate_market(
        state, policy, exposure, owner_id="owner-fixture",
        expires_at=(NOW + timedelta(minutes=20)).isoformat(),
        confirm=True, now=NOW,
    )
    activation_dir = tmp_path / "active-markets"
    activation_dir.mkdir()
    (activation_dir / "NFL-SPREAD.json").write_text(
        json.dumps(activation), encoding="utf-8"
    )
    source = _approved_source(
        wager_contract=contract,
        pick=contract["selection"], best_pick=contract["selection"],
        market_type=contract["market_type"], odds=contract["odds"],
        odds_american=contract["odds"], quote_source=contract["sportsbook"],
        quote_time=contract["quote_timestamp"],
        game_start_utc=contract["start"], spread_line=contract["line"],
        Play_Stake=contract["production_bet_amount"],
    )
    package = _package(monkeypatch, [source])
    settings = {
        "PARLAYPICKER_MARKET_ACTIVATIONS_DIR": str(activation_dir),
        "PARLAYPICKER_EXPOSURE_LEDGER": str(exposure_path),
    }
    return package, activation, settings


def _freeze_release_clocks(monkeypatch):
    monkeypatch.setattr("app_core.release_preflight.datetime", _Clock)
    monkeypatch.setattr("app_core.release_authority.datetime", _Clock)


def test_r01_actual_netlify_publisher_blocks_when_current_authority_missing(monkeypatch):
    package = _package(monkeypatch)
    _freeze_release_clocks(monkeypatch)
    from app_core import netlify_publishing
    calls = []
    monkeypatch.setattr(netlify_publishing, "api_call", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(ReleasePreflightError) as exc:
        netlify_publishing.deploy(package, "site-1234", "unused-test-token")

    assert calls == []
    assert exc.value.report["blocker_counts"]["CURRENT_AUTHORITY_NOT_APPROVED"] == 1
    assert exc.value.report["authority_resolution"]["source_status"] == "BLOCKED"


def test_r01_actual_publisher_uses_existing_owner_activation_and_exposure(
    tmp_path, monkeypatch,
):
    package, _, settings = _current_source(tmp_path, monkeypatch)
    _freeze_release_clocks(monkeypatch)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    from app_core import netlify_publishing
    calls = []
    monkeypatch.setattr(
        netlify_publishing, "api_call",
        lambda *args, **kwargs: calls.append(args) or {
            "id": "deploy-123", "site_id": "site-1234", "state": "processing",
        },
    )

    job = netlify_publishing.deploy(package, "site-1234", "unused-test-token")

    assert job["state"] == "processing"
    assert len(calls) == 1
    resolution = resolve_publication_authority(package, at=NOW + timedelta(minutes=1))
    assert resolution.required and resolution.source_status == "VERIFIED"
    assert resolution.provider_status == "FROZEN_QUOTE_POLICY"


def test_r01_revoked_expired_and_wrong_binding_authority_block(tmp_path, monkeypatch):
    package, activation, settings = _current_source(tmp_path, monkeypatch)
    resolution = resolve_publication_authority(
        package, at=NOW + timedelta(minutes=1), setting=lambda name: settings.get(name)
    )
    row = package["games"]["overall"][0]
    identity = row_identity(row)
    original = dict(resolution.authorities[identity])
    cases = [
        (dict(original, revoked=True), "CURRENT_AUTHORITY_WITHDRAWN"),
        (dict(original, expires_at=NOW.isoformat()), "CURRENT_AUTHORITY_EXPIRED"),
        (dict(original, binding=dict(original["binding"], odds=-125)),
         "CURRENT_AUTHORITY_PRICE_MISMATCH"),
    ]
    for current, reason in cases:
        report = evaluate_release(
            package, at=NOW + timedelta(minutes=1),
            current_authority={identity: current}, require_current_authority=True,
        )
        assert report["blocker_counts"][reason] == 1

    activation["expires_at"] = NOW.isoformat()
    from core.exposure_ledger import digest
    activation["activation_hash"] = digest({
        key: value for key, value in activation.items() if key != "activation_hash"
    })
    path = settings["PARLAYPICKER_MARKET_ACTIVATIONS_DIR"] + "/NFL-SPREAD.json"
    from pathlib import Path
    Path(path).write_text(json.dumps(activation), encoding="utf-8")
    expired = resolve_publication_authority(
        package, at=NOW + timedelta(minutes=1), setting=lambda name: settings.get(name)
    )
    assert expired.source_status == "BLOCKED"
    assert "CURRENT_MARKET_ACTIVATION_EXPIRED" in expired.reason_codes


def test_r01_actual_authority_rejects_synthetic_activation(tmp_path, monkeypatch):
    package, activation, settings = _current_source(tmp_path, monkeypatch)
    row = package["games"]["overall"][0]
    contract = row["wager_contract"]
    activation["model_id"] = contract["model_id"] = "synthetic-model"
    from core.exposure_ledger import digest
    activation["activation_hash"] = digest({
        key: value for key, value in activation.items() if key != "activation_hash"
    })
    from pathlib import Path
    Path(settings["PARLAYPICKER_MARKET_ACTIVATIONS_DIR"], "NFL-SPREAD.json").write_text(
        json.dumps(activation), encoding="utf-8"
    )

    resolution = resolve_publication_authority(
        package, at=NOW + timedelta(minutes=1),
        setting=lambda name: settings.get(name),
    )

    assert resolution.source_status == "BLOCKED"
    assert resolution.reason_codes == (
        "CURRENT_MARKET_AUTHORITY_VERIFICATION_FAILED",
    )


def test_r02_earliest_authority_review_or_provider_deadline_controls_release(monkeypatch):
    package = _package(monkeypatch)
    at = NOW + timedelta(minutes=1)
    row = package["games"]["overall"][0]
    identity = row_identity(row)
    binding = {
        "sport": row["sport"], "market": row["market"],
        "selection": row["pick"], "line": row["wager_contract"]["line"],
        "odds": row["odds"], "sportsbook": row["quote_source"],
        "quote_timestamp": row["quote_time"], "event_start": row["start"],
    }
    authority = {identity: {
        "status": "AUTHORIZED", "revoked": False,
        "effective_at": NOW.isoformat(),
        "expires_at": (NOW + timedelta(minutes=20)).isoformat(),
        "review_expires_at": (NOW + timedelta(minutes=10)).isoformat(),
        "provider_expires_at": (NOW + timedelta(minutes=5)).isoformat(),
        "binding": binding,
    }}
    fresh = evaluate_release(
        package, at=at, current_authority=authority,
        require_current_authority=True,
    )
    assert fresh["rows"][0]["expires_at"] == (NOW + timedelta(minutes=5)).isoformat()
    stale = evaluate_release(
        package, at=NOW + timedelta(minutes=6), current_authority=authority,
        require_current_authority=True,
    )
    assert stale["blocker_counts"]["CURRENT_PROVIDER_QUOTE_EXPIRED"] == 1


def _attestation_registry(evidence_root, secret):
    records = []
    for path in evidence_root.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        kind = payload["evidence_kind"]
        reference = payload["execution"]["reference"]
        records.append({
            "attestation_id": "trusted-" + kind,
            "evidence_kind": kind,
            "environment": payload["environment"],
            "source_revision": payload["release_candidate"]["source_revision"],
            "reference_provider": reference["provider"],
            "execution_id": str(reference.get("run_id") or reference.get("execution_id")),
            "artifact_sha256": [item["sha256"] for item in payload["artifacts"]],
            "scenario_ids": [item["id"] for item in payload["scenarios"]],
            "verified_at": payload["provenance"]["reviewed_at"],
        })
    registry = {
        "schema_version": 1,
        "registry_id": "owner-hosted-attestations-1",
        "attestations": records,
    }
    registry["signature_hmac_sha256"] = sign_registry(registry, secret)
    return registry


def test_r03_real_verifier_loads_authenticated_out_of_band_attestations(
    tmp_path, monkeypatch,
):
    evidence_root = _configure_verifier(monkeypatch, tmp_path)
    _write_required_evidence(evidence_root, status_only=False)
    secret = "fixture-secret-that-is-at-least-thirty-two-characters"
    registry_path = tmp_path / "trusted-attestations.json"
    registry_path.write_text(
        json.dumps(_attestation_registry(evidence_root, secret)), encoding="utf-8"
    )
    monkeypatch.setenv("PAID_TRUSTED_ATTESTATION_REGISTRY", str(registry_path))
    monkeypatch.setenv("PAID_TRUSTED_ATTESTATION_HMAC_SECRET", secret)

    code, report = verify_paid_launch.verify("staging")

    assert code == 0 and report["status"] == "PASS"
    assert report["checks"]["trusted_attestation_adapter"] == {
        "status": "READY", "registry_id": "owner-hosted-attestations-1",
        "record_count": 7, "reason_codes": [],
    }
    assert all(
        item["hosted_status"] == "PASS"
        for item in report["checks"]["evidence"].values()
    )


def test_r03_self_declared_or_tampered_attestation_remains_blocked(
    tmp_path, monkeypatch,
):
    evidence_root = _configure_verifier(monkeypatch, tmp_path)
    _write_required_evidence(evidence_root, status_only=False)
    secret = "fixture-secret-that-is-at-least-thirty-two-characters"
    registry = _attestation_registry(evidence_root, secret)
    registry["attestations"][0]["execution_id"] = "self-declared-change"
    registry_path = tmp_path / "trusted-attestations.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setenv("PAID_TRUSTED_ATTESTATION_REGISTRY", str(registry_path))
    monkeypatch.setenv("PAID_TRUSTED_ATTESTATION_HMAC_SECRET", secret)

    code, report = verify_paid_launch.verify("staging")

    assert code == 2 and report["status"] == "BLOCKED"
    assert report["checks"]["trusted_attestation_adapter"]["reason_codes"] == [
        "ATTESTATION_REGISTRY_SIGNATURE_INVALID"
    ]
    assert all(
        item["hosted_status"] == "BLOCKED"
        for item in report["checks"]["evidence"].values()
    )
