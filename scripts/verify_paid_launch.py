"""Read-only paid-launch verifier. It cannot activate markets, sales, or billing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONFIGURATION = (
    "PAID_DATABASE_URL", "PAID_PUBLIC_BASE_URL", "PAID_ALLOWED_ORIGINS", "PAID_ALLOWED_RETURN_URLS",
    "PAID_OIDC_ISSUER", "PAID_OIDC_CLIENT_ID", "PAID_OIDC_CLIENT_SECRET", "PAID_OIDC_CALLBACK_URL",
    "PAID_RELEASE_HMAC_SECRET", "PAID_GATEWAY_PROBE_TOKEN", "PAID_RELEASE_PROBE_URL",
    "PAID_STRIPE_SECRET_KEY", "PAID_STRIPE_WEBHOOK_SECRET", "PAID_STRIPE_ACCOUNT_ID", "PAID_STRIPE_PRICE_ID",
)


def check_scope() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_launch_change_scope.py")],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = {"status": "ERROR", "reason_codes": ["SCOPE_GUARD_NO_JSON"]}
    payload["exit_code"] = result.returncode
    return payload


def verify(environment: str) -> tuple[int, dict[str, Any]]:
    blockers: list[str] = []
    errors: list[str] = []
    checks: dict[str, Any] = {}
    scope = check_scope()
    checks["protected_scope"] = scope
    if scope.get("status") != "PASS":
        errors.append("PROTECTED_SCOPE_GUARD_FAILED")
    missing = [name for name in REQUIRED_CONFIGURATION if not os.environ.get(name, "").strip()]
    checks["configuration"] = {"status": "PASS" if not missing else "BLOCKED", "missing_names": missing}
    if missing:
        blockers.append("STAGING_CONFIGURATION_INCOMPLETE")
    live_enabled = os.environ.get("PAID_LIVE_BILLING_ENABLED", "false").strip().lower() == "true"
    checks["live_billing"] = {"status": "BLOCKED_AS_REQUIRED" if not live_enabled else "ERROR"}
    if live_enabled:
        errors.append("LIVE_BILLING_MUST_REMAIN_DISABLED_FOR_THIS_DELIVERY")
    base_url = os.environ.get("PAID_PUBLIC_BASE_URL", "").rstrip("/")
    if base_url and environment == "staging":
        try:
            response = httpx.get(base_url + "/api/v1/status", timeout=10, follow_redirects=False)
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            checks["staging_status_endpoint"] = {
                "status": "PASS" if response.status_code == 200 else "FAIL",
                "http_status": response.status_code,
                "schema_version": body.get("schema_version"),
                "source_revision": body.get("source_revision"),
                "premium_fields_present": any(key in body for key in ("recommendations", "selection", "odds_american")),
            }
            expected_revision = os.environ.get("PAID_SOURCE_REVISION", "").strip()
            checks["staging_status_endpoint"]["revision_matches"] = bool(expected_revision and body.get("source_revision") == expected_revision)
            if response.status_code != 200 or checks["staging_status_endpoint"]["premium_fields_present"] or not checks["staging_status_endpoint"]["revision_matches"]:
                blockers.append("STAGING_STATUS_ENDPOINT_UNVERIFIED")
        except Exception as exc:
            checks["staging_status_endpoint"] = {"status": "BLOCKED", "error_type": type(exc).__name__}
            blockers.append("STAGING_ENDPOINT_UNREACHABLE")
    else:
        checks["staging_status_endpoint"] = {"status": "NOT_RUN_EXTERNAL_BLOCKER"}
        blockers.append("STAGING_ENDPOINT_NOT_CONFIGURED")
    evidence_dir = ROOT / "docs" / "paid-launch" / "evidence"
    required_evidence = {
        "billing_sandbox": evidence_dir / "billing-sandbox-evidence.json",
        "publication_recovery": evidence_dir / "publication-recovery-evidence.json",
        "results_reconciliation": evidence_dir / "results-reconciliation-evidence.json",
        "alert_faults": evidence_dir / "alert-fault-evidence.json",
        "load": evidence_dir / "performance-load-evidence.json",
        "backup_restore": evidence_dir / "backup-restore-evidence.json",
        "pilot": evidence_dir / "pilot-ledger.json",
    }
    checks["evidence"] = {}
    for name, path in required_evidence.items():
        if not path.exists():
            checks["evidence"][name] = "MISSING"
            blockers.append(f"{name.upper()}_EVIDENCE_MISSING")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        checks["evidence"][name] = payload.get("status", "UNKNOWN")
        if payload.get("status") != "PASS":
            blockers.append(f"{name.upper()}_NOT_VERIFIED")
    report = {
        "schema_version": 1,
        "environment": environment,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "completion_label": "STAGING_VERIFIED_LIVE_BLOCKED" if not blockers and not errors else "IMPLEMENTED_LOCAL_VERIFIED",
        "status": "ERROR" if errors else ("BLOCKED" if blockers else "PASS"),
        "errors": sorted(set(errors)),
        "blockers": sorted(set(blockers)),
        "checks": checks,
    }
    return (1 if errors else (2 if blockers else 0)), report


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Paid launch readiness", "", f"- Status: `{report['status']}`", f"- Completion label: `{report['completion_label']}`", f"- Generated: `{report['generated_at']}`", "", "## Blockers", ""]
    lines.extend(f"- `{item}`" for item in report["blockers"] or ["None"])
    if report["errors"]:
        lines.extend(["", "## Integrity errors", ""] + [f"- `{item}`" for item in report["errors"]])
    lines.extend(["", "The verifier is read-only and cannot enable sales, markets, or live billing.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", required=True, choices=("local", "staging", "production"))
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        code, report = verify(args.environment)
    except Exception as exc:
        code = 1
        report = {"schema_version": 1, "status": "ERROR", "errors": ["VERIFIER_EXECUTION_ERROR"], "blockers": [], "generated_at": datetime.now(timezone.utc).isoformat(), "completion_label": "IMPLEMENTATION_INCOMPLETE", "details": type(exc).__name__}
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
