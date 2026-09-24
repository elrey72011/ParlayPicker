"""Sanitized, bounded record of one authenticated six-sport research cycle."""

from __future__ import annotations

import re
from datetime import datetime
import uuid


_CODE = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,79}\Z")
_SENSITIVE = re.compile(r"secret|token|password|credential|api.?key|https?|bearer|private", re.I)
_SAFE_LOWER = frozenset({"success", "failed", "not_attempted", "partial",
                         "budget_paused", "legacy_specialized",
                         "research_model_cycle_completed", "missing_frozen_model",
                         "stale_frozen_model", "missing_provider_keys",
                         "legacy_bounded_cycle_no_shared_ledger", "ready", "continue",
                         "not_configured", "ok", "error"})
_COUNT_FIELDS = ("discovered_events", "captured_events", "captured_quote_rows",
                 "research_predictions", "research_close_candidates", "graded_events")
_STATUS_FIELDS = ("restore", "capture", "grade", "close_capture", "backup",
                  "reconciliation_status")


def _code(value):
    if not isinstance(value, str):
        return "REDACTED_INVALID_CODE"
    if value in {"MISSING_CREDENTIALS", "missing_provider_keys", "NCAAF:missing_provider_keys",
                 "NBA:missing_provider_keys", "NHL:missing_provider_keys", "NFL:missing_provider_keys"}:
        return value
    if not _CODE.fullmatch(value) or _SENSITIVE.search(value):
        return "REDACTED_INVALID_CODE"
    if value in _SAFE_LOWER or re.fullmatch(r"[A-Z][A-Z0-9_]{2,79}", value):
        return value
    if ":" in value:
        prefix, code = value.split(":", 1)
        if prefix in {"NFL", "NCAAF", "NBA", "NCAAB", "MLB", "NHL",
                      "SPREAD", "TOTAL", "RUN_LINE", "PUCK_LINE",
                      "canonical", "scheduler", "public_grading"} and (
                code in _SAFE_LOWER or re.fullmatch(r"[A-Z][A-Z0-9_]{2,79}", code)
                or code in {"ValueError", "RuntimeError", "ProviderError", "BudgetLimit",
                            "EvidenceConflict", "KeyError", "TypeError", "OSError"}):
            return value
    return "REDACTED_INVALID_CODE"


def _stamp(value):
    if not isinstance(value, str) or len(value) > 40:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if parsed.tzinfo is not None else None


def _uuid(value):
    try:
        return str(uuid.UUID(value)) if isinstance(value, str) and len(value) == 36 else None
    except ValueError:
        return None


def _plan_id(value):
    return value if isinstance(value, str) and re.fullmatch(
        r"prospective-(?:nfl|ncaaf|nba|ncaab|mlb|nhl)-[a-z_]+-2026-09-23-v1", value
    ) else None


def _counts(value):
    if not isinstance(value, dict):
        return None
    allowed = ("remote_records_read", "records_restored", "records_verified", "new_records_verified",
               "objects_uploaded_verified", "verified_manifest_records")
    return {key: value[key] for key in allowed if type(value.get(key)) is int and value[key] >= 0}


def sanitize_cycle(report):
    """Allowlist only counts, hashes, status codes and bounded budget facts."""
    requested = report.get("requested_sports", [])
    if not isinstance(requested, list):
        requested = []
    result = {
        "schema": "research-cycle-audit-v1",
        "run_id": _uuid(report.get("run_id")),
        "source_commit": report.get("source_commit") if isinstance(report.get("source_commit"), str)
            and re.fullmatch(r"[0-9a-f]{40}", report["source_commit"]) else None,
        "started_at": _stamp(report.get("started_at")),
        "finished_at": _stamp(report.get("finished_at")),
        "execution_state": report.get("execution_state") if report.get("execution_state")
            in {"IN_PROGRESS", "COMPLETE", "FAILED", "SKIPPED"} else "IN_PROGRESS",
        "active_sport": _code(report.get("active_sport")) if report.get("active_sport") else None,
        "active_stage": _code(report.get("active_stage")) if report.get("active_stage") else None,
        "requested_sports": [_code(sport) for sport in requested],
        "requested_slate_success": report.get("requested_slate_success") is True,
        "production_eligible": False,
        "errors": [_code(error) for error in report.get("errors", []) if isinstance(error, str)],
        "canonical_restore": _counts(report.get("canonical_restore")),
        "canonical_backup": _counts(report.get("canonical_backup")),
        "sports": {},
    }
    plans = report.get("frozen_validation_plans", [])
    if isinstance(plans, list):
        result["frozen_validation_plans"] = [
            {"sport": _code(item.get("sport")),
             "market_family": _code(item.get("market_family")),
             "validation_plan_id": _plan_id(item.get("validation_plan_id")),
             "artifact_hash": item.get("artifact_hash") if isinstance(item.get("artifact_hash"), str)
                and re.fullmatch(r"[0-9a-f]{64}", item["artifact_hash"]) else None,
             "frozen_at": _stamp(item.get("frozen_at"))}
            for item in plans if isinstance(item, dict)]
    public = report.get("public_grading")
    if isinstance(public, dict):
        result["public_grading"] = {"status": _code(public.get("status"))}
    for sport in requested:
        health = report.get("health", {}).get(sport, {})
        if not isinstance(health, dict):
            health = {}
        row = {key: _code(health.get(key, "not_attempted")) for key in _STATUS_FIELDS}
        row.update({key: health[key] if type(health.get(key)) is int and health[key] >= 0 else 0
                    for key in _COUNT_FIELDS})
        row["verified_backup"] = health.get("verified_backup") is True
        row["provider_blockers"] = [_code(code) for code in health.get("provider_blockers", [])
                                    if isinstance(code, str)]
        cycle = report.get("sports", {}).get(sport, {})
        if isinstance(cycle, dict):
            if isinstance(cycle.get("reconciliation"), dict):
                reconciled = cycle["reconciliation"]
                row["reconciliation"] = {
                    key: reconciled[key] for key in ("source_records", "source_receipts",
                                                      "source_games", "source_market_rows",
                                                      "canonical_research_identities", "duplicates",
                                                      "new_source_artifacts", "new_reconciled_facts",
                                                      "canonical_predictions")
                    if type(reconciled.get(key)) is int and reconciled[key] >= 0}
            if isinstance(cycle.get("model_cycle"), dict):
                model = cycle["model_cycle"]
                row["model_cycle"] = {key: model[key] for key in (
                    "canonical_events", "canonical_quotes", "canonical_results",
                    "models_fitted", "predictions")
                    if type(model.get(key)) is int and model[key] >= 0}
                row["model_blockers"] = [_code(code) for code in model.get("blockers", [])
                                          if isinstance(code, str)]
        budget = health.get("api_budget")
        if isinstance(budget, dict):
            row["api_budget"] = {key: _code(budget[key]) for key in ("provider", "status")
                                 if isinstance(budget.get(key), str)}
        result["sports"][_code(sport)] = row
    return result
