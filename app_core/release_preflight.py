"""Fail-closed freshness and identity checks at the public release boundary.

The saved package remains immutable evidence of the producer decision.  This
module only answers whether a row that *already* claims APPROVED or TRIAL is
still usable at an observation time.  PASS/research/history packages remain
publishable even when old.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
import math
import re
from typing import Mapping

from app_core.public_quote_policy import supported_quote
from app_core.quote_freshness import package_age_minutes


VERSION = "public-release-preflight-v1"
ACTIONABLE = frozenset({"APPROVED", "TRIAL"})


class ReleasePreflightError(ValueError):
    """An actionable saved row is no longer safe to release."""

    def __init__(self, report: dict):
        self.report = report
        codes = sorted(report["blocker_counts"])
        super().__init__("Actionable release preflight failed: " + ", ".join(codes))


def _time(value) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _clock(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    parsed = _time(value) if isinstance(value, str) else value
    if not isinstance(parsed, datetime) or parsed.tzinfo is None:
        raise ValueError("Release preflight time must include a timezone")
    return parsed.astimezone(timezone.utc)


def _number(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _same_number(left, right) -> bool:
    a, b = _number(left), _number(right)
    return a is not None and b is not None and math.isclose(a, b, abs_tol=1e-9)


def _same_time(left, right) -> bool:
    a, b = _time(left), _time(right)
    return a is not None and b is not None and a == b


def _row_identity(row: Mapping) -> str:
    value = {
        key: row.get(key)
        for key in ("sport", "game", "market", "pick", "odds", "quote_source",
                    "quote_time", "as_of", "start", "status")
    }
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(raw.encode("utf-8")).hexdigest()


def _selection_line(row: Mapping) -> float | None:
    selection = str(row.get("pick") or "").strip()
    market = str(row.get("market") or "")
    pattern = r"^(?:Over|Under)\s+(-?\d+(?:\.\d+)?)$" if market.startswith("total") else r"\s([+-]\d+(?:\.\d+)?)$"
    match = re.search(pattern, selection, re.IGNORECASE)
    return _number(match.group(1)) if match else None


def _current_blockers(row: Mapping, at: datetime, minutes: int) -> tuple[list[str], dict]:
    blockers: list[str] = []
    start, quote, analysis = (_time(row.get(name)) for name in ("start", "quote_time", "as_of"))
    if start is None:
        blockers.append("START_TIME_UNAVAILABLE")
    elif start <= at:
        blockers.append("GAME_STARTED")
    if not supported_quote(row):
        blockers.append("QUOTE_UNAVAILABLE")
    if quote is None:
        blockers.append("QUOTE_TIME_UNAVAILABLE")
    elif quote > at:
        blockers.append("QUOTE_TIME_FUTURE")
    elif at - quote > timedelta(minutes=minutes):
        blockers.append("QUOTE_EXPIRED")
    if analysis is None:
        blockers.append("ANALYSIS_TIME_UNAVAILABLE")
    elif analysis > at:
        blockers.append("ANALYSIS_TIME_FUTURE")
    elif at - analysis > timedelta(minutes=minutes):
        blockers.append("ANALYSIS_EXPIRED")
    deadlines = [value for value in (
        quote + timedelta(minutes=minutes) if quote else None,
        analysis + timedelta(minutes=minutes) if analysis else None,
        start,
    ) if value is not None]
    expiry = min(deadlines) if len(deadlines) == 3 else None
    timing = {
        "quote_age_seconds": (at - quote).total_seconds() if quote else None,
        "analysis_age_seconds": (at - analysis).total_seconds() if analysis else None,
        "event_seconds_remaining": (start - at).total_seconds() if start else None,
        "expires_at": expiry.isoformat() if expiry else None,
        "remaining_validity_seconds": (expiry - at).total_seconds() if expiry else None,
    }
    return blockers, timing


def _strict_contract_blockers(row: Mapping) -> list[str]:
    contract = row.get("wager_contract")
    if not isinstance(contract, dict):
        return ["STRICT_AUTHORITY_MISSING"]
    try:
        from core.live_wager_contract import validate_snapshot
        validate_snapshot(contract)
    except (TypeError, ValueError):
        return ["STRICT_AUTHORITY_INVALID"]
    blockers: list[str] = []
    comparisons = (
        (contract.get("selection") == row.get("pick"), "SELECTION_CHANGED"),
        (contract.get("market_type") == row.get("market"), "MARKET_CHANGED"),
        (_same_number(contract.get("odds"), row.get("odds")), "PRICE_CHANGED"),
        (str(contract.get("sportsbook") or "").casefold() ==
         str(row.get("quote_source") or "").casefold(), "SPORTSBOOK_CHANGED"),
        (_same_time(contract.get("quote_timestamp"), row.get("quote_time")), "QUOTE_IDENTITY_CHANGED"),
        (_same_time(contract.get("start"), row.get("start")), "EVENT_START_CHANGED"),
        (_number(contract.get("line")) is not None and
         _same_number(contract.get("line"), _selection_line(row)), "LINE_CHANGED"),
        (contract.get("production_eligible") is True, "STRICT_AUTHORITY_WITHDRAWN"),
        ((_number(contract.get("production_bet_amount")) or 0) > 0, "STRICT_ALLOCATION_MISSING"),
        ((_number(contract.get("conservative_ev")) or 0) > 0, "NONPOSITIVE_CONSERVATIVE_EV"),
        (contract.get("identity_verified") is True, "IDENTITY_UNVERIFIED"),
        (contract.get("quote_verified") is True, "EXACT_QUOTE_UNVERIFIED"),
        (contract.get("quote_fresh") is True, "SAVED_QUOTE_NOT_FRESH"),
    )
    blockers.extend(code for passed, code in comparisons if not passed)
    return blockers


def _trial_contract_blockers(row: Mapping) -> list[str]:
    contract = row.get("controlled_trial_contract")
    if not isinstance(contract, dict):
        return ["TRIAL_AUTHORITY_MISSING"]
    try:
        from app_core.controlled_trial import validate_contract
        validate_contract(contract, read_only_legacy=True)
    except (TypeError, ValueError):
        return ["TRIAL_AUTHORITY_INVALID"]
    blockers: list[str] = []
    comparisons = (
        (contract.get("selection") == row.get("pick"), "SELECTION_CHANGED"),
        (contract.get("market_type") == row.get("market"), "MARKET_CHANGED"),
        (_same_number(contract.get("odds"), row.get("odds")), "PRICE_CHANGED"),
        (str(contract.get("sportsbook") or "").casefold() ==
         str(row.get("quote_source") or "").casefold(), "SPORTSBOOK_CHANGED"),
        (_same_time(contract.get("quote_timestamp"), row.get("quote_time")), "QUOTE_IDENTITY_CHANGED"),
        (_same_time(contract.get("start"), row.get("start")), "EVENT_START_CHANGED"),
        (_number(contract.get("line")) is not None and
         _same_number(contract.get("line"), _selection_line(row)), "LINE_CHANGED"),
        (contract.get("trial_eligible") is True, "TRIAL_AUTHORITY_WITHDRAWN"),
        ((_number(contract.get("recommended_bet_amount")) or 0) > 0, "TRIAL_ALLOCATION_MISSING"),
        ((_number(contract.get("estimated_expected_value")) or 0) > 0, "NONPOSITIVE_TRIAL_EV"),
        (contract.get("gemini_review_status") == "APPROVE", "TRIAL_REVIEW_NOT_APPROVED"),
        (contract.get("identity_verified") is True, "IDENTITY_UNVERIFIED"),
        (contract.get("quote_verified") is True, "EXACT_QUOTE_UNVERIFIED"),
        (contract.get("quote_fresh") is True, "SAVED_QUOTE_NOT_FRESH"),
    )
    blockers.extend(code for passed, code in comparisons if not passed)
    return blockers


def _external_authority_blockers(row_id: str, authority: Mapping | None, at: datetime) -> tuple[list[str], str]:
    if authority is None:
        return [], "NOT_AVAILABLE"
    current = authority.get(row_id) if isinstance(authority, Mapping) else None
    if not isinstance(current, Mapping):
        return ["CURRENT_AUTHORITY_UNKNOWN"], "UNKNOWN"
    blockers: list[str] = []
    if current.get("withdrawn") is True or current.get("revoked") is True:
        blockers.append("CURRENT_AUTHORITY_WITHDRAWN")
    expires = _time(current.get("expires_at"))
    if current.get("status") not in {"APPROVED", "AUTHORIZED"}:
        blockers.append("CURRENT_AUTHORITY_NOT_APPROVED")
    if expires is not None and expires <= at:
        blockers.append("CURRENT_AUTHORITY_EXPIRED")
    return blockers, "BLOCKED" if blockers else "VERIFIED"


def evaluate_release(package: dict, *, at: datetime | str | None = None,
                     published_at: datetime | str | None = None,
                     current_authority: Mapping | None = None,
                     validate: bool = True) -> dict:
    """Return saved-versus-current release evidence without mutating ``package``."""
    if validate:
        from app_core.public_board import validate_package
        validate_package(package)
    clock = _clock(at)
    published = _clock(published_at) if published_at is not None else None
    minutes = package_age_minutes(package)
    producer = (package.get("board_diagnostics") or {}).get("traces") or []
    rows: list[dict] = []
    seen: set[str] = set()
    for section, values in package.get("games", {}).items():
        for position, row in enumerate(values):
            row_id = _row_identity(row)
            duplicate_view = row_id in seen
            seen.add(row_id)
            current, timing = _current_blockers(row, clock, minutes)
            actionable = row.get("status") in ACTIONABLE
            producer_trace = (producer[position] if section == "overall" and
                              position < len(producer) and isinstance(producer[position], dict)
                              else None)
            contract_blockers = (
                _strict_contract_blockers(row) if row.get("status") == "APPROVED" else
                _trial_contract_blockers(row) if row.get("status") == "TRIAL" else []
            )
            contract = (row.get("wager_contract") if row.get("status") == "APPROVED" else
                        row.get("controlled_trial_contract") if row.get("status") == "TRIAL" else None)
            if (actionable and isinstance(contract, dict) and producer_trace and
                    producer_trace.get("game_id") and
                    str(contract.get("matchup_id") or contract.get("game_id") or "") !=
                    str(producer_trace["game_id"])):
                contract_blockers.append("EVENT_IDENTITY_CHANGED")
            external, external_status = _external_authority_blockers(
                row_id, current_authority, clock,
            ) if actionable else ([], "NOT_APPLICABLE")
            observed_blockers = list(dict.fromkeys(current + contract_blockers + external)) if actionable else list(dict.fromkeys(current))
            release_blockers = observed_blockers if actionable else []
            record = {
                "row_id": row_id, "section": section, "position": position,
                "duplicate_view": duplicate_view, "sport": row.get("sport"),
                "game": row.get("game"), "market": row.get("market"),
                "selection": row.get("pick"), "odds": row.get("odds"),
                "saved_status": row.get("status"), "saved_actionable": actionable,
                "saved_producer_primary_reason": (producer_trace or {}).get("producer_primary_reason"),
                "saved_producer_blockers": (producer_trace or {}).get("producer_blockers", []),
                "current_status": "CURRENT_ACTIONABLE" if actionable and not release_blockers else
                                  "BLOCKED_ACTIONABLE" if actionable else
                                  "EXPIRED_OR_INVALID_RESEARCH" if observed_blockers else "NONACTIONABLE_RESEARCH",
                "current_primary_reason": observed_blockers[0] if observed_blockers else
                                          "CURRENT_ACTIONABLE" if actionable else "SAVED_RESEARCH_PASS",
                "current_blockers": observed_blockers,
                "release_blockers": release_blockers,
                "current_authority_status": external_status,
                **timing,
            }
            if published is not None:
                quote, analysis = _time(row.get("quote_time")), _time(row.get("as_of"))
                record["quote_age_at_publication_seconds"] = (
                    (published - quote).total_seconds() if quote else None
                )
                record["analysis_age_at_publication_seconds"] = (
                    (published - analysis).total_seconds() if analysis else None
                )
            rows.append(record)
    unique_actionable = [row for row in rows if row["saved_actionable"] and not row["duplicate_view"]]
    blockers = Counter(code for row in unique_actionable for code in row["release_blockers"])
    primary = Counter(row["current_primary_reason"] for row in unique_actionable)
    selected_unique = [row for row in rows if not row["duplicate_view"]]
    observed_primary = Counter(row["current_primary_reason"] for row in selected_unique)
    observed_overlap = Counter(code for row in selected_unique for code in row["current_blockers"])
    overall = [row for row in rows if row["section"] == "overall"]
    overall_primary = Counter(row["current_primary_reason"] for row in overall)
    overall_overlap = Counter(code for row in overall for code in row["current_blockers"])
    original_primary = Counter(
        str(row.get("producer_primary_reason") or "UNKNOWN")
        for row in producer if isinstance(row, dict)
    )
    allowed = not any(row["release_blockers"] for row in unique_actionable)
    schema = int(_number(package.get("schema_version")) or 0)
    modern_authority_present = any(
        isinstance(row.get("wager_contract"), dict) or
        isinstance(row.get("controlled_trial_contract"), dict)
        for section in package.get("games", {}).values()
        for row in section if isinstance(row, Mapping) and row.get("status") in ACTIONABLE
    )
    # Schema-v1/v2 packages predate frozen release-authority contracts.  They
    # remain publishable only as legacy/history content under the existing
    # browser usability rules; current schema-v5 packages and any package that
    # carries modern authority are always subject to this release gate.
    enforced = schema >= 5 or modern_authority_present
    return {
        "schema_version": VERSION,
        "evaluated_at": clock.isoformat(),
        "published_at": published.isoformat() if published else None,
        "freshness_policy_minutes": minutes,
        "package_built_at": package.get("built_at"),
        "actionable_row_count": len(unique_actionable),
        "actionable_release_allowed": allowed,
        "preflight_enforced": enforced,
        "legacy_history_only": bool(unique_actionable) and not enforced,
        "saved_producer_primary_counts": dict(sorted(original_primary.items())),
        "overall_current_primary_counts": dict(sorted(overall_primary.items())),
        "overall_current_overlapping_blocker_counts": dict(sorted(overall_overlap.items())),
        "selected_current_primary_counts": dict(sorted(observed_primary.items())),
        "selected_current_overlapping_blocker_counts": dict(sorted(observed_overlap.items())),
        "primary_counts": dict(sorted(primary.items())),
        "blocker_counts": dict(sorted(blockers.items())),
        "rows": rows,
    }


def require_actionable_release(package: dict, *, at: datetime | str | None = None,
                               current_authority: Mapping | None = None,
                               validate: bool = True) -> dict:
    """Raise only when a saved actionable row fails current release checks."""
    report = evaluate_release(package, at=at, current_authority=current_authority,
                              validate=validate)
    if report["preflight_enforced"] and not report["actionable_release_allowed"]:
        raise ReleasePreflightError(report)
    return report
