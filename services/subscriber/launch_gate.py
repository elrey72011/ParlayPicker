"""Single fail-closed policy used by checkout, release and delivery."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Mapping

from .contracts import LaunchDecision


REQUIRED_COMMERCIAL_APPROVALS = frozenset(
    {
        "PRODUCT_CLAIMS",
        "DATA_RIGHTS",
        "PROCESSOR_SUPPORT",
        "LEGAL_JURISDICTION",
        "TERMS_PRIVACY_RENEWALS",
        "RESPONSIBLE_USE",
    }
)


def checkout_decision(
    *,
    engineering_status: str,
    commercial_status: str,
    sales_status: str,
    configured_terms: bool,
    qualified_markets: Iterable[str],
    commercially_enabled_markets: Iterable[str],
    current_approvals: Iterable[str],
    billing_mode: str,
    live_billing_enabled: bool,
    now: datetime | None = None,
) -> LaunchDecision:
    reasons: list[str] = []
    if engineering_status != "STAGING_VERIFIED":
        reasons.append("ENGINEERING_NOT_STAGING_VERIFIED")
    if commercial_status != "APPROVED":
        reasons.append("COMMERCIAL_APPROVAL_PENDING")
    if sales_status != "OWNER_ENABLED":
        reasons.append("SALES_NOT_OWNER_ENABLED")
    if not configured_terms:
        reasons.append("COMMERCIAL_TERMS_NOT_CONFIGURED")
    offered = set(qualified_markets) & set(commercially_enabled_markets)
    if not offered:
        reasons.append("NO_COMMERCIALLY_ENABLED_QUALIFIED_MARKET")
    missing = sorted(REQUIRED_COMMERCIAL_APPROVALS - set(current_approvals))
    reasons.extend(f"MISSING_APPROVAL_{item}" for item in missing)
    if billing_mode == "live" and not live_billing_enabled:
        reasons.append("LIVE_BILLING_DISABLED")
    return LaunchDecision(allowed=not reasons, reason_codes=reasons, evaluated_at=now or datetime.now(timezone.utc))


def release_decision(
    *,
    authority: Mapping[str, object],
    exact_markets: Iterable[str],
    commercially_enabled_markets: Iterable[str],
    expiry_at: datetime,
    event_starts: Iterable[datetime],
    reviewed_hash_matches: bool,
    rights_present: bool,
    now: datetime | None = None,
) -> LaunchDecision:
    at = now or datetime.now(timezone.utc)
    reasons: list[str] = []
    if authority.get("revoked") is True:
        reasons.append("AUTHORITY_REVOKED")
    if authority.get("market_status") != "QUALIFIED":
        reasons.append("MARKET_NOT_QUALIFIED")
    effective = authority.get("effective_at")
    expires = authority.get("expires_at")
    if not isinstance(effective, datetime) or not isinstance(expires, datetime) or not (effective <= at < expires):
        reasons.append("AUTHORITY_STALE_OR_UNKNOWN")
    if not set(exact_markets).issubset(set(commercially_enabled_markets)):
        reasons.append("MARKET_NOT_COMMERCIALLY_ENABLED")
    if not reviewed_hash_matches:
        reasons.append("REVIEW_HASH_MISMATCH")
    if not rights_present:
        reasons.append("CONTENT_RIGHTS_UNVERIFIED")
    if expiry_at <= at or any(start <= at for start in event_starts):
        reasons.append("RECOMMENDATION_EXPIRED_OR_STARTED")
    return LaunchDecision(allowed=not reasons, reason_codes=reasons, evaluated_at=at)
