"""Authenticate a submitted reviewed package without importing the research UI."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from services.subscriber.canonical import object_hash, verify_signature
from services.subscriber.contracts import ReleaseSubmission


class AuthorityError(ValueError):
    """The external authority package is missing, stale, changed, or untrusted."""


def verify_reviewed_submission(
    raw: dict[str, Any],
    *,
    signature: str,
    secret: str,
    environment: str,
    now: datetime | None = None,
) -> ReleaseSubmission:
    """Validate the exact signed input and its independent review binding.

    This adapter is intentionally pure and read-only.  It cannot create a lock,
    invoke analysis, fetch provider data, or mutate upstream authority.
    """
    if not signature or not verify_signature(raw, signature, secret):
        raise AuthorityError("SUBMISSION_SIGNATURE_INVALID")
    submission = ReleaseSubmission.model_validate(raw)
    if submission.environment != environment:
        raise AuthorityError("SUBMISSION_ENVIRONMENT_MISMATCH")
    if object_hash(submission.review_payload()) != submission.reviewed_payload_hash:
        raise AuthorityError("REVIEW_HASH_MISMATCH")
    current = now or datetime.now(timezone.utc)
    authority = submission.authority
    if authority.revoked:
        raise AuthorityError("AUTHORITY_REVOKED")
    if authority.upstream_gate_result != "APPROVED":
        raise AuthorityError("UPSTREAM_GATE_NOT_APPROVED")
    if authority.market_status.value != "QUALIFIED":
        raise AuthorityError("MARKET_NOT_QUALIFIED")
    if not (authority.effective_at <= current < authority.expires_at):
        raise AuthorityError("AUTHORITY_STALE_OR_UNKNOWN")
    return submission
