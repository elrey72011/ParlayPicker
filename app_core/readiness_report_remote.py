"""The latest market readiness report, kept on the Shared Drive for the app (owner 2026-11-10).

The readiness workflow restores the evidence read-only and builds the report in 25 minutes or
more, so the app cannot build it. With ``--publish`` the workflow uploads the finished report
as one object under ``PREFIX``, and the app reads the newest one.

* The object is outside every evidence prefix: the evidence restore never reads it, and the
  report's own evidence client still refuses every write (``read_only_drive``).
* Drive objects are create-only and a prefix listing walks the whole folder (minutes), so the
  name is the UTC hour of the report. The app asks for exact names, newest hour first.
  A second report in the same hour is not stored: the first one stands.
* Research only. Nothing here is read by a pick, a stake or a plan evaluation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

PREFIX = "parlaypicker/market-readiness-v1/report/"
MAX_BYTES = 20_000_000
LOOKBACK_HOURS = 48
MISSING = "Remote evidence object is missing"


def _utcnow():
    return datetime.now(timezone.utc)


def key_for(when) -> str:
    return PREFIX + when.astimezone(timezone.utc).strftime("%Y%m%dT%H") + ".json"


def publish(client, folder, report, markdown, *, clock=_utcnow):
    """Upload the report as this hour's object and read it back. Returns its status and key."""
    from app_core.evidence_drive import AlreadyExists
    now = clock()
    raw = json.dumps({"generated_at": now.isoformat(), "markdown": markdown, "report": report},
                     sort_keys=True, default=str).encode()
    if len(raw) > MAX_BYTES:
        return {"status": "TOO_LARGE", "bytes": len(raw)}
    key = key_for(now)
    try:
        client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
    except AlreadyExists:
        return {"status": "ALREADY_PUBLISHED_THIS_HOUR", "key": key}
    with client.get_object(Bucket=folder, Key=key)["Body"] as body:
        if body.read() != raw:
            raise ValueError("readiness_report_readback")
    return {"status": "PUBLISHED", "key": key, "bytes": len(raw)}


def latest(client, folder, *, now=None, hours=LOOKBACK_HOURS):
    """The newest published report within ``hours`` (``generated_at``, ``markdown``, ``report``), or None."""
    from app_core.evidence_config import EvidenceStorageError
    now = now or _utcnow()
    for back in range(hours + 1):
        try:
            with client.get_object(Bucket=folder, Key=key_for(now - timedelta(hours=back)))["Body"] as body:
                raw = body.read(MAX_BYTES + 1)
        except EvidenceStorageError as exc:
            if str(exc) == MISSING:
                continue
            raise
        if len(raw) > MAX_BYTES:
            raise ValueError("readiness_report_too_large")
        found = json.loads(raw)
        if not isinstance(found, dict) or not isinstance(found.get("markdown"), str) or not isinstance(found.get("report"), dict):
            raise ValueError("readiness_report_shape")
        return found
    return None
