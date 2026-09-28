"""Correlated, secret-free performance spans for interactive workflows.

Spans use a monotonic clock for duration and UTC wall time only for evidence.
Missing counters remain ``None`` so telemetry never turns "not measured" into
an apparently authoritative zero.
"""
from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from time import perf_counter
import uuid


_current_span: ContextVar[str | None] = ContextVar("performance_parent_span", default=None)
PROCESS_INSTANCE = uuid.uuid4().hex

COUNTERS = (
    "listing_traversals", "listing_pages", "metadata_items_seen",
    "objects_matched", "objects_downloaded", "objects_reused",
    "records_returned", "records_imported", "records_unchanged",
    "bytes_downloaded", "retries", "retry_wait_ms", "auth_refreshes",
    "cache_hits", "cache_misses",
)


def opaque_hash(*parts):
    """Return a stable non-secret identifier for a storage/database scope."""
    value = "\0".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(value.encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def operation_ids(**values):
    """Create correlation identifiers while accepting caller-owned IDs."""
    trace = values.get("trace_id") or uuid.uuid4().hex
    return {
        "trace_id": trace,
        "action_id": values.get("action_id") or uuid.uuid4().hex,
        "refresh_run_id": values.get("refresh_run_id"),
        "lock_operation_id": values.get("lock_operation_id"),
        "publication_attempt_id": values.get("publication_attempt_id"),
    }


class PerformanceSpan:
    """A mutable span whose finalized JSON record is available as ``record``."""

    def __init__(self, operation, *, ids=None, **fields):
        ids = ids or operation_ids()
        self.record = {
            "trace_id": ids.get("trace_id"),
            "action_id": ids.get("action_id"),
            "parent_span_id": fields.pop("parent_span_id", _current_span.get()),
            "span_id": uuid.uuid4().hex,
            "refresh_run_id": ids.get("refresh_run_id"),
            "lock_operation_id": ids.get("lock_operation_id"),
            "publication_attempt_id": ids.get("publication_attempt_id"),
            "source_commit": os.environ.get("COMMIT_SHA") or os.environ.get("GIT_SHA"),
            "process_instance": PROCESS_INSTANCE,
            "database_generation": None,
            "board_hash": None,
            "storage_scope_hash": None,
            "operation": operation,
            "table_or_kind": None,
            "started_at_utc": utc_now(),
            "elapsed_ms": None,
            "outcome": None,
            "reason_code": None,
            **{name: None for name in COUNTERS},
            "cache_invalidation_reason": None,
            "verification_status": None,
            "partial_result": False,
        }
        self.record.update(fields)
        self._started = None
        self._token = None

    def set(self, **fields):
        unknown = set(fields) - set(self.record)
        if unknown:
            raise ValueError("Unknown performance-span fields: " + ", ".join(sorted(unknown)))
        self.record.update(fields)
        return self

    def add(self, **counters):
        for name, amount in counters.items():
            if name not in COUNTERS:
                raise ValueError("Unknown performance counter: " + name)
            self.record[name] = (self.record[name] or 0) + amount
        return self

    def __enter__(self):
        self._started = perf_counter()
        self._token = _current_span.set(self.record["span_id"])
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.record["elapsed_ms"] = round((perf_counter() - self._started) * 1000, 3)
        if self.record["outcome"] is None:
            self.record["outcome"] = "ok" if exc_type is None else "error"
        if exc_type is not None and self.record["reason_code"] is None:
            # Class names are useful and do not expose provider messages, URLs,
            # headers, credentials, or user-supplied payloads.
            self.record["reason_code"] = exc_type.__name__
        if self._token is not None:
            _current_span.reset(self._token)
        logging.getLogger(__name__).warning(
            "PERFORMANCE_SPAN %s", json.dumps(self.record, sort_keys=True, separators=(",", ":")))
        return False
