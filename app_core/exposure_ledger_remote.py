"""A permanent home for the exposure ledger on Drive (go-live Part 3, owner 2026-11-08).

``core.exposure_ledger`` keeps the owner's bankroll, limits and committed wagers in a local
append-only SQLite file. On Streamlit Cloud that file is wiped on every reboot: the
bankroll configuration disappears (every stake is then blocked), and if the owner configures
it again, the open exposure already committed is forgotten, so the exposure caps would no
longer count it.

This mirrors the ledger to Drive, one write-once object per event in order
(``events/000001.json``, ``000002.json``, ...). Each event's id is the SHA-256 of its payload
(``core.exposure_ledger.digest``), so every restored event is verified.

* ``sync``: when the remote is a prefix of the local ledger, push the local-only events;
  when the local ledger is a prefix of the remote, restore the missing ones in order; when
  they diverge, change nothing and report ``DIVERGED`` (the owner resolves it). A write that
  finds its slot taken reports ``CONFLICT`` and is retried at the next sync.
* ``before_read`` / ``after_append``: the app's hooks, throttled, off unless
  ``PARLAYPICKER_EXPOSURE_LEDGER_REMOTE=true``; they never raise.

It cannot submit wagers and never deletes or rewrites an event.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
from pathlib import Path
import time

PREFIX = "parlaypicker/exposure-ledger-v1/events/"
REMOTE_ENV = "PARLAYPICKER_EXPOSURE_LEDGER_REMOTE"
READ_EVERY = 300  # seconds between restores before a read
MAX_EVENTS = 100_000


def key(seq: int) -> str:
    return f"{PREFIX}{seq:06d}.json"


def _payload(event: dict) -> str:
    value = {k: v for k, v in event.items() if k != "ledger_event_id"}
    return json.dumps(value, sort_keys=True, allow_nan=False)


def remote_events(client) -> list[dict]:
    """The remote events in order, each verified against its own hash."""
    from core.exposure_ledger import digest
    out = []
    for seq in range(1, MAX_EVENTS + 1):
        try:
            raw = client.get_object(Key=key(seq))["Body"].read()
        except Exception:
            break
        value = json.loads(raw)
        event_id = value.pop("ledger_event_id", None)
        if event_id != digest(value):
            raise ValueError(f"remote ledger event {seq} fails its hash")
        out.append(dict(value, ledger_event_id=event_id))
    return out


def sync(path, client) -> dict:
    from core.exposure_ledger import connect, events
    local, remote = events(path), remote_events(client)
    lids, rids = [e["ledger_event_id"] for e in local], [e["ledger_event_id"] for e in remote]
    if lids[:len(rids)] == rids:
        pushed = 0
        for seq, event in enumerate(local[len(rids):], start=len(rids) + 1):
            body = json.dumps(dict(json.loads(_payload(event)), ledger_event_id=event["ledger_event_id"]),
                              sort_keys=True, allow_nan=False).encode()
            try:
                client.put_object(Key=key(seq), Body=body, IfNoneMatch="*")
            except Exception as exc:
                if type(exc).__name__ == "AlreadyExists":
                    return {"status": "CONFLICT", "pushed": pushed}
                raise
            pushed += 1
        return {"status": "PUSHED" if pushed else "IN_SYNC", "pushed": pushed, "events": len(local)}
    if rids[:len(lids)] == lids:
        missing = remote[len(lids):]
        with closing(connect(path)) as db, db:
            for event in missing:
                db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?)",
                           (event["ledger_event_id"], event["recorded_at"], _payload(event)))
        return {"status": "RESTORED", "restored": len(missing), "events": len(remote)}
    return {"status": "DIVERGED", "local": len(local), "remote": len(remote)}


def enabled(setting=None) -> bool:
    return str((setting or os.environ.get)(REMOTE_ENV, "") or "").strip().lower() == "true"


def _client():
    from app_core.evidence_drive import DriveStore
    from app_core.evidence_remote import settings
    folder = settings()[0]
    if not folder or not os.getenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "").strip():
        return None
    return DriveStore(folder)


def _run(path, *, setting=None, client=None, throttle=None, clock=time.time) -> dict:
    if not enabled(setting):
        return {"status": "OFF"}
    try:
        stamp = Path(path).with_name(Path(path).name + ".remote-synced")
        if throttle is not None:
            try:
                last = float(stamp.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                last = None
            if last is not None and Path(path).exists() and 0 <= clock() - last < throttle:
                return {"status": "RECENT"}
        client = client or _client()
        if client is None:
            return {"status": "MISSING_CREDENTIALS"}
        result = sync(path, client)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(str(clock()), encoding="utf-8")
        return result
    except Exception as exc:  # the ledger itself still works locally; report, never raise
        return {"status": "SYNC_FAILED", "error": type(exc).__name__}


def before_read(path, **kwargs) -> dict:
    """Restore from Drive before the app reads the ledger (at most every ``READ_EVERY`` seconds,
    and always when the local file is missing, e.g. after a reboot)."""
    return _run(path, throttle=READ_EVERY, **kwargs)


def after_append(path, **kwargs) -> dict:
    """Push the new event to Drive right after the owner appends it."""
    return _run(path, **kwargs)
