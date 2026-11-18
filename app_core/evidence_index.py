"""A daily index of the evidence files in Drive, so a restart need not list the whole folder (owner 2026-11-18).

Every store shares one Drive folder. On 18 November it held 369,417 files, 1,087 of them evidence, and listing it took
551 of the 961 seconds of the first Refresh after the app was rebuilt. Whoever lists the whole folder for an evidence
restore now leaves a small file naming the evidence files it saw. The app's next restore reads that file and asks
Drive only for the files created since.

Why that is still every evidence file:

* an index comes from a complete listing, and says when that listing began;
* the second listing asks for every file created after that moment less ``MARGIN_SECONDS``, so a file is in one or the
  other unless it took longer than the margin to appear in listings;
* each time an index is written, the same two steps are run from the index before it and compared with the complete
  listing just made: every file the complete listing saw created in that span, of any store, must be in the second
  listing, and every evidence file must be in the earlier index or the second listing. The result is stored in the new
  index, and a restore takes the short route only from an index whose comparison found nothing missing. A wrong
  assumption about Drive leaves the short route switched off. It cannot leave a gap.

Each file an index names is still downloaded and checked exactly as before, and any failure on the short route falls
back to the complete listing. Nothing in Drive is updated or deleted: one index a day, created once.

What the short route does not see until the next index: a file somebody moved, trashed or restored by hand in Drive.
``PARLAYPICKER_EVIDENCE_INDEX=0`` switches all of this off.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import re
import time

ENV = "PARLAYPICKER_EVIDENCE_INDEX"
KIND = "evidence_inventory_index"
MARGIN_SECONDS = 6 * 3600   # how far before the indexed listing began the second listing reaches back
GRACE_SECONDS = 2           # the comparison leaves out files created this close to that boundary
READ_DAYS = 3               # a restore starts from an index made today or in the three days before
COMPARE_DAYS = 7            # a new index is compared with one up to a week old
CLOCK_SECONDS = 300         # an index dated further ahead of this clock than this is not believed


@dataclass(frozen=True)
class Index:
    day: str
    listed_from: float
    covers: tuple
    files: tuple            # (name, Drive file id, SHA-256 or "")
    check: dict | None


@dataclass(frozen=True)
class Plan:
    """Today's index is still to be written; ``previous`` is the index the new listing will be compared with."""
    previous: Index | None

    @property
    def watch(self):
        return None if self.previous is None else self.previous.listed_from - MARGIN_SECONDS + GRACE_SECONDS


def enabled():
    return os.environ.get(ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def supported(client):
    return all(callable(getattr(client, name, None))
               for name in ("named_files", "created_since", "put_object", "storage_scope_hash"))


def key(prefix, day):
    return f"{prefix}-index/{day}.json"


def _day(moment):
    return datetime.fromtimestamp(moment, timezone.utc).strftime("%Y-%m-%d")


def _stamp(moment):
    return datetime.fromtimestamp(int(moment), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def encode(prefix, covers, listed_from, files, check):
    return json.dumps({"schema": 1, "kind": KIND, "prefix": prefix, "covers": list(covers),
                       "listed_from": _stamp(listed_from), "files": [list(item) for item in files], "check": check},
                      sort_keys=True, separators=(",", ":")).encode()


def decode(raw, prefix, day, tables, now):
    """One index file as an ``Index``; ``ValueError`` when it is not one this restore can rely on."""
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        raise ValueError("not JSON") from None
    if not isinstance(value, dict) or value.get("schema") != 1 or value.get("kind") != KIND or value.get("prefix") != prefix:
        raise ValueError("not an evidence index for this store")
    covers = value.get("covers")
    if not isinstance(covers, list) or not all(isinstance(item, str) for item in covers) or not set(tables) <= set(covers):
        raise ValueError("does not cover every evidence table")       # written before a table existed
    try:
        listed = datetime.strptime(value.get("listed_from"), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except (TypeError, ValueError):
        raise ValueError("no listing time") from None
    if _day(listed) != day or listed > now + CLOCK_SECONDS:
        raise ValueError("listing time does not fit its name or this clock")
    files, seen = value.get("files"), set()
    if not isinstance(files, list):
        raise ValueError("no file list")
    for item in files:
        if (not isinstance(item, list) or len(item) != 3 or not all(isinstance(part, str) for part in item)
                or not item[0].startswith(tuple(covers)) or not re.fullmatch(r"[A-Za-z0-9_-]+", item[1])
                or not re.fullmatch(r"(?:[a-f0-9]{64})?", item[2]) or (item[0], item[1]) in seen):
            raise ValueError("bad file entry")
        seen.add((item[0], item[1]))
    check = value.get("check")
    if check is not None and not isinstance(check, dict):
        raise ValueError("bad comparison record")
    return Index(day, listed, tuple(covers), tuple(tuple(item) for item in files), check)


def _recent(client, prefix, tables, now, days, first=0):
    """The usable index of each day from ``first`` to ``days`` days back, newest first. A day without one is passed over."""
    for back in range(first, days + 1):
        day = _day(now - back * 86400)
        found = []
        for raw in client.named_files(key(prefix, day)):
            try:
                found.append(decode(raw, prefix, day, tables, now))
            except ValueError:
                continue
        if found:
            # Two writers can create the same day's name at once. Either is a complete listing; take the later.
            yield max(found, key=lambda index: index.listed_from)


def passed(index):
    return isinstance(index.check, dict) and index.check.get("passed") is True


def found_missing(index):
    return isinstance(index.check, dict) and bool(index.check.get("missing") or index.check.get("evidence_missing"))


def choose(client, prefix, tables, now=None):
    """The index a restore may start from, and why. Never raises: no index means the complete listing."""
    now = time.time() if now is None else now
    reason = "no_index"
    try:
        for index in _recent(client, prefix, tables, now, READ_DAYS):
            if passed(index):
                return index, "index"
            if found_missing(index):
                return None, "comparison_found_missing"
            reason = "awaiting_comparison"       # an index nobody has yet compared: look at the day before
    except Exception as exc:
        return None, "unreadable:" + type(exc).__name__
    return None, reason


def inventory(client, index, tables, *, operation_id, namespace, ids=None):
    """What a complete listing would hold for the evidence tables: the index, and every file created since."""
    from app_core.evidence_drive import DriveInventory
    newer, stats = client.created_since(index.listed_from - MARGIN_SECONDS, ids=ids)
    files = [{"id": file_id, "name": name, **({"sha256Checksum": checksum} if checksum else {})}
             for name, file_id, checksum in index.files if name.startswith(tuple(tables))]
    # A file in both keeps the fresher description: the later entry wins when the read groups them by id.
    files += [{name: item[name] for name in ("id", "name", "sha256Checksum") if name in item}
              for item in newer if item["name"].startswith(tuple(tables))]
    return DriveInventory(client.storage_scope_hash(), operation_id, namespace, tuple(files),
                          stats.get("listing_pages"), stats.get("metadata_items_seen"), covers=tuple(tables))


def plan(client, prefix, tables, now=None):
    """Before a complete listing: ``None`` when today already has an index, else what to compare with. Never raises."""
    now = time.time() if now is None else now
    try:
        if client.named_files(key(prefix, _day(now))):
            return None
        return Plan(next(_recent(client, prefix, tables, now, COMPARE_DAYS, first=1), None))
    except Exception:
        return None


def _compare(client, entries, listing, previous, ids):
    """Would the short route from ``previous`` have found everything this complete listing found?"""
    since = previous.listed_from - MARGIN_SECONDS
    record = {"previous": previous.day, "since": _stamp(since), "passed": False}
    if listing.recent_ids is None:
        return {**record, "reason": "creation_times_not_listed"}
    try:
        newer, _ = client.created_since(since, ids=ids)
    except Exception as exc:
        return {**record, "reason": "newer_listing_failed:" + type(exc).__name__}
    known = {(name, file_id) for name, file_id, _ in previous.files} | {(item["name"], item["id"]) for item in newer}
    missing = len(listing.recent_ids - {item["id"] for item in newer})
    evidence_missing = sum((name, file_id) not in known for name, file_id, _ in entries)
    record.update(expected=len(listing.recent_ids), listed=len(newer), missing=missing,
                  evidence_missing=evidence_missing, unknown=listing.recent_unknown)
    # Nothing created in the span proves nothing about the second listing, so it does not count as a pass.
    record["passed"] = bool(listing.recent_ids) and not missing and not evidence_missing and not listing.recent_unknown
    return record


def write(client, prefix, tables, listing, planned, *, ids=None):
    """After a restore from a complete listing: compare with the index before, then leave today's. Never raises."""
    try:
        if (getattr(listing, "listed_from", None) is None or not getattr(listing, "complete", False)
                or getattr(listing, "covers", None) is not None):
            return {"written": None, "reason": "not_a_complete_listing"}
        entries = set()
        for item in listing.files:
            if item["name"].startswith(tuple(tables)):
                checksum = item.get("sha256Checksum") or ""
                if not isinstance(checksum, str) or not re.fullmatch(r"(?:[a-f0-9]{64})?", checksum):
                    return {"written": None, "reason": "bad_checksum_in_listing"}
                entries.add((item["name"], item["id"], checksum))
        entries = sorted(entries)
        check = None if planned.previous is None else _compare(client, entries, listing, planned.previous, ids)
        day = _day(listing.listed_from)
        client.put_object(Key=key(prefix, day), Body=encode(prefix, tables, listing.listed_from, entries, check),
                          ContentType="application/json", IfNoneMatch="*")
        return {"written": day, "files": len(entries), "check": check}
    except Exception as exc:
        code = (getattr(exc, "response", None) or {}).get("Error", {}).get("Code")
        return {"written": None,
                "reason": "already_written" if code in {"PreconditionFailed", "412"} else "failed:" + type(exc).__name__}
