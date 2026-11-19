"""Say which evidence files Drive holds twice, when a restore stops on them (2026-11-19).

A restore refuses a name that two Drive files carry with different contents, and the refusal names neither. On the
night of 18 November that stopped every restore, and the job logs could not say which two files among 370,000.
This reads the listing the failed restore already holds and logs, for each evidence name with more than one file,
the table, each file's Drive id, creation time, size and checksum, and for the small tables which columns differ.

It prints a timestamp or an identifier that differs. Of a manifest, a payload or a saved run it prints only that it
differs. Nothing is written, moved or deleted, and whatever goes wrong here leaves the restore's own error to be
raised: this is a report, never a decision.
"""
import json
import logging

LARGE = frozenset({"snapshots"})                 # a saved run is megabytes: its copies are compared by checksum only
SHOWN = frozenset({"version", "frozen_at", "snapshot_id", "process_instance", "evidence_hash", "recorded_at",
                   "observation_id", "candidate_id", "plan_id", "sport", "generated_at", "payload_hash"})
MOST = 20                                        # names described in one report


def duplicates(inventory, prefixes):
    """Evidence names that more than one Drive file carries: ``{table: {name: [file, ...]}}``."""
    found = {table: {} for table in prefixes}
    for item in getattr(inventory, "files", None) or ():
        for table, prefix in prefixes.items():
            if item["name"].startswith(prefix):
                found[table].setdefault(item["name"], {})[item["id"]] = item
    return {table: {name: list(files.values()) for name, files in names.items() if len(files) > 1}
            for table, names in found.items()}


def _same(files):
    sums = [item.get("sha256Checksum") for item in files]
    return all(sums) and len(set(sums)) == 1


def _facts(client, item):
    fact = {"id": item["id"], "sha256": (item.get("sha256Checksum") or "")[:16] or None}
    try:
        seen = client.file_facts(item["id"])
        fact.update(created=seen.get("createdTime"), bytes=int(seen["size"]) if seen.get("size") is not None else None)
    except Exception as exc:
        fact["facts"] = "unavailable:" + type(exc).__name__
    return fact


def _columns(client, table, files, columns, decode):
    """Which columns differ between the copies of one small record, with the short values that say which is which."""
    rows = []
    for item in files:
        try:
            rows.append(decode(client._read_files([item]), table))
        except Exception as exc:
            return {"differs": ["unreadable:" + type(exc).__name__]}
    differs = [name for index, name in enumerate(columns) if len({row[index] for row in rows}) > 1]
    return {"differs": differs,
            "values": {name: [row[columns.index(name)] for row in rows] for name in differs if name in SHOWN}}


def describe(client, inventory, prefixes, tables, decode):
    """Log one ``EVIDENCE_CONFLICT`` line for each name whose copies differ, then one summary line. Never raises."""
    log = logging.getLogger(__name__)
    try:
        found = duplicates(inventory, prefixes)
        identical, different, described = {}, {}, 0
        for table, names in found.items():
            for name, files in sorted(names.items()):
                if _same(files):
                    identical[table] = identical.get(table, 0) + 1
                    continue
                different[table] = different.get(table, 0) + 1
                if described >= MOST:
                    continue
                described += 1
                record = {"table": table, "name": name, "files": [_facts(client, item) for item in files]}
                record.update({"differs": ["not read: a saved run"]} if table in LARGE
                              else _columns(client, table, files, tables[table], decode))
                log.warning("EVIDENCE_CONFLICT %s", json.dumps(record, sort_keys=True))
        log.warning("EVIDENCE_DUPLICATES %s", json.dumps(
            {"names_with_different_copies": different, "names_with_identical_copies": identical}, sort_keys=True))
    except Exception as exc:
        log.warning("EVIDENCE_DUPLICATES %s", json.dumps({"report": "failed:" + type(exc).__name__}))
