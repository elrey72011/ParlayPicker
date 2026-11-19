"""A restore stopped by two different copies of one evidence file says which files (2026-11-19).

SYNTHETIC Drive and evidence, as in tests/test_evidence_index.py. On the night of 18 November every restore stopped
with "Drive contains conflicting duplicate evidence names" and no log could say which two files of 370,000.
"""
from datetime import datetime, timezone
import json
import logging
from types import SimpleNamespace

import pytest

from app_core import evidence_conflicts
from app_core import evidence_remote as remote
from app_core.evidence_config import EvidenceStorageError
from app_core.evidence_drive import API
from test_evidence_index import DAY, MANIFEST, PREFIX, START, Response, _add_run, _sha, world  # noqa: F401  (world is a fixture)


@pytest.fixture
def drive(world, monkeypatch):
    """The synthetic Drive, also answering the question a report asks: when was this file created, how large is it?"""
    plain = world.drive.get

    def get(url, params, timeout):
        if url.startswith(API + "/") and url != API + "/folder" and params.get("alt") != "media":
            item = next(item for item in world.drive.files if url.endswith("/" + item["id"]))
            assert params["fields"] == "id,createdTime,size,trashed" and params["supportsAllDrives"] == "true"
            return Response({"id": item["id"], "size": str(len(item["content"])), "trashed": False,
                             "createdTime": datetime.fromtimestamp(item["created"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")})
        return plain(url, params, timeout)
    monkeypatch.setattr(world.drive, "get", get)
    return world


def _file(world, table):
    return next(item for item in world.drive.files if f"/{table}/" in item["name"])


def _again(world, table, change, created=START - 60):
    """A second Drive file under the same name as the first ``table`` record, with ``change`` applied to its row."""
    first = _file(world, table)
    row = list(remote._decode(first["content"], table))
    change(row)
    return first, world.drive.add(first["name"], remote._encode(table, tuple(row)), created=created)


def _lines(caplog, kind):
    return [json.loads(record.getMessage().split(" ", 1)[1]) for record in caplog.records
            if record.getMessage().startswith(kind + " ")]


def _fact(item):
    return {"id": item["id"], "sha256": item["sha256Checksum"][:16], "bytes": len(item["content"]),
            "created": datetime.fromtimestamp(item["created"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")}


def test_a_restore_stopped_by_two_different_copies_says_which(drive, caplog):
    _add_run(drive, START - DAY)
    # The same bundle registered by a second writer a moment later: one name, two freeze times.
    first, second = _again(drive, "bundles", lambda row: row.__setitem__(1, "2031-03-10T14:59:00Z"))
    path = drive.fresh()
    with caplog.at_level(logging.WARNING), pytest.raises(EvidenceStorageError, match="conflicting duplicate evidence names"):
        remote.restore(path)
    assert not path.exists()                                        # refused before any database, as before
    assert _lines(caplog, "EVIDENCE_CONFLICT") == [{
        "table": "bundles", "name": first["name"], "files": [_fact(first), _fact(second)],
        "differs": ["frozen_at"], "values": {"frozen_at": ["2031-01-01T00:00:00Z", "2031-03-10T14:59:00Z"]}}]
    assert _lines(caplog, "EVIDENCE_DUPLICATES") == [{"names_with_different_copies": {"bundles": 1},
                                                      "names_with_identical_copies": {}}]
    assert first["name"].startswith(f"{PREFIX}/bundles/") and first["id"] != second["id"]
    assert MANIFEST not in caplog.text                              # what differs is named; a manifest is never printed


def test_identical_copies_are_no_conflict_and_nothing_is_reported(drive, caplog):
    _add_run(drive, START - DAY)
    first = _file(drive, "bundles")
    drive.drive.add(first["name"], first["content"], created=START - 60)      # a harmless retry: the same bytes twice
    with caplog.at_level(logging.WARNING):
        assert remote.restore(drive.fresh()) == 1
    assert not _lines(caplog, "EVIDENCE_CONFLICT") and not _lines(caplog, "EVIDENCE_DUPLICATES")
    # Asked directly, the report counts it as identical and has nothing to describe.
    listing = drive.store.discover_complete_inventory()
    prefixes = {table: f"{PREFIX}/{table}/" for table in remote.TABLES}
    found = evidence_conflicts.duplicates(listing, prefixes)
    assert {table: len(names) for table, names in found.items() if names} == {"bundles": 1}
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        evidence_conflicts.describe(drive.store, listing, prefixes, remote.TABLES, remote._decode)
    assert _lines(caplog, "EVIDENCE_DUPLICATES") == [{"names_with_different_copies": {},
                                                      "names_with_identical_copies": {"bundles": 1}}]


def test_every_conflict_is_listed_and_a_saved_run_is_compared_by_checksum_only(drive, caplog):
    _add_run(drive, START - DAY)

    def other_run(row):
        row[3:7] = ["other-candidates", row[4], row[5], _sha("\0".join(["other-candidates", row[4], row[5]]))]
    run, run_again = _again(drive, "snapshots", other_run)
    score, score_again = _again(drive, "score_revisions", lambda row: row.__setitem__(2, "2031-03-05T00:00:00Z"))
    with caplog.at_level(logging.WARNING), pytest.raises(EvidenceStorageError):
        remote.restore(drive.fresh())
    assert _lines(caplog, "EVIDENCE_CONFLICT") == [
        {"table": "snapshots", "name": run["name"], "files": [_fact(run), _fact(run_again)],
         "differs": ["not read: a saved run"]},
        {"table": "score_revisions", "name": score["name"], "files": [_fact(score), _fact(score_again)],
         "differs": ["recorded_at"], "values": {"recorded_at": ["2031-03-02T00:00:00Z", "2031-03-05T00:00:00Z"]}}]
    assert _lines(caplog, "EVIDENCE_DUPLICATES") == [{"names_with_different_copies": {"snapshots": 1, "score_revisions": 1},
                                                      "names_with_identical_copies": {}}]
    assert "other-candidates" not in caplog.text and "scores-1" not in caplog.text


def test_only_so_many_names_are_described_and_all_are_counted(drive, caplog, monkeypatch):
    monkeypatch.setattr(evidence_conflicts, "MOST", 1)
    _add_run(drive, START - DAY)
    _again(drive, "bundles", lambda row: row.__setitem__(1, "2031-03-10T14:59:00Z"))
    _again(drive, "snapshot_runtime", lambda row: row.__setitem__(1, "another-process"))
    with caplog.at_level(logging.WARNING), pytest.raises(EvidenceStorageError):
        remote.restore(drive.fresh())
    assert [line["table"] for line in _lines(caplog, "EVIDENCE_CONFLICT")] == ["bundles"]
    assert _lines(caplog, "EVIDENCE_DUPLICATES")[0]["names_with_different_copies"] == {"bundles": 1, "snapshot_runtime": 1}


def test_the_report_never_takes_the_place_of_the_refusal(world, caplog):
    """``world`` here is the plain synthetic Drive, which cannot answer the report's question about a file."""
    _add_run(world, START - DAY)
    first, second = _again(world, "bundles", lambda row: row.__setitem__(1, "2031-03-10T14:59:00Z"))
    world.drive.unreadable.add(second["id"])                         # and one copy can no longer be fetched
    with caplog.at_level(logging.WARNING), pytest.raises(Exception) as refused:
        remote.restore(world.fresh())
    assert type(refused.value).__name__ in {"EvidenceStorageError", "HTTPError"}
    if isinstance(refused.value, EvidenceStorageError):
        (line,) = _lines(caplog, "EVIDENCE_CONFLICT")
        assert [fact["id"] for fact in line["files"]] == [first["id"], second["id"]]
        assert all(fact["facts"] == "unavailable:AssertionError" for fact in line["files"])
        assert line["differs"] == ["unreadable:HTTPError"]
    # A listing that is not a listing, and a client that cannot be asked anything: one line, no exception.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        evidence_conflicts.describe(object(), SimpleNamespace(operation_id="op"), {"bundles": "p/bundles/"}, remote.TABLES, remote._decode)
        evidence_conflicts.describe(object(), SimpleNamespace(files=[{"name": "p/bundles/a", "id": "1"}, {"name": "p/bundles/a", "id": "2"}]),
                                    {"bundles": "p/bundles/"}, remote.TABLES, remote._decode)
        evidence_conflicts.describe(None, None, None, None, None)
    summaries = _lines(caplog, "EVIDENCE_DUPLICATES")
    assert summaries[0] == {"names_with_different_copies": {}, "names_with_identical_copies": {}}
    assert summaries[1] == {"names_with_different_copies": {"bundles": 1}, "names_with_identical_copies": {}}
    assert summaries[2] == {"report": "failed:TypeError"}
    (line,) = _lines(caplog, "EVIDENCE_CONFLICT")
    assert line["files"] == [{"id": "1", "sha256": None, "facts": "unavailable:AttributeError"},
                             {"id": "2", "sha256": None, "facts": "unavailable:AttributeError"}]
    assert line["differs"] == ["unreadable:AttributeError"]


def test_a_refusal_for_any_other_reason_reports_no_duplicates(drive, caplog):
    _add_run(drive, START - DAY)
    damaged = _file(drive, "snapshot_runtime")
    damaged["sha256Checksum"] = "0" * 64                              # Drive's checksum no longer matches the bytes
    with caplog.at_level(logging.WARNING), pytest.raises(EvidenceStorageError, match="checksum changed"):
        remote.restore(drive.fresh())
    assert not _lines(caplog, "EVIDENCE_CONFLICT")
    assert _lines(caplog, "EVIDENCE_DUPLICATES") == [{"names_with_different_copies": {}, "names_with_identical_copies": {}}]
