"""The daily evidence index (owner 2026-11-18): a restart restores from it without listing the whole Drive folder.

SYNTHETIC Drive, clock and evidence throughout; no network. The stand-in below answers the same three questions the
code asks of Drive (every file, the files with one name, the files created after a date), a page at a time, and
can be told to misbehave the ways a real one might.
"""
from contextlib import closing
from datetime import datetime, timezone
from email import message_from_bytes
import hashlib
import json
import random
import time as real_time
from types import SimpleNamespace

import pytest
import requests

from app_core import evidence_drive, evidence_index
from app_core import evidence_remote as remote
from app_core import prediction_evidence as evidence
from app_core.evidence_config import EvidenceStorageError
from app_core.evidence_drive import API, DriveStore

PREFIX = "parlaypicker/evidence-v1"
HOUR, DAY = 3600, 86400
START = datetime(2031, 3, 10, 15, 0, tzinfo=timezone.utc).timestamp()       # a Monday afternoon, UTC
EVERY_FILE = "'folder' in parents and trashed = false"


def _sha(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode()).hexdigest()


class Response:
    def __init__(self, data=None, content=b""):
        self.data, self.content = data, content

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class Drive:
    """One Shared Drive folder. Files are created once and never change; listings come four to a page."""
    PAGE = 4

    def __init__(self, clock):
        self.clock, self.files, self.asked = clock, [], []
        self.incomplete_newer = False       # the "created after" listing reports itself incomplete
        self.reads_dates_late_by = 0        # seconds: a Drive that takes the date to be in another time zone
        self.unreadable = set()             # ids whose bytes can no longer be fetched

    def add(self, name, content, *, created=None, appears=0):
        """A file created at ``created`` that listings start to show ``appears`` seconds later."""
        created = self.clock() if created is None else created
        item = {"id": f"file{len(self.files) + 1}", "name": name, "content": content, "created": created,
                "appears": created + appears, "sha256Checksum": _sha(content)}
        self.files.append(item)
        return item

    def whole_folder_listings(self):
        return sum(kind == "every" for kind, _ in self.asked)

    def get(self, url, params, timeout):
        assert timeout > 0 and params.get("supportsAllDrives") == "true"
        if url == API + "/folder":
            return Response({"id": "folder", "driveId": "drive", "mimeType": "application/vnd.google-apps.folder",
                             "trashed": False})
        if params.get("alt") == "media":
            item = next(item for item in self.files if url.endswith("/" + item["id"]))
            self.asked.append(("bytes", item["name"]))
            if item["id"] in self.unreadable:
                error = requests.HTTPError("gone")
                error.response = SimpleNamespace(status_code=404)
                raise error
            return Response(content=item["content"])
        assert url == API and params["corpora"] == "drive" and params["driveId"] == "drive" and params["pageSize"] == 1000
        query, incomplete = params["q"], False
        assert query.startswith(EVERY_FILE), query
        rest = query[len(EVERY_FILE):]
        seen = [item for item in self.files if item["appears"] <= self.clock()]
        if rest.startswith(" and name = '") and rest.endswith("'"):
            name = rest[len(" and name = '"):-1]
            seen = [item for item in seen if item["name"] == name]
            kind = ("named", name)
        elif rest.startswith(" and createdTime > '") and rest.endswith("'"):
            stamp = rest[len(" and createdTime > '"):-1]
            after = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
            seen = [item for item in seen if item["created"] > after + self.reads_dates_late_by]
            kind, incomplete = ("newer", stamp), self.incomplete_newer
        else:
            assert rest == "", query
            kind = ("every", "")
        offset = int(params.get("pageToken", 0))
        if not offset:
            self.asked.append(kind)
        wanted = params["fields"].split("files(")[1].rstrip(")").split(",")
        page = []
        for item in seen[offset:offset + self.PAGE]:
            shown = {"id": item["id"], "name": item["name"], "sha256Checksum": item["sha256Checksum"],
                     "createdTime": datetime.fromtimestamp(item["created"], timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")}
            page.append({field: shown[field] for field in wanted})
        data = {"files": page, "incompleteSearch": incomplete}
        if offset + self.PAGE < len(seen):
            data["nextPageToken"] = str(offset + self.PAGE)
        return Response(data)

    def post(self, url, params, headers, data, timeout):
        assert url == "https://www.googleapis.com/upload/drive/v3/files" and params["uploadType"] == "multipart"
        message = message_from_bytes(("Content-Type: " + headers["Content-Type"] + "\r\n\r\n").encode() + data)
        parts = message.get_payload()
        metadata = json.loads(parts[0].get_payload(decode=True))
        assert metadata["parents"] == ["folder"]
        self.asked.append(("upload", metadata["name"]))
        return Response({"id": self.add(metadata["name"], parts[1].get_payload(decode=True))["id"]})


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def world(monkeypatch, tmp_path):
    for name in ("PARLAYPICKER_EVIDENCE_INDEX", "PARLAYPICKER_DRIVE_CACHE_DIR", "PARLAYPICKER_NETLIFY_SITE_ID",
                 "PARLAYPICKER_DRIVE_INVENTORY_REUSE_SECONDS", "PARLAYPICKER_SHARED_INVENTORY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PARLAYPICKER_DRIVE_FOLDER_ID", "folder")
    monkeypatch.setenv("PARLAYPICKER_DRIVE_PREFIX", PREFIX)
    monkeypatch.setenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "synthetic-principal")
    for name, value in (("_status", {}), ("_restored", set()), ("_verified", {})):
        monkeypatch.setattr(remote, name, value)
    clock = Clock(START)
    monkeypatch.setattr(evidence_index, "time", SimpleNamespace(time=clock))
    monkeypatch.setattr(evidence_drive, "time",
                        SimpleNamespace(time=clock, sleep=lambda seconds: None, monotonic=real_time.monotonic))
    drive = Drive(clock)
    store = DriveStore("folder", session=drive)
    monkeypatch.setattr(remote, "_client", lambda: store)
    paths = iter(tmp_path / f"store{number}" / "evidence.sqlite3" for number in range(1000))
    return SimpleNamespace(clock=clock, drive=drive, store=store, fresh=lambda: next(paths), runs=[0])


MANIFEST = '{"synthetic":true}'


def _add_run(world, created, appears=0):
    """One saved run as the app stores it: the run, its runtime row and a score, as three Drive files."""
    number = world.runs[0] = world.runs[0] + 1
    if number == 1:
        _put(world, "bundles", (_sha(MANIFEST), "2031-01-01T00:00:00Z", MANIFEST), created)
    run, parts, scores = f"run-{number}", (f"candidates-{number}", f"decisions-{number}", f"inputs-{number}"), f"scores-{number}"
    _put(world, "snapshots", (run, _sha(MANIFEST), "2031-03-01T12:00:00Z", *parts, _sha("\0".join(parts))), created, appears)
    _put(world, "snapshot_runtime", (run, "synthetic-process"), created, appears)
    _put(world, "score_revisions", (run, _sha(scores), "2031-03-02T00:00:00Z", scores), created, appears)
    return run


def _put(world, table, row, created, appears=0):
    return world.drive.add(remote._key(table, row), remote._encode(table, row), created=created, appears=appears)


def _add_other_stores(world, since, until, count=24):
    """Files of the other stores that share the folder, spread evenly over a span."""
    for number in range(count):
        world.drive.add(f"parlaypicker/nba-market-v1/quotes/{since:.0f}-{number}.json", b"{}",
                        created=since + (until - since) * (number + 0.5) / count)


def _stored(path):
    with closing(evidence.connect(path)) as db:
        return {table: sorted(db.execute(f"SELECT {','.join(columns)} FROM {table}").fetchall())
                for table, columns in remote.TABLES.items()}


def _index(world, day):
    """The index files Drive holds for a day, decoded as JSON."""
    return [json.loads(item["content"]) for item in world.drive.files if item["name"] == f"{PREFIX}-index/{day}.json"]


def _two_days(world):
    """Monday: old evidence, then a complete listing. Tuesday: more of everything, then another complete listing."""
    old = [_add_run(world, START - 3 * DAY), _add_run(world, START - 2 * DAY)]
    _add_other_stores(world, START - 3 * DAY, START)
    assert remote.restore(world.fresh()) == 2
    world.clock.now = START + 20 * HOUR
    newer = [_add_run(world, START + 2 * HOUR), _add_run(world, START + 12 * HOUR)]
    _add_other_stores(world, START, START + 20 * HOUR)
    assert remote.restore(world.fresh()) == 4
    return old + newer


def test_a_complete_listing_leaves_the_days_index_naming_only_the_evidence(world):
    runs = [_add_run(world, START - 3 * DAY), _add_run(world, START - DAY)]
    _add_other_stores(world, START - 3 * DAY, START)
    path = world.fresh()
    assert remote.restore(path) == 2 and world.drive.whole_folder_listings() == 1
    assert [row[0] for row in _stored(path)["snapshots"]] == runs
    (index,) = _index(world, "2031-03-10")
    assert index["schema"] == 1 and index["kind"] == "evidence_inventory_index" and index["prefix"] == PREFIX
    assert index["listed_from"] == "2031-03-10T15:00:00Z"
    assert index["covers"] == [f"{PREFIX}/{table}/" for table in remote.TABLES]
    evidence_files = sorted([item["name"], item["id"], item["sha256Checksum"]] for item in world.drive.files
                            if item["name"].startswith(PREFIX + "/"))
    assert index["files"] == evidence_files and len(evidence_files) == 7        # one bundle, two runs of three files
    assert index["check"] is None                                                # there was no index before it
    assert remote.inventory_status() == {"source": "complete_listing",
                                         "written": {"written": "2031-03-10", "files": 7, "check": None}}
    # The index sits beside the evidence tables, not in one: no restore reads it as evidence.
    assert not any(f"{PREFIX}-index/".startswith(covered) for covered in index["covers"])
    assert remote.restore(world.fresh()) == 2


def test_one_index_a_day_and_the_second_listing_costs_one_question(world):
    _add_run(world, START - DAY)
    remote.restore(world.fresh())
    before = len(world.drive.asked)
    world.clock.now += 3 * HOUR
    assert remote.restore(world.fresh()) == 1
    asked = world.drive.asked[before:]
    assert [kind for kind, _ in asked if kind != "bytes"] == ["named", "every"]      # is today's there? then the listing
    assert len(_index(world, "2031-03-10")) == 1 and "written" not in remote.inventory_status()


def test_an_index_nobody_has_compared_does_not_switch_the_short_route_on(world):
    _add_run(world, START - DAY)
    remote.restore(world.fresh())
    world.clock.now += HOUR
    before = world.drive.whole_folder_listings()
    assert remote.restore(world.fresh(), prefer_index=True) == 1
    assert world.drive.whole_folder_listings() == before + 1
    assert remote.inventory_status() == {"source": "complete_listing", "index": "awaiting_comparison"}


def test_the_next_days_listing_is_compared_with_the_index_before_it(world):
    _two_days(world)
    (monday,), (tuesday,) = _index(world, "2031-03-10"), _index(world, "2031-03-11")
    check = tuesday["check"]
    assert check["previous"] == "2031-03-10" and check["since"] == "2031-03-10T09:00:00Z"   # six hours before Monday's listing
    # Created after that moment, of every store: the last six hours of Monday's other-store files (6 of 24 over three
    # days come to 2), all 24 of Tuesday's, the six files of two runs, and Monday's index itself.
    newer = [item for item in world.drive.files if item["created"] > START - 6 * HOUR + 2
             and item["name"] != f"{PREFIX}-index/2031-03-11.json"]
    assert check["expected"] == len(newer) == 2 + 24 + 6 + 1
    assert (check["missing"], check["evidence_missing"], check["unknown"], check["passed"]) == (0, 0, 0, True)
    assert len(tuesday["files"]) == len(monday["files"]) + 6
    # The comparison is the second listing the short route itself would make.
    assert ("newer", "2031-03-10T09:00:00") in world.drive.asked


def test_the_short_route_restores_everything_without_listing_the_folder(world):
    runs = _two_days(world)
    world.clock.now = START + 25 * HOUR
    runs.append(_add_run(world, START + 22 * HOUR))                  # saved after Tuesday's index
    _add_other_stores(world, START + 20 * HOUR, START + 25 * HOUR)
    listings, before = world.drive.whole_folder_listings(), len(world.drive.asked)
    short = world.fresh()
    assert remote.restore(short, prefer_index=True) == 5
    assert world.drive.whole_folder_listings() == listings
    asked = [(kind, detail) for kind, detail in world.drive.asked[before:] if kind != "bytes" or "index" in detail]
    assert asked[0] == ("named", f"{PREFIX}-index/2031-03-11.json")                # today's index, by its name
    assert asked[-1] == ("newer", "2031-03-11T05:00:00")                           # then the files created since
    assert len(asked) == 3 and asked[1][0] == "bytes"                              # with the index's own bytes between
    assert remote.inventory_status() == {"source": "index", "index": "2031-03-11", "indexed_files": 13}
    complete = world.fresh()
    assert remote.restore(complete) == 5 and world.drive.whole_folder_listings() == listings + 1
    assert _stored(short) == _stored(complete)
    assert [row[0] for row in _stored(short)["snapshots"]] == sorted(runs)
    # Verified exactly as before: an incremental backup of the restored copy has nothing to upload.
    uploads = [detail for kind, detail in world.drive.asked if kind == "upload"]
    assert remote.sync(short, incremental=True)
    assert [detail for kind, detail in world.drive.asked if kind == "upload"] == uploads


def test_the_app_start_up_takes_the_short_route_and_an_explicit_restore_does_not(world, monkeypatch):
    _two_days(world)
    world.clock.now = START + 25 * HOUR
    listings = world.drive.whole_folder_listings()
    remote.restore_once(world.fresh())                              # what begin_run and iter_snapshots call
    assert world.drive.whole_folder_listings() == listings and remote.inventory_status()["source"] == "index"
    remote.restore(world.fresh())                                   # the daily job, and the app's own button
    assert world.drive.whole_folder_listings() == listings + 1
    remote.restore(world.fresh(), full_verification=True, prefer_index=True)
    assert world.drive.whole_folder_listings() == listings + 2      # a full verification never starts from an index
    asked = []
    monkeypatch.setattr(remote, "restore", lambda path, **options: asked.append(options))
    remote.restore_once(world.fresh())
    remote.restore_once(world.fresh(), full_verification=True)
    assert asked == [{"full_verification": False, "prefer_index": True}, {"full_verification": True, "prefer_index": False}]


def test_a_file_that_was_slow_to_appear_in_listings_is_still_restored(world):
    _add_run(world, START - 2 * DAY)
    _add_other_stores(world, START - DAY, START)
    assert remote.restore(world.fresh()) == 1                        # Monday
    world.clock.now = START + 20 * HOUR
    late = _add_run(world, START + 18 * HOUR, appears=3 * HOUR)      # saved two hours before Tuesday's listing, unseen by it
    _add_other_stores(world, START, START + 20 * HOUR)
    assert remote.restore(world.fresh()) == 1
    (tuesday,) = _index(world, "2031-03-11")
    assert tuesday["check"]["passed"] is True and len(tuesday["files"]) == 4     # the bundle and Monday's run only
    world.clock.now = START + 22 * HOUR                               # an hour after it appears
    listings, path = world.drive.whole_folder_listings(), world.fresh()
    # Not in the index, and created before the index's listing began: the six-hour reach is what finds it.
    assert remote.restore(path, prefer_index=True) == 2 and world.drive.whole_folder_listings() == listings
    assert late in [row[0] for row in _stored(path)["snapshots"]]


def test_a_file_slower_than_the_reach_is_caught_by_the_next_comparison(world):
    """The limit, stated: a file that took more than six hours to show in listings is missed by the short route
    until the next index. The next day's comparison sees that, and the short route is off until an index has it."""
    _add_run(world, START - 2 * DAY)
    _add_other_stores(world, START - DAY, START)
    remote.restore(world.fresh())                                    # Monday
    world.clock.now = START + 20 * HOUR
    _add_run(world, START + 13 * HOUR, appears=8 * HOUR)             # seven hours before Tuesday's listing, seen an hour after
    _add_other_stores(world, START, START + 20 * HOUR)
    remote.restore(world.fresh())                                    # Tuesday
    assert _index(world, "2031-03-11")[0]["check"]["passed"] is True
    world.clock.now = START + 22 * HOUR
    assert remote.restore(world.fresh(), prefer_index=True) == 1 and remote.restore(world.fresh()) == 2
    world.clock.now = START + 44 * HOUR                              # Wednesday
    _add_other_stores(world, START + 20 * HOUR, START + 44 * HOUR)
    remote.restore(world.fresh())
    check = _index(world, "2031-03-12")[0]["check"]
    assert (check["missing"], check["evidence_missing"], check["passed"]) == (0, 3, False)
    listings = world.drive.whole_folder_listings()
    assert remote.restore(world.fresh(), prefer_index=True) == 2 and world.drive.whole_folder_listings() == listings + 1
    assert remote.inventory_status()["index"] == "comparison_found_missing"
    world.clock.now = START + 68 * HOUR                              # Thursday: Wednesday's index has the file
    _add_other_stores(world, START + 44 * HOUR, START + 68 * HOUR)
    remote.restore(world.fresh())
    assert _index(world, "2031-03-13")[0]["check"]["passed"] is True
    listings = world.drive.whole_folder_listings()
    assert remote.restore(world.fresh(), prefer_index=True) == 2 and world.drive.whole_folder_listings() == listings


def test_a_drive_that_reads_the_date_differently_leaves_the_short_route_off(world):
    """The one thing that cannot be tried against the real Drive from a test. If Drive took the date to be in
    another time zone, the second listing would begin hours late. The daily comparison is what notices."""
    world.drive.reads_dates_late_by = 8 * HOUR
    runs = _two_days(world)
    check = _index(world, "2031-03-11")[0]["check"]
    assert check["passed"] is False and check["missing"] > 0 and check["evidence_missing"] == 3   # the run saved at 5 PM
    world.clock.now = START + 25 * HOUR
    listings, path = world.drive.whole_folder_listings(), world.fresh()
    assert remote.restore(path, prefer_index=True) == 4
    assert world.drive.whole_folder_listings() == listings + 1
    assert remote.inventory_status() == {"source": "complete_listing", "index": "comparison_found_missing"}
    assert [row[0] for row in _stored(path)["snapshots"]] == sorted(runs)


def test_a_quiet_span_proves_nothing_and_an_unread_creation_time_is_not_a_pass(world):
    _add_run(world, START - 2 * DAY)
    remote.restore(world.fresh())
    monday = next(item for item in world.drive.files if item["name"].startswith(f"{PREFIX}-index/"))
    monday["created"] = START - 7 * HOUR             # so that nothing at all was created in the span
    world.clock.now = START + 20 * HOUR
    remote.restore(world.fresh())
    check = _index(world, "2031-03-11")[0]["check"]
    assert (check["expected"], check["missing"], check["passed"]) == (0, 0, False)
    assert evidence_index.choose(world.store, PREFIX, tuple(f"{PREFIX}/{t}/" for t in remote.TABLES)) == (None, "awaiting_comparison")
    # A listing that could not say when its files were created cannot be compared at all.
    listing = world.store.discover_complete_inventory()
    assert listing.recent_ids is None and listing.listed_from == world.clock()
    previous = evidence_index.Index("2031-03-10", START, (), (), None)
    assert evidence_index._compare(world.store, [], listing, previous, None) == {
        "previous": "2031-03-10", "since": "2031-03-10T09:00:00Z", "passed": False, "reason": "creation_times_not_listed"}
    assert evidence_drive._created_epoch("2031-03-10T15:00:00.000Z") == START
    assert evidence_drive._created_epoch("2031-03-10T15:00:00") == START and evidence_drive._created_epoch("") is None
    assert evidence_drive._created_epoch("yesterday") is None and evidence_drive._created_epoch(None) is None


def _spoil(world, how):
    index = next(item for item in world.drive.files if item["name"] == f"{PREFIX}-index/2031-03-11.json")
    value = json.loads(index["content"])
    newest = max((item for item in world.drive.files if "/snapshots/" in item["name"]), key=lambda item: item["created"])
    if how == "an indexed file can no longer be read":
        world.drive.unreadable.add(value["files"][0][1])
    elif how == "an indexed file is not the bytes the index recorded":
        value["files"][0][2] = "0" * 64
    elif how == "the listing of newer files is incomplete":
        world.drive.incomplete_newer = True
    elif how == "the index is not JSON":
        value = None
    elif how == "the index was written before a table existed":
        value["covers"] = value["covers"][:-1]
        value["files"] = [item for item in value["files"] if item[0].startswith(tuple(value["covers"]))]
    elif how == "the index is dated ahead of the clock":
        value["listed_from"] = "2031-03-11T23:00:00Z"       # seven hours after this restart
    elif how == "the index is for another store":
        value["prefix"] = "parlaypicker/other-v1"
    elif how == "the index names a file outside the evidence tables":
        value["files"].append(["parlaypicker/nba-market-v1/quotes/a.json", "file1", ""])
    elif how == "a newer evidence file is damaged":
        newest["content"] = b"{}"
        newest["sha256Checksum"] = _sha(b"{}")
    else:
        raise AssertionError(how)
    index["content"] = b"not json" if value is None else json.dumps(value).encode()
    index["sha256Checksum"] = _sha(index["content"])


@pytest.mark.parametrize("how, reason", [
    ("an indexed file can no longer be read", "failed:HTTPError"),
    ("an indexed file is not the bytes the index recorded", "failed:EvidenceStorageError"),
    ("the listing of newer files is incomplete", "failed:EvidenceStorageError"),
    ("the index is not JSON", "no_index"),
    ("the index was written before a table existed", "no_index"),
    ("the index is dated ahead of the clock", "no_index"),
    ("the index is for another store", "no_index"),
    ("the index names a file outside the evidence tables", "no_index"),
])
def test_anything_wrong_on_the_short_route_falls_back_to_the_complete_listing(world, how, reason):
    runs = _two_days(world)
    world.clock.now = START + 25 * HOUR
    for item in world.drive.files:
        if item["name"] == f"{PREFIX}-index/2031-03-10.json":
            item["appears"] = float("inf")              # leave only Tuesday's index to rely on
    _spoil(world, how)
    listings, path = world.drive.whole_folder_listings(), world.fresh()
    if how == "an indexed file can no longer be read":
        # The complete listing still names the file, so the restore fails exactly as it did before there was an index.
        with pytest.raises(requests.HTTPError):
            remote.restore(path, prefer_index=True)
        assert world.drive.whole_folder_listings() == listings + 1 and not path.exists()
        return
    assert remote.restore(path, prefer_index=True) == 4
    assert world.drive.whole_folder_listings() == listings + 1
    assert remote.inventory_status()["source"] == "complete_listing" and remote.inventory_status()["index"] == reason
    assert [row[0] for row in _stored(path)["snapshots"]] == sorted(runs)


def test_damaged_evidence_is_refused_on_either_route_and_nothing_is_stored(world):
    _two_days(world)
    world.clock.now = START + 25 * HOUR
    _spoil(world, "a newer evidence file is damaged")
    listings, path = world.drive.whole_folder_listings(), world.fresh()
    with pytest.raises(EvidenceStorageError):
        remote.restore(path, prefer_index=True)
    assert world.drive.whole_folder_listings() == listings + 1 and not path.exists()


def test_an_inventory_from_the_index_cannot_answer_for_another_store(world):
    _two_days(world)
    tables = tuple(f"{PREFIX}/{table}/" for table in remote.TABLES)
    index, reason = evidence_index.choose(world.store, PREFIX, tables)
    assert reason == "index" and index.day == "2031-03-11"
    built = evidence_index.inventory(world.store, index, tables, operation_id="op", namespace="n")
    assert built.complete and built.covers == tables and built.listed_from is None
    assert set(world.store.read_verified_prefixes(Prefixes=list(tables), inventory=built)) == set(tables)
    for other in ("parlaypicker/nba-market-v1/", "parlaypicker/", PREFIX + "-index/"):
        with pytest.raises(EvidenceStorageError, match="does not cover"):
            world.store.read_verified_prefixes(Prefixes=[other], inventory=built)
    # And it is never written back as an index: only a complete listing is.
    assert evidence_index.write(world.store, PREFIX, tables, built, evidence_index.Plan(None)) == {
        "written": None, "reason": "not_a_complete_listing"}


def test_two_indexes_under_one_name_and_the_later_listing_is_used(world):
    _two_days(world)
    tuesday = next(item for item in world.drive.files if item["name"] == f"{PREFIX}-index/2031-03-11.json")
    value = json.loads(tuesday["content"])
    value["listed_from"] = "2031-03-11T11:30:00Z"                   # a second writer, half an hour later
    world.drive.add(tuesday["name"], json.dumps(value).encode(), created=START + 20.5 * HOUR)
    world.clock.now = START + 25 * HOUR
    tables = tuple(f"{PREFIX}/{table}/" for table in remote.TABLES)
    index, reason = evidence_index.choose(world.store, PREFIX, tables)
    assert reason == "index" and index.listed_from == START + 20.5 * HOUR
    listings = world.drive.whole_folder_listings()
    assert remote.restore(world.fresh(), prefer_index=True) == 4 and world.drive.whole_folder_listings() == listings
    assert ("newer", "2031-03-11T05:30:00") in world.drive.asked


def test_an_index_older_than_three_days_is_not_used_and_one_a_week_old_is_still_compared(world):
    _two_days(world)
    world.clock.now = START + 20 * HOUR + 3 * DAY                   # Friday: Tuesday's index is three days old
    listings = world.drive.whole_folder_listings()
    assert remote.restore(world.fresh(), prefer_index=True) == 4 and world.drive.whole_folder_listings() == listings
    world.clock.now += DAY                                          # Saturday: four
    assert remote.restore(world.fresh(), prefer_index=True) == 4 and world.drive.whole_folder_listings() == listings + 1
    assert remote.inventory_status()["index"] == "no_index"
    saturday = remote.inventory_status()["written"]
    assert saturday["written"] == "2031-03-15" and saturday["check"]["previous"] == "2031-03-11"
    assert saturday["check"]["passed"] is True                      # so the next restart is short again
    world.clock.now += HOUR
    assert remote.restore(world.fresh(), prefer_index=True) == 4 and world.drive.whole_folder_listings() == listings + 1


def test_the_switch_turns_all_of_it_off(world, monkeypatch):
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_INDEX", "0")
    assert not evidence_index.enabled()
    _add_run(world, START - DAY)
    assert remote.restore(world.fresh(), prefer_index=True) == 1
    assert [kind for kind, _ in world.drive.asked if kind != "bytes"] == ["every"]
    assert remote.inventory_status() == {"source": "complete_listing"}
    assert not any("-index/" in item["name"] for item in world.drive.files)
    for value in ("false", "No", " OFF "):
        monkeypatch.setenv("PARLAYPICKER_EVIDENCE_INDEX", value)
        assert not evidence_index.enabled()
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_INDEX", "1")
    assert evidence_index.enabled()


def test_a_client_without_the_new_questions_is_left_exactly_as_it_was(world, monkeypatch, tmp_path):
    class Older:
        """The shape of the stand-in clients older tests pass in: a listing and a bulk read, nothing more."""
        def __init__(self):
            self.kwargs = None

        def put_object(self, **kwargs):
            pytest.fail("no index is written through a client that cannot read one")

        def discover_complete_inventory(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(operation_id=kwargs["operation_id"])

        def read_verified_prefixes(self, *, Prefixes, **kwargs):
            return {prefix: [] for prefix in Prefixes}
    client = Older()
    assert not evidence_index.supported(client) and evidence_index.supported(world.store)
    assert remote.restore(tmp_path / "older.sqlite3", client=client, prefer_index=True) == 0
    assert set(client.kwargs) == {"operation_id", "namespace", "ids"}
    assert remote.inventory_status() == {"source": "complete_listing"}


def test_writing_the_index_never_fails_the_restore(world, monkeypatch):
    _add_run(world, START - DAY)
    monkeypatch.setattr(world.drive, "post", lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("down")))
    path = world.fresh()
    assert remote.restore(path) == 1 and len(_stored(path)["snapshots"]) == 1
    assert remote.inventory_status() == {"source": "complete_listing",
                                         "written": {"written": None, "reason": "failed:ConnectionError"}}
    assert not _index(world, "2031-03-10")


@pytest.mark.parametrize("seed", range(6))
def test_whatever_the_folder_holds_the_short_route_restores_what_the_complete_listing_does(world, seed):
    """Random histories: saved runs and other stores' files at random moments over five days, some slow to appear
    (never by more than the six-hour reach), a complete listing each morning, and a restart at a random later moment."""
    rng = random.Random(seed)
    events = sorted([("run", START + rng.uniform(-DAY, 4 * DAY), rng.choice([0, 0, 60, 5 * HOUR])) for _ in range(14)]
                    + [("other", START + rng.uniform(-DAY, 4 * DAY), 0) for _ in range(60)], key=lambda event: event[1])
    mornings = [START + day * DAY + rng.uniform(0, 3 * HOUR) for day in range(5)]
    restart = mornings[-1] + rng.uniform(60, 8 * HOUR)
    pending = list(events)

    def advance(moment):
        while pending and pending[0][1] <= moment:
            kind, created, appears = pending.pop(0)
            if kind == "run":
                _add_run(world, created, appears)
            else:
                world.drive.add(f"parlaypicker/nba-market-v1/quotes/{created:.3f}.json", b"{}", created=created)
        world.clock.now = moment

    for morning in mornings:
        advance(morning)
        remote.restore(world.fresh())
    assert [len(_index(world, f"2031-03-{10 + day}")) for day in range(5)] == [1] * 5
    assert all(_index(world, f"2031-03-{10 + day}")[0]["check"]["passed"] for day in range(1, 5))
    advance(restart)
    listings, short, complete = world.drive.whole_folder_listings(), world.fresh(), world.fresh()
    remote.restore(short, prefer_index=True)
    assert world.drive.whole_folder_listings() == listings and remote.inventory_status()["source"] == "index"
    remote.restore(complete)
    assert _stored(short) == _stored(complete)
    seen = {item["name"] for item in world.drive.files if item["appears"] <= restart and "/snapshots/" in item["name"]}
    assert len(_stored(short)["snapshots"]) == len(seen) > 0


def test_the_daily_job_reports_how_the_files_were_found(world, monkeypatch, capsys, tmp_path):
    import sys
    from scripts import bootstrap_evidence
    _add_run(world, START - DAY)
    monkeypatch.setattr(sys, "argv", ["bootstrap_evidence.py", "--database", str(tmp_path / "job" / "evidence.sqlite3"),
                                      "--restore", "--skip-validation"])
    assert bootstrap_evidence.main() == 0
    out = capsys.readouterr().out
    line = next(json.loads(text) for text in out.splitlines() if text.startswith('{"stage": "evidence_inventory"'))
    assert line["state"] == {"source": "complete_listing", "written": {"written": "2031-03-10", "files": 4, "check": None}}
    assert json.loads(out[out.index("{\n"):])["evidence_inventory"] == line["state"]
    # The job keeps the complete listing: it is what the index, and the comparison, are made from.
    assert world.drive.whole_folder_listings() == 1
