"""The closing-capture loop sees a run saved after it started (2026-11-19). SYNTHETIC Drive, clock and evidence.

The loop restores the evidence store once, then runs for hours. A Refresh made meanwhile saves a new run
with new candidate ids, and the loop read closes only for the picks it held at its start: the last run
before a game, the one the track record and TheOver's plan count, got none. ``evidence_remote.restore_newer``
merges what was saved since, from one listing of the files created after a moment; the loop asks before
each cycle. The Drive stand-in is the one the evidence index tests use.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app_core import evidence_remote as remote
from app_core.evidence_config import EvidenceStorageError
from test_evidence_index import DAY, HOUR, START, _add_other_stores, _add_run, _put, _sha, _stored, world  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("closing_capture_loop_refresh", ROOT / "scripts" / "closing_capture_loop.py")
loop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)
MARGIN = 15 * 60


def _runs(path):
    return [row[0] for row in _stored(path)["snapshots"]]


def _restored(world):
    """A job's store as it stands after its one restore, at the clock's present."""
    old = [_add_run(world, START - 3 * DAY), _add_run(world, START - DAY)]
    _add_other_stores(world, START - 3 * DAY, START)
    path = world.fresh()
    assert remote.restore(path) == 2
    return path, old


# ---- what was saved since ----

def test_a_run_saved_after_the_restore_is_merged_from_one_listing_of_newer_files(world):
    path, old = _restored(world)
    listings = world.drive.whole_folder_listings()
    world.clock.now = START + 2 * HOUR
    later = _add_run(world, START + HOUR)                            # the owner's Refresh, an hour into the job
    _add_other_stores(world, START, START + 2 * HOUR)
    asked = len(world.drive.asked)
    assert remote.restore_newer(path, START - MARGIN) == 1
    assert world.drive.whole_folder_listings() == listings          # the folder is not listed again
    assert [kind for kind, _ in world.drive.asked[asked:] if kind != "bytes"] == ["newer"]
    assert sorted(_runs(path)) == sorted(old + [later])
    fresh = world.fresh()
    assert remote.restore(fresh) == 3 and _stored(path) == _stored(fresh)      # exactly what a restore would hold


def test_asking_again_adds_nothing_and_downloads_nothing_twice(world):
    path, _ = _restored(world)
    world.clock.now = START + 2 * HOUR
    _add_run(world, START + HOUR)
    assert remote.restore_newer(path, START - MARGIN) == 1
    before, asked = _stored(path), len(world.drive.asked)
    assert remote.restore_newer(path, START - MARGIN) == 0
    assert _stored(path) == before
    assert not [name for kind, name in world.drive.asked[asked:] if kind == "bytes"]     # the cache holds them


def test_nothing_new_is_one_question_and_no_download(world):
    path, _ = _restored(world)
    world.clock.now = START + 2 * HOUR
    _add_other_stores(world, START, START + 2 * HOUR)               # other stores wrote; the evidence did not
    asked = len(world.drive.asked)
    assert remote.restore_newer(path, START - MARGIN) == 0
    assert [kind for kind, _ in world.drive.asked[asked:]] == ["newer"]


def test_a_run_the_restore_missed_because_drive_had_not_listed_it_yet_is_found_by_the_margin(world):
    _add_run(world, START - DAY)
    hidden = _add_run(world, START - 5 * 60, appears=20 * 60)       # saved five minutes before the job, unseen by its restore
    path = world.fresh()
    assert remote.restore(path) == 1 and hidden not in _runs(path)
    world.clock.now = START + HOUR
    assert remote.restore_newer(path, START - MARGIN) == 1 and hidden in _runs(path)
    assert loop.REFRESH_MARGIN_SECONDS == MARGIN


def test_a_damaged_new_file_is_refused_and_the_store_keeps_what_it_had(world):
    path, old = _restored(world)
    world.clock.now = START + 2 * HOUR
    _add_run(world, START + HOUR)
    newest = max((item for item in world.drive.files if "/snapshots/" in item["name"]), key=lambda item: item["created"])
    newest["content"] = b"{}"
    newest["sha256Checksum"] = _sha(b"{}")
    before = _stored(path)
    with pytest.raises(EvidenceStorageError):
        remote.restore_newer(path, START - MARGIN)
    assert _stored(path) == before and sorted(_runs(path)) == sorted(old)


def test_with_no_moment_no_remote_or_an_older_client_it_adds_nothing(world, monkeypatch, tmp_path):
    path, _ = _restored(world)
    world.clock.now = START + 2 * HOUR
    _add_run(world, START + HOUR)
    asked = len(world.drive.asked)
    assert remote.restore_newer(path) == 0                                        # no moment given
    older = SimpleNamespace(read_verified_prefixes=lambda **kwargs: pytest.fail("not asked"))
    assert remote.restore_newer(path, START - MARGIN, client=older) == 0          # cannot list by creation time
    monkeypatch.setattr(remote, "settings", lambda: ("", ""))
    assert remote.restore_newer(tmp_path / "local.sqlite3", START - MARGIN) == 0  # no remote store
    assert len(world.drive.asked) == asked


# ---- the loop asks before each cycle ----

def _clock():
    now = [0.0]
    return now, (lambda: now[0]), (lambda seconds: now.__setitem__(0, now[0] + seconds))


def test_the_loop_refreshes_before_every_capture_and_reports_what_it_added(capsys):
    now, clock, sleep = _clock()
    order, added = [], iter([0, 2, 0])

    def refresh(path):
        order.append("refresh")
        return next(added)

    def capture(path):
        order.append("capture")
        return {"verified": 0, "reason": "no_candidates_in_closing_window"}
    totals = loop.run("db", minutes=21, interval=10, capture=capture, restore=lambda path: None, backup=lambda path: None,
                      clock=clock, sleep=sleep, refresh=refresh)
    assert order == ["refresh", "capture"] * 3
    assert totals == {"cycles": 3, "verified": 0, "unavailable": 0, "backups": 0, "errors": 0, "runs_added": 2}
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line.get("runs_added") for line in lines] == [None, 2, None]


def test_a_failed_refresh_is_counted_and_the_cycle_still_reads_the_picks_it_holds(capsys):
    now, clock, sleep = _clock()
    captured = []

    def refresh(path):
        raise EvidenceStorageError("Drive listing was incomplete; restore cannot be verified")
    totals = loop.run("db", minutes=1, interval=10, capture=lambda path: captured.append(1) or {"verified": 1},
                      restore=lambda path: None, backup=lambda path: None, clock=clock, sleep=sleep, refresh=refresh)
    assert captured == [1] and totals["verified"] == 1 and totals["refresh_errors"] == 1 and totals["errors"] == 0
    assert '"refresh_error": "EvidenceStorageError"' in capsys.readouterr().out


def test_a_loop_given_its_own_restore_and_no_refresh_asks_drive_nothing(monkeypatch):
    now, clock, sleep = _clock()
    monkeypatch.setattr(remote, "restore_newer", lambda *args, **kwargs: pytest.fail("no refresh was asked for"))
    totals = loop.run("db", minutes=1, interval=10, capture=lambda path: {"verified": 0}, restore=lambda path: None,
                      backup=lambda path: None, clock=clock, sleep=sleep)
    assert totals == {"cycles": 1, "verified": 0, "unavailable": 0, "backups": 0, "errors": 0}


def test_the_real_loop_restores_once_then_merges_a_run_saved_meanwhile(world, monkeypatch):
    """The job's own wiring: its default restore, and the refresh it builds from the wall clock."""
    _add_run(world, START - DAY)
    path = world.fresh()
    monkeypatch.setattr(loop, "time", SimpleNamespace(time=world.clock))      # the loop's wall clock is the Drive's
    now, clock, _ = _clock()
    seen = []

    def capture(database):
        seen.append(_runs(database))
        return {"verified": 0, "reason": "no_candidates_in_closing_window"}

    def sleep(seconds):                     # between the first cycle and the second the owner refreshes
        now[0] += seconds
        world.clock.now = START + 10 * 60
        seen.append(_add_run(world, START + 5 * 60))
    totals = loop.run(path, minutes=11, interval=10, capture=capture, backup=lambda database: None, clock=clock, sleep=sleep)
    first, later, second = seen
    assert later not in first and later in second and totals["runs_added"] == 1
    assert world.drive.whole_folder_listings() == 1         # one restore; the rest are listings of newer files
