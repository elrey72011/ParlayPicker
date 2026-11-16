"""Opt-in listing reuse and persistent cache for scheduler evidence restores. Network blocked."""
import pytest

from app_core import evidence_drive
from app_core.evidence_drive import DriveStore
from scripts.benchmark_drive_history_loading import SyntheticDrive, blocked_network

PREFIX = "parlaypicker/public-history-v1/site-1234/scores/"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv(evidence_drive.DRIVE_CACHE_ENV, raising=False)
    monkeypatch.delenv(evidence_drive.DRIVE_INVENTORY_REUSE_ENV, raising=False)
    with blocked_network():
        yield


def _drive():
    session = SyntheticDrive(count=50)
    return session, DriveStore("folder", session=session)


def test_without_the_setting_every_restore_lists_fresh():
    session, drive = _drive()
    drive.read_objects(Prefix=PREFIX)
    drive.read_objects(Prefix=PREFIX)
    assert session.listings == 2


def test_restores_reuse_one_listing_within_the_window(monkeypatch):
    monkeypatch.setenv(evidence_drive.DRIVE_INVENTORY_REUSE_ENV, "900")
    session, drive = _drive()
    first = drive.read_objects(Prefix=PREFIX)
    second = drive.read_objects(Prefix="parlaypicker/public-history-v1/site-1234/imports/")
    assert session.listings == 1 and len(first) == 1 and len(second) == 1
    # Lock and authority reads call discover_complete_inventory directly: always fresh.
    drive.discover_complete_inventory(namespace="lock", coalesce=False)
    assert session.listings == 2


def test_the_window_expires_and_a_new_object_appears(monkeypatch):
    monkeypatch.setenv(evidence_drive.DRIVE_INVENTORY_REUSE_ENV, "900")
    clock = [1000.0]
    monkeypatch.setattr(evidence_drive.time, "monotonic", lambda: clock[0])
    session, drive = _drive()
    assert len(drive.read_objects(Prefix=PREFIX)) == 1
    session.add(PREFIX + "two.json", {"recorded_at": "2026-10-04T16:00:00Z", "scores": []})
    assert len(drive.read_objects(Prefix=PREFIX)) == 1      # reused listing, within the window
    clock[0] += 901
    assert len(drive.read_objects(Prefix=PREFIX)) == 2      # fresh listing after the window
    assert session.listings == 2


def test_own_uploads_are_visible_through_a_reused_listing(monkeypatch):
    monkeypatch.setenv(evidence_drive.DRIVE_INVENTORY_REUSE_ENV, "900")
    session, drive = _drive()
    drive.read_objects(Prefix=PREFIX)
    item = session.add(PREFIX + "mine.json", {"recorded_at": "2026-10-05T16:00:00Z", "scores": []})
    drive.created_ids[item["name"]] = item["id"]            # what put_object records
    names = [name for name, _ in drive.read_objects(Prefix=PREFIX)]
    assert item["name"] in names and session.listings == 1


def test_callers_cache_moves_to_the_kept_cache_and_still_verifies(monkeypatch, tmp_path):
    kept = tmp_path / "kept"
    monkeypatch.setenv(evidence_drive.DRIVE_CACHE_ENV, str(kept))
    session, drive = _drive()
    temporary = tmp_path / "evidence" / "db.remote-cache"
    drive.read_cached_objects(Prefix=PREFIX, cache_dir=temporary)
    assert not temporary.exists() and any(kept.rglob("*"))
    drive.read_cached_objects(Prefix=PREFIX, cache_dir=temporary)
    assert session.downloads == 1                           # second read served from the kept cache
    for path in kept.rglob("*"):
        if path.is_file():
            path.write_bytes(b"tampered")
    drive.read_cached_objects(Prefix=PREFIX, cache_dir=temporary)
    assert session.downloads == 2                           # checksum mismatch forces a download


# ---- one listing has to last a whole research cycle (owner 2026-11-16) ----

def test_a_listing_taken_sixteen_minutes_ago_is_reused_at_the_workflows_setting(monkeypatch):
    """With some 340,000 objects a listing takes 7 to 9 minutes. At 900 seconds the later sports of a
    cycle listed the whole folder again; the research workflows now set an hour."""
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    loop = (root / "research-loop.yml").read_text(encoding="utf-8")
    scheduler = (root / "research-scheduler.yml").read_text(encoding="utf-8")
    line = 'PARLAYPICKER_DRIVE_INVENTORY_REUSE_SECONDS: "3600"'
    assert loop.count(line) == 1 and scheduler.count(line) == 1
    assert yaml.safe_load(loop)["jobs"]["research-loop"]["env"]["PARLAYPICKER_DRIVE_INVENTORY_REUSE_SECONDS"] == "3600"
    clock = [1000.0]
    monkeypatch.setattr(evidence_drive.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv(evidence_drive.DRIVE_INVENTORY_REUSE_ENV, "3600")
    session, drive = _drive()
    drive.read_objects(Prefix=PREFIX)
    clock[0] += 16 * 60                                     # the NCAAB restore of 15 November, after NFL, NCAAF and NBA
    drive.read_objects(Prefix="parlaypicker/public-history-v1/site-1234/imports/")
    clock[0] += 16 * 60                                     # and the NHL restore after it
    drive.read_objects(Prefix=PREFIX)
    assert session.listings == 1                            # three listings at 900 seconds
    drive.discover_complete_inventory(namespace="lock", coalesce=False)
    assert session.listings == 2                            # a lock read still lists for itself
    clock[0] += 3601
    drive.read_objects(Prefix=PREFIX)
    assert session.listings == 3                            # and the window still ends
