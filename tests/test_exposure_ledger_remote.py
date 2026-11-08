"""The exposure ledger's Drive mirror: survives a reboot, never rewrites, reports divergence. SYNTHETIC."""
from datetime import datetime, timezone
import json

import pytest

from app_core import exposure_ledger_remote as remote
from core.exposure_ledger import append, events, snapshot

NOW = datetime(2026, 11, 8, 20, tzinfo=timezone.utc)
CONFIG = dict(status="CONFIGURED", bankroll=1000.0, unit_value=10.0, currency="USD", total_cap=0.1,
              daily_cap=0.05, weekly_cap=0.1, game_cap=0.02, team_cap=0.02)


class AlreadyExists(Exception):
    pass


class Drive:
    def __init__(self):
        self.objects = {}

    def get_object(self, *, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Key, Body, IfNoneMatch):
        if Key in self.objects:
            raise AlreadyExists(Key)
        self.objects[Key] = Body


def _wager(bet_id):
    return dict(status="COMMITTED", bet_id=bet_id, source_snapshot_id="s", sportsbook="Novig", stake_dollars=5.0,
                legs=[dict(sport="NBA", game_id="g1", team_ids=["t1", "t2"], market="spread_home", selection="Home",
                           odds=-110, line=-2.5)])


def test_ledger_survives_a_reboot_with_its_open_exposure(tmp_path):
    drive, path = Drive(), tmp_path / "exposure.sqlite3"
    append(path, CONFIG, confirmed=True, now=NOW)
    append(path, _wager("bet-1"), confirmed=True, now=NOW)
    assert remote.sync(path, drive) == {"status": "PUSHED", "pushed": 2, "events": 2}
    assert sorted(drive.objects) == [remote.key(1), remote.key(2)]
    assert remote.sync(path, drive)["status"] == "IN_SYNC"
    # Reboot: the local file is gone. The restore brings back the bankroll and the open wager in order.
    rebooted = tmp_path / "rebooted" / "exposure.sqlite3"
    assert remote.sync(rebooted, drive) == {"status": "RESTORED", "restored": 2, "events": 2}
    assert [e["ledger_event_id"] for e in events(rebooted)] == [e["ledger_event_id"] for e in events(path)]
    assert snapshot(rebooted, now=NOW)["committed"]["total"] == pytest.approx(0.005)


def test_a_tampered_remote_event_fails_its_hash(tmp_path):
    drive, path = Drive(), tmp_path / "exposure.sqlite3"
    append(path, CONFIG, confirmed=True, now=NOW)
    remote.sync(path, drive)
    value = json.loads(drive.objects[remote.key(1)])
    value["bankroll"] = 1_000_000.0
    drive.objects[remote.key(1)] = json.dumps(value).encode()
    with pytest.raises(ValueError, match="hash"):
        remote.remote_events(drive)


def test_diverged_ledgers_are_left_alone_and_a_taken_slot_is_a_conflict(tmp_path):
    drive, a, b = Drive(), tmp_path / "a.sqlite3", tmp_path / "b.sqlite3"
    append(a, CONFIG, confirmed=True, now=NOW)
    remote.sync(a, drive)
    append(b, dict(CONFIG, bankroll=2000.0), confirmed=True, now=NOW)          # a different first event
    assert remote.sync(b, drive)["status"] == "DIVERGED"
    assert len(events(b)) == 1 and len(drive.objects) == 1                     # nothing changed on either side
    # Two app sessions append at once: the second push finds its slot taken.
    remote.sync(b := tmp_path / "c.sqlite3", drive)                            # c restores a's first event
    append(a, _wager("bet-a"), confirmed=True, now=NOW)
    append(b, _wager("bet-c"), confirmed=True, now=NOW)
    assert remote.sync(a, drive)["status"] == "PUSHED"
    assert remote.sync(b, drive)["status"] == "DIVERGED"


def test_app_hooks_are_off_by_default_throttled_and_never_raise(tmp_path):
    path = tmp_path / "exposure.sqlite3"
    assert remote.before_read(path) == {"status": "OFF"}
    on = {remote.REMOTE_ENV: "true"}.get
    drive = Drive()
    append(path, CONFIG, confirmed=True, now=NOW)
    now = [1000.0]
    clock = lambda: now[0]  # noqa: E731
    assert remote.after_append(path, setting=on, client=drive, clock=clock)["status"] == "PUSHED"
    now[0] = 1100.0
    assert remote.before_read(path, setting=on, client=drive, clock=clock)["status"] == "RECENT"    # 100 s later
    now[0] = 1500.0
    assert remote.before_read(path, setting=on, client=drive, clock=clock)["status"] == "IN_SYNC"   # 500 s later

    class Broken:
        def get_object(self, **kwargs):
            raise RuntimeError("drive down")
    assert remote.after_append(path, setting=on, client=Broken())["status"] == "SYNC_FAILED"   # reported, not raised
