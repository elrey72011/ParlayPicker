"""Go-live A: a qualified market's own small store, published by decision capture, read by the app. SYNTHETIC."""
from datetime import datetime, timedelta, timezone
import gzip
import importlib
import io

from app_core import prospective_evidence as pe
from app_core import qualified_market_file as qmf
from tests.test_market_skill_validation import _game, _plan
from tests.test_prospective_evidence import _trained_market, at, clock  # noqa: F401  (clock is a fixture)

NOW = datetime(2026, 11, 8, 23, 7, tzinfo=timezone.utc)


class AlreadyExists(Exception):
    pass


class Drive:
    def __init__(self):
        self.objects = {}

    def get_object(self, *, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, *, Key, Body, IfNoneMatch):
        if Key in self.objects:
            raise AlreadyExists(Key)
        self.objects[Key] = Body


def _store(tmp_path, clock):  # noqa: F811
    path = tmp_path / "full.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    _plan(path)
    _game(path, clock, "validation", 11)
    _game(path, clock, "holdout", 20, close_stamp=19.5)
    clock(hours=26)
    return path


def test_the_slice_holds_only_its_market_and_evaluates_exactly_like_the_full_store(tmp_path, clock):  # noqa: F811
    path = _store(tmp_path, clock)
    target = tmp_path / "slice.sqlite3"
    counts = qmf.export(path, target, {("NBA", "SPREAD")})["NBA:SPREAD"]
    assert counts["prospective_validation_plan"] == 1 and counts["prospective_prediction"] >= 2
    assert qmf.export(path, tmp_path / "other.sqlite3", {("NBA", "TOTAL")})["NBA:TOTAL"]["prospective_prediction"] == 0
    # The authority chain recomputes the same evaluation on the slice as on the full store.
    assert pe.evaluate_validation_plan(target, "plan") == pe.evaluate_validation_plan(path, "plan")
    assert pe.deployment_state(target, "NBA", "SPREAD") == pe.deployment_state(path, "NBA", "SPREAD")


def test_publish_once_per_slot_and_the_app_reads_the_newest_slice(tmp_path, clock):  # noqa: F811
    path = _store(tmp_path, clock)
    drive = Drive()
    first = qmf.publish(path, drive, "NBA", "SPREAD", now=NOW)
    assert first["status"] == "PUBLISHED" and first["key"].endswith("NBA-SPREAD/2026-11-08T2300Z.sqlite.gz")
    assert qmf.publish(path, drive, "NBA", "SPREAD", now=NOW + timedelta(minutes=5))["status"] == "ALREADY_PUBLISHED"
    raw, name = qmf.fetch(drive, "NBA", "SPREAD", now=NOW + timedelta(minutes=50))     # three slots later
    assert name == first["key"] and raw == gzip.decompress(drive.objects[name])
    assert qmf.fetch(drive, "NBA", "SPREAD", now=NOW + timedelta(hours=4)) == (None, None)  # too old
    target = tmp_path / "app" / "qualified-markets.sqlite3"
    result = qmf.refresh(drive, {("NBA", "SPREAD"), ("NBA", "TOTAL")}, now=NOW, target=target)
    assert result["status"] == "REFRESHED" and result["missing"] == ["NBA:TOTAL"]
    assert pe.evaluate_validation_plan(target, "plan") == pe.evaluate_validation_plan(path, "plan")
    assert qmf.refresh(Drive(), {("NBA", "SPREAD")}, now=NOW, target=target)["status"] == "NO_SLICE"
    assert target.is_file()                                                  # the last good store is kept
    assert qmf.publish(path, object(), "NBA", "SPREAD", now=NOW)["status"] == "ERROR"   # reported, not raised


def test_decision_capture_publishes_only_markets_it_ran():
    loop = importlib.import_module("scripts.decision_capture_loop")
    calls = []
    report = {"markets": {"NCAAB SPREAD": {"capture": {}, "predictions": {}},
                          "NBA TOTAL": {"blocker": "NOT_QUALIFIED"}}}
    out = loop.publish_slices("store", "client", report,
                              publish=lambda path, client, sport, family: calls.append((sport, family)) or {"status": "PUBLISHED"})
    assert calls == [("NCAAB", "SPREAD")] and out == {"NCAAB SPREAD": {"status": "PUBLISHED"}}


def test_app_switch_is_off_by_default_and_the_bridge_reads_the_file_when_on(tmp_path, monkeypatch):
    from app_core import qualified_bridge as qb
    assert qmf.enabled({}.get) is False
    monkeypatch.setenv(qmf.MODE_ENV, "true")
    on = {qb.BRIDGE_ENV: "true"}.get
    seen = {}
    monkeypatch.setattr(qb, "refresh_market_files", lambda markets, **kw: seen.setdefault("markets", markets) and {"status": "X"})
    monkeypatch.setattr(qb, "refresh_store", lambda **kw: {"status": "FULL_STORE"})
    assert qb.refresh_if_enabled(on, activations=["NCAAB:SPREAD"], background=False) == {"status": "X"}
    assert seen["markets"] == ["NCAAB:SPREAD"]
    monkeypatch.delenv(qmf.MODE_ENV)
    assert qb.refresh_if_enabled(on, activations=["NCAAB:SPREAD"], background=False) == {"status": "FULL_STORE"}
