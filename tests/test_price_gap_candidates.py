"""Live price-gap candidates (owner 2026-11-10): research only, never a wager. SYNTHETIC quotes; no Drive."""
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json

import pytest
from streamlit.testing.v1 import AppTest

from app_core import price_gap_candidates as gaps
from app_core import price_watch as pw
from app_core.evidence_config import EvidenceStorageError
from app_core.evidence_drive import AlreadyExists
from tests.test_pinnacle_plan_selection import both

START = datetime(2030, 1, 1, 20, 0, tzinfo=timezone.utc)
NOW = START - timedelta(minutes=58)                      # two minutes after the newest capture


def _event(event_id, start=START):
    return {"event_id": event_id, "home_team": "H", "away_team": "A", "scheduled_start": start.isoformat(),
            "observed_at": (start - timedelta(hours=8)).isoformat()}


def _capture(event_id, minutes_before, novig_home, start=START, fanduel_home=-110):
    received = start - timedelta(minutes=minutes_before)
    stamp = (received - timedelta(seconds=30)).isoformat()
    return (both("pinnacle", -2.5, -110, -110, received.isoformat(), stamp, event_id)
            + both("Novig", -2.5, novig_home, -110, received.isoformat(), stamp, event_id)
            + both("FanDuel", -2.5, fanduel_home, -110, received.isoformat(), stamp, event_id)
            + both("OtherBook", -2.5, +120, -110, received.isoformat(), stamp, event_id))


def _scope():
    events = {name: _event(name) for name in ("open", "closed", "stale")}
    events["started"] = _event("started", START - timedelta(hours=2))
    events["far"] = _event("far", START + timedelta(hours=6))
    quotes = (_capture("open", 90, -110) + _capture("open", 60, +106)            # the gap is in the newest capture
              + _capture("closed", 90, +106) + _capture("closed", 60, -110)      # an earlier gap that has closed
              + _capture("stale", 90, +106)                                      # newest capture is 32 minutes old
              + _capture("started", 60, +106, START - timedelta(hours=2))
              + _capture("far", 60, +106, START + timedelta(hours=6)))
    return events, quotes


def test_a_candidate_is_a_stake_book_above_pinnacles_fair_price_in_the_newest_live_capture():
    events, quotes = _scope()
    rows = gaps.current("db", NOW, sports=("NCAAB",), read_scope=lambda path, sport, family: (
        (events, quotes, []) if family == "SPREAD" else ({}, [], [])))
    assert [(r["away_team"], r["sportsbook"], r["pick"]) for r in rows] == [("A", "Novig", "H -2.5")]
    row = rows[0]
    assert row["fair_probability"] == 0.5 and row["fair_american_odds"] == -100
    assert row["edge"] == pytest.approx(0.03) and row["captured_at"] == (START - timedelta(minutes=60)).isoformat()
    # only stake books; a closed gap, a stale capture, a started game and one outside the window are left out
    assert gaps.current("db", NOW, sports=(), read_scope=lambda *a: (events, quotes, [])) == []
    assert gaps._american(0.6) == -150 and gaps._american(0.4) == 150 and gaps._american(1.0) is None


class Drive:
    def __init__(self):
        self.objects = {}

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **kwargs):
        if Key in self.objects:
            raise AlreadyExists()
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise EvidenceStorageError(gaps.MISSING)
        return {"Body": BytesIO(self.objects[Key])}


def test_the_list_is_published_per_five_minute_slot_outside_the_evidence_and_read_newest_first():
    drive, at = Drive(), datetime(2026, 11, 10, 23, 7, 40, tzinfo=timezone.utc)
    first = gaps.publish(drive, "f", [{"pick": "H -2.5"}], now=at, sports=("NBA",))
    assert first == {"status": "PUBLISHED", "key": "parlaypicker/price-gap-candidates-v1/20261110T2305.json", "candidates": 1}
    assert not first["key"].startswith("parlaypicker/evidence-v1")
    assert gaps.publish(drive, "f", [], now=at + timedelta(minutes=1))["status"] == "ALREADY_PUBLISHED_THIS_SLOT"
    assert gaps.publish(drive, "f", [], now=at + timedelta(minutes=5))["status"] == "PUBLISHED"   # an empty list too
    found = gaps.latest(drive, "f", now=at + timedelta(minutes=9))
    assert found["candidates"] == [] and found["stale"] is False and found["notice"] == gaps.NOTICE
    old = gaps.latest(drive, "f", now=at + timedelta(minutes=30))
    assert old["stale"] is True and old["age_minutes"] == pytest.approx(25.0)
    assert gaps.latest(drive, "f", now=at + timedelta(hours=3)) is None
    drive.objects[gaps.key_for(at + timedelta(hours=5))] = json.dumps({"candidates": "x"}).encode()
    with pytest.raises(ValueError, match="price_gap_candidates_shape"):
        gaps.latest(drive, "f", now=at + timedelta(hours=5))


def test_the_price_watch_publishes_after_a_capture_and_a_failure_never_stops_it(monkeypatch):
    on = {pw.WATCH_ENV: "NBA"}.get
    clock = lambda: datetime(2026, 11, 10, 23, 7, tzinfo=timezone.utc)  # noqa: E731
    stored = {"capture": {"requested": True, "stored_quotes": 52}}
    assert pw.after_poll("db", Drive(), "f", {"capture": None}, setting=on, clock=clock) is None
    monkeypatch.setattr(gaps, "current", lambda path, now, *, sports: [{"pick": "H -2.5", "sports": list(sports)}])
    drive = Drive()
    assert pw.after_poll("db", drive, "f", stored, setting=on, clock=clock)["status"] == "PUBLISHED"
    assert json.loads(next(iter(drive.objects.values())))["sports"] == ["NBA"]

    def broken(*a, **k):
        raise RuntimeError("scope unreadable")
    monkeypatch.setattr(gaps, "current", broken)
    assert pw.after_poll("db", drive, "f", stored, setting=on, clock=clock) == {
        "status": "CANDIDATES_FAILED", "error": "RuntimeError"}


def test_the_shared_loop_publishes_before_the_backup(tmp_path, monkeypatch):
    import importlib
    loop = importlib.import_module("scripts.news_capture_loop")
    order = []

    class Watcher:
        def poll(self, path, **kwargs):
            return {"news": [], "capture": {"requested": True, "stored_quotes": 8}}
    monkeypatch.setattr(pw, "load_state", lambda client, folder: {})
    monkeypatch.setattr(pw, "save_state", lambda *a, **k: None)
    monkeypatch.setattr(pw, "after_poll", lambda path, client, folder, report, setting=None: order.append("publish") or {"status": "PUBLISHED"})
    ticks = iter([0, 400])
    loop.run(5, 5, root=tmp_path, client=None, folder="f", key="k", request_get=None, setting={pw.WATCH_ENV: "NBA"}.get,
             sync=lambda *a: order.append("sync"), clock=lambda: next(ticks), sleep=lambda s: None,
             watcher=Watcher(), source=pw)
    assert order == ["sync", "publish", "sync"]                      # restore, then publish, then the backup


def test_the_panel_marks_the_candidates_unproven_and_flags_a_stale_list():
    payload = {"generated_at": "2026-11-10T23:05:00+00:00", "min_edge": 0.01, "age_minutes": 22.0, "stale": True,
               "notice": gaps.NOTICE, "candidates": [dict(sport="NBA", market_family="SPREAD", away_team="A", home_team="H",
                                                         scheduled_start="2026-11-11T00:10:00+00:00", sportsbook="Novig",
                                                         pick="H -2.5", american_odds=106, fair_american_odds=-100,
                                                         fair_probability=0.5, edge=0.03,
                                                         captured_at="2026-11-10T23:05:00+00:00")]}
    script = ("import json\nimport app.ui.price_gap_candidates as panel\n"
              f"panel._from_drive = lambda: json.loads({json.dumps(json.dumps(payload))})\n"
              "panel.render_price_gap_candidates()\n")
    at = AppTest.from_string(script)
    at.run(timeout=30)
    assert not at.exception and len(at.dataframe) == 0
    assert any("Research only" in w.value and "Not approved to bet" in w.value for w in at.warning)
    next(b for b in at.button if b.key == "price_gap_candidates_load").click().run(timeout=30)
    assert not at.exception and len(at.dataframe) == 1
    assert any("more than 15 minutes old" in e.value for e in at.error)
    assert any("1 candidate(s) at 1% or more" in m.value for m in at.markdown)
    from pathlib import Path
    assert "render_price_gap_candidates()" in (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
