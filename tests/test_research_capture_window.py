"""Research capture window and batching (roadmap D16). SYNTHETIC provider rows; no network."""
from datetime import timedelta

from app_core import odds_research_adapter as ora


def test_capture_window_defaults_to_two_hours_and_is_bounded():
    assert ora.capture_hours(lambda *_: "") == 2.0
    assert ora.capture_hours(lambda *_: "18") == 18.0
    assert ora.capture_hours(lambda *_: "72") == 24.0
    assert ora.capture_hours(lambda *_: "0.5") == 2.0
    assert ora.capture_hours(lambda *_: "abc") == 2.0


def test_a_long_window_captures_a_full_slate_in_batches(monkeypatch, tmp_path):
    now = ora.utcnow()
    upcoming = {f"g{i:02d}": {"id": f"g{i:02d}", "commence_time": (now + timedelta(hours=1.05 + i / 10)).isoformat()}
                for i in range(60)}
    upcoming["late"] = {"id": "late", "commence_time": (now + timedelta(hours=30)).isoformat()}
    calls = []

    def fake_fetch(sport, endpoint, key, request_get, *, params=None):
        calls.append((endpoint, params))
        return {} if endpoint == "participants" else []

    monkeypatch.setattr(ora, "fetch", fake_fetch)
    monkeypatch.setattr(ora, "participant_ids", lambda response: {})
    monkeypatch.setenv(ora.CAPTURE_HOURS_ENV, "18")
    adapter = ora.OddsResearchAdapter("NCAAB")
    result = adapter.capture_pregame(tmp_path / "SYNTHETIC.sqlite3", {}, "key", None, lambda: None,
                                     upcoming=upcoming)
    odds = [params["eventIds"].split(",") for endpoint, params in calls if endpoint == "odds"]
    assert result["due"] == 60 and result["capture_window_hours"] == 18.0
    assert [len(batch) for batch in odds] == [24, 24, 12]
    assert "late" not in sum(odds, [])
    assert result["deferred_due_events"] == 0

    monkeypatch.delenv(ora.CAPTURE_HOURS_ENV)
    calls.clear()
    result = adapter.capture_pregame(tmp_path / "SYNTHETIC-2.sqlite3", {}, "key", None, lambda: None,
                                     upcoming=upcoming)
    assert result["due"] == 10  # the default 2-hour window is unchanged
