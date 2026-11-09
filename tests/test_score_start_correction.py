"""A score whose provider start moved later is graded; one that moved earlier is not (owner 2026-11-09). SYNTHETIC."""
from datetime import datetime, timedelta, timezone

import pytest

from app_core import odds_research_adapter as odds
from app_core import prospective_sport_adapters as registry
from tests.test_six_sport_adapters import fixture_payload, provider_for


@pytest.mark.parametrize("moved, graded, reason", [
    (timedelta(minutes=60), 1, None),                                   # tip-off corrected an hour later
    (timedelta(minutes=10), 1, None),                                   # within the old tolerance
    (timedelta(hours=13), 0, "score_event_identity_changed"),           # a different day: not the same game
    (-timedelta(minutes=60), 0, "score_event_identity_changed"),        # earlier: captured prices may be in-game
])
def test_a_later_corrected_start_is_still_graded(tmp_path, monkeypatch, moved, graded, reason):
    now = datetime.now(timezone.utc).replace(microsecond=0)
    clock = [now]
    monkeypatch.setattr(odds, "utcnow", lambda: clock[0])
    event, participants, odds_event, score = fixture_payload("NCAAB", now)
    get, _ = provider_for(event, participants, odds_event, None)
    adapter = registry.get_adapter("NCAAB")
    path = tmp_path / adapter.path_name
    adapter.capture_pregame(path, {}, "key", get, lambda: None)
    start = datetime.fromisoformat(event["commence_time"]) + moved
    corrected = dict(score, commence_time=start.isoformat(), last_update=(start + timedelta(hours=2)).isoformat())
    clock[0] = max(start, datetime.fromisoformat(event["commence_time"])) + timedelta(hours=3)   # due for a score check
    get, _ = provider_for(event, participants, odds_event, corrected)
    result = adapter.grade(path, "key", get, lambda: None, {})
    assert result["graded"] == graded
    assert result["score_rejections"].get("abc123") == reason
