"""Credit batching for the research loop (owner 2026-11-06). SYNTHETIC provider; no network."""
from datetime import timedelta

import pytest

from app_core import odds_research_adapter as ora


@pytest.fixture
def provider(monkeypatch):
    calls = []

    def fake_fetch(sport, endpoint, key, request_get, *, params=None):
        calls.append(endpoint)
        return {} if endpoint == "participants" else []
    monkeypatch.setattr(ora, "fetch", fake_fetch)
    monkeypatch.setattr(ora, "participant_ids", lambda response: {})
    monkeypatch.setenv(ora.CAPTURE_HOURS_ENV, "18")
    return calls


def _upcoming(now, hours):
    return {f"g{i}": {"id": f"g{i}", "commence_time": (now + timedelta(hours=h)).isoformat()} for i, h in enumerate(hours)}


def _capture(tmp_path, state, upcoming):
    return ora.OddsResearchAdapter("NCAAB").capture_pregame(tmp_path / "SYNTHETIC.sqlite3", state, "key", None,
                                                           lambda: None, upcoming=upcoming)


def test_off_by_default_every_due_game_is_requested_at_once(provider, tmp_path, monkeypatch):
    monkeypatch.delenv(ora.BATCHING_ENV, raising=False)
    now = ora.utcnow()
    state = {"ncaab_last_capture": now.isoformat()}
    result = _capture(tmp_path, state, _upcoming(now, [10]))
    assert "batched_wait" not in result and "odds" in provider


def test_batching_waits_for_soon_full_or_stale(provider, tmp_path, monkeypatch):
    monkeypatch.setenv(ora.BATCHING_ENV, "true")
    now = ora.utcnow()
    recent = {"ncaab_last_capture": (now - timedelta(hours=1)).isoformat()}
    waited = _capture(tmp_path, dict(recent), _upcoming(now, [10, 12]))
    assert waited["batched_wait"] == 2 and provider == []                # far games, recent capture: no credits
    soon = _capture(tmp_path, dict(recent), _upcoming(now, [10, 2.5]))
    assert "batched_wait" not in soon and "odds" in provider              # a game within 3 hours
    provider.clear()
    full = _capture(tmp_path, dict(recent), _upcoming(now, [10 + i / 10 for i in range(24)]))
    assert "batched_wait" not in full and "odds" in provider              # a full batch
    provider.clear()
    stale_state = {"ncaab_last_capture": (now - timedelta(hours=3, minutes=1)).isoformat()}
    _capture(tmp_path, stale_state, _upcoming(now, [10]))
    assert "odds" in provider and ora.timestamp(stale_state["ncaab_last_capture"]) >= now   # stamped
    provider.clear()
    _capture(tmp_path, {}, _upcoming(now, [10]))
    assert "odds" in provider                                             # never captured before


def test_grading_batches_new_finals_hourly_and_corrections_six_hourly(provider, tmp_path, monkeypatch):
    monkeypatch.setenv(ora.BATCHING_ENV, "true")
    adapter = ora.OddsResearchAdapter("NCAAB")
    now = ora.utcnow()
    finished = {"event_id": "g1", "start": (now - timedelta(hours=4)).isoformat()}
    monkeypatch.setattr(ora, "captures", lambda records: {"g1": finished})
    monkeypatch.setattr(adapter.store, "records", lambda path: [])
    path = tmp_path / "SYNTHETIC.sqlite3"
    monkeypatch.setattr(ora, "scores", lambda records: {})                # a new final is due
    recent = {"ncaab_last_score_fetch": (now - timedelta(minutes=30)).isoformat()}
    assert adapter.grade(path, "key", None, lambda: None, dict(recent))["batched_wait"] == 1
    assert provider == []
    hour = {"ncaab_last_score_fetch": (now - timedelta(minutes=61)).isoformat()}
    adapter.grade(path, "key", None, lambda: None, hour)
    assert provider == ["scores"]
    provider.clear()
    monkeypatch.setattr(ora, "scores", lambda records: {"g1": {"home_score": 1, "away_score": 0}})  # corrections only
    two_hours = {"ncaab_last_score_fetch": (now - timedelta(hours=2)).isoformat()}
    assert adapter.grade(path, "key", None, lambda: None, two_hours)["batched_wait"] == 1 and provider == []
    six = {"ncaab_last_score_fetch": (now - timedelta(hours=6, minutes=1)).isoformat()}
    adapter.grade(path, "key", None, lambda: None, six)
    assert provider == ["scores"]
