"""NHL settlement against the NHL's own official final (roadmap D17). SYNTHETIC evidence; no network."""
from datetime import datetime, timezone

import pytest

from app_core import nhl_official_scores as official
from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from tests.test_prospective_research_models import _at, _event

# Shape of the NHL score feed (api-web.nhle.com/v1/score/2026-11-03), trimmed.
FEED = {"games": [
    {"id": 2026020239, "gameState": "OFF", "startTimeUTC": "2026-11-04T00:30:00Z",
     "gameOutcome": {"lastPeriodType": "SO"},
     "homeTeam": {"abbrev": "NJD", "name": {"default": "Devils"}, "score": 4},
     "awayTeam": {"abbrev": "EDM", "name": {"default": "Oilers"}, "score": 3}},
    {"id": 2026020240, "gameState": "LIVE", "startTimeUTC": "2026-11-04T00:00:00Z",
     "homeTeam": {"name": {"default": "Rangers"}, "score": 1},
     "awayTeam": {"name": {"default": "Blues"}, "score": 1}},
]}


def test_official_finals_include_the_shootout_and_match_one_game():
    games = official.finals(FEED)
    assert len(games) == 1 and games[0]["last_period_type"] == "SO"
    assert (games[0]["home_score"], games[0]["away_score"]) == (4, 3)
    event = {"scheduled_start": "2026-11-04T00:35:00+00:00",
             "home_team": "New Jersey Devils", "away_team": "Edmonton Oilers"}
    assert official.match(event, games)["game_id"] == 2026020239
    assert official.match(dict(event, scheduled_start="2026-11-04T01:00:00+00:00"), games) is None
    assert official.match(dict(event, home_team="Edmonton Oilers", away_team="New Jersey Devils"), games) is None
    assert official.match(event, games + games) is None  # ambiguous
    assert official.game_dates([event]) == {"2026-11-04", "2026-11-03"}


@pytest.fixture
def clock(monkeypatch):
    state = [_at(0)]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    monkeypatch.setattr(models, "source_commit", lambda: "a" * 40)
    monkeypatch.setattr(pc, "CANDIDATE_MIN_TRAINING_EVENTS", 60)
    return state


def _official_for(path, event_id, delta=0):
    score = pe.load_record(path, "prospective_result", event_id.replace("event-", "score-"))
    return {"game_id": 1, "source_hash": "f" * 64, "last_period_type": "REG",
            "home_score": score["home_score"] + delta, "away_score": score["away_score"]}


def test_nhl_settles_only_with_an_equal_official_final(tmp_path, clock):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    event_id = _event(path, clock, "NHL", 0)
    quote_id = f"quote-{event_id}-PUCK_LINE"
    score_id = f"score-NHL-0"
    with pytest.raises(ValueError, match="official final"):
        pe.insert_score_settlement(path, score_result_id=score_id, quote_id=quote_id)
    with pytest.raises(ValueError, match="disagrees"):
        pe.insert_score_settlement(path, score_result_id=score_id, quote_id=quote_id,
                                   official=_official_for(path, event_id, delta=1))
    result_id = pe.insert_score_settlement(path, score_result_id=score_id, quote_id=quote_id,
                                           official=_official_for(path, event_id))
    stored = pe.load_record(path, "prospective_result", result_id)
    assert stored["outcome"] in ("WIN", "LOSS") and b"official_check" in stored["raw_source"]


def test_settle_scope_needs_the_setting_and_the_official_feed(tmp_path, clock, monkeypatch):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    for index in range(70):
        _event(path, clock, "NHL", index)
    clock[0] = _at(71)
    models.train_scope(path, "NHL", "PUCK_LINE")
    for index in range(72, 76):
        _event(path, clock, "NHL", index)
        clock[0] = _at(index, 1)
        pc.predict_candidate(path, "NHL", "PUCK_LINE", now=clock[0])
    clock[0] = _at(77)

    # Off by default: the existing contract still holds.
    assert pc.settle_scope(path, "NHL", "PUCK_LINE")["errors"] == ["SCORE_SETTLEMENT_NOT_ENABLED"]
    on = lambda key, default="": "official_check" if key == pc.NHL_SETTLEMENT_ENV else default

    def unavailable(url, timeout):
        raise OSError("feed down")
    report = pc.settle_scope(path, "NHL", "PUCK_LINE", request_get=unavailable, setting=on)
    assert report["settled"] == 0 and report["errors"][0].startswith("OFFICIAL_SCORES_UNAVAILABLE")

    class Feed:
        def raise_for_status(self):
            pass

        def json(self):
            return {"games": []}
    matched = {"event-NHL-72", "event-NHL-73"}
    monkeypatch.setattr(official, "match",
                        lambda event, games: _official_for(path, event["event_id"]) if event["event_id"] in matched else None)
    report = pc.settle_scope(path, "NHL", "PUCK_LINE", request_get=lambda url, timeout: Feed(), setting=on)
    assert report["settled"] == 2 and report["awaiting_official"] == 2 and report["errors"] == []
