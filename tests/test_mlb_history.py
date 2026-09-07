from copy import deepcopy
import pytest
from app_core.mlb_history import build_dataset, normalize_game


def game(i, cutoff, end, home=1, away=2):
    return dict(game_id=i, season=2023, cutoff=cutoff, completed_at=end,
                home_id=home, away_id=away, home_score=5, away_score=2)


def test_completed_before_cutoff_and_doubleheaders():
    rows = [game(1, "2023-04-01T12:00:00Z", "2023-04-01T15:00:00Z"),
            game(2, "2023-04-01T18:00:00Z", "2023-04-01T21:00:00Z")]
    result = build_dataset(rows, minimum_games=1)
    assert len(result["features"]) == 1
    assert result["features"][0]["home_prior_game_ids"] == [1]
    assert "home_score" not in result["features"][0]


def test_suspended_game_cannot_leak_and_exact_cutoff_excluded():
    rows = [game(1, "2023-04-01T12:00:00Z", "2023-04-03T15:00:00Z"),
            game(2, "2023-04-02T18:00:00Z", "2023-04-02T21:00:00Z"),
            game(3, "2023-04-03T15:00:00Z", "2023-04-03T21:00:00Z")]
    result = build_dataset(rows, minimum_games=1)
    assert [f["game_id"] for f in result["features"]] == [3]
    assert result["features"][0]["home_prior_game_ids"] == [2]


def test_no_cross_season_or_conflicting_duplicates():
    a = game(1, "2023-04-01T12:00:00Z", "2023-04-01T15:00:00Z")
    b = game(2, "2024-04-01T12:00:00Z", "2024-04-01T15:00:00Z"); b["season"] = 2024
    assert not build_dataset([a, b], 1)["features"]
    c = deepcopy(a); c["home_score"] = 8
    with pytest.raises(ValueError, match="conflicting"):
        build_dataset([a, c])


def test_nonfinal_rejected():
    with pytest.raises(ValueError, match="not_final"):
        normalize_game({"gameData": {"status": {"abstractGameState": "Live"}}})


def test_collector_budget_resume_and_separate_targets(tmp_path, monkeypatch):
    from scripts import collect_mlb_history as collector
    calls = []
    def get(url, params=None, timeout=None):
        calls.append(url)
        class Response:
            def raise_for_status(self):
                pass
            def json(self):
                if url.endswith("schedule"):
                    return {"dates": [{"games": [{"gamePk": 1}, {"gamePk": 2}]}]}
                return {"id": int(url.split("/")[-3])}
        return Response()
    monkeypatch.setattr(collector.requests, "get", get)
    monkeypatch.setattr(collector, "normalize_game", lambda p: game(p["id"], "2023-04-01T12:00:00Z", "2023-04-01T15:00:00Z"))
    first = collector.collect(tmp_path, [2023], 2)
    assert len(calls) == 2 and first["pending_games"] == 1
    second = collector.collect(tmp_path, [2023], 2)
    assert len(calls) == 3 and second["collection_complete"]
    assert second["games"] == 2
    assert (tmp_path / "targets.json").exists()


def test_normalizer_preserves_actual_end_and_rejects_missing_time():
    feed = {"gamePk": 1, "gameData": {
        "status": {"abstractGameState": "Final"}, "game": {"type": "R", "season": "2023"},
        "datetime": {"dateTime": "2023-04-01T12:00:00Z"},
        "teams": {"home": {"id": 1, "name": "A"}, "away": {"id": 2, "name": "B"}}},
        "liveData": {"plays": {"allPlays": [{"about": {"isComplete": True,
            "startTime": "2023-04-01T12:05:00Z", "endTime": "2023-04-03T15:00:00Z"}}]},
            "linescore": {"teams": {"home": {"runs": 5}, "away": {"runs": 2}}}}}
    row = normalize_game(feed)
    assert row["completed_at"].startswith("2023-04-03")
    assert row["cutoff"].startswith("2023-04-01T12:00")
    del feed["liveData"]["plays"]["allPlays"][0]["about"]["endTime"]
    with pytest.raises(KeyError):
        normalize_game(feed)
