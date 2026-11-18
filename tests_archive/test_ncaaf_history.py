import io
import json
import zipfile
from copy import deepcopy
from unittest.mock import Mock, patch
import pytest
import requests
from app_core.ncaaf_history import (new_collection, collect_batch, pending_requests,
    checkpoint_bytes, load_checkpoint, build_dataset, archive_bytes, backup_checkpoint)


def game(gid, year=2023, week=1, day=1, home=1, away=2, hp=20, ap=10):
    return dict(id=gid, season=year, week=week, seasonType="regular",
                startDate=f"{year}-09-{day:02d}T12:00:00Z", startTimeTBD=False,
                completed=True, neutralSite=False, homeId=home, awayId=away,
                homeTeam="Home", awayTeam="Away", homePoints=hp, awayPoints=ap)


def batch(kind, year, records, week=1):
    req = dict(kind=kind, year=year)
    if kind == "stats":
        req.update(week=week, season_type="regular")
    return dict(request=req, retrieved_at="2026-09-07T20:00:00Z", records=records)


def teams(g):
    return dict(id=g["id"], teams=[dict(teamId=g[s+"Id"], points=g[s+"Points"],
                    stats=[dict(category="totalYards", stat="300")]) for s in ("home", "away")])


def response(payload, status=200):
    return Mock(status_code=status, json=Mock(return_value=payload))


def api(url, **kwargs):
    p = kwargs["params"]
    if url.endswith("/games"):
        return response([game(p["year"] * 100 + w, p["year"], week=w, day=w) for w in range(1, 4)])
    return response([teams(game(p["year"] * 100 + p["week"], p["year"], week=p["week"], day=p["week"]))])


def test_bounded_resume_no_repeated_successful_requests():
    state = new_collection(2026)
    get = Mock(side_effect=api)
    state, status = collect_batch(state, "sensitive-token", get=get)
    assert status == "batch_saved" and get.call_count == 6
    state = load_checkpoint(checkpoint_bytes(state))
    state, status = collect_batch(state, "sensitive-token", get=get)
    assert status == "requests_complete" and get.call_count == 12
    assert not pending_requests(state)
    collect_batch(state, "sensitive-token", get=get)
    assert get.call_count == 12
    requests_seen = [json.dumps(c.kwargs["params"], sort_keys=True) for c in get.call_args_list]
    assert len(set(requests_seen)) == 12
    assert "sensitive-token" not in checkpoint_bytes(state).decode()
    assert all(c.kwargs["allow_redirects"] is False and c.kwargs["timeout"] == 6 for c in get.call_args_list)


@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
def test_failure_preserves_progress_retries_only_failed_request(status):
    get = Mock(side_effect=[api("/games", params={"year": 2023}), response([], status)])
    state, error = collect_batch(new_collection(2026), "key", get=get)
    assert error == f"http_{status}" and len(state["batches"]) == 1
    assert pending_requests(state)[0] == dict(kind="games", year=2024)


@pytest.mark.parametrize("failure", [requests.Timeout("secret"), ValueError("secret")])
def test_errors_safe(failure):
    state, error = collect_batch(new_collection(2026), "secret", get=Mock(side_effect=failure))
    assert "secret" not in error and state["batches"] == []


def test_missing_key_no_request():
    get = Mock()
    _, error = collect_batch(new_collection(2026), None, get=get)
    assert error == "missing_or_invalid_key"
    get.assert_not_called()


def test_reject_arbitrary_checkpoint_request():
    state = new_collection(2026)
    state["batches"] = [batch("games", 2023, [])]
    state["batches"][0]["request"]["url"] = "https://example.com"
    with pytest.raises(ValueError, match="Invalid NCAAF"):
        load_checkpoint(checkpoint_bytes(state))


@pytest.mark.parametrize("payload", [b"no json", b"[]", b"null", b'{"schema": 2}'])
def test_invalid_checkpoint(payload):
    with pytest.raises(ValueError, match="Invalid NCAAF"):
        load_checkpoint(payload)


def test_empty_schedule_not_coverage_success():
    state, _ = collect_batch(new_collection(2026), "key", get=Mock(return_value=response([])))
    audit, features, targets = build_dataset(state)
    assert audit["requests_remaining"] == 0
    assert all(r["usable_games"] == 0 for r in audit["seasons"])
    assert features == targets == []


def research_state():
    state = new_collection(2026)
    games = [game(1, day=1), game(2, week=2, day=9, hp=30), game(3, week=3, day=17, hp=40),
             game(4, week=4, day=25, hp=50)]
    state["batches"] = [batch("games", 2023, games)]
    state["batches"] += [batch("stats", 2023, [teams(g)], week=g["week"]) for g in games]
    return state


def test_features_exclude_target_future_and_recent_games():
    state = research_state()
    _, rows, targets = build_dataset(state)
    assert rows[0]["home_ppg"] is None
    assert rows[-1]["home_prior_game_ids"] == "1;2;3"
    assert rows[-1]["home_ppg"] == 30
    assert rows[-1]["scoring_features_available"] is True
    assert rows[-1]["production_eligible"] is False
    assert targets[-1]["home_margin"] == 40
    changed = deepcopy(state)
    changed["batches"][0]["records"][-1]["homePoints"] = 999
    assert build_dataset(changed)[1] == rows
    recent = game(5, week=4, day=24, hp=999)
    state["batches"][0]["records"].append(recent)
    target = next(r for r in build_dataset(state)[1] if r["game_id"] == 4)
    assert target["home_ppg"] == 30


def test_prior_season_and_equal_cutoff_excluded():
    state = research_state()
    state["batches"][0]["records"][1]["startDate"] = "2023-09-08T12:00:00Z"
    rows = build_dataset(state)[1]
    assert rows[1]["home_prior_games"] == 0
    state["batches"].append(batch("games", 2024, [game(10, year=2024)]))
    assert build_dataset(state)[1][-1]["home_prior_games"] == 0


def test_conflicting_games_excluded_and_stats_scores_checked():
    state = research_state()
    conflicting = deepcopy(state["batches"][0]["records"][0])
    conflicting["homePoints"] = 99
    state["batches"][0]["records"].append(conflicting)
    state["batches"][2]["records"][0]["teams"][0]["points"] = 888
    audit, rows, _ = build_dataset(state)
    assert 1 not in [r["game_id"] for r in rows]
    assert any(i["issue"] == "conflicting_game_excluded" for i in audit["issues"])
    assert any(i["game_id"] == 2 and i["issue"] == "missing_or_conflicting_team_stats" for i in audit["issues"])


def test_archive_contains_separate_targets_and_checkpoint():
    state = research_state()
    with zipfile.ZipFile(io.BytesIO(archive_bytes(state))) as z:
        assert set(z.namelist()) == {"checkpoint.json", "coverage-audit.json", "research-features.csv", "targets.csv"}
        assert "home_score" not in z.read("research-features.csv").decode().splitlines()[0]
        assert load_checkpoint(z.read("checkpoint.json")) == state


def test_backup_readback_and_idempotence():
    from app_core.evidence_drive import AlreadyExists
    state = new_collection(2026)
    client = Mock()
    client.get_object.side_effect = lambda **kw: {"Body": io.BytesIO(checkpoint_bytes(state))}
    key = backup_checkpoint(state, client=client, folder="folder")
    assert key.startswith("parlaypicker/ncaaf-history-v1/")
    client.put_object.side_effect = AlreadyExists()
    assert backup_checkpoint(state, client=client, folder="folder") == key
    client.get_object.return_value = {"Body": io.BytesIO(b"wrong")}
    client.get_object.side_effect = None
    with pytest.raises(ValueError, match="verification failed"):
        backup_checkpoint(state, client=client, folder="folder")


def test_ui_click_only_and_keeps_checkpoint_on_rerun():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string("from app.ui.ncaaf_history import render_ncaaf_history\nrender_ncaaf_history()")
    with patch("app.ui.ncaaf_history._token", return_value="key"), patch(
        "app.ui.ncaaf_history.collect_batch", return_value=(new_collection(2026), "http_429")
    ) as collect:
        app.run()
        collect.assert_not_called()
        app.button(key="ncaaf_history_collect").click().run()
        assert collect.call_count == 1 and not app.exception
        app.run()
        assert collect.call_count == 1 and not app.exception
        assert app.session_state["ncaaf_history"]["years"] == [2023, 2024, 2025]
