from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import sqlite3
import pandas as pd
import pytest
from app_core import mlb_pregame_receipts as r
from app_core import mlb_spread_total_model as model

NOW=datetime(2026,9,15,12,tzinfo=timezone.utc)
START=NOW+timedelta(hours=6)


def schedule_game(gid, start, final=False):
    return {"gamePk":gid,"season":"2026","gameType":"R","gameDate":start.isoformat(),
        "status":{"abstractGameState":"Final" if final else "Preview"},
        "teams":{"home":{"team":{"id":112,"name":"Chicago Cubs"},"score":5 if final else 0},
                 "away":{"team":{"id":134,"name":"Pittsburgh Pirates"},"score":2 if final else 0}}}


def odds_game():
    return {"id":"odds-100","sport_key":"baseball_mlb","matchup_id":"pirates-cubs",
        "home_team":"Chicago Cubs","away_team":"Pittsburgh Pirates","commence_time":START.isoformat(),
        "live_receipt_observed_at":NOW.isoformat(),"live_receipt_price_format":"american",
        "bookmakers":[{"key":"novig","last_update":(NOW-timedelta(minutes=1)).isoformat(),"markets":[
            {"key":"spreads","outcomes":[{"name":"Chicago Cubs","point":-1.5,"price":110},
                                               {"name":"Pittsburgh Pirates","point":1.5,"price":-120}]},
            {"key":"totals","outcomes":[{"name":"Over","point":8.5,"price":-110},
                                              {"name":"Under","point":8.5,"price":-105}]}]}]}


def feed(game):
    start=datetime.fromisoformat(game["gameDate"])
    return {"gamePk":game["gamePk"],"gameData":{
        "game":{"type":"R","season":"2026"},"datetime":{"dateTime":game["gameDate"]},
        "status":{"abstractGameState":"Final"},
        "teams":{s:game["teams"][s]["team"] for s in ("home","away")}},
        "liveData":{"plays":{"allPlays":[{"about":{"isComplete":True,
            "startTime":(start+timedelta(minutes=1)).isoformat(),"endTime":(start+timedelta(hours=3)).isoformat()}}]},
            "linescore":{"teams":{"home":{"runs":5},"away":{"runs":2}}}}}


@pytest.fixture
def fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(r,"now",lambda:NOW)
    games=[schedule_game(i,START-timedelta(days=i),True) for i in range(1,11)]+[schedule_game(100,START)]
    calls=[]
    def fetch(endpoint,params=None):
        calls.append(endpoint)
        if endpoint.endswith("schedule"):
            payload={"dates":[{"games":games}]}
        else:
            gid=int(endpoint.split("/")[-3])
            payload=feed(next(g for g in games if g["gamePk"]==gid))
        return {"source":"mlb_statsapi","endpoint":endpoint,"params":params or {},
                "observed_at":r.now().isoformat(),"payload":deepcopy(payload)}
    return tmp_path/"receipts.db",games,fetch,calls


def collect(fixture,**kwargs):
    db,games,fetch,calls=fixture
    return r.capture_live_games([odds_game()],path=db,fetch=fetch,**kwargs)


def test_live_collector_contract_orientation_hash_and_authority(fixture):
    original=odds_game();original.update(production_eligible=False,wager_approved=False,stake=0,maturity="RESEARCH")
    db,_,fetch,_=fixture
    output,health=r.capture_live_games([original],path=db,fetch=fetch)
    assert health["receipts_created"]==4,health
    assert original.get("team_ids") is None
    assert output[0]["team_ids"]==["mlb:112","mlb:134"]
    for key in ("production_eligible","wager_approved","stake","maturity"):
        assert output[0][key]==original[key]
    records=r.export_records(db)
    assert len(records)==4 and all(x["outcome"] is None for x in records)
    expected={"spread_home":-1.5,"spread_away":1.5,"total_over":8.5,"total_under":8.5}
    for record in records:
        snapshot=record["snapshot"];p,_=model.receipt_features(snapshot)
        assert snapshot["sha256"]==model.digest(p)
        assert p["quote"]["line"]==expected[p["quote"]["market_type"]]
        assert p["quote"]["provider_namespace"]=="odds_api"
        assert p["provider_event_id"]=="100" and p["quote"]["provider_event_id"]=="odds-100"
        changed=deepcopy(p);changed["quote"]["line"]+=.5
        assert model.digest(changed)!=snapshot["sha256"]
        assert r.timestamp(p["captured_at"])<=r.timestamp(p["prediction_cutoff"])<r.timestamp(p["game_start_utc"])
    assert r.health(db)["automatic_training"] is False


@pytest.mark.parametrize("side",["home","away"])
def test_missing_team_ids_rejected(fixture,side):
    fixture[1][-1]["teams"][side]["team"].pop("id")
    output,health=collect(fixture)
    assert health["receipts_created"]==0
    assert "missing_team_ids" in health["reasons"]
    assert "team_ids" not in output[0]


def test_missing_provider_id_rejected_without_network(fixture):
    game=odds_game();game.pop("id")
    _,health=r.capture_live_games([game],path=fixture[0],fetch=fixture[2])
    assert health["receipts_created"]==0 and "missing_provider_event_id" in health["reasons"]
    assert fixture[3]==[]


@pytest.mark.parametrize("change",["missing","future","stale","updated_future"])
def test_quote_time_fail_closed_without_network(fixture,change):
    game=odds_game()
    if change=="missing":game.pop("live_receipt_observed_at")
    if change=="future":game["live_receipt_observed_at"]=(NOW+timedelta(seconds=1)).isoformat()
    if change=="stale":game["live_receipt_observed_at"]=(NOW-timedelta(minutes=31)).isoformat()
    if change=="updated_future":game["bookmakers"][0]["last_update"]=(NOW+timedelta(seconds=1)).isoformat()
    _,health=r.capture_live_games([game],path=fixture[0],fetch=fixture[2])
    assert health["receipts_created"]==0
    assert "invalid_quote_time" in health["reasons"]
    assert not fixture[3]


def test_game_start_during_collection_rejected(fixture,monkeypatch):
    original=fixture[2]
    def fetch(endpoint,params=None):
        obs=original(endpoint,params)
        if endpoint.endswith("feed/live"):monkeypatch.setattr(r,"now",lambda:START)
        return obs
    _,health=r.capture_live_games([odds_game()],path=fixture[0],fetch=fetch)
    assert health["receipts_created"]==0
    assert not r.read("receipts",fixture[0])


@pytest.mark.parametrize("change",["future","target","duplicate","namespace","tie","status"])
def test_prior_validation(fixture,change):
    collect(fixture)
    record=next(iter(r.read("receipts",fixture[0]).values()))
    p=record["payload"];prior=p["prior_games"]
    if change=="future":prior[0]["available_at"]=(NOW+timedelta(seconds=1)).isoformat()
    if change=="target":prior[0]["game_id"]=p["provider_event_id"]
    if change=="duplicate":prior.append(deepcopy(prior[0]))
    if change=="namespace":prior[0]["provider_namespace"]="espn"
    if change=="tie":prior[0]["away_score"]=prior[0]["home_score"]
    if change=="status":prior[0]["status"]="LIVE"
    record["sha256"]=model.digest(p)
    with pytest.raises(ValueError):r.save_receipt(record,fixture[0])


def test_immutable_conflict_and_sql_replace(fixture):
    collect(fixture)
    db=fixture[0];records=r.read("receipts",db);key,record=next(iter(records.items()))
    assert r.save_receipt(record,db) is False
    changed=deepcopy(record);changed["payload"]["quote"]["line"]=-2.5
    changed["sha256"]=model.digest(changed["payload"])
    with pytest.raises(r.Rejected,match="duplicate_conflict"):r.save_receipt(changed,db)
    with sqlite3.connect(db) as conn:
        for sql in ("UPDATE receipts SET payload='bad'", "DELETE FROM receipts", "INSERT OR REPLACE INTO receipts SELECT * FROM receipts"):
            with pytest.raises(sqlite3.IntegrityError,match="append-only"):conn.execute(sql)
    assert r.read("receipts",db)==records


def test_outcome_append_exact_identity_and_training_export(fixture,monkeypatch):
    collect(fixture);db=fixture[0];before=r.read("receipts",db)
    monkeypatch.setattr(r,"now",lambda:START+timedelta(hours=4))
    report=r.reconcile(db,fetch=fixture[2])
    assert report["outcomes_created"]==1,report
    assert r.read("receipts",db)==before
    rows=r.export_records(db,settled_only=True)
    assert len(rows)==4 and len(model.prepare_rows(rows))==4
    assert all(x["outcome"]["available_at"]==r.now().isoformat() for x in rows)
    assert r.reconcile(db,fetch=fixture[2])["outcomes_created"]==0
    bad=deepcopy(rows[0]["outcome"]);bad["home_team_id"]="mlb:999"
    with pytest.raises(ValueError):r.save_outcome(bad,db)
    assert not (db.parent/"evidence.sqlite3").exists()


def test_reconciliation_tied_final_rejected(fixture,monkeypatch):
    collect(fixture);original=fixture[2]
    monkeypatch.setattr(r,"now",lambda:START+timedelta(hours=4))
    def tied(endpoint,params=None):
        obs=original(endpoint,params)
        obs["payload"]["liveData"]["linescore"]["teams"]["away"]["runs"]=5
        return obs
    report=r.reconcile(fixture[0],fetch=tied)
    assert report["outcomes_created"]==0
    assert not r.read("outcomes",fixture[0])


def test_budget_resume_duplicate_and_no_refetch(fixture):
    _,first=collect(fixture,max_feeds=5)
    assert first["prior_feeds_requested"]==5 and first["receipts_created"]==0
    _,second=collect(fixture,max_feeds=5)
    assert second["prior_feeds_requested"]==5 and second["receipts_created"]==4
    _,third=collect(fixture,max_feeds=5)
    assert third["prior_feeds_requested"]==0 and third["receipts_created"]==0
    assert third["reasons"]["duplicate_receipt"]==4


def test_doubleheader_and_namespace_not_guessed(fixture):
    fixture[1].append(schedule_game(101,START+timedelta(hours=4)))
    _,health=collect(fixture)
    assert health["receipts_created"]==0 and "event_identity_ambiguous" in health["reasons"]


def test_fresh_observation_clock_comes_from_response(monkeypatch):
    class Response:
        def raise_for_status(self):pass
        def json(self):return {"facts":1}
    monkeypatch.setattr(r.requests,"get",lambda *a,**k:Response())
    monkeypatch.setattr(r,"now",lambda:NOW)
    assert r.observe("api/v1/schedule")["observed_at"]==NOW.isoformat()


def test_live_ingestion_and_expansion_preserve_stable_ids(fixture,monkeypatch):
    from app_core import odds_api
    import core.streamlit_pipeline as sp
    class Client:
        def __init__(self,**kwargs):pass
        def get_odds(self,sk,date=None):return [odds_game()] if sk=="baseball_mlb" else []
    monkeypatch.setattr(odds_api,"TheOddsAPIClient",Client)
    monkeypatch.setattr(odds_api,"filter_games_today_only",lambda games:games)
    monkeypatch.setattr(sp,"_get_odds_api_key",lambda:"test")
    real=r.capture_live_games
    monkeypatch.setattr(r,"capture_live_games",lambda games:real(games,path=fixture[0],fetch=fixture[2]))
    frame=sp.fetch_live_odds_dataframe(["MLB"])
    assert frame.attrs["mlb_receipt_health"]["receipts_created"]==4
    expanded,_=sp._expand_live_odds_to_bet_rows(frame)
    assert not expanded.empty
    from app_core.mlb_live_model_binding import verify
    import json
    for candidate in expanded.to_dict("records"):
        if candidate["market_type"] in model.TARGETS:
            receipt = json.loads(candidate["mlb_pregame_receipts"])[candidate["market_type"]]
            verify(candidate, receipt, now=NOW)
    assert all(ids==["mlb:112","mlb:134"] for ids in expanded.team_ids)
    assert expanded.home_team_id.eq("mlb:112").all()
    assert all(ids["odds_api"]=="odds-100" for ids in expanded.provider_ids)
    from app_core.candidate_evidence_schema import authority_projection
    from core.wager_decisions import allocate_exposure
    private = authority_projection(expanded, [])
    assert all(ids == ["mlb:112", "mlb:134"] for ids in private.team_ids)
    # The factual IDs clear only the identity blocker; collection grants no stake.
    allocated = allocate_exposure(private.to_dict("records"), 1000,
                                  total_cap=.01, game_cap=.001, sport_caps={"MLB": .01})
    assert all(x["recommended_stake"] == 0 for x in allocated)
    assert all("missing_stable_team_ids" not in x.get("production_gate_reason", "") for x in allocated)


@pytest.mark.parametrize("requested_date", [None, "2026-09-15"])
def test_odds_response_marks_only_live_requests(monkeypatch, tmp_path, requested_date):
    from app_core.odds_api import TheOddsAPIClient
    import app_core.odds_api as adapter
    raw = odds_game()
    raw.pop("live_receipt_observed_at")
    raw.pop("live_receipt_price_format")
    class Response:
        status_code = 200
        headers = {}
        def raise_for_status(self): pass
        def json(self): return [deepcopy(raw)]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(adapter.requests, "get", lambda *a, **k: Response())
    before = datetime.now(timezone.utc)
    games = TheOddsAPIClient("test").get_odds("baseball_mlb", date=requested_date)
    after = datetime.now(timezone.utc)
    if requested_date is None:
        assert before <= r.timestamp(games[0]["live_receipt_observed_at"]) <= after
        assert games[0]["live_receipt_price_format"] == "american"
    else:
        assert "live_receipt_observed_at" not in games[0]


def test_save_rechecks_start_after_lock_wait(fixture, monkeypatch):
    collect(fixture)
    record = next(iter(r.read("receipts", fixture[0]).values()))
    monkeypatch.setattr(r, "now", lambda: START)
    with pytest.raises(r.Rejected, match="invalid_capture_time"):
        r.save_receipt(record, fixture[0].with_name("late.db"))


def test_source_failure_keeps_research_game_and_reports_skips(fixture):
    def unavailable(*args, **kwargs):
        raise r.requests.Timeout("provider unavailable")
    original = odds_game()
    output, report = r.capture_live_games([original], path=fixture[0], fetch=unavailable)
    assert output == [original]
    assert report["receipts_created"] == 0 and report["receipts_skipped"] == 4
    assert report["reasons"] == {"receipt_source_or_storage_unavailable": 1}


def test_live_receipts_returned_without_rewriting_first_training_receipt(fixture):
    db, _, fetch, _ = fixture
    first, _ = r.capture_live_games([odds_game()], path=db, fetch=fetch)
    original = r.read("receipts", db)
    changed = odds_game()
    changed["bookmakers"][0]["markets"][1]["outcomes"][0]["point"] = 9.5
    second, report = r.capture_live_games([changed], path=db, fetch=fetch)
    assert second[0]["mlb_pregame_receipts"]["total_over"]["payload"]["quote"]["line"] == 9.5
    assert first[0]["mlb_pregame_receipts"]["total_over"]["payload"]["quote"]["line"] == 8.5
    assert r.read("receipts", db) == original
    assert report["receipts_created"] == 0


def test_identical_schedule_duplicates_continue_collection(fixture):
    games = fixture[1]
    games.extend(deepcopy(games))
    _, health = collect(fixture)
    assert health['receipts_created'] == 4
    assert health['reasons']['identical_schedule_duplicates_collapsed'] == 11


def test_conflicting_target_never_picks_a_variant(fixture):
    games = fixture[1]
    changed = deepcopy(games[-1])
    changed['gameDate'] = (START + timedelta(hours=1)).isoformat()
    games.append(changed)
    _, health = collect(fixture)
    assert health['receipts_created'] == 0
    assert health['reasons']['conflicting_schedule_event'] == 1


def test_unrelated_conflict_does_not_stop_receipts(fixture):
    games = fixture[1]
    other = schedule_game(999, START)
    other['teams']['home']['team'] = {'id': 111, 'name': 'Boston Red Sox'}
    other['teams']['away']['team'] = {'id': 147, 'name': 'New York Yankees'}
    changed = deepcopy(other)
    changed['gameDate'] = (START + timedelta(hours=1)).isoformat()
    games.extend([other, changed])
    original = deepcopy(games)
    _, health = collect(fixture)
    assert health['receipts_created'] == 4
    assert health['reasons']['conflicting_schedule_events_quarantined'] == 1
    assert games == original


def test_conflicting_prior_cannot_be_replaced_by_older_history(fixture):
    games = fixture[1]
    changed = deepcopy(games[0])
    changed['teams']['home']['score'] = 9
    games.extend([changed, schedule_game(20, START-timedelta(days=20), True)])
    _, health = collect(fixture)
    assert health['receipts_created'] == 0
    assert health['reasons']['conflicting_prior_schedule_event'] == 1


@pytest.mark.parametrize('minutes', [-10, -1, 1, 10])
def test_small_unique_event_start_difference_collects(fixture, minutes):
    raw = odds_game()
    raw['commence_time'] = (START + timedelta(minutes=minutes)).isoformat()
    db, games, fetch, _ = fixture
    output, health = r.capture_live_games([raw], path=db, fetch=fetch)
    assert health['receipts_created'] == 4, health
    receipt = output[0]['mlb_pregame_receipts']['spread_home']['payload']
    assert r.timestamp(receipt['game_start_utc']) == START
    assert r.timestamp(receipt['quote']['source_game_start_utc']) == START + timedelta(minutes=minutes)


def test_large_start_difference_still_blocked(fixture):
    raw = odds_game()
    raw['commence_time'] = (START + timedelta(minutes=11)).isoformat()
    with pytest.raises(ValueError, match='event_start_mismatch'):
        r.resolve_event(raw, fixture[1])


def test_earlier_provider_start_blocks_capture(fixture):
    fixture[1][-1]['gameDate'] = (NOW + timedelta(minutes=1)).isoformat()
    raw = odds_game()
    raw['commence_time'] = (NOW - timedelta(minutes=1)).isoformat()
    with pytest.raises(ValueError, match='invalid_capture_time'):
        r.resolve_event(raw, fixture[1])
