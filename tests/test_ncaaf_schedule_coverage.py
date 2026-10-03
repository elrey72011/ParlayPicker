"""Offline schedule -> real caller -> coverage regressions; no acquisition."""
from contextlib import nullcontext
from copy import deepcopy
import json
import socket
import urllib.request

import pandas as pd
import pytest
import requests

from app_core import ncaaf_schedule as ns
from app_core import espn_ncaaf_odds as fcs

CANARY = "SCHEDULE_CREDENTIAL_CANARY_19"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []
    def block(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Network access is forbidden")
    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, block)
    monkeypatch.setattr(socket, "getaddrinfo", block)
    monkeypatch.setattr(socket, "create_connection", block)
    monkeypatch.setattr(requests.sessions.Session, "request", block)
    monkeypatch.setattr(urllib.request, "urlopen", block)
    yield
    assert not attempts


def event(eid="1", kickoff="2026-10-03T23:45Z", home="LSU Tigers", away="McNeese Cowboys", state="STATUS_SCHEDULED"):
    return {"id": eid, "date": kickoff, "status": {"type": {"name": state, "state": "pre"}},
            "competitions": [{"id": eid, "date": kickoff, "competitors": [
                {"homeAway": "home", "team": {"id": "99", "displayName": home}},
                {"homeAway": "away", "team": {"id": "88", "displayName": away}}]}]}


def inv(events=None, **kwargs):
    return ns.inventory_from_events([("FBS", events or [event()]), ("FCS", [event()])],
                                    "2026-10-03", "2026-10-03", complete=True, **kwargs)


def game(eid="primary", away="McNeese State", kickoff="2026-10-03T23:45:00Z", quote_time="2026-10-03T16:40Z"):
    return {"id": eid, "matchup_id": "historical:" + eid, "home_team": "LSU", "away_team": away,
            "sport_key": "americanfootball_ncaaf", "commence_time": kickoff,
            "bookmakers": [{"key": "novig", "last_update": quote_time, "markets": [
                {"key": "spreads", "outcomes": [{"name": "LSU", "point": -53.5, "price": -108},
                                                {"name": away, "point": 53.5, "price": -102}]}]}]}


def response(data, status=200):
    result = requests.Response()
    result.status_code = status
    result.url = "https://schedule.invalid?apiKey=" + CANARY
    result._content = json.dumps(data).encode()
    return result


def index(ids, *, page=1, count=None, pages=1, size=300):
    return {"count": len(ids) if count is None else count, "pageIndex": page, "pageSize": size, "pageCount": pages,
            "items": [{"$ref": ns.INDEX.replace("https:", "http:") + "/" + i} for i in ids]}


def mock_feed(monkeypatch, boards=None, indexes=None):
    calls=[]
    boards = boards or {"80": [event()], "81": [event(), event("2", away="Other Team")]}
    def get(url, params, timeout):
        assert timeout == (3, 5)
        assert params.get("dates") in (None, "20261003", "20261003-20261004")
        calls.append((url, deepcopy(params)))
        if url == ns.SCOREBOARD:
            return response({"events": deepcopy(boards[params["groups"]]), "leagues": calendar()})
        return response(indexes[params["groups"]] if indexes else index([e["id"] for e in boards[params["groups"]]]))
    monkeypatch.setattr(ns.requests, "get", get)
    return calls


def calendar():
    return [{"season": {"year": 2026}, "calendar": [{"value": "2", "entries": [
        {"value": "5", "startDate": "2026-09-28T07:00Z", "endDate": "2026-10-05T06:59Z"}]}]}]


def test_complete_division_overlap_and_canonical_identity(monkeypatch):
    calls = mock_feed(monkeypatch)
    result = ns.fetch_schedule("2026-10-03", "2026-10-03")
    assert result["complete"] and len(result["events"]) == 2 and len(calls) == 5
    assert result["events"][0]["divisions"] == ["FBS", "FCS"]
    assert ns.match_event(game(), result)[1] == "MATCHED"
    assert ns.match_event(game(away="McNeese Cowboys"), result)[0]["schedule_provider_id"] == "1"


@pytest.mark.parametrize("cause", ["missing", "mismatch", "duplicate", "invalid_ref", "wrong_page", "count_drift"])
def test_incomplete_index_never_claims_complete(monkeypatch, cause):
    idx = index(["1"])
    if cause == "missing": idx.pop("items")
    if cause == "mismatch": idx = index(["9"])
    if cause == "duplicate": idx["items"] *= 2
    if cause == "invalid_ref": idx["items"] = [{"$ref": "https://private.invalid/" + CANARY}]
    if cause == "wrong_page": idx["pageIndex"] = 2
    if cause == "count_drift": idx["count"] = 8
    mock_feed(monkeypatch, indexes={"80": idx, "81": index(["1", "2"])})
    result = ns.fetch_schedule("2026-10-03", "2026-10-03")
    assert not result["complete"] and result["status"] == "PARTIAL"
    assert len(result["events"]) == 2
    assert CANARY not in json.dumps(result)


def test_real_pagination_consumes_pages_within_budget(monkeypatch):
    calls=[]
    def get(url, params, timeout):
        calls.append(params)
        if url == ns.SCOREBOARD:
            return response({"events": [event("1"), event("2")] if params["groups"] == "80" else [], "leagues": calendar()})
        if params["groups"] == "81": return response(index([]))
        return response(index([str(params["page"])], count=2, pages=2, page=params["page"], size=1))
    monkeypatch.setattr(ns.requests, "get", get)
    assert ns.fetch_schedule("2026-10-03", "2026-10-03")["complete"]
    assert len(calls) == 6
    limited = ns.fetch_schedule("2026-10-03", "2026-10-03", max_requests=3)
    assert not limited["complete"] and "SCHEDULE_REQUEST_BUDGET_EXHAUSTED" in limited["reasons"]


@pytest.mark.parametrize("failure", [response({"secret": CANARY}, 403), requests.Timeout(CANARY), requests.ConnectionError(CANARY)])
def test_schedule_failures_are_controlled(monkeypatch, failure, caplog):
    def get(*args, **kwargs):
        if isinstance(failure, Exception): raise failure
        return failure
    monkeypatch.setattr(ns.requests, "get", get)
    result = ns.fetch_schedule("2026-10-03", "2026-10-03")
    assert not result["complete"] and not result["events"]
    assert CANARY not in json.dumps(result) + caplog.text


def test_midnight_eastern_boundary_and_started_games():
    es = [event("1", "2026-10-04T03:59Z"), event("2", "2026-10-04T04:00Z"),
          event("3", "2026-10-03T03:59Z"), event("4", "2026-10-03T16:00Z")]
    inventory = ns.inventory_from_events([("FBS", es)], "2026-10-03", "2026-10-03", complete=True)
    assert {e["schedule_provider_id"] for e in inventory["events"]} == {"1", "4"}
    report = ns.coverage(inventory, now="2026-10-03T17:00Z")
    assert report["counts"]["scheduled"] == 2 and report["counts"]["qualified"] == 0
    assert "STARTED" in report["rows"][1]["exclusion_reasons"]


@pytest.mark.parametrize("state", ["STATUS_POSTPONED", "STATUS_CANCELED"])
def test_cancelled_postponed_keep_inventory_with_zero_qualification(state):
    inventory = ns.inventory_from_events([("FCS", [event(state=state)])], "2026-10-03", "2026-10-03", complete=True)
    report = ns.coverage(inventory, now="2026-10-03T17:00Z")
    assert len(report["rows"]) == 1 and report["rows"][0]["selection_status"] == "PASS"
    assert "POSTPONED" in report["rows"][0]["exclusion_reasons"] or "CANCELLED" in report["rows"][0]["exclusion_reasons"]


def test_incomplete_record_retains_canonical_event_and_revision_conflict():
    incomplete = event("2")
    incomplete["competitions"] = []
    inventory = inv([event(), incomplete, event("1", "2026-10-04T01:00Z")])
    assert len(inventory["events"]) == 2 and not inventory["complete"]
    assert inventory["events"][0]["identity_conflict"]
    assert len(inventory["events"][0]["kickoff_revisions"]) == 2
    assert ns.match_event(game(), inventory)[1] == "KICKOFF_OR_IDENTITY_CONFLICT"


@pytest.mark.parametrize("competition", [None, ["malformed"], [{"competitors": None}], [{"competitors": [{"homeAway":"home","team":None}]}]])
def test_malformed_schedule_record_keeps_id_and_healthy_inventory(competition):
    broken = event("2")
    broken["competitions"] = competition
    inventory = inv([event(), broken])
    assert len(inventory["events"]) == 2 and inventory["status"] == "PARTIAL"
    assert ns.match_event(game(), inventory)[1] == "MATCHED"


def test_distinct_events_and_ambiguous_names_are_not_merged():
    inventory = inv([event(), event("2", "2026-10-04T01:00Z")])
    primary = game()
    later = game("espn-2", "McNeese", "2026-10-04T01:00Z")
    assert len(ns.merge_schedule_odds([primary], [later], inventory)) == 2
    assert len(fcs.merge_missing_ncaaf_games([primary], [later])) == 2
    ambiguous = inv([event(), event("2")])
    assert ns.match_event(primary, ambiguous)[1] == "AMBIGUOUS"
    assert len(ns.merge_schedule_odds([primary], [game("espn-2")], ambiguous)) == 2
    assert ns.match_event(game("espn-2"), ambiguous)[0]["schedule_provider_id"] == "2"
    assert len(fcs.merge_missing_ncaaf_games([game("espn-1")], [game("espn-2")])) == 2


def test_primary_quotes_and_historical_ids_are_preserved_on_alias_merge():
    primary, fallback = game(), game("espn-1", "McNeese", quote_time=None)
    before = deepcopy(primary)
    merged = ns.merge_schedule_odds([primary], [fallback], inv())
    assert len(merged) == 1 and primary == before
    assert merged[0]["id"] == "primary" and merged[0]["bookmakers"] == primary["bookmakers"]
    assert merged[0]["historical_matchup_id"] == "historical:primary"
    assert merged[0]["matchup_id"] == "espn:college-football:1"


def test_retained_gardnerwebb_alias_is_an_exact_school_alias():
    inventory = ns.inventory_from_events([("FCS", [event(away="Gardner-Webb Runnin' Bulldogs")])],
                                        "2026-10-03", "2026-10-03", complete=True)
    assert ns.match_event(game(away="Gardnerwebb"), inventory)[1] == "MATCHED"


def test_observation_time_does_not_supply_provider_quote_time_or_authority():
    q = {"recorded_at": None, "observed_at": "2026-10-03T16:40Z", "price": -108,
         "provider_namespace": "espn_ncaaf_fcs_scoreboard", "provider_event_id": "espn-1"}
    row = {**game(), "provider_quotes": json.dumps([q]), "production_eligible": True, "wager_approved": True}
    report = ns.coverage(inv(), [row], [row], [row], now="2026-10-03T17:00Z")
    assert report["counts"] == {"scheduled": 1, "matched": 1, "quoted": 1, "timestamped": 0, "ranked": 1, "qualified": 0}
    assert report["rows"][0]["quote_coverage"] == "PROVENANCE_INCOMPLETE"
    assert q["recorded_at"] is None


def test_actual_caller_inventory_survives_empty_and_partial_provider_failure(monkeypatch):
    from core import streamlit_pipeline as sp
    from app_core import odds_api, college_novig
    mock_feed(monkeypatch)
    scenario = {"basketball_nba": [], "americanfootball_ncaaf": []}
    class Client:
        def __init__(self, **kwargs): pass
        def get_odds(self, sport_key, date=None):
            result=scenario[sport_key]
            if isinstance(result, Exception): raise result
            return deepcopy(result)
    monkeypatch.setattr(odds_api, "TheOddsAPIClient", Client)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: CANARY)
    monkeypatch.setattr(college_novig, "recover_college_novig", lambda games, key: games)
    monkeypatch.setattr(fcs, "fetch_espn_ncaaf_fcs_odds", lambda date: [])
    monkeypatch.setattr(odds_api, "filter_games_today_only", lambda games: games)
    empty = sp.fetch_live_odds_dataframe(["NBA", "NCAAF"], schedule_start="2026-10-03", schedule_end="2026-10-03")
    assert empty.empty and len(empty.attrs["ncaaf_schedule"]["events"]) == 2
    assert empty.attrs["provider_health"]["sports"]["americanfootball_ncaaf"]["outcome"] == "SUCCESS_EMPTY"
    from test_provider_caller_health import game as nba_game
    scenario["basketball_nba"] = [nba_game()]
    scenario["americanfootball_ncaaf"] = requests.Timeout(CANARY)
    frame = sp.fetch_live_odds_dataframe(["NBA", "NCAAF"], schedule_start="2026-10-03", schedule_end="2026-10-03")
    assert len(frame) == 1 and frame.iloc[0]["novig_home_price"] == -108
    diagnostics = {"ncaaf_schedule": frame.attrs["ncaaf_schedule"], "provider_health": frame.attrs["provider_health"]}
    ns.refresh_coverage(diagnostics)
    assert diagnostics["ncaaf_coverage"]["counts"]["scheduled"] == 2
    assert all("PROVIDER_FAILURE" in r["exclusion_reasons"] for r in diagnostics["ncaaf_coverage"]["rows"])
    assert CANARY not in json.dumps(diagnostics)


@pytest.mark.parametrize("fallback_fails", [False, True])
def test_actual_caller_normalized_games_and_expansion_keep_distinct_event_ids(monkeypatch, fallback_fails):
    from core import streamlit_pipeline as sp
    from app_core import odds_api, college_novig
    inventory=inv([event(), event("2", "2026-10-04T01:00Z")])
    monkeypatch.setattr(ns, "fetch_schedule", lambda *args: inventory)
    class Client:
        def __init__(self, **kwargs): pass
        def get_odds(self, *args, **kwargs): return [game(), game("second", kickoff="2026-10-04T01:00Z")]
    monkeypatch.setattr(odds_api, "TheOddsAPIClient", Client)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: "offline")
    monkeypatch.setattr(college_novig, "recover_college_novig", lambda games, key: games)
    def fallback(date):
        if fallback_fails:raise requests.Timeout(CANARY)
        return [game("espn-1", "McNeese")]
    monkeypatch.setattr(fcs, "fetch_espn_ncaaf_fcs_odds", fallback)
    monkeypatch.setattr(odds_api, "filter_games_today_only", lambda games: [])
    frame = sp.fetch_live_odds_dataframe(["NCAAF"], schedule_start="2026-10-03", schedule_end="2026-10-03")
    assert len(frame) == 2 and frame["game_id"].tolist() == ["primary", "second"]
    frame["matchup_id"] = sp._matchup_id(frame)
    expanded, _ = sp._expand_live_odds_to_bet_rows(frame)
    assert expanded["matchup_id"].nunique() == 2
    assert expanded["schedule_event_id"].nunique() == 2


def test_unresolved_distinct_provider_events_keep_their_inventory_keys():
    from core import streamlit_pipeline as sp
    frame = pd.DataFrame(ns.merge_schedule_odds([game("first"), game("second")], [], inv([event(), event("2")])))
    frame["league"] = "NCAAF"
    assert sp._matchup_id(frame).nunique() == 2


def test_actual_caller_keeps_opaque_same_kickoff_provider_ids_unresolved(monkeypatch):
    from core import streamlit_pipeline as sp
    from app_core import odds_api, college_novig
    inventory = inv()
    monkeypatch.setattr(ns, "fetch_schedule", lambda *args: inventory)
    original = [game("first"), game("second")]
    original[1]["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = -115
    class Client:
        def __init__(self, **kwargs): pass
        def get_odds(self, *args, **kwargs): return deepcopy(original)
    monkeypatch.setattr(odds_api, "TheOddsAPIClient", Client)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: "offline")
    monkeypatch.setattr(college_novig, "recover_college_novig", lambda games, key: games)
    monkeypatch.setattr(fcs, "fetch_espn_ncaaf_fcs_odds", lambda date: [])
    frame = sp.fetch_live_odds_dataframe(["NCAAF"], schedule_start="2026-10-03", schedule_end="2026-10-03")
    assert frame["game_id"].tolist() == ["first", "second"]
    assert frame["novig_home_price"].tolist() == [-108, -115]
    assert frame["schedule_match_status"].tolist() == ["AMBIGUOUS_PROVIDER_ID"] * 2
    assert frame["football_identity_status"].tolist() == ["CONFLICT"] * 2
    assert frame["historical_matchup_id"].tolist() == ["historical:first", "historical:second"]
    receipts = [json.loads(value) for value in frame["provider_quotes"]]
    assert {q["provider_event_id"] for quotes in receipts for q in quotes} == {"first", "second"}
    expanded, _ = sp._expand_live_odds_to_bet_rows(frame)
    assert expanded["matchup_id"].nunique() == 2
    assert set(expanded["football_identity_status"]) == {"CONFLICT"}
    report = ns.coverage(inventory, frame, expanded, now="2026-10-03T17:00Z")
    assert report["counts"]["scheduled"] == 1 and report["counts"]["matched"] == 0
    assert report["counts"]["qualified"] == 0
    assert "IDENTITY_UNRESOLVED" in report["rows"][0]["exclusion_reasons"]
    assert original[0]["matchup_id"] == "historical:first"


@pytest.mark.parametrize("start,end", [("invalid","2026-10-03"),("2026-10-04","2026-10-03"),("2026-10-01","2026-11-01")])
def test_invalid_or_unbounded_date_ranges_are_rejected_before_requests(start,end):
    with pytest.raises(ValueError):ns.fetch_schedule(start,end)


def test_inventory_display_download_works_without_candidate_rows(monkeypatch):
    from app.ui import ncaaf_inventory as panel
    captured = {}
    class UI:
        def subheader(self, *args): pass
        def caption(self, *args): pass
        def warning(self, value): captured["warning"] = value
        def dataframe(self, value, **kwargs): captured["table"] = value
        def download_button(self, label, data, **kwargs): captured["csv"] = data
    monkeypatch.setattr(panel, "st", UI())
    report = ns.coverage(inv(), now="2026-10-03T17:00Z")
    panel.render_inventory({"ncaaf_coverage": report})
    assert len(captured["table"]) == 1 and "espn:college-football:1" in captured["csv"]
    assert "QUALIFIED_SELECTION_UNAVAILABLE" in captured["csv"]


def test_date_range_changes_analysis_signature():
    import streamlit_app as app
    first={"sports":["NCAAF"], "schedule_start":"2026-10-03", "schedule_end":"2026-10-03"}
    assert app._analysis_input_signature(first) != app._analysis_input_signature({**first,"schedule_end":"2026-10-04"})


def test_actual_streamlit_handler_keeps_inventory_when_analysis_is_empty(monkeypatch):
    import streamlit_app as app
    from app_core import prediction_evidence
    diagnostics={"ncaaf_schedule": inv()}
    ns.refresh_coverage(diagnostics)
    calls=[]
    def pipeline(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(), pd.DataFrame(), diagnostics
    monkeypatch.setattr(app, "run_analysis_pipeline", pipeline)
    monkeypatch.setattr(app, "load_theover_csv", lambda upload: (pd.DataFrame(), None))
    monkeypatch.setattr(prediction_evidence, "begin_run", lambda controls: None)
    controls={"sports":["NCAAF"],"use_ml":False,"schedule_start":"2026-10-03","schedule_end":"2026-10-03"}
    state, warnings, errors=app._run_pipeline(controls)
    assert state["analysis_df"].empty and not errors
    assert state["diagnostics"]["ncaaf_coverage"]["counts"]["scheduled"] == 1
    assert calls[0]["schedule_start"] == "2026-10-03"


def test_schedule_only_results_become_stale_after_date_change():
    import streamlit_app as app
    controls={"sports":["NCAAF"],"schedule_start":"2026-10-03","schedule_end":"2026-10-03"}
    state={"analysis_df":pd.DataFrame(),"diagnostics":{"ncaaf_schedule":inv()},
           "last_successful_pipeline_signature":app._analysis_input_signature(controls)}
    assert not app._analysis_inputs_stale(state,controls)
    assert app._analysis_inputs_stale(state,{**controls,"schedule_end":"2026-10-04"})
