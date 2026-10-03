"""Exercise the real provider caller and diagnostic boundary with no sockets."""
from contextlib import nullcontext
from copy import deepcopy
import json
import socket
import urllib.request

import pandas as pd
import pytest
import requests


CANARY = "PRIVATE_CREDENTIAL_CANARY_73f8"
URL = "https://provider.invalid/odds?apiKey=" + CANARY
from app_core.espn_ncaaf_odds import fetch_espn_ncaaf_fcs_odds as ORIGINAL_FCS_FETCH


def http_error(status):
    response = requests.Response()
    response.status_code = status
    response.url = URL
    response._content = ("private response body " + CANARY).encode()
    response.request = requests.Request("GET", URL).prepare()
    return requests.HTTPError("private exception " + CANARY, response=response, request=response.request)


def game(sport="basketball_nba", event="healthy"):
    return {"id": event, "matchup_id": event, "sport_key": sport,
            "home_team": "Boston Celtics", "away_team": "Miami Heat",
            "commence_time": "2026-10-04T23:00:00Z",
            "bookmakers": [{"key": "novig", "markets": [{"key": "spreads", "outcomes": [
                {"name": "Boston Celtics", "point": -3.5, "price": -108},
                {"name": "Miami Heat", "point": 3.5, "price": -102}]}]}]}


@pytest.fixture
def caller(monkeypatch, tmp_path, caplog):
    attempted = []

    def blocked(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("Real network is forbidden in provider-caller regressions")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.chdir(tmp_path)
    from core import streamlit_pipeline as sp
    from app_core import odds_api, college_novig, nfl_novig, espn_ncaaf_odds, football_identity_capture
    scenario = {}
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def get_odds(self, sport_key, date=None):
            calls.append((sport_key, date))
            value = scenario.get(sport_key, [])
            if isinstance(value, Exception):
                raise value
            return deepcopy(value)

    monkeypatch.setattr(odds_api, "TheOddsAPIClient", Client)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: CANARY)
    monkeypatch.setattr(odds_api, "filter_games_today_only", lambda games: games)
    monkeypatch.setattr(college_novig, "recover_college_novig", lambda games, key: games)
    monkeypatch.setattr(nfl_novig, "recover_nfl_novig", lambda games, key: games)
    monkeypatch.setattr(espn_ncaaf_odds, "fetch_espn_ncaaf_fcs_odds", lambda date: [])
    monkeypatch.setattr(football_identity_capture, "collect", lambda games, sport: games)
    caplog.set_level("INFO", logger=sp.logger.name)
    yield sp, scenario, calls, odds_api
    assert attempted == [], "An unmocked dependency attempted real network access"
    assert CANARY not in caplog.text
    assert URL not in caplog.text


def health(frame):
    value = frame.attrs["provider_health"]
    assert CANARY not in json.dumps(value)
    assert URL not in json.dumps(value)
    return value


@pytest.mark.parametrize("result,category,status,aggregate", [
    (http_error(403).response, "AUTHENTICATION_FAILURE", 403, "PARTIAL_FAILURE"),
    (requests.Timeout(CANARY, request=requests.Request("GET", URL).prepare()), "TIMEOUT", None, "PARTIAL_FAILURE"),
    (requests.ConnectionError(CANARY, request=requests.Request("GET", URL).prepare()), "TRANSPORT_FAILURE", None, "PARTIAL_FAILURE"),
    ({"events": [], "private_body": CANARY}, "SUCCESS_EMPTY", 200, "SUCCESS"),
    ([CANARY], "INVALID_RESPONSE", 200, "PARTIAL_FAILURE"),
])
def test_actual_caller_with_real_fcs_helper_never_logs_provider_secrets(caller, monkeypatch, caplog,
                                                                     result, category, status, aggregate):
    sp, scenario, _, _ = caller
    from app_core import espn_ncaaf_odds
    scenario.update(basketball_nba=[game()], americanfootball_ncaaf=[])
    monkeypatch.setattr(espn_ncaaf_odds, "fetch_espn_ncaaf_fcs_odds", ORIGINAL_FCS_FETCH)

    def private_failure(*args, **kwargs):
        assert kwargs["timeout"] == 15
        if isinstance(result, Exception):
            raise result
        if isinstance(result, requests.Response):
            return result
        response = requests.Response()
        response.status_code = 200
        response.url = URL
        response.request = requests.Request("GET", URL).prepare()
        response._content = json.dumps(result).encode()
        return response

    monkeypatch.setattr(espn_ncaaf_odds.requests, "get", private_failure)
    frame = sp.fetch_live_odds_dataframe(["NBA", "NCAAF"], date="2026-10-04")
    assert len(frame) == 1 and frame.iloc[0]["game_id"] == "healthy"
    assert frame.iloc[0]["novig_home_price"] == -108
    value = health(frame)
    assert value["status"] == aggregate
    primary = value["sports"]["americanfootball_ncaaf"]
    assert primary["outcome"] == "SUCCESS_EMPTY" and primary["http_status"] is None
    detail = {"outcome": category, "http_status": status}
    assert primary["fallback_outcomes"]["ncaaf_fcs"] == detail
    failed = category != "SUCCESS_EMPTY"
    assert primary["fallback_errors"] == ({"ncaaf_fcs": detail} if failed else {})
    if failed and category != "INVALID_RESPONSE":
        assert f"outcome={category} http_status={status}" in caplog.text

    from core.run_readiness import build_readiness
    from app.ui import readiness_dashboard as panel
    readiness = build_readiness(None, None, diagnostics={"provider_health": value})
    assert readiness["provider_health"] == value
    ui = UI()
    monkeypatch.setattr(panel, "st", ui)
    panel.render_readiness_dashboard(diagnostics={"provider_health": readiness["provider_health"]})
    tables = [args[0] for name, args, _ in ui.outputs if name == "dataframe" and "Sport" in args[0].columns]
    displayed = tables[0].set_index("Sport").loc["americanfootball_ncaaf"]
    assert category in displayed["Fallback outcomes"]
    assert (category in displayed["Fallback errors"]) == failed
    if status is not None:
        assert f"HTTP {status}" in displayed["Fallback outcomes"]
    downloads = [args[1] for name, args, _ in ui.outputs if name == "download_button" and args[0] == "Download provider outcomes"]
    assert json.loads(downloads[0]) == value
    assert CANARY not in repr(ui.outputs) and URL not in repr(ui.outputs)


def test_real_fcs_receipt_is_list_compatible_and_refresh_clears_fallback_failure(caller, monkeypatch):
    sp, scenario, _, _ = caller
    from app_core import espn_ncaaf_odds
    scenario.update(basketball_nba=[game()], americanfootball_ncaaf=[])
    monkeypatch.setattr(espn_ncaaf_odds, "fetch_espn_ncaaf_fcs_odds", ORIGINAL_FCS_FETCH)
    failed_response = http_error(403).response
    monkeypatch.setattr(espn_ncaaf_odds.requests, "get", lambda *a, **k: failed_response)
    failed = health(sp.fetch_live_odds_dataframe(["NBA", "NCAAF"], date="2026-10-04"))
    assert failed["status"] == "PARTIAL_FAILURE"

    response = requests.Response()
    response.status_code = 200
    response._content = b'{"events": []}'
    monkeypatch.setattr(espn_ncaaf_odds.requests, "get", lambda *a, **k: response)
    games = ORIGINAL_FCS_FETCH("2026-10-04")
    assert isinstance(games, list) and games == [] and json.dumps(games) == "[]"
    assert deepcopy(games).provider_outcome == {"outcome": "SUCCESS_EMPTY", "http_status": 200}
    refreshed = health(sp.fetch_live_odds_dataframe(["NBA", "NCAAF"], date="2026-10-04"))
    assert refreshed["status"] == "SUCCESS"
    assert refreshed["sports"]["americanfootball_ncaaf"]["fallback_errors"] == {}
    assert refreshed["sports"]["americanfootball_ncaaf"]["fallback_outcomes"]["ncaaf_fcs"] == games.provider_outcome
    assert failed["sports"]["americanfootball_ncaaf"]["fallback_errors"]["ncaaf_fcs"]["http_status"] == 403


@pytest.mark.parametrize("exc,expected,code", [
    (http_error(401), "AUTHENTICATION_FAILURE", 401),
    (http_error(403), "AUTHENTICATION_FAILURE", 403),
    (http_error(429), "RATE_LIMIT_EXHAUSTED", 429),
    (http_error(500), "HTTP_FAILURE", 500),
    (http_error(204), "HTTP_FAILURE", 204),
    (requests.Timeout(CANARY, request=requests.Request("GET", URL).prepare()), "TIMEOUT", None),
    (requests.ConnectionError(CANARY, request=requests.Request("GET", URL).prepare()), "TRANSPORT_FAILURE", None),
    (RuntimeError(CANARY), "PROVIDER_FAILURE", None),
])
def test_actual_caller_preserves_healthy_sport_and_sanitizes_failures(caller, exc, expected, code):
    sp, scenario, calls, _ = caller
    scenario.update(basketball_nba=[game()], icehockey_nhl=exc)
    frame = sp.fetch_live_odds_dataframe(["NBA", "NHL"])
    assert frame["game_id"].tolist() == ["healthy"]
    assert frame.iloc[0]["novig_home_price"] == -108
    report = health(frame)
    assert report["status"] == "PARTIAL_FAILURE"
    assert report["returned_games"] == 1
    assert report["sports"]["basketball_nba"]["outcome"] == "SUCCESS"
    failed = report["sports"]["icehockey_nhl"]
    assert (failed["outcome"], failed["http_status"]) == (expected, code)
    assert failed["received_games"] == 0
    assert calls == [("basketball_nba", None), ("icehockey_nhl", None)]


def test_success_empty_and_failed_empty_remain_distinct_across_refresh(caller):
    sp, scenario, _, _ = caller
    scenario["basketball_nba"] = requests.Timeout(CANARY)
    failed = sp.fetch_live_odds_dataframe(["NBA"])
    saved = deepcopy(health(failed))
    assert failed.empty and saved["status"] == "FAILED"
    scenario["basketball_nba"] = [game()]
    assert health(sp.fetch_live_odds_dataframe(["NBA"]))["status"] == "SUCCESS"
    scenario["basketball_nba"] = []
    empty = sp.fetch_live_odds_dataframe(["NBA"])
    assert empty.empty and health(empty)["status"] == "SUCCESS_EMPTY"
    assert health(empty)["sports"]["basketball_nba"]["outcome"] == "SUCCESS_EMPTY"
    assert health(failed) == saved  # New fetch cannot mutate an earlier result.


def test_filtered_empty_is_not_inferred_provider_failure(caller, monkeypatch):
    sp, scenario, _, api = caller
    scenario["basketball_nba"] = [game()]
    monkeypatch.setattr(api, "filter_games_today_only", lambda games: [])
    frame = sp.fetch_live_odds_dataframe(["NBA"])
    assert frame.empty
    assert health(frame)["status"] == "SUCCESS"
    assert health(frame)["sports"]["basketball_nba"]["processing"] == "FILTERED_EMPTY"


@pytest.mark.parametrize("value", [{"message": CANARY, "status": 401, "url": URL}, None, CANARY])
def test_untrusted_payload_is_invalid_not_authentication_or_success(caller, value):
    sp, scenario, _, _ = caller
    scenario["basketball_nba"] = value
    item = health(sp.fetch_live_odds_dataframe(["NBA"]))["sports"]["basketball_nba"]
    assert item["outcome"] == "INVALID_RESPONSE"
    assert item["http_status"] is None


@pytest.mark.parametrize("status", [True, "401 " + CANARY, -1, 600])
def test_only_verified_typed_http_status_reaches_diagnostics(caller, status):
    sp, scenario, _, _ = caller
    scenario["basketball_nba"] = http_error(status)
    item = health(sp.fetch_live_odds_dataframe(["NBA"]))["sports"]["basketball_nba"]
    assert item["outcome"] == "HTTP_FAILURE" and item["http_status"] is None


def test_missing_configuration_is_not_success_empty(caller, monkeypatch):
    sp, _, calls, _ = caller
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: None)
    frame = sp.fetch_live_odds_dataframe(["NBA"])
    assert frame.empty and calls == []
    assert health(frame)["sports"]["basketball_nba"]["outcome"] == "NOT_CONFIGURED"


def test_normalization_failure_preserves_other_sport_without_exception_text(caller, monkeypatch):
    sp, scenario, _, api = caller
    scenario.update(basketball_nba=[game()], icehockey_nhl=[game("icehockey_nhl", "bad")])

    def filtering(games):
        if games[0]["id"] == "bad":
            raise ValueError(CANARY + URL)
        return games

    monkeypatch.setattr(api, "filter_games_today_only", filtering)
    frame = sp.fetch_live_odds_dataframe(["NBA", "NHL"])
    assert frame["game_id"].tolist() == ["healthy"]
    item = health(frame)["sports"]["icehockey_nhl"]
    assert item["outcome"] == "SUCCESS" and item["processing"] == "FAILED"
    assert item["processing_error"] == {"outcome": "PROVIDER_FAILURE", "http_status": None}
    assert health(frame)["status"] == "PARTIAL_FAILURE"


def test_fallback_rows_do_not_hide_primary_failure(caller, monkeypatch):
    sp, scenario, _, _ = caller
    from app_core import espn_ncaaf_odds
    scenario["americanfootball_ncaaf"] = http_error(403)
    fallback = game("americanfootball_ncaaf", "fallback")
    fallback["odds_feed_source"] = "espn_ncaaf_fcs_scoreboard"
    monkeypatch.setattr(espn_ncaaf_odds, "fetch_espn_ncaaf_fcs_odds", lambda date: [fallback])
    frame = sp.fetch_live_odds_dataframe(["NCAAF"])
    assert frame["game_id"].tolist() == ["fallback"]
    assert frame.iloc[0]["odds_feed_source"] == "espn_ncaaf_fcs_scoreboard"
    assert health(frame)["sports"]["americanfootball_ncaaf"]["outcome"] == "AUTHENTICATION_FAILURE"


@pytest.mark.parametrize("source", ["college_novig", "ncaaf_fcs", "nfl_novig"])
def test_fallback_exception_is_sanitized_and_original_games_survive(caller, monkeypatch, source):
    sp, scenario, _, _ = caller
    from app_core import college_novig, nfl_novig, espn_ncaaf_odds
    sport = "americanfootball_nfl" if source == "nfl_novig" else "americanfootball_ncaaf"
    scenario[sport] = [game(sport)]

    def failed(*args):
        raise requests.Timeout(CANARY, request=requests.Request("GET", URL).prepare())

    module, name = {"college_novig": (college_novig, "recover_college_novig"),
                    "nfl_novig": (nfl_novig, "recover_nfl_novig"),
                    "ncaaf_fcs": (espn_ncaaf_odds, "fetch_espn_ncaaf_fcs_odds")}[source]
    monkeypatch.setattr(module, name, failed)
    frame = sp.fetch_live_odds_dataframe(["NFL" if source == "nfl_novig" else "NCAAF"])
    assert frame["game_id"].tolist() == ["healthy"]
    assert health(frame)["sports"][sport]["fallback_errors"][source]["outcome"] == "TIMEOUT"


@pytest.mark.parametrize("statuses,category", [([429, 200], "SUCCESS_EMPTY"), ([429] * 4, "RATE_LIMIT_EXHAUSTED")])
def test_actual_caller_and_real_client_keep_bounded_429_contract(caller, monkeypatch, statuses, category):
    sp, _, _, api = caller
    import time
    # Replace the fake with the production client; only its HTTP dependency is mocked.
    # Captured before fixture patching; no production module reload is needed.
    client_type = ORIGINAL_CLIENT
    monkeypatch.setattr(api, "TheOddsAPIClient", client_type)
    calls = []

    def response(*args, **kwargs):
        calls.append(kwargs)
        r = requests.Response(); r.status_code = statuses[len(calls) - 1]
        r.url = URL; r.request = requests.Request("GET", URL).prepare()
        r._content = b"[]" if r.status_code == 200 else CANARY.encode()
        return r

    monkeypatch.setattr(requests, "get", response)
    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    frame = sp.fetch_live_odds_dataframe(["NBA"], date="2026-10-03")
    assert health(frame)["sports"]["basketball_nba"]["outcome"] == category
    assert len(calls) == len(statuses)
    assert all(c["timeout"] == 15 for c in calls)


class UI:
    def __init__(self, source="Current run"):
        self.session_state = {"readiness_snapshots": [("old", pd.DataFrame(), pd.DataFrame())]}
        self.source = source
        self.outputs = []

    def expander(self, *args, **kwargs):
        return nullcontext()

    def button(self, *args, **kwargs):
        return False

    def selectbox(self, label, options, **kwargs):
        return self.source if label == "Readiness source" else options[0]

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.outputs.append((name, args, kwargs))


def test_actual_readiness_panel_and_download_sanitize_stored_health(caller, monkeypatch):
    sp, scenario, _, _ = caller
    from app.ui import readiness_dashboard as panel
    scenario["basketball_nba"] = http_error(401)
    raw = deepcopy(health(sp.fetch_live_odds_dataframe(["NBA"])))
    raw.update(message=CANARY, request_url=URL)
    raw["sports"]["basketball_nba"].update(response_body=CANARY)
    raw["sports"]["basketball_nba"]["fallback_outcomes"] = {
        "ncaaf_fcs": {"outcome": "AUTHENTICATION_FAILURE", "http_status": 403,
                      "response_body": CANARY, "request_url": URL},
        "college_novig": {"outcome": CANARY, "http_status": CANARY},
        CANARY: {"outcome": CANARY, "http_status": 403},
    }
    raw["sports"][CANARY] = {"outcome": CANARY}
    ui = UI(); monkeypatch.setattr(panel, "st", ui)
    panel.render_readiness_dashboard(diagnostics={"provider_health": raw})
    assert "AUTHENTICATION_FAILURE" in repr(ui.outputs)
    assert CANARY not in repr(ui.outputs) and URL not in repr(ui.outputs)
    assert any(name == "download_button" and args[0] == "Download provider outcomes" for name, args, _ in ui.outputs)
    saved = UI("Saved snapshot"); monkeypatch.setattr(panel, "st", saved)
    panel.render_readiness_dashboard(diagnostics={"provider_health": raw})
    assert "Download provider outcomes" not in repr(saved.outputs)


def test_analysis_boundary_and_readiness_preserve_empty_fetch_health_on_refresh(caller, monkeypatch):
    sp, scenario, _, _ = caller
    from app_core import external_data_fetcher
    from core.run_readiness import build_readiness
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(external_data_fetcher, "enrich_with_external_data", lambda frame: frame)
    scenario["basketball_nba"] = requests.Timeout(CANARY)
    _, _, first = sp.run_analysis_pipeline(sports=["NBA"], use_ml=False)
    assert first["provider_health"]["status"] == "FAILED"
    assert CANARY not in json.dumps(first["provider_health"])
    scenario["basketball_nba"] = []
    _, _, refreshed = sp.run_analysis_pipeline(sports=["NBA"], use_ml=False)
    assert refreshed["provider_health"]["status"] == "SUCCESS_EMPTY"
    assert first["provider_health"]["status"] == "FAILED"
    report = build_readiness(None, None, diagnostics=refreshed)
    assert report["provider_health"] == refreshed["provider_health"]


def test_actual_streamlit_handler_returns_empty_health_and_replaces_refresh_state(caller, monkeypatch):
    sp, scenario, _, _ = caller
    import streamlit_app as application
    from app_core import external_data_fetcher, prediction_evidence
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(external_data_fetcher, "enrich_with_external_data", lambda frame: frame)
    monkeypatch.setattr(prediction_evidence, "begin_run", lambda controls: None)
    monkeypatch.setattr(application, "load_theover_csv", lambda upload: (None, None))
    controls = {"sports": ["NBA"], "use_ml": False}
    scenario["basketball_nba"] = http_error(401)
    updates, warnings, errors = application._run_pipeline(controls)
    state = dict(updates)
    assert state["analysis_df"].empty
    assert state["diagnostics"]["provider_health"]["status"] == "FAILED"
    assert CANARY not in repr((warnings, errors, state["diagnostics"]))
    scenario["basketball_nba"] = []
    refreshed, warnings, errors = application._run_pipeline(controls)
    state.update(refreshed)  # The same update contract used by main().
    assert state["diagnostics"]["provider_health"]["status"] == "SUCCESS_EMPTY"
    assert updates["diagnostics"]["provider_health"]["status"] == "FAILED"
    assert CANARY not in repr((warnings, errors, state["diagnostics"]))


# No request occurs at import; capturing the class avoids reloading production modules.
from app_core.odds_api import TheOddsAPIClient as ORIGINAL_CLIENT
