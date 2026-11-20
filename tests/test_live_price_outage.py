"""When the Odds API refuses every request, the board says so instead of ranking -110 placeholders.

2026-11-20, about 4:30 PM ET: the month's credits ran out, every sport answered HTTP 401, the
Refresh fell back to the uploaded rows and priced all 264 candidate rows at a -110 placeholder.
The board showed them as picks and said nothing. At 12:33 PM the same day all 264 rows read
``odds_api``.
"""
import logging

import pandas as pd
import pytest
import requests
from unittest.mock import Mock

from app_core import live_prices, odds_api

SECRET = "PRIVATE-KEY-abc123"


def _answer(status, body=None, headers=None):
    answer = requests.Response()
    answer.status_code = status
    answer.headers.update(headers or {})
    answer.json = Mock(return_value=[] if body is None else body) if not isinstance(body, Exception) else Mock(side_effect=body)
    return answer


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(odds_api, "last_refusal", {})
    monkeypatch.setattr(odds_api, "last_usage", {})
    return odds_api.TheOddsAPIClient(SECRET)


# ---- the provider's own reason ----

@pytest.mark.parametrize("body, code", [
    ({"message": "Usage quota has been reached", "error_code": "OUT_OF_USAGE_CREDITS"}, "OUT_OF_USAGE_CREDITS"),
    ({"error_code": "INVALID_KEY"}, "INVALID_KEY"),
    ({"error_code": "invalid key " + SECRET}, None),          # free text never passes
    ({"error_code": "X" * 41}, None),
    ({"error_code": 401}, None),
    (["OUT_OF_USAGE_CREDITS"], None),
    (ValueError("not JSON"), None),
])
def test_only_a_short_upper_case_error_code_is_kept(body, code):
    assert odds_api.provider_error_code(_answer(401, body)) == code


def test_a_refusal_is_recorded_with_its_code_and_remaining_credits_and_never_the_message_or_key(client, monkeypatch, caplog):
    body = {"message": f"Usage quota has been reached for {SECRET}", "error_code": "OUT_OF_USAGE_CREDITS"}
    monkeypatch.setattr(requests, "get", Mock(return_value=_answer(401, body, {"x-requests-remaining": "0"})))
    with caplog.at_level(logging.ERROR, logger="app_core.odds_api"), pytest.raises(odds_api.OddsAPIAuthError):
        client.get_odds("basketball_ncaab")
    assert odds_api.last_refusal == {"sport": "basketball_ncaab", "http_status": 401,
                                     "code": "OUT_OF_USAGE_CREDITS", "remaining": "0"}
    assert "provider_code=OUT_OF_USAGE_CREDITS" in caplog.text and "Odds API failed for basketball_ncaab [HTTP 401]" in caplog.text
    assert SECRET not in caplog.text and "quota" not in caplog.text


def test_a_good_answer_clears_the_refusal_and_still_keeps_the_usage_counters(client, monkeypatch):
    odds_api.last_refusal.update(sport="basketball_nba", http_status=401, code="OUT_OF_USAGE_CREDITS", remaining="0")
    headers = {"x-requests-last": "3", "x-requests-used": "120", "x-requests-remaining": "880"}
    monkeypatch.setattr(requests, "get", Mock(return_value=_answer(200, [], headers)))
    client.get_odds("basketball_nba")
    assert odds_api.last_refusal == {}
    assert odds_api.last_usage == {"sport": "basketball_nba", "last": "3", "used": "120", "remaining": "880"}


def test_other_failures_leave_the_refusal_alone(client, monkeypatch):
    monkeypatch.setattr(requests, "get", Mock(return_value=_answer(500, {"error_code": "SERVER"})))
    with pytest.raises(requests.exceptions.HTTPError):
        client.get_odds("basketball_nba")
    assert odds_api.last_refusal == {}


# ---- what the board says ----

def _candidates(*sources):
    return pd.DataFrame({"matchup_id": [f"g{i // 2}" for i in range(len(sources))], "odds_source": list(sources)})


OUT = {"sport": "basketball_ncaab", "http_status": 401, "code": "OUT_OF_USAGE_CREDITS", "remaining": "0"}


def test_the_day_the_credits_ran_out_the_board_says_so():
    gap = live_prices.outage(_candidates(*["fallback_novig"] * 4), OUT)
    assert gap["games"] == 2 and gap["code"] == "OUT_OF_USAGE_CREDITS"
    message = gap["message"]
    assert message.startswith("No live prices.") and "HTTP 401, OUT_OF_USAGE_CREDITS" in message
    assert "credits are used up" in message and "-110 placeholder" in message and "nothing on this page is a pick" in message


@pytest.mark.parametrize("refusal, expected", [
    ({"http_status": 401, "code": None, "remaining": "0"}, "credits are used up"),
    ({"http_status": 401, "code": "INVALID_KEY", "remaining": None}, "Check the Odds API key"),
    ({"http_status": 401, "code": None, "remaining": None}, "usual cause is the month's credits running out"),
    ({}, "returned no prices"),
])
def test_the_reason_follows_what_the_provider_said(refusal, expected):
    assert expected in live_prices.outage(_candidates("fallback_novig", "espn_draftkings_fallback"), refusal)["message"]


@pytest.mark.parametrize("candidates", [
    _candidates("odds_api", "odds_api"),                       # an ordinary day
    _candidates("odds_api", "fallback_novig"),                 # one live price is enough to say nothing
    _candidates(None, None),                                   # no recorded source is not an outage
    pd.DataFrame({"matchup_id": ["g1"]}),
    pd.DataFrame(),
    None,
])
def test_nothing_is_said_while_any_live_price_exists(candidates):
    assert live_prices.outage(candidates, OUT) is None


def test_the_placeholder_sources_match_the_ones_the_pipeline_never_funds():
    from core.streamlit_pipeline import _is_nonproduction_odds_source
    for source in ("fallback_novig", "espn_draftkings_fallback", "odds_api", "novig", "", None):
        assert live_prices.is_placeholder(source) == _is_nonproduction_odds_source(source)


def _outage_app():
    import pandas as pd
    import streamlit as st
    from app.ui.daily_dashboard import render_daily_dashboard
    from app_core import odds_api
    odds_api.last_refusal.clear()
    odds_api.last_refusal.update(sport="basketball_ncaab", http_status=401, code="OUT_OF_USAGE_CREDITS", remaining="0")
    today, details = st.tabs(["Today", "Details"])
    board = pd.DataFrame([{"league": "NCAAB", "Home": "Rice", "Away": "Tulsa", "best_pick": "Under 145.5",
                           "market_type": "total_under", "matchup_id": "g1", "production_edge": 0.0, "Play_Stake": 0,
                           "production_eligible": False, "odds_american": -110, "production_win_probability": 0.5,
                           "production_expected_value": -0.045, "rating_basis": "Price only"}])
    candidates = pd.DataFrame([{"matchup_id": "g1", "market_type": "total_under", "best_pick": "Under 145.5",
                                "rating_basis": "Price only", "odds_source": "fallback_novig"}])
    render_daily_dashboard(today.empty(), details.empty(), board, candidates)
    odds_api.last_refusal.clear()


def test_the_homepage_board_shows_the_outage_above_the_picks():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(_outage_app).run()
    assert not app.exception
    errors = [e.value for e in app.error]
    assert len(errors) == 1 and "this month's credits are used up" in errors[0]


# ---- the closing-capture job ----

def test_the_capture_loop_prints_a_refusal_instead_of_only_the_last_good_count(monkeypatch, capsys):
    import json
    import importlib.util
    from pathlib import Path
    import core.streamlit_pipeline as pipeline
    spec = importlib.util.spec_from_file_location(
        "closing_capture_loop", Path(__file__).resolve().parents[1] / "scripts" / "closing_capture_loop.py")
    loop = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loop)
    monkeypatch.setattr(odds_api, "last_usage", {"sport": "basketball_ncaab", "last": "3", "used": "19979", "remaining": "21"})
    monkeypatch.setattr(odds_api, "last_refusal", {})
    answers = iter([None, OUT])

    def fetch_live_odds_dataframe(sports):
        odds_api.last_refusal.clear()
        odds_api.last_refusal.update(next(answers) or {})
        return pd.DataFrame()
    monkeypatch.setattr(pipeline, "fetch_live_odds_dataframe", fetch_live_odds_dataframe)
    now = [0.0]

    def capture(database, *, fetch=None):
        fetch(["NCAAB"])
        return {"verified": 0, "unavailable": 1}

    def sleep(seconds):
        now[0] += seconds
    loop.run("db", minutes=21, interval=10, capture=capture, restore=lambda path: None, backup=lambda path: None,
             clock=lambda: now[0], sleep=sleep)
    cycles = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert "provider_refusal" not in cycles[0] and cycles[0]["provider_usage"]["remaining"] == "21"
    assert cycles[1]["provider_refusal"] == OUT                  # the refused cycle says so beside the stale count
