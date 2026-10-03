"""Provider outcomes are mocked; successful responses write only to pytest scratch."""
from unittest.mock import Mock

import pytest
import requests

from app_core.odds_api import OddsAPIAuthError, TheOddsAPIClient


def response(status=200, games=None, headers=None):
    result = requests.Response()
    result.status_code = status
    result.url = "https://provider.invalid/odds?apiKey=PRIVATE"
    result._content = b"PRIVATE provider error body"
    result.headers.update(headers or {})
    result.json = Mock(return_value=[] if games is None else games)
    return result


@pytest.fixture
def provider(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    get = Mock()
    sleep = Mock()
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr("time.sleep", sleep)
    return TheOddsAPIClient("PRIVATE"), get, sleep


@pytest.mark.parametrize("games", [[], [{"id": "game", "bookmakers": [{"key": "novig"}]}]])
def test_success_including_genuinely_empty_response(provider, games):
    client, get, sleep = provider
    get.return_value = response(games=games)
    assert client.get_odds("basketball_wnba", date="2026-10-02") == games
    assert get.call_count == 1
    assert get.call_args.kwargs["timeout"] == 15
    assert get.call_args.kwargs["params"]["commenceTimeFrom"] == "2026-10-02T04:00:00Z"
    assert get.call_args.kwargs["params"]["commenceTimeTo"] == "2026-10-03T03:59:59Z"
    sleep.assert_not_called()


@pytest.mark.parametrize("status", [401, 403])
def test_authentication_failure_is_explicit(provider, status, caplog):
    client, get, sleep = provider
    get.return_value = response(status)
    with pytest.raises(OddsAPIAuthError, match=f"basketball_wnba.*HTTP {status}") as caught:
        client.get_odds("basketball_wnba")
    assert caught.value.response.status_code == status
    assert "PRIVATE" not in str(caught.value) + caplog.text
    assert get.call_count == 1
    sleep.assert_not_called()


def test_rate_limit_retries_then_retains_success(provider):
    client, get, sleep = provider
    games = [{"id": "game", "bookmakers": [{"key": "novig"}]}]
    get.side_effect = [response(429), response(games=games)]
    assert client.get_odds("basketball_wnba") == games
    assert get.call_count == 2
    sleep.assert_called_once_with(2.0)


def test_rate_limit_exhaustion_respects_original_request_budget(provider, caplog):
    client, get, sleep = provider
    get.return_value = response(429)
    with pytest.raises(requests.exceptions.HTTPError, match="rate limit exhausted") as caught:
        client.get_odds("basketball_wnba")
    assert caught.value.response.status_code == 429
    assert get.call_count == 4
    assert [call.args[0] for call in sleep.call_args_list] == [2.0, 4.0, 8.0]
    assert all(call.kwargs["timeout"] == 15 for call in get.call_args_list)
    assert "PRIVATE" not in str(caught.value) + caplog.text


@pytest.mark.parametrize("status", [500, 204, 302])
def test_non_success_is_not_an_empty_slate(provider, status, caplog):
    client, get, sleep = provider
    get.return_value = response(status)
    with pytest.raises(requests.exceptions.HTTPError, match=f"HTTP {status}") as caught:
        client.get_odds("icehockey_nhl")
    assert caught.value.response.status_code == status
    assert "PRIVATE" not in str(caught.value) + caplog.text
    assert get.call_count == 1
    sleep.assert_not_called()


def test_timeout_is_not_an_empty_slate(provider, caplog):
    client, get, sleep = provider
    get.side_effect = requests.exceptions.Timeout("mock timeout")
    with pytest.raises(requests.exceptions.Timeout):
        client.get_odds("icehockey_nhl")
    assert "icehockey_nhl (Timeout)" in caplog.text
    assert get.call_count == 1
    sleep.assert_not_called()


def test_partial_slate_keeps_successes_and_per_sport_failures(provider):
    client, get, sleep = provider
    games = [{"id": "game", "bookmakers": [{"key": "novig"}]}]
    get.side_effect = [response(games=games), response(), response(401), response(500),
                       requests.exceptions.Timeout("mock timeout")]
    sports = ["basketball_wnba", "baseball_mlb", "americanfootball_nfl", "icehockey_nhl",
              "americanfootball_ncaaf"]
    results = client.get_odds_for_sports(sports)
    assert results == dict(zip(sports, [games, [], [], [], []]))
    assert client.last_fetch_errors == {
        "americanfootball_nfl": {"error_type": "OddsAPIAuthError", "http_status": 401},
        "icehockey_nhl": {"error_type": "HTTPError", "http_status": 500},
        "americanfootball_ncaaf": {"error_type": "Timeout", "http_status": None},
    }
    assert get.call_count == 5
    sleep.assert_not_called()
    get.side_effect = [response()]
    assert client.get_odds_for_sports(["americanfootball_nfl"]) == {"americanfootball_nfl": []}
    assert client.last_fetch_errors == {}  # A successful empty refresh clears old failures.


def test_rate_limit_failure_does_not_drop_a_later_successful_sport(provider):
    client, get, sleep = provider
    games = [{"id": "game", "bookmakers": [{"key": "novig"}]}]
    get.side_effect = [response(429)] * 4 + [response(games=games)]
    assert client.get_odds_for_sports(["icehockey_nhl", "basketball_wnba"]) == {
        "icehockey_nhl": [], "basketball_wnba": games,
    }
    assert client.last_fetch_errors == {"icehockey_nhl": {"error_type": "HTTPError", "http_status": 429}}
    assert get.call_count == 5
    assert sleep.call_count == 3
