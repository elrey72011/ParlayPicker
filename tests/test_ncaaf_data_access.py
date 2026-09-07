import json
from unittest.mock import Mock, patch
import pytest
import requests
from app_core.ncaaf_data_access import probe_cfbd_access


def response(payload, status=200):
    return Mock(status_code=status, json=Mock(return_value=payload))


def good_get(url, **kwargs):
    if url.endswith("/games"):
        return response([dict(id=1, season=kwargs["params"]["year"], completed=True,
                              startDate="2025-08-30T12:00:00Z", startTimeTBD=False,
                              homeId=2, awayId=3, homePoints=20, awayPoints=10)])
    return response([dict(id=1, teams=[dict(teamId=t, stats=[dict(category="yards", stat="300")]) for t in (2, 3)])])


def test_good_samples_bounded_and_safe():
    get = Mock(side_effect=good_get)
    report = probe_cfbd_access("secret-value", current_year=2026, get=get)
    assert report["status"] == "samples_accessible"
    assert report["requests_made"] == get.call_count == 6
    assert [r["season"] for r in report["seasons"]] == [2025, 2024, 2023]
    assert "secret-value" not in json.dumps(report)
    for call in get.call_args_list:
        assert call.kwargs["allow_redirects"] is False
        assert call.kwargs["timeout"] == 6


@pytest.mark.parametrize("token", [None, "", "bad key"])
def test_missing_key_no_calls(token):
    get = Mock()
    assert probe_cfbd_access(token, get=get)["status"] == "missing_or_invalid_key"
    get.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 429, 500, 302])
def test_http_stops(status):
    get = Mock(return_value=response([], status))
    report = probe_cfbd_access("key", get=get)
    assert report["requests_made"] == 1
    assert report["seasons"][0]["status"] == f"http_{status}"


@pytest.mark.parametrize("error", [requests.Timeout("secret-value"), ValueError("secret-value")])
def test_failures_sanitized(error):
    get = Mock(side_effect=error)
    report = probe_cfbd_access("secret-value", get=get)
    assert report["status"] == "incomplete"
    assert report["requests_made"] == 1
    assert "secret-value" not in json.dumps(report)


@pytest.mark.parametrize("payload", [[], {}, [None], [{"completed": True}]])
def test_empty_malformed_not_success(payload):
    report = probe_cfbd_access("key", get=Mock(return_value=response(payload)))
    assert report["status"] == "incomplete"


def test_wrong_game_stats_not_success():
    def get(url, **kwargs):
        return good_get(url, **kwargs) if url.endswith("/games") else response([dict(id=999, teams=[])])
    report = probe_cfbd_access("key", get=get)
    assert not any(r["team_stats_accessible"] for r in report["seasons"])


def test_ui_only_probes_on_click():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string("from app.ui.ncaaf_data_access import render_ncaaf_data_access\nrender_ncaaf_data_access()")
    report = probe_cfbd_access("key", get=Mock(return_value=response([])))
    with patch("app.ui.ncaaf_data_access._token", return_value="key"), patch(
        "app.ui.ncaaf_data_access.probe_cfbd_access", return_value=report
    ) as probe:
        app.run()
        probe.assert_not_called()
        app.button[0].click().run()
        probe.assert_called_once_with("key")
        app.run()
        assert probe.call_count == 1
        assert not app.exception
