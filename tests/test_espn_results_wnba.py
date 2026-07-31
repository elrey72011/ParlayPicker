from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from app_core.espn_results import ESPN_ENDPOINTS, fetch_espn_results


class _Response:
    status_code = 200

    def __init__(self, home: str, away: str, home_score: str, away_score: str):
        self._payload = {
            "events": [
                {
                    "competitions": [
                        {
                            "status": {"type": {"state": "post"}},
                            "competitors": [
                                {
                                    "homeAway": "home",
                                    "score": home_score,
                                    "team": {"displayName": home},
                                },
                                {
                                    "homeAway": "away",
                                    "score": away_score,
                                    "team": {"displayName": away},
                                },
                            ],
                        }
                    ]
                }
            ]
        }

    def json(self):
        return self._payload


def test_wnba_scoreboard_is_configured():
    assert ESPN_ENDPOINTS["WNBA"].endswith("/basketball/wnba/scoreboard")


def test_mixed_mlb_wnba_results_and_zero_score(monkeypatch):
    def fake_get(url, timeout):
        assert timeout == 10
        if "/wnba/" in url:
            return _Response("Las Vegas Aces", "New York Liberty", "104", "99")
        return _Response("Boston Red Sox", "Oakland Athletics", "5", "0")

    monkeypatch.setattr("app_core.espn_results.requests.get", fake_get)
    results = fetch_espn_results(
        ["MLB", "wnba", "MLB"],
        target_date=date(2026, 7, 30),
        attempts=1,
    )

    assert results["league"].tolist() == ["MLB", "WNBA"]
    assert results.loc[0, "away_score"] == 0
    assert results.attrs["requested_leagues"] == ["MLB", "WNBA"]
    assert results.attrs["unsupported_leagues"] == []


def test_transient_failure_is_retried(monkeypatch):
    calls = {"count": 0}

    def flaky_get(url, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.RequestException("temporary failure")
        return _Response("Las Vegas Aces", "New York Liberty", "104", "99")

    monkeypatch.setattr("app_core.espn_results.requests.get", flaky_get)
    results = fetch_espn_results(
        ["WNBA"],
        target_date=date(2026, 7, 30),
        attempts=2,
    )

    assert calls["count"] == 2
    assert len(results) == 1


def test_unsupported_league_is_reported(monkeypatch, caplog):
    monkeypatch.setattr(
        "app_core.espn_results.requests.get",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    results = fetch_espn_results(["WNBA", "XYZ"], attempts=1)

    assert results.attrs["unsupported_leagues"] == ["XYZ"]
    assert "No ESPN scoreboard provider configured" in caplog.text
