from unittest.mock import mock_open

import pandas as pd
import pytest

from app_core import kalshi_integrator as ki


def _row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "league": ["NCAAF"],
            "market_type": ["spread_home"],
            "home_team": ["Lamar"],
            "away_team": ["Abilene Christian"],
            "game_date": ["2026-08-29T00:00:00Z"],
            "spread_line": [-7.5],
            "best_pick": ["Lamar -7.5"],
        }
    )


@pytest.fixture(autouse=True)
def _clear_kalshi_cache(monkeypatch):
    ki._SERIES_FETCH_ERRORS.clear()
    for attr in ("series_cache", "series_cache_meta"):
        if hasattr(ki.enrich_with_kalshi_markets, attr):
            getattr(ki.enrich_with_kalshi_markets, attr).clear()
    monkeypatch.setattr("builtins.open", mock_open())


def test_reason_distinguishes_series_fetch_failure(monkeypatch) -> None:
    monkeypatch.setattr(ki, "_fetch_series_events", lambda _series: [])
    ki._SERIES_FETCH_ERRORS["KXNCAAFSPREAD"] = "KalshiAPIError: timeout"

    out = ki.enrich_with_kalshi_markets(_row())

    assert out.loc[0, "kalshi_match_reason"] == "series_fetch_failed"
    assert out.loc[0, "kalshi_series_fetch_error"] == "KalshiAPIError: timeout"


def test_reason_distinguishes_empty_series(monkeypatch) -> None:
    monkeypatch.setattr(ki, "_fetch_series_events", lambda _series: [])

    out = ki.enrich_with_kalshi_markets(_row())

    assert out.loc[0, "kalshi_match_reason"] == "no_series_events"
    assert out.loc[0, "kalshi_series_ticker"] == "KXNCAAFSPREAD"
    assert int(out.loc[0, "kalshi_candidate_event_count"]) == 0
    assert int(out.loc[0, "kalshi_dated_candidate_event_count"]) == 0


def test_reason_distinguishes_events_outside_date_window(monkeypatch) -> None:
    monkeypatch.setattr(
        ki,
        "_fetch_series_events",
        lambda _series: [
            {
                "event_ticker": "KXNCAAFSPREAD-26SEP20OTHER",
                "title": "Other Team at Another Team",
                "sub_title": "Sep 20",
                "close_time": "2026-09-20T23:00:00Z",
            }
        ],
    )

    out = ki.enrich_with_kalshi_markets(_row())

    assert out.loc[0, "kalshi_match_reason"] == "no_events_in_date_window"
    assert int(out.loc[0, "kalshi_candidate_event_count"]) == 1
    assert int(out.loc[0, "kalshi_dated_candidate_event_count"]) == 0


def test_reason_distinguishes_low_team_match_score(monkeypatch) -> None:
    monkeypatch.setattr(ki, "_event_match_score", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr(
        ki,
        "_fetch_series_events",
        lambda _series: [
            {
                "event_ticker": "KXNCAAFSPREAD-26AUG29OTHER",
                "title": "Other Team at Another Team",
                "sub_title": "Aug 29",
                "close_time": "2026-08-29T23:00:00Z",
            }
        ],
    )

    out = ki.enrich_with_kalshi_markets(_row())

    assert out.loc[0, "kalshi_match_reason"] == "team_match_below_threshold"
    assert int(out.loc[0, "kalshi_candidate_event_count"]) == 1
    assert int(out.loc[0, "kalshi_dated_candidate_event_count"]) == 1
    assert float(out.loc[0, "kalshi_best_event_score"]) < 10
