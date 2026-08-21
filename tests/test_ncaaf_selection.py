from types import SimpleNamespace

import pandas as pd

from app.ui.sidebar_controls import FALLBACK_SPORTS, _resolve_sports_options
from app_core import espn_results, feature_processing, odds_api
from app_core.kalshi_integrator import (
    MAX_LINE_TOLERANCE,
    MAX_TOTAL_LINE_TOLERANCE,
    league_series_ticker,
)
from app_core.market_probability_model import predict_market_probabilities
from collect_historical_data import ODDS_API_SPORTS
from core.schema.base_schema import ensure_base_schema
import core.streamlit_pipeline as sp


def _ncaaf_live_game():
    return {
        "id": "ncaaf-1",
        "matchup_id": "americanfootball_ncaaf:texas longhorns:ohio state buckeyes:2026-08-29",
        "sport_key": "americanfootball_ncaaf",
        "home_team": "Ohio State Buckeyes",
        "away_team": "Texas Longhorns",
        "commence_time": "2026-08-29T16:00:00Z",
        "bookmakers": [
            {
                "key": "novig",
                "markets": [
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": "Ohio State Buckeyes", "point": -3.5, "price": -108},
                            {"name": "Texas Longhorns", "point": 3.5, "price": -102},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": 55.5, "price": -105},
                            {"name": "Under", "point": 55.5, "price": -115},
                        ],
                    },
                ],
            }
        ],
    }


def test_ncaaf_is_selectable_and_uses_official_odds_api_key(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def get_odds(self, sport_key, date=None):
            calls.append((sport_key, date))
            return [_ncaaf_live_game()]

    monkeypatch.setattr(odds_api, "TheOddsAPIClient", FakeClient)
    monkeypatch.setattr(sp, "_get_odds_api_key", lambda: "fake")

    frame = sp.fetch_live_odds_dataframe(
        sports=["NCAAF"], date="2026-08-29T12:00:00Z"
    )

    assert "NCAAF" in FALLBACK_SPORTS
    assert "NCAAF" in _resolve_sports_options()
    assert ODDS_API_SPORTS["NCAAF"] == "americanfootball_ncaaf"
    assert calls == [("americanfootball_ncaaf", "2026-08-29T12:00:00Z")]
    assert frame["league"].eq("NCAAF").all()
    assert float(frame.iloc[0]["novig_home_point"]) == -3.5
    assert float(frame.iloc[0]["novig_over_point"]) == 55.5


def test_ncaaf_source_hint_wins_over_shared_college_mascots():
    raw = pd.DataFrame(
        {
            "league": [pd.NA],
            "sport": ["americanfootball_ncaaf"],
            "home_team": ["Georgia Bulldogs"],
            "away_team": ["Auburn Tigers"],
        }
    )

    recovered = sp._infer_missing_league_from_team_sets(raw, ["NCAAB", "NCAAF"])

    assert recovered.loc[0, "league"] == "NCAAF"


def test_base_schema_recovers_historical_sport_as_league():
    historical = pd.DataFrame(
        {
            "sport": ["NCAAF", "NBA"],
            "league": [pd.NA, ""],
            "home_team": ["Georgia Bulldogs", "Boston Celtics"],
        }
    )

    out = ensure_base_schema(historical)

    assert out["league"].tolist() == ["NCAAF", "NBA"]


def test_ncaaf_kalshi_series_and_line_tolerances_are_configured():
    assert league_series_ticker("NCAAF", "moneyline_home") == "KXNCAAFGAME"
    assert league_series_ticker("NCAAF", "spread_home") == "KXNCAAFSPREAD"
    assert league_series_ticker("NCAAF", "total_over") == "KXNCAAFTOTAL"
    assert MAX_LINE_TOLERANCE["NCAAF"] == 2.5
    assert MAX_TOTAL_LINE_TOLERANCE["NCAAF"] == 3.5


def test_ncaaf_results_are_gradeable_through_espn(monkeypatch):
    payload = {
        "events": [
            {
                "competitions": [
                    {
                        "status": {"type": {"state": "post"}},
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": "31",
                                "team": {"displayName": "Ohio State Buckeyes"},
                            },
                            {
                                "homeAway": "away",
                                "score": "24",
                                "team": {"displayName": "Texas Longhorns"},
                            },
                        ],
                    }
                ]
            }
        ]
    }
    requested_urls = []

    class FakeResponse:
        status_code = 200

        def json(self):
            return payload

    def fake_get(url, timeout):
        requested_urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(espn_results.requests, "get", fake_get)
    results = espn_results.fetch_espn_results(
        ["NCAAF"], target_date=pd.Timestamp("2026-08-29").date(), attempts=1
    )

    assert espn_results.ESPN_ENDPOINTS["NCAAF"].endswith(
        "/football/college-football/scoreboard"
    )
    assert requested_urls and "dates=20260829" in requested_urls[0]
    assert results.attrs["unsupported_leagues"] == []
    assert results.loc[0, "league"] == "NCAAF"
    assert int(results.loc[0, "home_score"]) == 31


def test_disabled_cfbd_session_uses_espn_ncaaf_fallback(monkeypatch):
    fallback = [{"team_norm": "ohio state", "league_key": "NCAAF"}]
    fake_streamlit = SimpleNamespace(
        session_state={"cfbd_disabled_reason": "UNAUTHORIZED_401"}
    )
    monkeypatch.setattr(feature_processing, "st", fake_streamlit)
    monkeypatch.setattr(
        feature_processing, "fetch_from_espn_ncaaf", lambda season_year: fallback
    )
    fetch = getattr(
        feature_processing.fetch_ncaaf_stats,
        "__wrapped__",
        feature_processing.fetch_ncaaf_stats,
    )

    assert fetch(2026) == fallback


def test_successful_cfbd_stats_keep_ncaaf_league_binding(monkeypatch):
    season_payload = [
        {
            "team": "Ohio State",
            "statName": "scoringOffense",
            "statValue": 32.0,
        },
        {
            "team": "Ohio State",
            "statName": "scoringDefense",
            "statValue": 18.0,
        },
    ]
    games_payload = [
        {
            "home_team": "Ohio State",
            "away_team": "Texas",
            "home_points": 31,
            "away_points": 24,
        }
    ]

    class FakeResponse:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, **kwargs):
        if url.endswith("/stats/season"):
            return FakeResponse(season_payload)
        if url.endswith("/games"):
            return FakeResponse(games_payload)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(
        feature_processing,
        "st",
        SimpleNamespace(session_state={}, secrets={}),
    )
    monkeypatch.setattr(feature_processing, "_get_secret", lambda name: "token")
    monkeypatch.setattr(feature_processing.requests, "get", fake_get)
    monkeypatch.setattr(feature_processing, "fetch_from_espn_ncaaf", lambda year: [])
    fetch = getattr(
        feature_processing.fetch_ncaaf_stats,
        "__wrapped__",
        feature_processing.fetch_ncaaf_stats,
    )

    stats = fetch(2026)

    assert len(stats) == 1
    assert stats[0]["league_key"] == "NCAAF"
    assert stats[0]["stats_team_key"] == "ohio state"
    assert stats[0]["team_name_source"] == "Ohio State"
    assert stats[0]["win_pct"] == 1.0


def test_ncaaf_spread_total_model_fails_closed_until_calibrated():
    rows = pd.DataFrame(
        {
            "League": ["NCAAF", "NCAAF"],
            "market_type": ["spread_home", "total_over"],
            "spread_line": [-3.5, pd.NA],
            "total_line": [pd.NA, 55.5],
            "feature_home_win_pct": [0.70, 0.70],
            "feature_away_win_pct": [0.55, 0.55],
            "feature_home_ppg": [140.0, 140.0],
            "feature_away_ppg": [112.0, 112.0],
            "feature_home_oppg": [84.0, 84.0],
            "feature_away_oppg": [120.0, 120.0],
            "feature_diff_last5": [0.15, 0.15],
            "ml_feature_eligible": [True, True],
            "stats_resolution_status": ["resolved", "resolved"],
        }
    )

    predictions = predict_market_probabilities(rows)

    assert predictions["ml_probability"].isna().all()
    assert predictions["ml_feature_quality"].eq("unavailable").all()
