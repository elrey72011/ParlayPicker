import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core import kalshi_integrator as ki


def test_enrich_with_kalshi_markets_sets_match_fields(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "market_type": ["spread_home"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "spread_line": [-4.5],
            "best_pick": ["Boston Celtics -4.5"]
        }
    )

    class FakeResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code

        def json(self):
            return self._json_data

    def fake_make_request(url, **kwargs):
        if url.endswith("/events") and "series_ticker" in kwargs.get("params", {}):
            return FakeResponse({
                "events": [
                    {
                        "event_ticker": "KXNBASPREAD-26MAR10LALBOS",
                        "title": "Boston Celtics at Los Angeles Lakers: Spread",
                        "sub_title": "LAL at BOS (Mar 10)",
                        "close_time": "2026-03-10T05:00:00Z",
                    }
                ]
            })
        if "KXNBASPREAD-26MAR10LALBOS" in url:
            return FakeResponse({
                "event": {
                    "markets": [
                        {
                            "ticker": "KXNBASPREAD-26MAR10LALBOS",
                            "event_ticker": "KXNBASPREAD-26MAR10LALBOS",
                            "title": "Boston Celtics wins by over 4",
                                "yes_bid_dollars": 0.58,
                                "yes_ask_dollars": 0.62,
                        }
                    ]
                }
            })
        return FakeResponse({"error": "not found"}, 404)

    # Clear cache before running test
    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()

    monkeypatch.setattr(ki, "_make_kalshi_request", fake_make_request)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert out.loc[0, "kalshi_match_reason"] in ["spread_match", "spread_match_exact"]
    assert float(out.loc[0, "kalshi_probability"]) == 0.60


def test_enrich_with_kalshi_markets_missing_team_code_reason(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "market_type": ["total_over"],
            "home_team": ["Team That Does Not Exist"],
            "away_team": ["Boston Celtics"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "best_pick": [""],
        }
    )

    class FakeResponse404:
        def __init__(self, json_data, status_code=404):
            self._json_data = json_data
            self.status_code = status_code
        def json(self): return self._json_data

    def fake_make_request(url, **kwargs):
        from requests.exceptions import HTTPError
        err = HTTPError()
        err.response = FakeResponse404({})
        raise err

    # Clear cache before running test
    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()

    monkeypatch.setattr(ki, "_make_kalshi_request", fake_make_request)
    monkeypatch.setattr(ki, "team_code_map", lambda league, team: "" if "does not exist" in team.lower() else "BOS")
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "miss"
    assert out.loc[0, "kalshi_match_reason"] in ["event_not_found", "no_market_for_tickers", "missing_team_code", "no_fuzzy_event_match"]


def test_enrich_with_kalshi_markets_title_fallback_when_event_ticker_codes_differ(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NCAAB"],
            "market_type": ["total_over"],
            "home_team": ["Saint Mary's"],
            "away_team": ["Pepperdine"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "total_line": [140.5],
            "total_pick_side": ["over"],
            "best_pick": ["Over 140.5"]
        }
    )

    class FakeResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code
        def json(self): return self._json_data

    def fake_make_request(url, **kwargs):
        if url.endswith("/events") and "series_ticker" in kwargs.get("params", {}):
            return FakeResponse({
                "events": [
                    {
                        "event_ticker": "KXNCAAMBTOTAL-26MAR10ABCD",
                        "title": "Pepperdine at Saint Mary's: Total",
                        "sub_title": "PEP at SMC (Mar 10)",
                        "close_time": "2026-03-10T05:00:00Z",
                    }
                ]
            })
        if "KXNCAAMBTOTAL" in url:
            return FakeResponse({
                "event": {
                    "markets": [
                        {
                            "ticker": "KXNCAAMBTOTAL-26MAR10XXXX",
                            "event_ticker": "KXNCAAMBTOTAL-26MAR10ABCD",
                            "title": "Saint Mary's vs Pepperdine total over 140.5",
                            "subtitle": "total",
                            "yes_bid_dollars": 0.49,
                            "yes_ask_dollars": 0.51,
                        }
                    ]
                }
            })
        return FakeResponse({}, 404)

    # Clear cache before running test
    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()

    monkeypatch.setattr(ki, "_make_kalshi_request", fake_make_request)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert float(out.loc[0, "kalshi_probability"]) == 0.50

def test_enrich_with_kalshi_markets_matches_from_event_ticker_codes(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "market_type": ["total_over"],
            "home_team": ["Los Angeles Clippers"],
            "away_team": ["Sacramento Kings"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "total_line": [229.5],
            "best_pick": ["Over 229.5"],
        }
    )

    class FakeResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code

        def json(self):
            return self._json_data

    def fake_make_request(url, **kwargs):
        if url.endswith("/events") and "series_ticker" in kwargs.get("params", {}):
            return FakeResponse(
                {
                    "events": [
                        {
                            "event_ticker": "KXNBATOTAL-26MAR10LACSAC",
                            # intentionally no full names in title/subtitle
                            "title": "LAC vs SAC total",
                            "sub_title": "NBA Total",
                            "close_time": "2026-03-10T05:00:00Z",
                        }
                    ]
                }
            )
        if "KXNBATOTAL-26MAR10LACSAC" in url:
            return FakeResponse(
                {
                    "event": {
                        "markets": [
                            {
                                "ticker": "KXNBATOTAL-26MAR10LACSAC-2295",
                                "event_ticker": "KXNBATOTAL-26MAR10LACSAC",
                                "title": "Total points over 229.5",
                                "subtitle": "over/under",
                                "yes_bid_dollars": 0.48,
                                "yes_ask_dollars": 0.52,
                            }
                        ]
                    }
                }
            )
        return FakeResponse({}, 404)

    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()

    monkeypatch.setattr(ki, "_make_kalshi_request", fake_make_request)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert float(out.loc[0, "kalshi_probability"]) == 0.50

def test_enrich_with_kalshi_markets_accepts_nearby_alt_line_within_league_tolerance(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "market_type": ["spread_home"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Washington Wizards"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "spread_line": [-7.5],
            "best_pick": ["Boston Celtics -7.5"],
        }
    )

    class FakeResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code

        def json(self):
            return self._json_data

    def fake_make_request(url, **kwargs):
        if url.endswith("/events") and "series_ticker" in kwargs.get("params", {}):
            return FakeResponse(
                {
                    "events": [
                        {
                            "event_ticker": "KXNBASPREAD-26MAR10BOSWAS",
                            "title": "BOS vs WAS spread",
                            "sub_title": "NBA Spread",
                            "close_time": "2026-03-10T05:00:00Z",
                        }
                    ]
                }
            )
        if "KXNBASPREAD-26MAR10BOSWAS" in url:
            return FakeResponse(
                {
                    "event": {
                        "markets": [
                            {
                                "ticker": "KXNBASPREAD-26MAR10BOSWAS-125",
                                "event_ticker": "KXNBASPREAD-26MAR10BOSWAS",
                                "title": "Boston Celtics wins by over 12.5",
                                "subtitle": "spread",
                                "yes_bid_dollars": 0.47,
                                "yes_ask_dollars": 0.53,
                            }
                        ]
                    }
                }
            )
        return FakeResponse({}, 404)

    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()

    monkeypatch.setattr(ki, "_make_kalshi_request", fake_make_request)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert out.loc[0, "kalshi_match_reason"] in ["spread_match_nearest", "nearest_line_proxy"]

def test_enrich_with_kalshi_markets_single_candidate_event_fallback(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "market_type": ["total_over"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "total_line": [220.5],
            "best_pick": ["Over 220.5"],
        }
    )

    class FakeResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code

        def json(self):
            return self._json_data

    def fake_make_request(url, **kwargs):
        if url.endswith("/events") and "series_ticker" in kwargs.get("params", {}):
            return FakeResponse(
                {
                    "events": [
                        {
                            "event_ticker": "KXNBATOTAL-26MAR10XXYY",
                            "title": "NBA Total",
                            "sub_title": "Game total",
                            "close_time": "2026-03-10T05:00:00Z",
                        }
                    ]
                }
            )
        if "KXNBATOTAL-26MAR10XXYY" in url:
            return FakeResponse(
                {
                    "event": {
                        "markets": [
                            {
                                "ticker": "KXNBATOTAL-26MAR10XXYY-2205",
                                "event_ticker": "KXNBATOTAL-26MAR10XXYY",
                                "title": "Total points over 220.5",
                                "subtitle": "over/under",
                                "yes_bid_dollars": 0.49,
                                "yes_ask_dollars": 0.51,
                            }
                        ]
                    }
                }
            )
        return FakeResponse({}, 404)

    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()

    monkeypatch.setattr(ki, "_make_kalshi_request", fake_make_request)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert str(out.loc[0, "kalshi_event_ticker"]).endswith("XXYY")
