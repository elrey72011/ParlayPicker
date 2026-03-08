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

    def fake_api_get_markets(**params):
        if "tickers" in params:
            return [
                {
                    "ticker": "KXNBASPREAD-26MAR10LALBOS",
                    "event_ticker": "KXNBASPREAD-26MAR10LALBOS",
                    "title": "Boston Celtics wins by over 4",
                    "yes_bid_dollars": 58,
                    "yes_ask_dollars": 62,
                }
            ]
        return []

    monkeypatch.setattr(ki, "api_get_markets", fake_api_get_markets)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert out.loc[0, "kalshi_match_reason"] == "spread_match"
    assert float(out.loc[0, "kalshi_probability"]) == 0.60


def test_enrich_with_kalshi_markets_missing_team_code_reason(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "market_type": ["total_over"],
            "home_team": ["Team That Does Not Exist"],
            "away_team": ["Boston Celtics"],
            "game_date": ["2026-03-10T00:00:00Z"],
        }
    )

    monkeypatch.setattr(ki, "api_get_markets", lambda **_kwargs: [])
    monkeypatch.setattr(ki, "team_code_map", lambda league, team: "" if "does not exist" in team.lower() else "BOS")
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "miss"
    assert out.loc[0, "kalshi_match_reason"] == "missing_team_code"


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
            "best_pick": ["over"]
        }
    )

    def fake_api_get_markets(**params):
        if "tickers" in params:
            return {"markets": []}

        return {
            "markets": [
                {
                    "ticker": "KXNCAAMBTOTAL-26MAR10XXXX",
                    "event_ticker": "KXNCAAMBTOTAL-26MAR10ABCD",
                    "title": "Saint Mary's vs Pepperdine over 141",
                    "yes_bid_dollars": 49,
                    "yes_ask_dollars": 51,
                }
            ]
        }

    monkeypatch.setattr(ki, "api_get_markets", fake_api_get_markets)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert float(out.loc[0, "kalshi_probability"]) == 0.50
