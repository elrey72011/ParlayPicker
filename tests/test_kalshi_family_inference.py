import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core import kalshi_integrator as ki


def test_enrich_uses_best_pick_to_infer_spread_family_when_market_type_missing(monkeypatch):
    df = pd.DataFrame(
        {
            "league": ["NBA"],
            "home_team": ["Boston Celtics"],
            "away_team": ["Los Angeles Lakers"],
            "game_date": ["2026-03-10T00:00:00Z"],
            "best_pick": ["Boston Celtics -4.5"],
        }
    )

    seen = []

    def fake_api_get_markets(**params):
        seen.append(params)
        if "tickers" in params:
            return {"data": []}
        if params.get("series_ticker") == "KXNBASPREAD":
            return {
                "data": [
                    {
                        "ticker": "KXNBASPREAD-26MAR10LALBOS",
                        "event_ticker": "KXNBASPREAD-26MAR10LALBOS",
                        "title": "Lakers vs Celtics",
                        "yes_bid_dollars": 54,
                        "yes_ask_dollars": 56,
                    }
                ]
            }
        return {"data": []}

    monkeypatch.setattr(ki, "api_get_markets", fake_api_get_markets)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert any(call.get("series_ticker") == "KXNBASPREAD" for call in seen)
