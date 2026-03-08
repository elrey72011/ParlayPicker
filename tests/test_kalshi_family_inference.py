import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core import kalshi_integrator as ki


def test_enrich_uses_best_pick_to_infer_spread_family_when_market_type_missing(monkeypatch):
    df = pd.DataFrame(
        [{
            "league": "NBA",
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
            "game_date": pd.Timestamp("2024-03-08", tz="UTC"),
            "best_pick": "Boston Celtics -4.5",
            "spread_line": -4.5,
            "pick_team": "Boston Celtics"
        }]
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
                        "ticker": "KXNBASPREAD-24MAR08LALBOS",
                        "event_ticker": "KXNBASPREAD-24MAR08LALBOS",
                        "title": "Boston Celtics wins by over 4.5",
                        "subtitle": "",
                        "yes_bid_dollars": 0.45,
                        "yes_ask_dollars": 0.55,
                    }
                ]
            }
        return {"data": []}

    monkeypatch.setattr(ki, "api_get_markets", fake_api_get_markets)
    out = ki.enrich_with_kalshi_markets(df)

    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert any(call.get("series_ticker") == "KXNBASPREAD" for call in seen)
