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

    class FakeResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code
        def json(self): return self._json_data

    def fake_make_request(url, **kwargs):
        seen.append(url)
        if url.endswith("/events") and "series_ticker" in kwargs.get("params", {}):
            return FakeResponse({
                "events": [
                    {
                        "event_ticker": "KXNBASPREAD-24MAR08BOSLAL",
                        "title": "Boston Celtics at Los Angeles Lakers: Spread",
                        "sub_title": "BOS at LAL (Mar 8)",
                        "close_time": "2024-03-08T05:00:00Z",
                    }
                ]
            })
        if "KXNBASPREAD-24MAR08BOSLAL" in url:
            return FakeResponse({
                "event": {
                    "markets": [
                        {
                            "ticker": "KXNBASPREAD-24MAR08BOSLAL",
                            "event_ticker": "KXNBASPREAD-24MAR08BOSLAL",
                            "title": "Boston Celtics wins by over 4.5",
                            "subtitle": "",
                            "yes_bid_dollars": 0.45,
                            "yes_ask_dollars": 0.55,
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
    assert any("KXNBASPREAD" in call for call in seen)
