import sys
import pandas as pd
import logging
sys.path.append('.')
import app_core.kalshi_integrator as ki
from _pytest.monkeypatch import MonkeyPatch

logging.basicConfig(level=logging.DEBUG)

def test_enrich():
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
                        "close_time": "2026-03-11T05:00:00Z",
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

    mp = MonkeyPatch()
    mp.setattr(ki, "_make_kalshi_request", fake_make_request)
    out = ki.enrich_with_kalshi_markets(df)
    print("Match Status:", out.loc[0, "kalshi_match_status"])
    print("Match Reason:", out.loc[0, "kalshi_match_reason"])

test_enrich()
