"""A nearest-line Kalshi proxy is moved to the pick's line before it can create EV."""
import pandas as pd

import app_core.kalshi_integrator as ki


def test_shift_directions():
    # Favourite -9.5 priced from a 7-point contract: harder, so lower.
    assert ki.shift_proxy_probability(0.579, "NCAAB", 2.5, is_total=False, pick_is_favorite=True) < 0.579
    # Underdog +9.5 from a 7-point contract: easier, so higher.
    assert ki.shift_proxy_probability(0.42, "NCAAB", 2.5, is_total=False, pick_is_favorite=False) > 0.42
    # Over a higher total is harder; under a higher total is easier.
    assert ki.shift_proxy_probability(0.55, "NBA", 3.0, is_total=True) < 0.55
    assert ki.shift_proxy_probability(0.45, "NBA", 3.0, is_total=True, is_under=True) > 0.45
    assert ki.shift_proxy_probability(0.6, "NBA", 0.0, is_total=True) == 0.6
    assert ki.shift_proxy_probability(float("nan"), "NBA", 1.0, is_total=True) is None


def test_shift_size_matches_normal_scale():
    # 2.5 points on an 11-point margin sd moves 0.579 to about 0.49: the Iona case.
    shifted = ki.shift_proxy_probability(0.579, "NCAAB", 2.5, is_total=False, pick_is_favorite=True)
    assert 0.47 < shifted < 0.51


class _Response:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


def test_enrich_shifts_far_spread_proxy(monkeypatch):
    df = pd.DataFrame({"league": ["NBA"], "market_type": ["spread_home"], "home_team": ["Boston Celtics"],
                       "away_team": ["Los Angeles Lakers"], "game_date": ["2026-03-10T00:00:00Z"],
                       "spread_line": [-9.5], "best_pick": ["Boston Celtics -9.5"]})

    def fake(url, **kwargs):
        if url.endswith("/events") and "series_ticker" in kwargs.get("params", {}):
            return _Response({"events": [{"event_ticker": "KXNBASPREAD-26MAR10LALBOS",
                                          "title": "Boston Celtics at Los Angeles Lakers: Spread",
                                          "sub_title": "LAL at BOS (Mar 10)",
                                          "close_time": "2026-03-10T05:00:00Z"}]})
        if "KXNBASPREAD-26MAR10LALBOS" in url:
            return _Response({"event": {"markets": [{
                "ticker": "KXNBASPREAD-26MAR10LALBOS-BOS65", "event_ticker": "KXNBASPREAD-26MAR10LALBOS",
                "title": "Boston Celtics wins by over 6.5", "yes_bid_dollars": 0.58, "yes_ask_dollars": 0.62}]}})
        return _Response({"error": "not found"}, 404)

    if hasattr(ki.enrich_with_kalshi_markets, "series_cache"):
        ki.enrich_with_kalshi_markets.series_cache.clear()
    monkeypatch.setattr(ki, "_make_kalshi_request", fake)
    out = ki.enrich_with_kalshi_markets(df)
    assert out.loc[0, "kalshi_match_status"] == "matched"
    assert out.loc[0, "kalshi_match_reason"] == "nearest_line_proxy"
    assert float(out.loc[0, "kalshi_line_offset"]) == -3.0
    # Decay alone gave ~0.57; with the shift the -9.5 price falls below the 6.5 contract's 0.60.
    assert float(out.loc[0, "kalshi_probability"]) < 0.53
