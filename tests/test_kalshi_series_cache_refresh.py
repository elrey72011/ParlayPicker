import pandas as pd

from app_core import kalshi_integrator as ki


def test_series_cache_refreshes_when_slate_date_changes(monkeypatch):
    calls = []

    def fake_fetch(series):
        calls.append(series)
        return []

    monkeypatch.setattr(ki, "_fetch_series_events", fake_fetch)
    for attr in ("series_cache", "series_cache_meta"):
        if hasattr(ki.enrich_with_kalshi_markets, attr):
            delattr(ki.enrich_with_kalshi_markets, attr)

    base = {
        "league": ["MLB"],
        "market_type": ["total_over"],
        "home_team": ["Pittsburgh Pirates"],
        "away_team": ["Milwaukee Brewers"],
        "total_line": [8.5],
        "best_pick": ["Over 8.5"],
    }
    ki.enrich_with_kalshi_markets(pd.DataFrame({**base, "game_date": ["2026-07-10T00:00:00Z"]}))
    ki.enrich_with_kalshi_markets(pd.DataFrame({**base, "game_date": ["2026-07-11T00:00:00Z"]}))

    assert calls == ["KXMLBTOTAL", "KXMLBTOTAL"]
