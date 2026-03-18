import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app_core.kalshi_integrator as kalshi_integrator


def test_kalshi_enrichment_infers_ncaab_for_blank_league_and_college_teams(monkeypatch):
    captured = {"series": []}

    def fake_fetch_series_events(series: str):
        captured["series"].append(series)
        return []

    monkeypatch.setattr(kalshi_integrator, "_fetch_series_events", fake_fetch_series_events)
    if hasattr(kalshi_integrator.enrich_with_kalshi_markets, "series_cache"):
        delattr(kalshi_integrator.enrich_with_kalshi_markets, "series_cache")

    df = pd.DataFrame(
        {
            "league": [pd.NA],
            "home_team": ["Wichita St"],
            "away_team": ["Oklahoma State"],
            "game_date": [pd.Timestamp("2026-03-12", tz="UTC")],
            "market_type": ["spread_home"],
            "best_pick": ["Wichita St -2.5"],
            "spread_line": [-2.5],
        }
    )

    out = kalshi_integrator.enrich_with_kalshi_markets(df)

    assert len(out) == 1
    assert captured["series"] == ["KXNCAAMBSPREAD"]
    assert out.iloc[0]["kalshi_match_reason"] != "missing_series"
