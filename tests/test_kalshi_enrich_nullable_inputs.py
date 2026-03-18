import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.kalshi_integrator import enrich_with_kalshi_markets


def test_kalshi_enrichment_handles_nullable_scalars_without_boolean_errors():
    df = pd.DataFrame(
        {
            "league": [pd.NA],
            "home_team": ["Boston Celtics"],
            "away_team": ["Miami Heat"],
            "game_date": [pd.Timestamp("2026-03-12", tz="UTC")],
            "market_type": ["spread_home"],
            "best_pick": [pd.NA],
            "spread_line": [pd.NA],
        }
    )

    out = enrich_with_kalshi_markets(df)

    assert len(out) == 1
    assert "kalshi_match_status" in out.columns
    # Most important: function returns normally and does not raise
    # "boolean value of NA is ambiguous".
