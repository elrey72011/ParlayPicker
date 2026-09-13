"""Legacy Kalshi direction diagnostics cannot override probability-first selection."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.streamlit_pipeline import build_best_picks_df


def _totals_pair(k_over: float | None, k_under: float | None,
                 p_over: float = 0.60, p_under: float = 0.48) -> pd.DataFrame:
    """An MLB total over/under pair where the MODEL strongly favors the Over."""
    def row(mt, prob, kalshi):
        return {
            "game_id": "g1", "league": "MLB",
            "home_team": "Cincinnati", "away_team": "Baltimore",
            "game_date": pd.Timestamp("2026-07-10", tz="UTC"),
            "market_type": mt, "total_line": 8.5, "spread_line": pd.NA,
            "calibrated_probability": prob,
            "expected_value": 0.05 if prob > 0.5 else 0.01,
            "edge": 0.04 if prob > 0.5 else 0.01,
            "market_probability": 0.50, "ml_probability": prob,
            "kalshi_probability": kalshi,
            "kalshi_match_status": "matched" if kalshi is not None else "miss",
            "consensus_agreement": "Neutral",
            "odds_american": -110, "odds_source": "test",
            "used_stale_features": False,
        }
    return pd.DataFrame([
        row("total_over", p_over, k_over),
        row("total_under", p_under, k_under),
    ])


def test_kalshi_component_cannot_override_candidate_win_probability():
    # Kalshi prices the OVER at 42% (i.e. favors the Under by 8 pts) while the
    # final candidate favors Over. The component must not override that estimate.
    best = build_best_picks_df(_totals_pair(k_over=0.42, k_under=0.58))
    totals = best[best["market_type"].astype(str).str.contains("total")]
    assert len(totals) == 1
    assert totals.iloc[0]["market_type"] == "total_over"
    assert totals.iloc[0]["best_available_probability"] > .55


def test_weak_kalshi_does_not_veto():
    # 52/48 is inside the conviction threshold: the model's side stands.
    best = build_best_picks_df(_totals_pair(k_over=0.52, k_under=0.48))
    totals = best[best["market_type"].astype(str).str.contains("total")]
    assert len(totals) == 1
    assert totals.iloc[0]["market_type"] == "total_over"


def test_missing_kalshi_sentinel_never_vetoes():
    # kalshi_probability 0.0 is the "No Kalshi" miss sentinel, not a price —
    # it must not read as maximum conviction against the Over.
    best = build_best_picks_df(_totals_pair(k_over=0.0, k_under=0.0))
    totals = best[best["market_type"].astype(str).str.contains("total")]
    assert len(totals) == 1
    assert totals.iloc[0]["market_type"] == "total_over"


def test_kalshi_agreeing_with_model_changes_nothing():
    best = build_best_picks_df(_totals_pair(k_over=0.58, k_under=0.42))
    totals = best[best["market_type"].astype(str).str.contains("total")]
    assert len(totals) == 1
    assert totals.iloc[0]["market_type"] == "total_over"
