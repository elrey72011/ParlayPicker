"""The app's post-Kalshi recompute de-vigs against the real opposite quote when it has one."""
import pandas as pd

from streamlit_app import _recompute_consensus_from_kalshi


def _row(**extra):
    row = {"league": "NCAAB", "market_type": "spread_away", "home_team": "A", "away_team": "B",
           "odds_american": -108.0, "market_probability": 0.51192, "calibrated_probability": 0.51192,
           "ml_probability": float("nan"), "kalshi_probability": float("nan")}
    row.update(extra)
    return row


def test_paired_quote_is_used_and_edge_is_not_manufactured():
    out = _recompute_consensus_from_kalshi(pd.DataFrame([_row(opposing_odds_american=102.0)]))
    assert abs(float(out.loc[0, "market_probability"]) - 0.51192) < 1e-4


def test_synthetic_fallback_without_a_paired_quote():
    out = _recompute_consensus_from_kalshi(pd.DataFrame([_row()]))
    # -108 against a synthetic -112: about 0.4957, never the old "+88" value 0.4940.
    assert abs(float(out.loc[0, "market_probability"]) - 0.49567) < 1e-3
