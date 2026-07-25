import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core import empirical_tiers, probability_calibration
from core.streamlit_pipeline import optimize_portfolio_allocation


def _row(name: str, *, calibrated=0.80, effective=None, empirical=None):
    row = {
        "league": "MLB",
        "market_type": "total_over",
        "consensus_agreement": "Agrees",
        "best_pick": name,
        "Pick_Status": "Actionable",
        "calibrated_probability": calibrated,
        "odds_american": -110,
        "market_line_source": "live",
        "line_provenance_warning": "",
        "market_line_used": 8.5,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "canonical_pick_key": name,
    }
    if effective is not None:
        row["effective_win_probability"] = effective
    if empirical is not None:
        row["empirical_win_probability"] = empirical
    return row


def test_kelly_prefers_empirical_probability_over_model_blend():
    out = optimize_portfolio_allocation(
        pd.DataFrame([_row("A Over 8.5", empirical=0.50)]),
        bankroll=1000.0,
    )
    assert float(out.loc[0, "kelly_probability_used"]) == 0.50
    assert out.loc[0, "kelly_probability_source"] == "empirical_win_probability"
    assert float(out.loc[0, "production_bet_amount"]) == 0.0


def test_kelly_fits_effective_probability_before_sizing(monkeypatch):
    monkeypatch.setattr(probability_calibration, "load_calibration", lambda: [[0.0, 0.50], [1.0, 0.50]])
    monkeypatch.setattr(empirical_tiers, "load_bucket_stats", lambda: None)
    out = optimize_portfolio_allocation(
        pd.DataFrame([_row("B Over 8.5", effective=0.80)]),
        bankroll=1000.0,
    )
    assert float(out.loc[0, "kelly_probability_used"]) == 0.50
    assert out.loc[0, "kelly_probability_source"] == "fitted_effective_probability"
    assert float(out.loc[0, "production_bet_amount"]) == 0.0


def test_effective_probability_without_fitted_calibration_is_not_staked(monkeypatch):
    monkeypatch.setattr(probability_calibration, "load_calibration", lambda: None)
    monkeypatch.setattr(empirical_tiers, "load_bucket_stats", lambda: None)
    out = optimize_portfolio_allocation(
        pd.DataFrame([_row("C Over 8.5", effective=0.80)]),
        bankroll=1000.0,
    )
    assert out.loc[0, "kelly_probability_source"] == "missing_fitted_calibration"
    assert not bool(out.loc[0, "production_eligible"])
    assert float(out.loc[0, "production_bet_amount"]) == 0.0


def test_production_caps_apply_per_pick_and_per_slate():
    rows = [_row(f"P{i} Over 8.5", empirical=0.90) for i in range(20)]
    out = optimize_portfolio_allocation(pd.DataFrame(rows), bankroll=1000.0)
    assert float(out["production_bet_amount"].max()) <= 20.0
    assert float(out["production_bet_amount"].sum()) <= 100.0


def test_absolute_pick_cap_protects_large_bankroll_input():
    out = optimize_portfolio_allocation(
        pd.DataFrame([_row("D Over 8.5", empirical=0.90)]),
        bankroll=100000.0,
    )
    assert float(out.loc[0, "production_bet_amount"]) <= 50.0

