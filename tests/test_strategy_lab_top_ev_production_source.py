import pandas as pd

from app.ui.strategy_lab_dashboard import _build_production_top_ev


def test_top_ev_excludes_rejected_live_no_play_and_unresolved_rows():
    best = pd.DataFrame([
        {"best_pick": "Over 8.5", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True, "market_line_used": 8.5, "line_provenance_warning": "", "canonical_pick_key": "k1", "pick_id": "p1", "expected_value": 0.2},
        {"best_pick": "Over unresolved", "Pick_Status": "No Play", "market_line_source": "rejected_live", "line_consistency_flag": False, "line_event_identity_match_flag": False, "market_line_used": pd.NA, "line_provenance_warning": "Total line unresolved", "canonical_pick_key": "k2", "pick_id": "p2", "expected_value": 0.5},
    ])
    top, diag = _build_production_top_ev(best)
    assert len(top) == 1
    assert top.iloc[0]["canonical_pick_key"] == "k1"
    assert diag["top_ev_excluded_rejected_line_count"] == 1


def test_top_ev_requires_identity_fields_and_actionable_status():
    best = pd.DataFrame([
        {"best_pick": "Over 8.5", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True, "market_line_used": 8.5, "line_provenance_warning": "", "canonical_pick_key": "", "pick_id": "", "expected_value": 0.3},
        {"best_pick": "Over 9.5", "Pick_Status": "High Variance/Speculative", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True, "market_line_used": 9.5, "line_provenance_warning": "", "canonical_pick_key": "k2", "pick_id": "p2", "expected_value": 0.4},
    ])
    top, diag = _build_production_top_ev(best)
    assert top.empty
    assert diag["top_ev_missing_identity_count"] >= 1
    assert diag["top_ev_excluded_non_actionable_count"] >= 1


def test_top_ev_includes_portfolio_kelly_fields_when_available():
    best = pd.DataFrame([
        {"best_pick": "Over 8.5", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True, "market_line_used": 8.5, "line_provenance_warning": "", "canonical_pick_key": "k1", "pick_id": "p1", "expected_value": 0.2}
    ])
    portfolio = pd.DataFrame([{"canonical_pick_key": "k1", "raw_kelly_amount": 100.0, "production_bet_amount": 40.0, "kelly_cap_reason": "Capped by pick exposure"}])
    top, _ = _build_production_top_ev(best, portfolio)
    assert "raw_kelly_amount" in top.columns
    assert float(top.iloc[0]["production_bet_amount"]) == 40.0
