import pandas as pd

from app.ui.strategy_lab_dashboard import _build_kelly_export, _build_production_top_ev


def _best_rows():
    return pd.DataFrame([
        {"pick_id": "p1", "canonical_pick_key": "k1", "league": "MLB", "home_team": "A", "away_team": "B", "market_type": "total_over", "best_pick": "Over 8.5", "Pick_Status": "Actionable", "market_line_used": 8.5, "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
        {"pick_id": "p2", "canonical_pick_key": "k2", "league": "MLB", "home_team": "C", "away_team": "D", "market_type": "total_over", "best_pick": "Over 7.5", "Pick_Status": "High Variance/Speculative", "market_line_used": 7.5, "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
        {"pick_id": "p3", "canonical_pick_key": "k3", "league": "MLB", "home_team": "E", "away_team": "F", "market_type": "total_over", "best_pick": "Total line unresolved", "Pick_Status": "No Play", "market_line_used": pd.NA, "market_line_source": "rejected_live", "line_consistency_flag": False, "line_event_identity_match_flag": False},
    ])


def test_kelly_alignment_uses_canonical_key_not_row_order():
    best = _best_rows()
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k2", "pick_id": "p2", "raw_kelly_amount": 80, "production_bet_amount": 20, "production_eligible": False},
        {"canonical_pick_key": "k1", "pick_id": "p1", "raw_kelly_amount": 100, "production_bet_amount": 40, "production_eligible": True},
        {"canonical_pick_key": "k3", "pick_id": "p3", "raw_kelly_amount": 90, "production_bet_amount": 30, "production_eligible": False},
    ])
    out, _ = _build_kelly_export(best, portfolio)
    assert float(out.loc[out["canonical_pick_key"] == "k1", "production_bet_amount"].iloc[0]) == 40.0


def test_reordered_best_rows_do_not_break_kelly_alignment():
    best = _best_rows().iloc[::-1].reset_index(drop=True)
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k1", "pick_id": "p1", "raw_kelly_amount": 100, "production_bet_amount": 40, "production_eligible": True},
    ])
    out, _ = _build_kelly_export(best, portfolio)
    assert float(out.loc[out["canonical_pick_key"] == "k1", "production_bet_amount"].iloc[0]) == 40.0


def test_non_production_rows_are_zero_after_safety_pass():
    best = _best_rows()
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k2", "pick_id": "p2", "raw_kelly_amount": 100, "production_bet_amount": 40, "production_eligible": True},
        {"canonical_pick_key": "k3", "pick_id": "p3", "raw_kelly_amount": 100, "production_bet_amount": 40, "production_eligible": True},
    ])
    out, diag = _build_kelly_export(best, portfolio)
    assert float(out.loc[out["canonical_pick_key"] == "k2", "production_bet_amount"].iloc[0]) == 0.0
    assert float(out.loc[out["canonical_pick_key"] == "k3", "production_bet_amount"].iloc[0]) == 0.0
    assert diag["kelly_positive_non_actionable_count"] == 0
    assert diag["kelly_positive_rejected_line_count"] == 0
    assert diag["kelly_positive_unresolved_count"] == 0


def test_positive_kelly_rows_match_top_ev_keys():
    best = _best_rows()
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k1", "pick_id": "p1", "raw_kelly_amount": 100, "production_bet_amount": 40, "kelly_cap_reason": "Capped by pick exposure", "production_eligible": True},
    ])
    kelly, _ = _build_kelly_export(best, portfolio)
    top_ev, _ = _build_production_top_ev(best, portfolio)
    kelly_keys = set(kelly.loc[kelly["production_bet_amount"] > 0, "canonical_pick_key"])
    top_ev_keys = set(top_ev["canonical_pick_key"])
    assert kelly_keys.issubset(top_ev_keys)


def test_kelly_export_has_identity_and_safety_fields():
    out, _ = _build_kelly_export(_best_rows(), pd.DataFrame())
    required = {"source_table", "pick_id", "canonical_pick_key", "Pick_Status", "production_eligible", "raw_kelly_amount", "production_bet_amount", "kelly_cap_reason"}
    assert required.issubset(set(out.columns))
