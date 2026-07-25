import pandas as pd

from core.streamlit_pipeline import BEST_PICK_COLUMNS
from streamlit_app import _attach_kelly_to_best_picks, _safe_str_series


def test_kelly_bet_size_order_after_best_pick_column():
    idx = BEST_PICK_COLUMNS.index("best_pick")
    assert BEST_PICK_COLUMNS[idx + 1] == "Kelly_Bet_Size"


def test_kelly_attach_joins_by_canonical_key_and_handles_reorder():
    best = pd.DataFrame([
        {"canonical_pick_key": "k2", "best_pick": "B", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
        {"canonical_pick_key": "k1", "best_pick": "A", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
    ])
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k1", "production_bet_amount": 30.12},
        {"canonical_pick_key": "k2", "production_bet_amount": 10.34},
    ])
    out = _attach_kelly_to_best_picks(best, portfolio, {})
    assert float(out.loc[out["canonical_pick_key"] == "k1", "Kelly_Bet_Size"].iloc[0]) == 30.12
    assert float(out.loc[out["canonical_pick_key"] == "k2", "Kelly_Bet_Size"].iloc[0]) == 10.34


def test_kelly_attach_zeroes_missing_non_actionable_rejected_and_unresolved():
    best = pd.DataFrame([
        {"canonical_pick_key": "", "best_pick": "Over 8.5", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
        {"canonical_pick_key": "k2", "best_pick": "Over 8.5", "Pick_Status": "No Play", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
        {"canonical_pick_key": "k3", "best_pick": "Over 8.5", "Pick_Status": "Actionable", "market_line_source": "rejected_live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
        {"canonical_pick_key": "k4", "best_pick": "Total line unresolved", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
    ])
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k2", "production_bet_amount": 20},
        {"canonical_pick_key": "k3", "production_bet_amount": 20},
        {"canonical_pick_key": "k4", "production_bet_amount": 20},
    ])
    out = _attach_kelly_to_best_picks(best, portfolio, {})
    assert (out["Kelly_Bet_Size"] == 0).all()


def test_safe_str_series_handles_categorical_pick_status():
    df = pd.DataFrame({"Pick_Status": pd.Categorical(["Actionable", None, "No Play"])})
    out = _safe_str_series(df, "Pick_Status")
    assert str(out.dtype) == "string"
    assert out.tolist() == ["Actionable", "", "No Play"]


def test_safe_str_series_handles_category_default_not_present():
    df = pd.DataFrame({"Pick_Status": pd.Categorical(["Actionable", None], categories=["Actionable", "No Play"])})
    out = _safe_str_series(df, "Pick_Status", default="")
    assert out.iloc[1] == ""


def test_attach_kelly_does_not_crash_with_categorical_pick_status_and_maps_by_key():
    best = pd.DataFrame([
        {"canonical_pick_key": "k1", "best_pick": "Over 8.5", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
        {"canonical_pick_key": "k2", "best_pick": "Over 8.5", "Pick_Status": "No Play", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
    ])
    best["Pick_Status"] = pd.Categorical(best["Pick_Status"], categories=["Actionable", "No Play"])
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k1", "production_bet_amount": 12.34},
        {"canonical_pick_key": "k2", "production_bet_amount": 56.78},
    ])
    out = _attach_kelly_to_best_picks(best, portfolio, {})
    assert float(out.loc[out["canonical_pick_key"] == "k1", "Kelly_Bet_Size"].iloc[0]) == 12.34
    assert float(out.loc[out["canonical_pick_key"] == "k2", "Kelly_Bet_Size"].iloc[0]) == 0.0


def test_clean_actionable_zero_kelly_gets_reason_and_diagnostics():
    best = pd.DataFrame([
        {"canonical_pick_key": "k1", "best_pick": "Over 220.5", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True, "production_eligible": True},
    ])
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k1", "production_bet_amount": 0.0, "raw_kelly_amount": 50.0, "kelly_cap_reason": "", "production_eligible": True},
    ])
    diags = {}
    out = _attach_kelly_to_best_picks(best, portfolio, diags)
    assert out.loc[0, "kelly_zero_reason"] != ""
    assert diags["actionable_rows_with_zero_kelly_count"] == 1
    assert diags["clean_actionable_rows_with_zero_kelly_count"] == 1


def test_attach_kelly_carries_portfolio_audit_fields():
    best = pd.DataFrame([
        {"canonical_pick_key": "k1", "best_pick": "Under 213.5", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
    ])
    portfolio = pd.DataFrame([
        {
            "canonical_pick_key": "k1",
            "production_bet_amount": 22.0,
            "raw_kelly_amount": 88.0,
            "kelly_cap_reason": "Scaled by slate exposure",
            "production_eligible": True,
            "kelly_uncalibrated_probability": 0.70,
            "kelly_probability_used": 0.65,
            "kelly_probability_source": "empirical_win_probability",
        },
    ])
    out = _attach_kelly_to_best_picks(best, portfolio, {})
    assert float(out.loc[0, "production_bet_amount"]) == 22.0
    assert float(out.loc[0, "raw_kelly_amount"]) == 88.0
    assert out.loc[0, "kelly_cap_reason"] == "Scaled by slate exposure"
    assert float(out.loc[0, "kelly_uncalibrated_probability"]) == 0.70
    assert float(out.loc[0, "kelly_probability_used"]) == 0.65
    assert out.loc[0, "kelly_probability_source"] == "empirical_win_probability"


def test_attach_kelly_handles_portfolio_without_production_bet_amount():
    best = pd.DataFrame([
        {"canonical_pick_key": "k1", "best_pick": "Over 220.5", "Pick_Status": "Actionable", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
    ])
    portfolio = pd.DataFrame([
        {"canonical_pick_key": "k1", "raw_kelly_amount": 25.0, "production_eligible": False},
    ])
    out = _attach_kelly_to_best_picks(best, portfolio, {})
    assert float(out.loc[0, "Kelly_Bet_Size"]) == 0.0
    assert float(out.loc[0, "raw_kelly_amount"]) == 25.0
    assert out.loc[0, "kelly_zero_reason"] == "production_ineligible"


def test_every_zero_kelly_row_has_reason():
    best = pd.DataFrame([
        {"canonical_pick_key": "k1", "best_pick": "Over 220.5", "Pick_Status": "No Play", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
        {"canonical_pick_key": "k2", "best_pick": "Over 221.5", "Pick_Status": "High Variance/Speculative", "market_line_source": "live", "line_consistency_flag": True, "line_event_identity_match_flag": True},
    ])
    out = _attach_kelly_to_best_picks(best, pd.DataFrame(), {})
    zeros = out[pd.to_numeric(out["Kelly_Bet_Size"], errors="coerce").fillna(0).eq(0)]
    assert zeros["kelly_zero_reason"].astype(str).str.strip().ne("").all()
