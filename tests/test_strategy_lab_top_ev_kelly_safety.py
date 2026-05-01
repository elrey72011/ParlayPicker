import pandas as pd

from core.streamlit_pipeline import optimize_portfolio_allocation


def _rows():
    return pd.DataFrame([
        {"league":"MLB","home_team":"A","away_team":"B","best_pick":"Over 8.5","Pick_Status":"Actionable","line_consistency_flag":True,"line_event_identity_match_flag":True,"market_line_source":"live","line_provenance_warning":"","market_line_used":8.5,"calibrated_probability":0.6,"decimal_odds":2.0,"expected_value":0.1,"edge":0.1},
        {"league":"MLB","home_team":"C","away_team":"D","best_pick":"Over unresolved","Pick_Status":"No Play","line_consistency_flag":False,"line_event_identity_match_flag":False,"market_line_source":"rejected_live","line_provenance_warning":"Total line unresolved","market_line_used":pd.NA,"calibrated_probability":0.6,"decimal_odds":2.0,"expected_value":0.2,"edge":0.2},
        {"league":"MLB","home_team":"E","away_team":"F","best_pick":"Under 7.5","Pick_Status":"High Variance/Speculative","line_consistency_flag":True,"line_event_identity_match_flag":True,"market_line_source":"live","line_provenance_warning":"","market_line_used":7.5,"calibrated_probability":0.6,"decimal_odds":2.0,"expected_value":0.2,"edge":0.2},
    ])


def test_kelly_non_actionable_rows_are_zeroed_and_flagged():
    out = optimize_portfolio_allocation(_rows(), bankroll=1000.0)
    non_actionable = out[out["Pick_Status"] != "Actionable"]
    assert (non_actionable["production_bet_amount"] == 0).all()
    assert non_actionable["kelly_cap_reason"].eq("Non-production row").all()


def test_kelly_pick_and_slate_caps_are_enforced():
    rows = pd.concat([_rows().iloc[[0]]] * 10, ignore_index=True)
    out = optimize_portfolio_allocation(rows, bankroll=1000.0)
    assert out["production_bet_amount"].max() <= 40.0 + 1e-9
    assert out["production_bet_amount"].sum() <= 250.0 + 1e-9


def test_kelly_contains_raw_and_production_amounts():
    out = optimize_portfolio_allocation(_rows(), bankroll=1000.0)
    assert "raw_kelly_amount" in out.columns
    assert "production_bet_amount" in out.columns
