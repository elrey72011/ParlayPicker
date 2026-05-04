import pandas as pd
import streamlit_app as app
import core.streamlit_pipeline as sp


def test_identity_columns_ready_before_portfolio(monkeypatch):
    captured = {}

    def fake_run_analysis_pipeline(**kwargs):
        analysis = pd.DataFrame([
            {"league":"NBA","home_team":"A","away_team":"B","game_date":"2026-05-03","market_type":"total_over","expected_value":0.2,"edge":0.2,"calibrated_probability":0.65,"line_consistency_flag":True,"line_event_identity_match_flag":True,"market_line_source":"live","line_provenance_warning":"","total_line":220.5,"decimal_odds":2.0}
        ])
        return analysis, pd.DataFrame(), {}

    def fake_build_best_picks_df(analysis_df, diagnostics_out=None):
        return pd.DataFrame([
            {"league":"NBA","home_team":"A","away_team":"B","game_date":"2026-05-03","market_type":"total_over","best_pick":"Over 220.5","Pick_Status":"Actionable","expected_value":0.2,"edge":0.2,"effective_expected_value":0.2,"effective_edge":0.2,"calibrated_probability":0.65,"line_consistency_flag":True,"line_event_identity_match_flag":True,"market_line_source":"live","line_provenance_warning":"","market_line_used":220.5}
        ])

    def fake_optimize(best_picks_df, bankroll=1000.0):
        captured["cols"] = list(best_picks_df.columns)
        assert "canonical_pick_key" in best_picks_df.columns
        assert best_picks_df["canonical_pick_key"].astype(str).str.strip().ne("").all()
        return pd.DataFrame([
            {"canonical_pick_key": best_picks_df.iloc[0]["canonical_pick_key"], "production_bet_amount": 10.0, "raw_kelly_amount": 40.0, "kelly_cap_reason": "", "production_eligible": True}
        ])

    monkeypatch.setattr(app, "run_analysis_pipeline", fake_run_analysis_pipeline)
    monkeypatch.setattr(sp, "build_best_picks_df", fake_build_best_picks_df)
    monkeypatch.setattr(app, "optimize_portfolio_allocation", fake_optimize)
    monkeypatch.setattr(app, "generate_parlays", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "run_bankroll_simulation", lambda *args, **kwargs: {})
    monkeypatch.setattr(app, "_enrich_with_kalshi_safe", lambda df: (df, None))
    monkeypatch.setattr(app, "_recompute_consensus_from_kalshi", lambda df, require_ml=False: df)
    
    controls = {"sports":["NBA"],"use_ml":False,"theover_spreads":None,"theover_totals":None,"bankroll":1000.0,"use_gemini":False}
    state, _warnings, _errors = app._run_pipeline(controls)
    assert state["diagnostics"]["identity_columns_ready_before_portfolio"] is True
    assert state["diagnostics"].get("best_pick_export_missing_columns", []) == []


def test_empty_production_card_sets_final_empty_diagnostics(monkeypatch):
    def fake_run_analysis_pipeline(**kwargs):
        analysis = pd.DataFrame([{"league":"NBA","home_team":"A","away_team":"B","game_date":"2026-05-03","market_type":"total_over","expected_value":0.2,"edge":0.2,"calibrated_probability":0.65}])
        return analysis, pd.DataFrame(), {}

    def fake_build_best_picks_df(analysis_df, diagnostics_out=None):
        return pd.DataFrame([{"league":"NBA","home_team":"A","away_team":"B","game_date":"2026-05-03","market_type":"total_over","best_pick":"Over 220.5","Pick_Status":"High Variance/Speculative","expected_value":0.2,"edge":0.2,"line_consistency_flag":True,"line_event_identity_match_flag":True,"market_line_source":"live","line_provenance_warning":"","market_line_used":220.5}])

    monkeypatch.setattr(app, "run_analysis_pipeline", fake_run_analysis_pipeline)
    monkeypatch.setattr(sp, "build_best_picks_df", fake_build_best_picks_df)
    monkeypatch.setattr(app, "optimize_portfolio_allocation", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "generate_parlays", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "run_bankroll_simulation", lambda *args, **kwargs: {})
    monkeypatch.setattr(app, "_enrich_with_kalshi_safe", lambda df: (df, None))
    monkeypatch.setattr(app, "_recompute_consensus_from_kalshi", lambda df, require_ml=False: df)

    controls = {"sports":["NBA"],"use_ml":False,"theover_spreads":None,"theover_totals":None,"bankroll":1000.0,"use_gemini":False}
    state, _warnings, _errors = app._run_pipeline(controls)
    assert state["diagnostics"]["final_actionable_count"] == 0
    assert state["diagnostics"]["production_card_empty_flag"] is True
    assert state["diagnostics"]["actionable_family_counts"] == {}
