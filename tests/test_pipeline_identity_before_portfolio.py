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
    required_audit = {"production_eligible", "raw_kelly_amount", "production_bet_amount", "kelly_cap_reason", "kelly_zero_reason", "final_actionable_count", "production_card_empty_flag"}
    assert required_audit.issubset(set(state["best_picks_df"].columns))


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
    assert str(state["diagnostics"]["production_card_empty_reason"]).strip() != ""
    assert state["diagnostics"]["actionable_family_counts"] == {}


def test_empty_card_recovery_publishes_separate_controlled_value_card(monkeypatch):
    def fake_run_analysis_pipeline(**kwargs):
        analysis = pd.DataFrame([{"league":"NBA","home_team":"A","away_team":"B","game_date":"2026-05-03","market_type":"spread_home","expected_value":0.2,"edge":0.2,"calibrated_probability":0.65}])
        return analysis, pd.DataFrame(), {}

    def fake_build_best_picks_df(analysis_df, diagnostics_out=None):
        rows = []
        for i, mt in enumerate(["spread_home", "total_under", "total_over"]):
            # consensus_agreement + odds_american + effective_win_probability are now required
            # by the recovery guards (Neutral consensus, real price, calibrated win > break-even).
            rows.append({"league":"NBA","home_team":f"H{i}","away_team":f"A{i}","game_date":"2026-05-03","market_type":mt,"best_pick":"Team -3.5" if "spread" in mt else ("Under 220.5" if mt=="total_under" else "Over 220.5"),"Pick_Status":"High Variance/Speculative","expected_value":0.2,"edge":0.2,"effective_expected_value":0.2,"effective_edge":0.2,"effective_win_probability":0.62,"empirical_win_probability":0.62,"consensus_agreement":"Neutral","odds_american":-110,"production_expected_value":0.12,"production_edge":0.08,"production_win_probability":0.62,"line_consistency_flag":True,"line_event_identity_match_flag":True,"market_line_source":"live","line_provenance_warning":"","market_line_used":220.5,"best_available_selection_verified":True,"best_available_ranking_verified":True,"final_pick_valid":True})
        return pd.DataFrame(rows)

    # Pin the recovery calibration gate to raw mode (no table) so the test is deterministic and
    # not coupled to the live fitted calibration: raw win 0.62 beats the -110 break-even (.524).
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda *a, **k: None)
    monkeypatch.setattr(app, "run_analysis_pipeline", fake_run_analysis_pipeline)
    monkeypatch.setattr(sp, "build_best_picks_df", fake_build_best_picks_df)
    monkeypatch.setattr(app, "optimize_portfolio_allocation", lambda df, bankroll=1000.0: pd.DataFrame([{"canonical_pick_key": k, "production_bet_amount": 40.0, "raw_kelly_amount": 80.0, "kelly_cap_reason": "", "production_eligible": True} for k in df["canonical_pick_key"].tolist()]))
    monkeypatch.setattr(app, "generate_parlays", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "run_bankroll_simulation", lambda *args, **kwargs: {})
    monkeypatch.setattr(app, "_enrich_with_kalshi_safe", lambda df: (df, None))
    monkeypatch.setattr(app, "_recompute_consensus_from_kalshi", lambda df, require_ml=False: df)
    controls = {"sports":["NBA"],"use_ml":False,"theover_spreads":None,"theover_totals":None,"bankroll":1000.0,"use_gemini":False}
    state, _, _ = app._run_pipeline(controls)
    out = state["best_picks_df"]
    actionable = out[out["Pick_Status"] == "Actionable"]
    # Legacy recovery lacks validated canonical evidence and cannot fund a new run.
    assert actionable.empty
    assert not out["wager_approved"].any()
    assert not out["sellable_as_premium"].any()
    assert state["diagnostics"]["controlled_value_pick_count"] == 0



def test_controlled_value_recovery_rejects_low_probability_plus_money_rows(monkeypatch):
    """Price edge cannot bypass the owner's likely-to-win production floor."""

    def fake_run_analysis_pipeline(**kwargs):
        analysis = pd.DataFrame([
            {
                "league": "MLB",
                "home_team": "Cincinnati",
                "away_team": "Athletics",
                "game_date": "2026-08-05",
                "market_type": "spread_home",
                "expected_value": 0.0341137278,
                "edge": 0.0282599410,
                "calibrated_probability": 0.4400483948,
            }
        ])
        return analysis, pd.DataFrame(), {}

    def fake_build_best_picks_df(analysis_df, diagnostics_out=None):
        return pd.DataFrame([
            {
                "league": "MLB",
                "home_team": "Cincinnati",
                "away_team": "Athletics",
                "game_date": "2026-08-05",
                "market_type": "spread_home",
                "best_pick": "Cincinnati -1.5",
                "Pick_Status": "High Variance/Speculative",
                "expected_value": 0.0341137278,
                "edge": 0.0282599410,
                "effective_expected_value": 0.0341137278,
                "effective_edge": 0.0282599410,
                "effective_win_probability": 0.4400483948,
                "empirical_win_probability": 0.4624687484,
                "consensus_agreement": "Disagrees",
                "odds_american": 135,
                "production_expected_value": 0.0341137278,
                # This is intentionally below the legacy 2% edge floor. The
                # empirical price edge is +3.69 points and is authoritative.
                "production_edge": 0.0145164793,
                "production_win_probability": 0.4400483948,
                "line_consistency_flag": True,
                "line_event_identity_match_flag": True,
                "market_line_source": "live",
                "market_line_used": -1.5,
                "best_available_selection_verified": True,
                "best_available_ranking_verified": True,
                "final_pick_valid": True,
            },
            {
                "league": "MLB",
                "home_team": "New York Yankees",
                "away_team": "Saint Louis",
                "game_date": "2026-08-05",
                "market_type": "spread_home",
                "best_pick": "New York Yankees -1.5",
                "Pick_Status": "High Variance/Speculative",
                "expected_value": 0.0430743718,
                "edge": 0.0306599743,
                "effective_expected_value": 0.0430743718,
                "effective_edge": 0.0306599743,
                "effective_win_probability": 0.4328109426,
                "empirical_win_probability": 0.4485899256,
                "consensus_agreement": "Disagrees",
                "odds_american": 141,
                "production_expected_value": 0.0430743718,
                "production_edge": 0.0178727687,
                "production_win_probability": 0.4328109426,
                "line_consistency_flag": True,
                "line_event_identity_match_flag": True,
                "market_line_source": "live",
                "market_line_used": -1.5,
                "best_available_selection_verified": True,
                "best_available_ranking_verified": True,
                "final_pick_valid": True,
            },
        ])

    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda *a, **k: None)
    monkeypatch.setattr(app, "run_analysis_pipeline", fake_run_analysis_pipeline)
    monkeypatch.setattr(sp, "build_best_picks_df", fake_build_best_picks_df)
    monkeypatch.setattr(
        app,
        "optimize_portfolio_allocation",
        lambda df, bankroll=1000.0: pd.DataFrame([
            {
                "canonical_pick_key": key,
                "production_bet_amount": 5.0,
                "raw_kelly_amount": 10.0,
                "kelly_cap_reason": "",
                "production_eligible": True,
            }
            for key in df["canonical_pick_key"].tolist()
        ]),
    )
    monkeypatch.setattr(app, "generate_parlays", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(app, "run_bankroll_simulation", lambda *args, **kwargs: {})
    monkeypatch.setattr(app, "_enrich_with_kalshi_safe", lambda df: (df, None))
    monkeypatch.setattr(app, "_recompute_consensus_from_kalshi", lambda df, require_ml=False: df)

    controls = {
        "sports": ["MLB"],
        "use_ml": False,
        "theover_spreads": None,
        "theover_totals": None,
        "bankroll": 1000.0,
        "use_gemini": False,
    }
    state, _, _ = app._run_pipeline(controls)
    out = state["best_picks_df"]
    controlled = out[out["controlled_card_recovery"].fillna(False).astype(bool)]

    assert controlled.empty
    assert not out["wager_approved"].any()
    assert state["diagnostics"]["empty_card_recovery_candidate_count"] == 0
    assert state["diagnostics"]["empty_card_recovery_promoted_count"] == 0


def test_expanded_prepared_pool_is_evaluated_and_persisted(monkeypatch,tmp_path):
    import sqlite3
    from io import StringIO
    from activation_fixture import setup,NOW
    from app_core import prediction_evidence as evidence
    from core import live_wager_contract as authority
    r,policy,config=setup(tmp_path/'ledger.db')
    r.update(game_date='2026-09-14',game_start_utc=r['start'],calibrated_probability=.62,
             expected_value=.1,edge=.1,line_consistency_flag=True,line_event_identity_match_flag=True,
             market_line_source='live',market_line_used=-2.5,Pick_Status='PASS',odds_source='DraftKings',
             best_available_candidate_count=2,best_available_selected=True,candidate_id='original')
    alternative=dict(r,candidate_id='expanded',market_type='spread_away',line=2.5,spread_line=2.5,
                     market_line_used=2.5,selection='Away +2.5',best_pick='Away +2.5',best_available_selected=False)
    import json
    quotes=json.dumps([dict(book='draftkings', market_type=kind, point=line, price=-110,
                           recorded_at=r['quote_time'], provider_event_id='one', provider_namespace='odds_api')
                       for kind,line in [('spread_home',-2.5),('spread_away',2.5)]])
    r['provider_quotes']=quotes;alternative['provider_quotes']=quotes
    r['identity_verified']=False
    base=pd.DataFrame([r]);expanded=pd.DataFrame([r,alternative])
    seen={}
    monkeypatch.setattr(app,'run_analysis_pipeline',lambda **kwargs:(base.copy(),pd.DataFrame(),{}))
    def build(frame,diagnostics_out=None):
        diagnostics_out['candidate_authority_df']=expanded.copy()
        diagnostics_out['candidate_audit_df']=expanded.copy()
        return base.copy()
    monkeypatch.setattr(sp,'build_best_picks_df',build)
    monkeypatch.setattr(app,'optimize_portfolio_allocation',lambda *args,**kwargs:pd.DataFrame())
    monkeypatch.setattr(app,'generate_parlays',lambda *args,**kwargs:pd.DataFrame())
    monkeypatch.setattr(app,'run_bankroll_simulation',lambda *args,**kwargs:{})
    monkeypatch.setattr(app,'_enrich_with_kalshi_safe',lambda df:(df,None))
    monkeypatch.setattr(app,'_recompute_consensus_from_kalshi',lambda df,require_ml=False:df)
    def prepare(frame):
        assert set(frame.candidate_id)=={'original','expanded'}
        frame=frame.copy();frame['evidence_snapshot_id']=frame.candidate_id.map(lambda x:'prepared-'+x)
        seen['prepared']=frame.copy()
        return frame
    monkeypatch.setattr('core.prospective_uncertainty.prepare_live',prepare)
    real_finalize=authority.finalize_live_wagers
    def finalize(frame,best,bankroll):
        assert frame.evidence_snapshot_id.tolist()==['prepared-original','prepared-expanded']
        return real_finalize(frame,best,bankroll,now=NOW,policies={'NFL':policy},config=config,reviews=frame)
    monkeypatch.setattr(authority,'finalize_live_wagers',finalize)
    db=tmp_path/'evidence.sqlite3'
    real_begin=evidence.begin_run
    monkeypatch.setattr(evidence,'begin_run',lambda controls:real_begin(controls,path=db))
    real_capture=evidence.capture_run
    monkeypatch.setattr(evidence,'capture_run',lambda context,audit,card,inputs,**kwargs:real_capture(context,audit,card,inputs,path=db,**kwargs))
    state,_,_=app._run_pipeline({'sports':['NFL'],'use_ml':False,'theover_spreads':None,
        'theover_totals':None,'bankroll':1000.,'use_gemini':False})
    assert state['diagnostics']['prediction_snapshot_saved'],state['diagnostics'].get('prediction_snapshot_error')
    with sqlite3.connect(db) as conn:
        saved=pd.read_csv(StringIO(conn.execute('SELECT candidates FROM snapshots').fetchone()[0]))
    assert set(saved.candidate_id)=={'original','expanded'}
    indexed=saved.set_index('candidate_id')
    assert indexed.loc['expanded','evidence_snapshot_id']=='prepared-expanded'
    assert indexed.loc['original','market_type']=='spread_home'
    assert indexed.loc['expanded','market_type']=='spread_away'
    assert bool(indexed.loc['expanded','best_available_selected'])
    assert not bool(indexed.loc['original','best_available_selected'])
    assert state['best_picks_df'].iloc[0]['candidate_id']=='expanded'
