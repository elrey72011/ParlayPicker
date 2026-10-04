"""Actual score producer to immutable snapshot/export/package; synthetic facts only."""
import json
from io import StringIO

import pandas as pd
import pytest

from app_core.market_probability_model import predict_market_probabilities
from app_core.research_display import captured_legacy_half_point, preserve_source_semantics
from core.streamlit_pipeline import _coerce_export_to_canonical
from test_research_probability_producer import forecast, real_path, assert_pass
from test_research_probability_display import NOW, ANALYSIS, QUOTE, START


def produced(**changes):
    raw=forecast(model_status="Market Score Model",market_type="total_over",best_pick="Over 6.5",
        total_line=6.5,spread_line=None,live_total_line=6.5,live_spread_line=None,
        odds_american=-120,calibrated_probability=.56,model_probability=.56,
        expected_value=.56*(1+100/120)-1)
    for k in ("inference_status","push_probability","market_push_probability","probability_semantics"):
        raw.pop(k,None)
    raw.update(feature_home_ppg=4.5,feature_away_ppg=4.5,
        feature_home_oppg=4.5,feature_away_oppg=4.5,ml_feature_eligible=True,
        stats_resolution_status="resolved",feature_home_win_pct=.55,feature_away_win_pct=.45,
        feature_diff_last5=.1)
    raw.update(changes)
    raw["provider_quotes"]=json.dumps([dict(book="novig",market_type=raw["market_type"],
        point=raw["total_line"],price=raw["odds_american"],recorded_at=QUOTE,
        provider_event_id="Home",provider_namespace="odds_api")])
    output=predict_market_probabilities(pd.DataFrame([raw])).iloc[0].to_dict()
    raw.update(output)
    return raw


def test_captured_half_point_compatibility_reaches_actual_renderer(monkeypatch,tmp_path):
    raw=produced()
    result=real_path(monkeypatch,tmp_path,raw)
    assert_pass(result)
    for stage in ("authority","captured"):
        row=result[stage].iloc[0]
        assert row.ml_probability==raw["ml_probability"]
        assert row.calibrated_probability==.56
        assert row.ml_probability!=row.calibrated_probability
        assert row.ml_inference_status=="success"
        assert row.ml_estimate_metadata==raw["ml_estimate_metadata"]
    captured=result["captured"].iloc[0]
    assert captured.probability_semantics=="win_conditional_on_decision"
    assert captured_legacy_half_point(captured,6.5)
    for family in ("overall","totals"):
        row=result["package"]["games"][family][0]
        display=row["research_display"]
        assert display["availability_reason"]=="AVAILABLE"
        assert display["probability"]==pytest.approx(.56)
        assert display["ev"]==pytest.approx(.56*(1+100/120)-1)
        # A raw-model success is not a statement that the blend is a validated model.
        assert display["inference_status"]=="UNKNOWN"
        assert display["source_field"]=="best_available_probability"
        assert display["basis"].startswith("Candidate win estimate")
        assert row["status"]=="PASS"
        if "wager_contract" in row:
            assert row["wager_contract"]["production_bet_amount"]==0
    assert "research_estimate_trace" not in json.dumps(result["package"])
    from test_research_probability_browser import inspect_browser
    browser=inspect_browser(result["package"],tmp_path/"browser",NOW,rendered_html=result["html"])
    assert browser["initial"]["shown"][0]["probability"]==pytest.approx(.56)
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert browser["initial"]["saved"]==[dict(probability=None,ev=None,status="PASS",stake=0)]


@pytest.mark.parametrize("change,reason",[
    ({"quote_id":""},"ESTIMATE_PROVENANCE_NOT_RECORDED"),
    ({"market_period":"","period":""},"ESTIMATE_PROVENANCE_NOT_RECORDED"),
    ({"settlement_rules":""},"ESTIMATE_PROVENANCE_NOT_RECORDED"),
    ({"quote_id":"source-quote","prospective_quote_id":"other"},"ESTIMATE_IDENTITY_MISMATCH"),
    ({"market_period":"full_game","period":"first_half"},"ESTIMATE_IDENTITY_MISMATCH"),
    ({"ml_target":"home_win"},"TARGET_MISMATCH"),
    ({"ml_target":"total_under"},"TARGET_MISMATCH"),
    ({"inference_status":"FAILED"},"INFERENCE_FAILED"),
    ({"probability_semantics":"unsupported"},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability":.1},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability":float("nan")},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability":True},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
])
def test_producer_route_preserves_missing_and_rejected_facts(monkeypatch,tmp_path,change,reason):
    raw=produced();raw.update(change)
    result=real_path(monkeypatch,tmp_path,raw)
    assert_pass(result)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]==reason
    assert display["probability"] is None and display["ev"] is None


def test_original_timestamps_and_sanitized_trace_survive_csv(monkeypatch,tmp_path):
    raw=produced(prediction_generated_at="2026-10-01T19:25:13.123456+00:00",
        game_start_utc="2026-10-01T22:40:31+00:00",game_time_est="2026-10-01 6:40 PM ET")
    result=real_path(monkeypatch,tmp_path,raw)
    row=result["frames"][0].iloc[0]
    assert row.prediction_generated_at==raw["prediction_generated_at"]
    assert row.start==raw["game_start_utc"]
    assert "game_start_utc" not in result["frames"][0].columns
    trace=json.loads(pd.read_csv(StringIO(result["frames"][0].to_csv(index=False))).iloc[0].research_estimate_trace)
    assert trace["source"]["ml_probability"]["value"]==raw["ml_probability"]
    assert trace["source"]["calibrated_probability"]["value"]==.56
    assert trace["source"]["inference_status"]=={"state":"MISSING"}
    assert trace["display"]["availability_reason"]=="AVAILABLE"
    public=result["package"]["games"]["overall"][0]
    assert public["as_of"]==raw["prediction_generated_at"]
    assert public["start"]==raw["game_start_utc"]
    assert public["research_display"]["identity"]["analysis_time"]==public["as_of"]
    assert "MUST-NOT-LEAK" not in row.research_estimate_trace
    assert "provider_quotes" not in trace["source"]


def test_negative_value_displays_without_funding(monkeypatch,tmp_path):
    result=real_path(monkeypatch,tmp_path,produced(calibrated_probability=.4,expected_value=.4*(1+100/120)-1))
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["probability"]==pytest.approx(.4)
    assert display["ev"]==pytest.approx(.4*(1+100/120)-1)
    assert_pass(result)


def test_integer_push_is_never_assumed(monkeypatch,tmp_path):
    raw=produced(total_line=6,best_pick="Over 6",live_total_line=6)
    metadata=json.loads(raw["ml_estimate_metadata"])
    assert metadata["inference_status"]=="success"
    assert metadata["push_probability"] is None
    assert metadata["probability_semantics"]=="UNDECLARED_PUSH_MODEL"
    result=real_path(monkeypatch,tmp_path,raw)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["probability"] is None and display["ev"] is None
    assert not captured_legacy_half_point(result["captured"].iloc[0],6)
    assert_pass(result)


@pytest.mark.parametrize("league",["NCAAF","NHL"])
def test_unsupported_leagues_record_unavailable_inference(league):
    raw=produced(league=league)
    assert pd.isna(raw["ml_probability"])
    assert raw["ml_inference_status"]=="unavailable"
    metadata=json.loads(raw["ml_estimate_metadata"])
    assert metadata["probability"]["state"]=="INVALID"
    assert metadata["push_probability"] is None
    assert "No market-specific model" in metadata["reason"]


def test_origin_metadata_survives_retained_projection_without_numeric_promotion():
    raw=produced()
    frame=_coerce_export_to_canonical(pd.DataFrame([raw]),["MLB"])
    assert frame.iloc[0].ml_estimate_metadata==raw["ml_estimate_metadata"]
    assert frame.iloc[0].ml_inference_status=="success"
    legacy=dict(raw);legacy.pop("ml_estimate_metadata");legacy.pop("ml_inference_status")
    projected=_coerce_export_to_canonical(pd.DataFrame([legacy]),["MLB"]).iloc[0]
    assert pd.isna(projected.ml_inference_status) and pd.isna(projected.ml_estimate_metadata)


def test_ui_analysis_pipeline_transports_actual_origin(monkeypatch):
    from types import SimpleNamespace
    from core import streamlit_pipeline as sp
    from scripts.benchmark_drive_history_loading import blocked_network
    raw=produced()
    for key in ("ml_inference_status","ml_estimate_metadata"):
        raw.pop(key)
    monkeypatch.setattr(sp,"load_base_data",lambda:pd.DataFrame())
    monkeypatch.setattr(sp,"build_theover_bet_rows",lambda *a,**k:pd.DataFrame([raw]))
    monkeypatch.setattr(sp,"fetch_live_odds_dataframe",lambda *a,**k:pd.DataFrame())
    def features(frame,*args):
        out=frame.copy()
        out["League"]="MLB"
        for key,value in raw.items():
            if key.startswith("feature_") or key in {"ml_feature_eligible","stats_resolution_status"}:
                out[key]=value
        return out
    monkeypatch.setattr("app_core.feature_processing.enrich_with_model_features",features)
    monkeypatch.setattr("app_core.mlb_team_stats.enrich_mlb_model_features",lambda f:f)
    monkeypatch.setattr("app_core.external_data_fetcher.enrich_with_external_data",lambda f:f)
    monkeypatch.setattr(sp,"ML_AVAILABLE",True)
    monkeypatch.setattr(sp,"PredictionEngine",object)
    monkeypatch.setattr(sp,"get_cached_prediction_engine",lambda:SimpleNamespace(
        use_fallback=False,predict_batch=lambda f:[.8]*len(f)))
    monkeypatch.setattr("app_core.candidate_chronology.now_utc",lambda:pd.Timestamp(NOW))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats",lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration",lambda:None)
    with blocked_network():
        analysis,best,diagnostics=sp.run_analysis_pipeline(sports=["MLB"],use_ml=True,max_rows=5)
    assert not analysis.empty
    row=analysis.iloc[0]
    assert row.ml_inference_status=="success"
    assert row.ml_probability==pytest.approx(raw["ml_probability"])
    assert row.ml_target=="total_over"
    metadata=json.loads(row.ml_estimate_metadata)
    assert metadata["probability"]["value"]==pytest.approx(row.ml_probability)
    assert metadata["inference_status"]=="success"
    assert diagnostics["market_specific_ml_predictions"]==1


def test_native_aware_clocks_keep_their_original_value(monkeypatch):
    from test_research_probability_display import source,package_for
    raw=source(prediction_generated_at=pd.Timestamp(ANALYSIS),game_start_utc=pd.Timestamp(START))
    frames,package=package_for(monkeypatch,raw)
    assert frames[0].iloc[0].prediction_generated_at==ANALYSIS
    assert frames[0].iloc[0].start==START
    display=package["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]=="AVAILABLE"
    trace=json.loads(frames[0].iloc[0].research_estimate_trace)
    assert trace["source"]["prediction_generated_at"]=={"state":"VALUE","value":ANALYSIS}


def test_origin_diagnostics_do_not_change_authority_payload_hash():
    from app_core.candidate_evidence_schema import project
    raw=produced()
    legacy={k:v for k,v in raw.items() if k not in {"ml_inference_status","ml_estimate_metadata"}}
    with_metadata=project(pd.DataFrame([raw])).iloc[0]
    without_metadata=project(pd.DataFrame([legacy])).iloc[0]
    assert with_metadata.payload_hash==without_metadata.payload_hash
    assert with_metadata.candidate_id==without_metadata.candidate_id
    assert with_metadata.ml_estimate_metadata==raw["ml_estimate_metadata"]


def test_start_override_cannot_borrow_original_research_or_hide_date_lock_rule(monkeypatch,tmp_path):
    from app_core.public_board import build_package
    from app_core.locked_picks import lock_audit,lock_candidates
    result=real_path(monkeypatch,tmp_path,produced())
    frames=[f.copy() for f in result["frames"]]
    frames[0]["start"]="2026-10-02T22:40:00+00:00"
    package=build_package(*frames)
    row=package["games"]["overall"][0]
    assert row["research_display"]["availability_reason"]=="ESTIMATE_IDENTITY_MISMATCH"
    assert row["status"]=="PASS"
    assert lock_audit(package,NOW.isoformat())[0]["Lock status"]=="Other date"
    assert lock_candidates(package,NOW.isoformat())==[]
