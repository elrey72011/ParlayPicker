"""Prospective sanitized evidence through actual application/capture/export/replay."""
from copy import deepcopy
import json
import base64
import hashlib
from types import SimpleNamespace
import pandas as pd
import pytest
from app_core import nfl_inference_evidence as packet
from app_core import research_replay
from app_core.research_estimate_trace import fact
from core import streamlit_pipeline as sp
from scripts.benchmark_drive_history_loading import blocked_network
from test_source_contract_pipeline import fixture, exported, NOW, QUOTE, INFERENCE, START, FrozenDateTime

VALUES=dict(zip(packet.FEATURES,[24.,22.,21.,23.,7.,7.,.55,.45,.1,1.5,-1.5]))

@pytest.fixture(autouse=True)
def no_network():
    with blocked_network():yield


def actual(monkeypatch, *, changes=None, dependencies=True, game_changes=None):
    from app_core import odds_api, football_identity_capture
    game,_=fixture(monkeypatch)
    if game_changes:game_changes(game)
    class Client:
        def __init__(self,**kwargs):pass
        def get_odds(self,sport,date=None):return [deepcopy(game)] if sport=="americanfootball_nfl" else []
    monkeypatch.setattr(odds_api,"TheOddsAPIClient",Client)
    monkeypatch.setattr(odds_api,"datetime",FrozenDateTime)
    monkeypatch.setattr(sp,"_game_date_fallback",lambda:pd.Timestamp("2026-10-06"))
    monkeypatch.setattr(sp,"_get_odds_api_key",lambda:"offline-unused")
    monkeypatch.setattr(sp,"load_base_data",lambda:pd.DataFrame())
    monkeypatch.setattr("app_core.nfl_novig.recover_nfl_novig",lambda games,key:games)
    monkeypatch.setattr(football_identity_capture.requests,"get",lambda *a,**k:SimpleNamespace(raise_for_status=lambda:None,json=lambda:{"events":[]}))
    def features(frame,*args):
        out=frame.copy()
        for k,v in dict(VALUES,League="NFL",ml_feature_eligible=True,stats_resolution_status="resolved").items():out[k]=v
        if changes:
            for k,v in changes.items():out[k]=v
        if dependencies:
            deps={k:{"payload":dict(feature=k,value=v,source_id="synthetic:scoring-source",provider_event_id=game["id"],
                  available_at="2026-10-06T18:00:00Z",observed_at="2026-10-06T19:00:00Z")} for k,v in VALUES.items()}
            for d in deps.values():
                raw=json.dumps(dict(event_id=game["id"],value=d["payload"]["value"])).encode()
                d["payload"].update(source_artifact=dict(bytes_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()),value_path=["value"],event_path=["event_id"])
                d["sha256"]=packet.digest(d["payload"])
            out["nfl_feature_dependencies"]=json.dumps(deps)
        return out
    monkeypatch.setattr("app_core.feature_processing.enrich_with_model_features",features)
    monkeypatch.setattr("app_core.external_data_fetcher.enrich_with_external_data",lambda f:f)
    monkeypatch.setattr(sp,"ML_AVAILABLE",True)
    monkeypatch.setattr(sp,"PredictionEngine",object)
    monkeypatch.setattr(sp,"get_cached_prediction_engine",lambda:SimpleNamespace(use_fallback=False,predict_batch=lambda f:[.8]*len(f)))
    class FeatureClock(FrozenDateTime):
        @classmethod
        def now(cls,tz=None):
            return (pd.Timestamp(INFERENCE)-pd.Timedelta(seconds=1)).to_pydatetime()
    monkeypatch.setattr("app_core.football_feature_capture.datetime",FeatureClock)
    monkeypatch.setattr("app_core.research_estimate_trace.generated_time",lambda:INFERENCE)
    monkeypatch.setattr("app_core.candidate_chronology.now_utc",lambda:pd.Timestamp(NOW))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats",lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration",lambda:None)
    analysis,_,_=sp.run_analysis_pipeline(sports=["NFL"],use_ml=True,max_rows=20)
    assert len(analysis)==2
    return analysis


def mutate(row, change):
    item=json.loads(row["ml_estimate_metadata"])
    change(item["nfl_inputs"]["payload"])
    item["nfl_inputs"]["sha256"]=packet.digest(item["nfl_inputs"]["payload"])
    row=dict(row,ml_estimate_metadata=json.dumps(item))
    return row


def test_complete_actual_pipeline_private_capture_replay_download_and_display(monkeypatch,tmp_path):
    analysis=actual(monkeypatch)
    assert all(packet.diagnose(r)["status"]=="COMPLETE" for r in analysis.to_dict("records"))
    raw=analysis.ml_probability.tolist();blend=analysis.calibrated_probability.tolist();ev=analysis.expected_value.tolist()
    result=exported(monkeypatch,tmp_path,analysis)
    receipt=research_replay.retain_export(result["frames"],result["package"],result["card"],result["captured"],path=result["db"])
    saved,sources=research_replay.read_export(receipt["export_id"],path=result["db"])
    source=next(iter(sources.values()))
    producer=research_replay.frame_from_payload(source["original"]["producer"])
    assert producer.ml_probability.tolist()==raw and producer.calibrated_probability.tolist()==blend and producer.expected_value.tolist()==ev
    for row in producer.to_dict("records"):
        replay=packet.replay(row);assert replay["raw_probability"]==row["ml_probability"] and replay["blended_probability"]==row["calibrated_probability"]
        assert replay["scientific_acceptance"] is False and replay["wagering_authority"] is False
        p=json.loads(row["ml_estimate_metadata"])["nfl_inputs"]["payload"]
        assert p["feature_order"]==list(packet.FEATURES) and p["features"][packet.FEATURES[0]]==fact(24.)
        assert p["inference_time"]==INFERENCE and p["event_offer"]["offer"]["source_time"]==pd.Timestamp(QUOTE).isoformat()
        assert p["event_offer"]["event"]["start"]==pd.Timestamp(START).isoformat()
        assert p["raw_probability"]!=p["blend"]["probability"]
    rebuilt=[__import__('app_core.per_game_boards',fromlist=['per_game_board']).per_game_board(
        research_replay.frame_from_payload(source["captured_card"]),research_replay.frame_from_payload(source["captured_candidates"]),family=f,novig_only=True) for f in ("overall","sides","totals")]
    from app_core.public_board import build_package
    assert build_package(*rebuilt)==saved["package"]
    public=json.dumps(saved["package"])
    assert all(x not in public for x in ("nfl_inputs","bytes_base64","synthetic:scoring-source","source_dependencies"))
    assert saved["package"]["games"]["overall"][0]["research_display"]["availability_reason"]=="AVAILABLE"
    assert saved["package"]["games"]["overall"][0]["research_display"]["ev"] is None # FVS protection remains.
    assert not result["captured"].production_eligible.fillna(False).any()
    assert all(r["production_bet_amount"]==0 for r in result["card"].wager_contract)
    bundle,download=research_replay.download_bundle(receipt,expected_package_hash=receipt["package_hash"],path=result["db"])
    assert bundle and download["source_boundary"]=="RETAINED"
    unchanged=actual(monkeypatch,dependencies=False)
    assert unchanged.ml_probability.tolist()==raw and unchanged.calibrated_probability.tolist()==blend and unchanged.expected_value.tolist()==ev
    assert all(packet.diagnose(r)["status"]=="INCOMPLETE" for r in unchanged.to_dict("records"))


@pytest.mark.parametrize("change,diagnostic",[
    (lambda p:p["features"].update(feature_home_ppg={"state":"MISSING"}),"features.feature_home_ppg"),
    (lambda p:p["feature_order"].reverse(),"features.contract_order"),
    (lambda p:p["event_offer"]["event"].update(provider_event_id="other-event"),"event_offer.conflict"),
    (lambda p:p["event_offer"]["offer"].update(line=4.5),"event_offer.conflict"),
    (lambda p:p["event_offer"]["offer"].update(price=110),"event_offer.conflict"),
    (lambda p:p.update(inference_time=START),"origin.inference_time"),
    (lambda p:p["source_dependencies"][packet.FEATURES[0]]["payload"].update(available_at=START),"features.availability_clock:feature_home_ppg"),
    (lambda p:p["source_dependencies"][packet.FEATURES[0]]["payload"].update(provider_event_id="unrelated"),"features.dependency_event:feature_home_ppg"),
    (lambda p:p["artifacts"][packet.ARTIFACTS[0]].update(bytes_base64="Y29ycnVwdA=="),"artifact.integrity:app_core/market_probability_model.py"),
    (lambda p:p.update(wagering_authority=True),"packet.authority_forbidden"),
    (lambda p:p["source_dependencies"][packet.FEATURES[0]]["payload"]["source_artifact"].update(bytes_base64="e30="),"features.source_artifact_integrity:feature_home_ppg"),
    (lambda p:p["source_dependencies"][packet.FEATURES[0]]["payload"].update(observed_at="not-a-clock"),None),
])
def test_rehashed_negative_packets_reject_export_display_and_no_authority(monkeypatch,tmp_path,change,diagnostic):
    analysis=actual(monkeypatch);row=mutate(analysis.iloc[0].to_dict(),change)
    if diagnostic:assert diagnostic in packet.diagnose(row)["errors"]
    else:assert packet.diagnose(row)["status"] in {"REJECTED","INCOMPLETE"}
    with pytest.raises(ValueError):packet.replay(row)
    analysis.at[analysis.index[0],"ml_estimate_metadata"]=row["ml_estimate_metadata"]
    result=exported(monkeypatch,tmp_path,analysis)
    rejected=0
    for frame in result["frames"]:
        for r in frame.to_dict("records"):
            if r.get("research_display") and 'producer.nfl_inputs' in str(r.get("research_estimate_trace")):
                display=json.loads(r["research_display"]) if isinstance(r["research_display"],str) else r["research_display"]
                rejected+=1
                assert display["probability"] is None and display["ev"] is None
    assert rejected>0
    from scripts.publish_board import render
    assert 'nfl_inputs' not in render(result["package"])
    assert not result["captured"].production_eligible.fillna(False).any()
    assert all(r["production_bet_amount"]==0 for r in result["card"].wager_contract)


def test_future_missing_required_feature_fails_at_original_predictor(monkeypatch,tmp_path):
    analysis=actual(monkeypatch,changes={"feature_home_ppg":None})
    assert analysis.ml_inference_status.eq("unavailable").all()
    assert analysis.ml_probability.isna().all()
    assert all(packet.diagnose(r)["status"]=="REJECTED" for r in analysis.to_dict("records"))


@pytest.mark.parametrize("field,value",[("spread_line",4.5),("odds_american",110),("home_team","Other"),("prediction_generated_at",START)])
def test_changed_candidate_bindings_reject(monkeypatch,field,value):
    row=actual(monkeypatch).iloc[0].to_dict();row[field]=value
    assert packet.diagnose(row)["status"]=="REJECTED"
    with pytest.raises(ValueError):packet.replay(row)


def test_missing_runtime_and_changed_configuration_replay_fail_closed(monkeypatch):
    row=actual(monkeypatch).iloc[0].to_dict()
    bad=mutate(row,lambda p:p.update(runtime={}))
    assert "runtime.missing" in packet.diagnose(bad)["errors"]
    with pytest.raises(ValueError):packet.replay(bad)
    bad=mutate(row,lambda p:p["configuration"]["score_parameters"].update(reliability=.8))
    with pytest.raises(ValueError,match="configuration/runtime mismatch"):packet.replay(bad)


def test_legacy_retained_metadata_is_not_backfilled(monkeypatch):
    row=actual(monkeypatch).iloc[0].to_dict();item=json.loads(row["ml_estimate_metadata"]);item.pop("nfl_inputs")
    row["ml_estimate_metadata"]=json.dumps(item);before=deepcopy(row)
    assert packet.diagnose(row)["status"]=="UNKNOWN" and row==before
    with pytest.raises(ValueError):packet.replay(row)


@pytest.mark.parametrize("failure",["missing_artifact","invalid_dependency_json"])
def test_capture_failure_does_not_change_original_inference_or_blend(monkeypatch,tmp_path,failure):
    original=actual(monkeypatch)
    if failure=="missing_artifact":
        def absent():raise OSError("synthetic unreadable consumed artifact")
        monkeypatch.setattr(packet,"artifacts",absent)
        after=actual(monkeypatch)
    else:
        after=actual(monkeypatch,dependencies=False,changes={"nfl_feature_dependencies":"{invalid json"})
    assert after.ml_probability.tolist()==original.ml_probability.tolist()
    assert after.calibrated_probability.tolist()==original.calibrated_probability.tolist()
    assert after.expected_value.tolist()==original.expected_value.tolist()
    assert all(packet.diagnose(r)["status"]=="REJECTED" for r in after.to_dict("records"))
    result=exported(monkeypatch,tmp_path,after)
    assert all(r["research_display"]["probability"] is None and r["research_display"]["ev"] is None for r in result["package"]["games"]["overall"])
    assert all(r["production_bet_amount"]==0 for r in result["card"].wager_contract)



def test_replay_binds_consumed_callables_and_imported_blend_weights(monkeypatch):
    row=actual(monkeypatch).iloc[0].to_dict()
    p=json.loads(row["ml_estimate_metadata"])["nfl_inputs"]["payload"]
    assert p["predictor_callables"]==packet.predictor_callables()
    assert p["blend"]["consumed"]["weights"]=={name:getattr(sp,name) for name in packet.BLEND_WEIGHTS}
    monkeypatch.setattr(sp,"FALLBACK_ML_WEIGHT",.001)
    with pytest.raises(ValueError,match="consumed blend configuration mismatch"):packet.replay(row)
