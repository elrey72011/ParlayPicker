"""Actual native adapters through inference/capture/export/replay, with no network."""
import base64
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import pandas as pd
import pytest
from app_core import nfl_native_provenance as native, nfl_inference_evidence as packet, research_replay
from app_core import feature_processing as fp, odds_api, football_identity_capture, source_contract as contracts
from core import streamlit_pipeline as sp
from scripts.benchmark_drive_history_loading import blocked_network
from test_source_contract_pipeline import fixture, exported, FrozenDateTime, INFERENCE, NOW, START
from test_nfl_inference_evidence import mutate

SOURCE_TIME="2026-10-06T19:20:00Z"


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def schedule():
    rows=[]
    for club,opponent,base in [("DAL","PHI",25),("BUF","ATL",20)]:
        for i in range(6):
            home,away=(club,opponent) if i%2==0 else (opponent,club)
            scored=base+i%3;allowed=20+i%4
            h,a=(scored,allowed) if i%2==0 else (allowed,scored)
            rows.append(dict(game_id=f"synthetic-prior:{club}:{i}",season=2026,gameday=f"2026-09-{1+i*4:02d}",
                home_team=home,away_team=away,home_score=h,away_score=a,result=h-a))
    rows.append(dict(game_id="synthetic-future",season=2026,gameday="2026-10-07",home_team="DAL",away_team="BUF",home_score=99,away_score=0,result=99))
    return pd.DataFrame(rows)


def native_pipeline(monkeypatch, *, data=None, retain=True, cached_stats=None, observation_time=SOURCE_TIME):
    game,catalog=fixture(monkeypatch)
    game.update(home_team="Dallas Cowboys",away_team="Buffalo Bills",matchup_id="Dallas Cowboys|Buffalo Bills|2026-10-06")
    market=game["bookmakers"][0]["markets"][0]
    market["outcomes"][0]["name"]=game["home_team"]
    market["outcomes"][1]["name"]=game["away_team"]
    for outcome in market["outcomes"]:
        entry=catalog[outcome["source_contract_ref"]]
        entry["receipt"]["identity"]=contracts.identity(game,game["bookmakers"][0],market,outcome)
        entry["receipt"]["reference_team"]=entry["receipt"]["identity"]["selection"]
        entry["sha256"]=contracts.digest(entry["receipt"])
    class Client:
        def __init__(self,**kwargs):pass
        def get_odds(self,sport,date=None):return [deepcopy(game)] if sport=="americanfootball_nfl" else []
    calls=[]
    def import_schedules(years):
        calls.append(years)
        return (data if data is not None else schedule()).copy()
    monkeypatch.setattr(fp,"nfl",SimpleNamespace(__name__="synthetic_nfl_schedule_transport",__version__="offline-v1",import_schedules=import_schedules))
    observed_stats=[]
    real_fetch=fp.fetch_nfl_stats.__wrapped__
    def stats_adapter(*a,**kw):
        stats=deepcopy(cached_stats) if cached_stats is not None else real_fetch(*a,**kw)
        observed_stats.extend(deepcopy(stats))
        return stats
    monkeypatch.setattr(fp,"fetch_nfl_stats",stats_adapter)
    monkeypatch.setattr(native,"now",lambda:observation_time)
    if not retain:monkeypatch.setattr(native,"bind",lambda frame,*args:frame)
    monkeypatch.setattr(odds_api,"TheOddsAPIClient",Client)
    monkeypatch.setattr(odds_api,"datetime",FrozenDateTime)
    monkeypatch.setattr(sp,"_game_date_fallback",lambda:pd.Timestamp("2026-10-06"))
    monkeypatch.setattr(sp,"_get_odds_api_key",lambda:"offline-unused")
    monkeypatch.setattr(sp,"load_base_data",lambda:pd.DataFrame())
    monkeypatch.setattr("app_core.nfl_novig.recover_nfl_novig",lambda games,key:games)
    monkeypatch.setattr(football_identity_capture.requests,"get",lambda *a,**k:SimpleNamespace(raise_for_status=lambda:None,json=lambda:{"events":[]}))
    monkeypatch.setattr("app_core.external_data_fetcher.enrich_with_external_data",lambda f:f)
    monkeypatch.setattr(sp,"ML_AVAILABLE",True)
    monkeypatch.setattr(sp,"PredictionEngine",object)
    monkeypatch.setattr(sp,"get_cached_prediction_engine",lambda:SimpleNamespace(use_fallback=False,predict_batch=lambda f:[.8]*len(f)))
    class FeatureClock(FrozenDateTime):
        @classmethod
        def now(cls,tz=None):return (pd.Timestamp(INFERENCE)-pd.Timedelta(seconds=1)).to_pydatetime()
    monkeypatch.setattr("app_core.football_feature_capture.datetime",FeatureClock)
    monkeypatch.setattr("app_core.research_estimate_trace.generated_time",lambda:INFERENCE)
    monkeypatch.setattr("app_core.candidate_chronology.now_utc",lambda:pd.Timestamp(NOW))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats",lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration",lambda:None)
    analysis,_,_=sp.run_analysis_pipeline(sports=["NFL"],use_ml=True,max_rows=20)
    assert len(analysis)==2 and len(calls)==(0 if cached_stats is not None else 1)
    analysis.attrs["native_stats"]=observed_stats
    return analysis


def original_of(row,name=None):
    name=name or packet.FEATURES[0]
    return json.loads(base64.b64decode(json.loads(row["ml_estimate_metadata"])["nfl_inputs"]["payload"]["source_dependencies"][name]["payload"]["source_artifact"]["bytes_base64"]))


def rehash(row, change, name=None):
    name=name or packet.FEATURES[0]
    def update(p):
        r=p["source_dependencies"][name];d=r["payload"]
        original=json.loads(base64.b64decode(d["source_artifact"]["bytes_base64"]))
        change(original,d)
        for member in original.get("inputs",[]):
            member["receipt"]["sha256"]=native.digest(member["receipt"]["payload"])
        raw=json.dumps(original).encode()
        d.update(scope=original["scope"],source_artifact=dict(bytes_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest()))
        r["sha256"]=packet.digest(d)
    return mutate(row,update)


def test_actual_native_adapter_pipeline_capture_export_replay_and_public_boundary(monkeypatch,tmp_path):
    analysis=native_pipeline(monkeypatch)
    assert all(packet.diagnose(r)["status"]=="COMPLETE" for r in analysis.to_dict("records"))
    raw=analysis.ml_probability.tolist();blend=analysis.calibrated_probability.tolist()
    for row in analysis.to_dict("records"):
        p=json.loads(row["ml_estimate_metadata"])["nfl_inputs"]["payload"]
        assert set(p["source_dependencies"])==set(packet.FEATURES)
        for name in packet.FEATURES:
            original=original_of(row,name)
            assert original["scope"]["availability_basis"]=="original_adapter_first_observation"
            assert original["scope"]["publisher_available_at"] is None
            assert original["scope"]["available_at"]==pd.Timestamp(SOURCE_TIME).isoformat()
            for item in original["inputs"]:
                native_input=item["receipt"]["payload"]
                assert len(native_input["original"]["rows"])==6
                assert all("synthetic-prior:" in c["game_id"]["value"] for c in native_input["original"]["rows"])
                assert native_input["original"]["source"]["wire_bytes_retained"] is False
        result=packet.replay(row)
        assert result["raw_probability"]==row["ml_probability"] and result["blended_probability"]==row["calibrated_probability"]
        assert not result["scientific_acceptance"] and not result["wagering_authority"]
    result=exported(monkeypatch,tmp_path,analysis)
    receipt=research_replay.retain_export(result["frames"],result["package"],result["card"],result["captured"],path=result["db"])
    saved,sources=research_replay.read_export(receipt["export_id"],path=result["db"])
    producer=research_replay.frame_from_payload(next(iter(sources.values()))["original"]["producer"])
    assert producer.ml_probability.tolist()==raw and producer.calibrated_probability.tolist()==blend
    for row in producer.to_dict("records"):assert packet.replay(row)["raw_probability"]==row["ml_probability"]
    public=json.dumps(saved["package"])
    assert all(x not in public for x in ["source_dependencies","bytes_base64","synthetic-prior:",native.COLUMN])
    assert not result["captured"].production_eligible.fillna(False).any()
    assert all(r["production_bet_amount"]==0 for r in result["card"].wager_contract)
    assert saved["package"]["games"]["overall"][0]["research_display"]["availability_reason"]=="AVAILABLE"


def test_receipts_leave_actual_native_features_raw_and_blend_unchanged(monkeypatch):
    with monkeypatch.context() as disabled:
        before=native_pipeline(disabled,retain=False)
        before_status=[packet.diagnose(r)["status"] for r in before.to_dict("records")]
    with monkeypatch.context() as enabled:
        after=native_pipeline(enabled)
        after_status=[packet.diagnose(r)["status"] for r in after.to_dict("records")]
    for key in ["ml_probability","calibrated_probability","expected_value"]:
        assert before[key].tolist()==after[key].tolist()
    for a,b in zip(before.to_dict("records"),after.to_dict("records")):
        assert json.loads(a["ml_estimate_metadata"])["nfl_inputs"]["payload"]["features"]==json.loads(b["ml_estimate_metadata"])["nfl_inputs"]["payload"]["features"]
    assert before_status==["INCOMPLETE","INCOMPLETE"]
    assert after_status==["COMPLETE","COMPLETE"]


@pytest.mark.parametrize("change,reason",[
 (lambda o,d:o["scope"]["event"].update(provider_event_id="unrelated"),"target_event"),
 (lambda o,d:o["scope"]["event"].update(home=o["scope"]["event"]["away"],away=o["scope"]["event"]["home"]),"target_event"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"].update(team="ATLANTA FALCONS"),"aggregate_team"),
 (lambda o,d:o["inputs"][0].update(slot="away"),"input_slots"),
 (lambda o,d:o["inputs"][0].update(stat="win_pct"),"stat_mapping"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"].update(cutoff="2026-10-07"),"cutoff_after_event"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"].update(season=2025),"schema"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"]["original"]["rows"][0].update(home_team={"type":"scalar","value":"BUF"},away_team={"type":"scalar","value":"KC"}),"unrelated_or_ineligible_member"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"]["original"]["rows"][0].update(gameday={"type":"scalar","value":"2026-10-07"}),"unrelated_or_ineligible_member"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"]["original"].update(observed_at=START),"availability_origin"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"]["original"].update(publisher_available_at=START),"publisher_availability"),
 (lambda o,d:o["scope"].update(availability_basis="invented_publisher_time"),"availability_basis"),
 (lambda o,d:o["scope"].update(available_at="invalid-clock"),"invalid_scope_clock"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"]["original"]["source"].update(callable_source="contradictory source code"),"source_callable_integrity"),
 (lambda o,d:o.update(transformation="unreviewed"),"transformation"),
 (lambda o,d:o.update(adapter_code={}),"adapter_identity"),
 (lambda o,d:o["inputs"][0]["receipt"]["payload"]["values"].update(points_per_game={"type":"scalar","value":100.}),"aggregate_values"),
])
def test_consistently_rehashed_native_applicability_attacks_reject_export_and_replay(monkeypatch,tmp_path,change,reason):
    row=rehash(native_pipeline(monkeypatch).iloc[0].to_dict(),change)
    assessment=packet.diagnose(row)
    assert assessment["status"]=="REJECTED" and any(reason in x for x in assessment["errors"])
    with pytest.raises(ValueError):packet.replay(row)
    result=exported(monkeypatch,tmp_path,pd.DataFrame([row]))
    traces=[json.loads(r["research_estimate_trace"]) for f in result["frames"] for r in f.to_dict("records") if r.get("research_estimate_trace")]
    assert any(t.get("first_rejection_stage")=="producer.nfl_inputs" and any(reason in x for x in t["nfl_evidence"]["errors"]) for t in traces)
    assert all(g["research_display"]["availability_reason"]!="AVAILABLE" for family in result["package"]["games"].values() for g in family)


@pytest.mark.parametrize("missing",["dependencies","observation","member_id","member_season","source_version","one_feature","aggregate_season","aggregate_team","aggregate_cutoff","aggregate_values","members","member_date","member_score","member_team","scope_available_at","scope_observed_at","dependency_available_at","dependency_observed_at"])
def test_missing_native_facts_remain_unknown_and_cannot_complete_replay(monkeypatch,missing):
    row=native_pipeline(monkeypatch).iloc[0].to_dict()
    if missing=="dependencies":row=mutate(row,lambda p:p.update(source_dependencies=None))
    elif missing=="one_feature":row=mutate(row,lambda p:p["source_dependencies"].pop(packet.FEATURES[0]))
    else:
        def change(o,d):
            original=o["inputs"][0]["receipt"]["payload"]["original"]
            if missing=="observation":original["observed_at"]=None
            elif missing=="source_version":original["source"]["version"]=None
            elif missing.startswith("scope_"):o["scope"][missing.removeprefix("scope_")]=None
            elif missing.startswith("dependency_"):d[missing.removeprefix("dependency_")]=None
            elif missing.startswith("aggregate_"):
                key=missing.removeprefix("aggregate_")
                o["inputs"][0]["receipt"]["payload"][key]=None
            elif missing=="members":original["rows"]=[]
            else:
                key={"member_id":"game_id","member_season":"season","member_date":"gameday","member_score":"home_score","member_team":"home_team"}[missing]
                original["rows"][0][key]={"type":"none"}
        row=rehash(row,change)
    assessment=packet.diagnose(row)
    assert assessment["status"]=="INCOMPLETE" and assessment["unknown"]
    with pytest.raises(ValueError):packet.replay(row)


def test_missing_derivation_member_is_unknown_not_complete(monkeypatch):
    row=rehash(native_pipeline(monkeypatch).iloc[0].to_dict(),lambda o,d:o["inputs"].pop(),name="feature_diff_last5")
    result=packet.diagnose(row)
    assert result["status"]=="INCOMPLETE" and any("input_coverage" in x for x in result["unknown"])
    with pytest.raises(ValueError):packet.replay(row)


def test_native_corruption_rejects_without_executing_saved_bytes(monkeypatch):
    row=mutate(native_pipeline(monkeypatch).iloc[0].to_dict(),lambda p:p["source_dependencies"][packet.FEATURES[0]]["payload"]["source_artifact"].update(bytes_base64="e30="))
    assert packet.diagnose(row)["status"]=="REJECTED"
    with pytest.raises(ValueError):packet.replay(row)


def test_native_capture_failure_preserves_values_and_unknown(monkeypatch):
    with monkeypatch.context() as control:before=native_pipeline(control)
    def failed(*args):raise OSError("synthetic receipt failure")
    monkeypatch.setattr(native,"code_identity",failed)
    after=native_pipeline(monkeypatch)
    assert before.ml_probability.tolist()==after.ml_probability.tolist()
    assert before.calibrated_probability.tolist()==after.calibrated_probability.tolist()
    assert all(packet.diagnose(r)["status"]=="INCOMPLETE" for r in after.to_dict("records"))


def test_cached_native_sources_keep_first_observation_separate_from_mapping(monkeypatch,tmp_path):
    with monkeypatch.context() as first:
        original=native_pipeline(first)
        cached=original.attrs["native_stats"]
    later="2026-10-06T19:25:00Z"
    with monkeypatch.context() as second:
        rebuilt=native_pipeline(second,cached_stats=cached,observation_time=later)
        for row in rebuilt.to_dict("records"):
            assert packet.diagnose(row)["status"]=="COMPLETE"
            for name in packet.FEATURES:
                dependency=original_of(row,name)
                assert dependency["scope"]["available_at"]==pd.Timestamp(SOURCE_TIME).isoformat()
                assert dependency["scope"]["observed_at"]==later
                assert all(x["receipt"]["payload"]["original"]["observed_at"]==SOURCE_TIME for x in dependency["inputs"])
            assert packet.replay(row)["raw_probability"]==row["ml_probability"]
        result=exported(second,tmp_path,rebuilt)
        assert result["package"]["games"]["overall"][0]["research_display"]["availability_reason"]=="AVAILABLE"
    assert original.ml_probability.tolist()==rebuilt.ml_probability.tolist()
    assert original.calibrated_probability.tolist()==rebuilt.calibrated_probability.tolist()


@pytest.mark.parametrize("availability,status",[
    ("2026-10-06T19:10:00Z","COMPLETE"),
    ("2026-10-06T19:30:00Z","REJECTED"),
    ("not-a-clock","REJECTED"),
])
def test_original_member_availability_through_actual_adapter_export_replay(monkeypatch,tmp_path,availability,status):
    data=schedule();data["source_available_at"]=availability
    frame=native_pipeline(monkeypatch,data=data)
    for row in frame.to_dict("records"):
        assert packet.diagnose(row)["status"]==status
        if status=="COMPLETE":
            assert packet.replay(row)["raw_probability"]==row["ml_probability"]
        else:
            assert any("member_availability" in r for r in packet.diagnose(row)["errors"])
            with pytest.raises(ValueError):packet.replay(row)
    result=exported(monkeypatch,tmp_path,frame)
    receipt=research_replay.retain_export(result["frames"],result["package"],result["card"],result["captured"],path=result["db"])
    saved,sources=research_replay.read_export(receipt["export_id"],path=result["db"])
    producer=research_replay.frame_from_payload(next(iter(sources.values()))["original"]["producer"])
    assert producer.ml_probability.tolist()==frame.ml_probability.tolist()
    assert producer.calibrated_probability.tolist()==frame.calibrated_probability.tolist()
    assert all(packet.diagnose(r)["status"]==status for r in producer.to_dict("records"))
    assert all(r["production_bet_amount"]==0 for r in result["card"].wager_contract)


def test_completed_native_member_cannot_postdate_original_observation(monkeypatch):
    frame=native_pipeline(monkeypatch,observation_time="2026-09-02T19:20:00Z")
    for row in frame.to_dict("records"):
        assert packet.diagnose(row)["status"]=="REJECTED"
        assert any("member_after_observation" in r for r in packet.diagnose(row)["errors"])
        with pytest.raises(ValueError):packet.replay(row)


def test_consistently_rehashed_original_member_availability_rejects(monkeypatch):
    def contradiction(o,d):
        retained=o["inputs"][0]["receipt"]["payload"]["original"]
        retained["columns"].append("source_available_at")
        retained["rows"][0]["source_available_at"]={"type":"scalar","value":"2026-10-06T19:30:00Z"}
    row=rehash(native_pipeline(monkeypatch).iloc[0].to_dict(),contradiction)
    assert packet.diagnose(row)["status"]=="REJECTED"
    assert any("member_availability" in r for r in packet.diagnose(row)["errors"])
    with pytest.raises(ValueError):packet.replay(row)
