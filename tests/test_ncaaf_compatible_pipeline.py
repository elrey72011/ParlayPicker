"""Labelled SYNTHETIC new inference; authentic historical packets stay static."""
import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from app_core import ncaaf_compatible_pipeline as caller, ncaaf_pipeline_evidence as adapter
from app_core import ncaaf_compatible_observation as observation, ncaaf_model_compatibility as model
from app_core import ncaaf_history as history, ncaaf_research as research
from app_core import market_probability_model, research_replay
from app_core.research_estimate_trace import encode
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_model_compatibility import synthetic, make_packet, AT
from test_source_contract_pipeline import exported
from test_research_probability_browser import inspect_browser

NOW = datetime.fromisoformat(AT) + timedelta(seconds=2)


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def bind(packet, monkeypatch, *, accept=True):
    o = packet["payload"]["observation"]
    p = o["payload"]
    p["mapping_review"]["binding"] = observation.mapping_binding(p["event"], p["schedule"], p["crosswalk"])
    p["source_review"]["quote_sha256"] = model.digest(p["quote"])
    o["sha256"] = model.digest(p)
    packet["sha256"] = model.digest(packet["payload"])
    monkeypatch.setattr(observation, "ACCEPTED_EVENT_MAPPINGS", {p["mapping_review"]["review_id"]:model.digest(p["mapping_review"])})
    monkeypatch.setattr(observation, "ACCEPTED_SOURCE_REVIEWS", {p["source_review"]["review_id"]:model.digest(p["source_review"])})
    monkeypatch.setattr(adapter, "ACCEPTED_PACKETS", {packet["sha256"]:dict(
        source_review_sha256=model.digest(p["source_review"]), public_derived_output="permitted",
        dependency_source_review=dict(provider="cfbd",endpoints=["games","games/teams"],
            dependency_hashes=[v["sha256"] for v in packet["payload"]["dependency_objects"]],
            permitted_use="prospective_research_features",public_derived_output="permitted",
            rights_document="SYNTHETIC feature permissions",reviewed_at="2026-10-09T11:00:00Z",effective_until="2026-11-01T00:00:00Z"))} if accept else {})
    return packet


def fixture(synthetic, monkeypatch, kind="spread_home", line=-3.5):
    original = make_packet(synthetic, monkeypatch, kind, line)
    p = original["payload"]
    p["event"].update(home_team="Alabama", away_team="Georgia")
    p["schedule"][0].update(homeTeam="Alabama", awayTeam="Georgia", week=7, seasonType="regular")
    p["crosswalk"] = [dict(provider_name=name, schedule_name=name, team_id=tid)
        for name, tid in (("Alabama",1),("Georgia",2))]
    p["quote"].update(event_home_team="Alabama", event_away_team="Georgia",book="draftkings",operator="draftkings",
        period_source="market.period", rules_source="market.settlement_rules",provenance_version="provider-offer-facts-v1")
    p["source_review"]["operator"] = "draftkings"
    p["features"]["values"] = [30.,20.,20.,30.,400.,330.,0.]
    games = [dict(id=100+i,season=2026,week=i+1,seasonType="regular",startDate=f"2026-09-{10+i:02d}T12:00:00Z",
        completed=True,startTimeTBD=False,neutralSite=False,homeId=1,awayId=2,homeTeam="Alabama",awayTeam="Georgia",
        homePoints=30,awayPoints=20) for i in range(3)]
    batches = [dict(request=dict(kind="games",year=2026),retrieved_at=AT,records=history._clean("games",games))]
    for g in games:
        stats = dict(id=g["id"],teams=[dict(teamId=tid,points=points,stats=[dict(category="totalYards",stat=str(yards))])
            for tid, points, yards in ((1,30,400),(2,20,330))])
        batches.append(dict(request=dict(kind="stats",year=2026,week=g["week"],season_type="regular"),retrieved_at=AT,records=[stats]))
    objects = [dict(sha256=hashlib.sha256(model.encode(b)).hexdigest(),bytes_b64=base64.b64encode(model.encode(b)).decode()) for b in batches]
    p["feature_dependencies"] = [dict(game_id=g["id"],team_id=tid,season=2026,start_utc=g["startDate"],
        available_at=AT,kind=k,source_sha256=objects[0 if k=="scoring" else i+1]["sha256"])
        for i,g in enumerate(games) for tid in (1,2) for k in ("scoring","yardage")]
    packet = dict(payload=dict(version=caller.VERSION,evidence_label="SYNTHETIC",observation=original,dependency_objects=objects),sha256="")
    bind(packet,monkeypatch)
    monkeypatch.setattr(adapter, "generated_time", lambda:NOW.isoformat())
    q=p["quote"]
    row = dict(league="NCAAF",League="NCAAF",home_team="Alabama",away_team="Georgia",provider_namespace="odds_api",
        provider_event_id=q["provider_event_id"],matchup_id="ALABAMA|GEORGIA|2026-10-10",game_start_utc=q["event_start_utc"],
        market_type=kind,odds_american=q["price"],odds_source=q["book"],provider_quotes=encode([q]),
        spread_line=line if kind.startswith("spread") else None,total_line=line if kind.startswith("total") else None,
        best_pick=("Alabama" if kind=="spread_home" else "Georgia" if kind=="spread_away" else kind.split("_")[1].title()) + f" {line:+.1f}")
    return packet,row


def actual(monkeypatch, packet):
    from core import streamlit_pipeline as pipeline
    from app_core import odds_api, college_novig, ncaaf_schedule as ns, football_identity_capture
    from test_ncaaf_schedule_coverage import event as schedule_event
    q = caller.view(packet)["original_quote"]
    kind=q["market_type"]
    opposite={"spread_home":"spread_away","spread_away":"spread_home","total_over":"total_under","total_under":"total_over"}[kind]
    offers = [dict(name="Alabama" if k=="spread_home" else "Georgia" if k=="spread_away" else k.split("_")[1].title(),
        point=q["point"] if k==kind or k.startswith("total") else -q["point"],price=q["price"]) for k in (kind,opposite)]
    book = dict(key=q["book"],last_update=q["recorded_at"],markets=[dict(key="spreads" if kind.startswith("spread") else "totals",
        period=q["period"],settlement_rules=q["rules"],outcomes=offers)])
    second=deepcopy(book);second["key"]="betmgm"
    game = dict(id=q["provider_event_id"],home_team="Alabama",away_team="Georgia",sport_key="americanfootball_ncaaf",
        commence_time=q["event_start_utc"],bookmakers=[book,second])
    class Client:
        def __init__(self, **kwargs):pass
        def get_odds(self,sport,date=None):return [deepcopy(game)] if sport=="americanfootball_ncaaf" else []
    inventory=ns.inventory_from_events([("FBS",[schedule_event("9",q["event_start_utc"],"Alabama","Georgia")]),
        ("FCS",[schedule_event("10",q["event_start_utc"],"McNeese Cowboys","Other FCS")])],"2026-10-10","2026-10-10",complete=True,observed_at=NOW.isoformat())
    monkeypatch.setattr(ns,"fetch_schedule",lambda *a:deepcopy(inventory))
    monkeypatch.setattr(odds_api,"TheOddsAPIClient",Client)
    monkeypatch.setattr(odds_api,"filter_games_today_only",lambda games:games)
    monkeypatch.setattr(college_novig,"recover_college_novig",lambda games,key:games)
    monkeypatch.setattr("app_core.espn_ncaaf_odds.fetch_espn_ncaaf_fcs_odds",lambda day:[])
    monkeypatch.setattr(pipeline,"_get_odds_api_key",lambda:"SYNTHETIC_UNUSED_SECRET_CANARY")
    monkeypatch.setattr(pipeline,"load_base_data",lambda:pd.DataFrame())
    monkeypatch.setattr(football_identity_capture.requests,"get",lambda *a,**k:SimpleNamespace(raise_for_status=lambda:None,json=lambda:{"events":[]}))
    monkeypatch.setattr("app_core.feature_processing.enrich_with_model_features",lambda f,*a:f.assign(League="NCAAF",ml_feature_eligible=False,stats_resolution_status="unavailable"))
    monkeypatch.setattr("app_core.external_data_fetcher.enrich_with_external_data",lambda f:f)
    monkeypatch.setattr(pipeline,"ML_AVAILABLE",True)
    monkeypatch.setattr(pipeline,"PredictionEngine",object)
    monkeypatch.setattr(pipeline,"get_cached_prediction_engine",lambda:SimpleNamespace(use_fallback=False,predict_batch=lambda f:[.99]*len(f)))
    monkeypatch.setattr("app_core.candidate_chronology.now_utc",lambda:pd.Timestamp(NOW))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats",lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration",lambda:None)
    with adapter.selected([packet]):
        analysis,_,diagnostics=pipeline.run_analysis_pipeline(sports=["NCAAF"],use_ml=True,max_rows=20,schedule_start="2026-10-10",schedule_end="2026-10-10")
    return analysis,diagnostics


@pytest.mark.parametrize("kind,line",[("spread_home",-3.5),("spread_away",3.5),("total_over",50.5),("total_under",50.5)])
def test_actual_new_caller_capture_export_reader_browser(synthetic,monkeypatch,tmp_path,kind,line):
    packet,_=fixture(synthetic,monkeypatch,kind,line)
    original=deepcopy(packet)
    analysis,diagnostics=actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq(kind)].iloc[0].to_dict()
    assert row["ml_inference_status"]=="success",row["ml_unavailable_reason"]
    assert adapter.diagnose(row)==dict(status="COMPLETE",reason="AVAILABLE")
    checked=observation.read_observation(packet["payload"]["observation"])
    assert checked["probability"] is None and not checked["feature_derivation_verified"] and not checked["dependency_objects_verified"]
    feature=dict(zip(research.FEATURES,checked["ordered_features"]))
    target="total" if kind.startswith("total") else "margin"
    expected=research.probabilities(float(research.centers(checked["fit"],[feature],target)[0]),checked["fit"]["sigma"],-line if kind=="spread_home" else line,total=target=="total")
    assert row["ml_probability"]==expected["over" if kind in {"spread_home","total_over"} else "under"]
    assert row["ml_probability"] != .99 and len(diagnostics["ncaaf_schedule"]["events"])==2
    import test_source_contract_pipeline as ef
    monkeypatch.setattr(ef,"NOW",NOW);monkeypatch.setattr(ef,"CAPTURE",(NOW+timedelta(seconds=1)).isoformat())
    result=exported(monkeypatch,tmp_path/"export",analysis)
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package,validate_package
    from scripts.publish_board import render,assets_from_html
    result["frames"]=[per_game_board(result["card"],result["captured"],family=f,novig_only=True,college_fallback=True) for f in ("overall","sides","totals")]
    package=json.loads(assets_from_html(render(build_package(*result["frames"]))) ["board-data.json"])
    validate_package(package)
    family="sides" if kind.startswith("spread") else "totals"
    shown=package["games"][family][0]["research_display"]
    assert shown["availability_reason"]=="AVAILABLE",shown
    assert shown["probability"]==row["ml_probability"] and shown["ev"] is None and shown["edge"] is None
    assert package["games"]["overall"][0]["research_display"]["probability"]==row["ml_probability"]
    assert "SYNTHETIC" in shown["basis"] and "uncalibrated" in shown["basis"]
    receipt=research_replay.retain_export(result["frames"],package,result["card"],result["captured"],path=result["db"])
    _,sources=research_replay.read_export(receipt["export_id"],path=result["db"])
    producer=research_replay.frame_from_payload(next(iter(sources.values()))["original"]["producer"])
    retained=producer.loc[producer.market_type.eq(kind)].iloc[0]
    p=json.loads(retained.ml_estimate_metadata)["ncaaf_inputs"]["payload"]
    assert p["original_packet"]==packet and p["inference_time"]==NOW.isoformat()
    assert p["computation"]["payload"]["feature_derivation_verified"] and p["computation"]["payload"]["dependency_objects_verified"]
    assert p["original_blend"]["probability"]["value"]==retained.calibrated_probability and p["ui_refresh"] is None
    assert p["board_schedule"]["payload"]["identity_events"][0]["id"]=="9"
    assert packet==original and all(c["production_bet_amount"]==0 for c in result["card"].wager_contract)
    assert not result["captured"].production_eligible.fillna(False).any()
    public=json.dumps(package)
    assert all(k not in public for k in ("dependency_objects","bytes_b64","original_record_b64","ncaaf_inputs","SYNTHETIC_UNUSED_SECRET_CANARY"))
    browser=inspect_browser(package,tmp_path/"browser",NOW)
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert browser["initial"]["shown"][0]["probability"]==shown["probability"]


def mutate_batch(packet,index,change,*,rehash=True):
    obj=packet["payload"]["dependency_objects"][index]
    batch=json.loads(base64.b64decode(obj["bytes_b64"]))
    old=obj["sha256"]
    change(batch)
    raw=model.encode(batch);obj["bytes_b64"]=base64.b64encode(raw).decode()
    if rehash:
        obj["sha256"]=hashlib.sha256(raw).hexdigest()
        for r in packet["payload"]["observation"]["payload"]["feature_dependencies"]:
            if r["source_sha256"]==old:r["source_sha256"]=obj["sha256"]


@pytest.mark.parametrize("change,reason",[
    ("missing","NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT"),("corrupt","NCAAF_COMPAT_DEPENDENCY_BYTES_CORRUPT"),
    ("unrelated","NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT"),("wrong_name","NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT"),
    ("feature","NCAAF_COMPAT_FEATURE_DERIVATION_CONFLICT"),("order","NCAAF_COMPAT_FEATURE_CONFLICT"),
    ("future","NCAAF_COMPAT_FEATURE_CLOCK_MISSING_OR_FUTURE"),("history","NCAAF_COMPAT_MINIMUM_HISTORY_MISSING"),
    ("lag","NCAAF_COMPAT_DEPENDENCY_CLOCK_CONFLICT"),("mapping","NCAAF_COMPAT_EVENT_REVIEW_NOT_ACCEPTED"),
    ("review","NCAAF_SOURCE_REVIEW_NOT_ACCEPTED"),("rights","NCAAF_PUBLIC_DERIVED_RIGHTS_UNAVAILABLE"),
    ("neutral","NCAAF_COMPAT_EVENT_FACT_CONFLICT"),("orientation","NCAAF_COMPAT_EVENT_AMBIGUOUS_OR_ORIENTATION_CONFLICT"),
    ("ambiguous","NCAAF_COMPAT_EVENT_AMBIGUOUS_OR_ORIENTATION_CONFLICT"),("collision","NCAAF_COMPAT_ALIAS_COLLISION"),
    ("integer","NCAAF_INTEGER_PUSH_MODEL_UNVALIDATED"),("novig","NCAAF_COMPAT_SOURCE_REVIEW_MISSING_OR_CONFLICT"),
    ("recorded","NCAAF_COMPAT_RECORDED_INFERENCE_FORBIDDEN"),("stale","NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT"),
    ("future_quote","NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT"),("missing_quote","NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT"),
    ("started","NCAAF_COMPAT_NEW_INFERENCE_CLOCK_CONFLICT"),
    ("bool_team","NCAAF_COMPAT_DEPENDENCY_IDENTITY_CONFLICT"),
    ("missing_model","NCAAF_COMPAT_PACKET_SCHEMA"),("corrupt_model","NCAAF_COMPAT_RECORD_SCHEMA"),
    ("altered_model","NCAAF_COMPAT_RECORD_HASH"),
])
def test_reject_before_inference(synthetic,monkeypatch,change,reason):
    packet,row=fixture(synthetic,monkeypatch)
    p=packet["payload"]["observation"]["payload"]
    if change=="missing":packet["payload"]["dependency_objects"].pop()
    elif change in {"corrupt","unrelated","wrong_name"}:
        mutate_batch(packet,0,lambda b:b["records"][0].update(homeId=999) if change!="wrong_name" else b["records"][0].update(homeTeam="Unrelated team"),rehash=change!="corrupt")
    elif change=="bool_team":p["feature_dependencies"][0]["team_id"]=True
    elif change=="missing_model":p["model"]=None
    elif change=="corrupt_model":p["model"]["original_record_b64"]="bad"
    elif change=="altered_model":p["model"]["original_record_b64"]=base64.b64encode(base64.b64decode(p["model"]["original_record_b64"])+b" ").decode()
    elif change=="feature":p["features"]["values"][0]+=1
    elif change=="order":p["features"]["order"].reverse()
    elif change=="future":p["features"]["available_at"]="2027-01-01T00:00:00Z"
    elif change=="history":p["feature_dependencies"]=p["feature_dependencies"][:8]
    elif change=="lag":p["feature_dependencies"][0]["start_utc"]="2026-10-04T12:00:00Z"
    elif change=="neutral":p["event"]["neutral_site"]=True
    elif change=="orientation":p["schedule"][0].update(homeId=2,awayId=1)
    elif change=="ambiguous":p["schedule"].append(deepcopy(p["schedule"][0]))
    elif change=="collision":p["crosswalk"].append(dict(p["crosswalk"][0],team_id=999))
    elif change=="integer":p["quote"]["point"]=row["spread_line"]=-3;row["provider_quotes"]=encode([p["quote"]])
    elif change=="novig":p["quote"].update(book="novig",operator="novig");p["source_review"]["operator"]="novig";row.update(odds_source="novig",provider_quotes=encode([p["quote"]]))
    elif change=="recorded":p["original_inference_time"]=AT
    elif change in {"stale","started"}:monkeypatch.setattr(adapter,"generated_time",lambda:(NOW+timedelta(days=2 if change=="started" else 1)).isoformat())
    elif change in {"future_quote","missing_quote"}:
        p["quote"]["recorded_at"]=None if change=="missing_quote" else (NOW+timedelta(seconds=1)).isoformat();row["provider_quotes"]=encode([p["quote"]])
    bind(packet,monkeypatch)
    if change=="mapping":monkeypatch.setattr(observation,"ACCEPTED_EVENT_MAPPINGS",{})
    elif change=="review":monkeypatch.setattr(observation,"ACCEPTED_SOURCE_REVIEWS",{})
    elif change=="rights":adapter.ACCEPTED_PACKETS[packet["sha256"]]["public_derived_output"]="UNKNOWN"
    monkeypatch.setattr(research,"centers",Mock(side_effect=AssertionError("Rejected inputs must not infer")))
    with adapter.selected([packet]):out=adapter.predict(row)
    assert out["ml_unavailable_reason"]==reason and out["ml_inference_status"]=="unavailable"
    research.centers.assert_not_called()
    assert not __import__('math').isfinite(out["ml_probability"])
    saved=json.loads(out["ml_estimate_metadata"])["ncaaf_inputs"]["payload"]
    assert saved["original_packet"]==packet and saved["live_stake"]==0 and not saved["wagering_authority"]


def test_static_readers_do_not_reconstruct_recorded_features_or_probabilities(synthetic,monkeypatch):
    packet,row=fixture(synthetic,monkeypatch)
    with adapter.selected([packet]):out=adapter.predict(row)
    source=dict(row,**out)
    monkeypatch.setattr(history,"build_dataset",Mock(side_effect=AssertionError("Historical feature reconstruction forbidden")))
    monkeypatch.setattr(research,"centers",Mock(side_effect=AssertionError("Historical inference forbidden")))
    monkeypatch.setattr(research,"probabilities",Mock(side_effect=AssertionError("Historical probability rebuilding forbidden")))
    assert adapter.diagnose(source)==dict(status="COMPLETE",reason="AVAILABLE")
    from app_core.research_display import from_export
    # Static diagnosis is the same reader used by display; no numeric calls.
    for field in ("ml_probability","provider_event_id","home_team","odds_american"):
        changed=dict(source,**{field:.99 if field=="ml_probability" else "foreign"})
        assert adapter.diagnose(changed)["status"]=="REJECTED"
    item=json.loads(source["ml_estimate_metadata"])
    item["ncaaf_inputs"]["payload"]["computation"]["payload"]["raw_probability"] = .99
    item["ncaaf_inputs"]["payload"]["computation"]["sha256"]=model.digest(item["ncaaf_inputs"]["payload"]["computation"]["payload"])
    item["ncaaf_inputs"]["sha256"]=model.digest(item["ncaaf_inputs"]["payload"])
    assert adapter.diagnose(dict(source,ml_estimate_metadata=encode(item)))["reason"]=="NCAAF_PROBABILITY_BINDING_CONFLICT"


@pytest.mark.parametrize("change,reason",[("duplicate","NCAAF_PACKET_AMBIGUOUS"),("quote","NCAAF_EXACT_OFFER_NOT_SELECTED"),
    ("none","NCAAF_EXACT_OFFER_NOT_SELECTED"),("failure","NCAAF_INFERENCE_FAILED"),
    ("unapproved","NCAAF_SOURCE_REVIEW_NOT_ACCEPTED")])
def test_selection_and_failed_inference_never_fall_back(synthetic,monkeypatch,change,reason):
    packet,row=fixture(synthetic,monkeypatch)
    packets=[packet]*2 if change=="duplicate" else [] if change=="none" else [packet]
    if change=="quote":row["odds_american"]=-115
    if change=="unapproved":monkeypatch.setattr(adapter,"ACCEPTED_PACKETS",{})
    monkeypatch.setattr(research,"centers",Mock(side_effect=RuntimeError("private detail; no fallback")))
    with adapter.selected(packets):out=adapter.predict(dict(row,calibrated_probability=.99,expected_value=2.))
    assert out["ml_inference_status"]=="unavailable" and out["ml_unavailable_reason"]==reason
    assert not __import__('math').isfinite(out["ml_probability"]) and "private detail" not in out["ml_estimate_metadata"]
    assert research.centers.call_count == (1 if change=="failure" else 0)


def test_owner_staging_is_static_and_production_catalogs_do_not_gain_entries(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    packet["payload"]["evidence_label"]=packet["payload"]["observation"]["payload"]["evidence_label"]="RETAINED"
    bind(packet,monkeypatch)
    catalogs=deepcopy((observation.ACCEPTED_EVENT_MAPPINGS,observation.ACCEPTED_SOURCE_REVIEWS,adapter.ACCEPTED_PACKETS))
    monkeypatch.setattr(caller,"infer",Mock(side_effect=AssertionError("Selection cannot run analysis")))
    monkeypatch.setattr(history,"build_dataset",Mock(side_effect=AssertionError("Selection cannot derive")))
    assert adapter.load(encode(packet).encode(),owner_upload=True)==packet
    with adapter.selected([packet]):assert adapter.selection_requested()
    assert not adapter.selection_requested()
    assert catalogs==(observation.ACCEPTED_EVENT_MAPPINGS,observation.ACCEPTED_SOURCE_REVIEWS,adapter.ACCEPTED_PACKETS)
    assert adapter.ACCEPTED_PACKETS is not observation.ACCEPTED_SOURCE_REVIEWS


@pytest.mark.parametrize("field,value,reason",[("dependency_hashes",[],"NCAAF_COMPAT_DEPENDENCY_SOURCE_REVIEW_MISSING_OR_CONFLICT"),
    ("rights_document","","NCAAF_COMPAT_DEPENDENCY_SOURCE_REVIEW_MISSING_OR_CONFLICT"),
    ("provider","unrelated","NCAAF_COMPAT_DEPENDENCY_SOURCE_REVIEW_MISSING_OR_CONFLICT"),
    ("permitted_use","unknown","NCAAF_COMPAT_DEPENDENCY_RIGHTS_UNAVAILABLE"),
    ("public_derived_output","unknown","NCAAF_COMPAT_DEPENDENCY_RIGHTS_UNAVAILABLE"),
    ("reviewed_at","2027-01-01T00:00:00Z","NCAAF_COMPAT_SOURCE_REVIEW_CLOCK_CONFLICT")])
def test_quote_rights_do_not_supply_feature_provider_permissions(synthetic,monkeypatch,field,value,reason):
    packet,row=fixture(synthetic,monkeypatch)
    adapter.ACCEPTED_PACKETS[packet["sha256"]]["dependency_source_review"][field]=value
    monkeypatch.setattr(research,"centers",Mock(side_effect=AssertionError("Missing source facts must not infer")))
    with adapter.selected([packet]):out=adapter.predict(row)
    assert out["ml_unavailable_reason"]==reason and out["ml_inference_status"]=="unavailable"
    research.centers.assert_not_called()


def test_rejected_actual_pipeline_keeps_entire_independent_slate_and_precise_blocker(synthetic,monkeypatch):
    packet,_=fixture(synthetic,monkeypatch)
    packet["payload"]["observation"]["payload"]["features"]["values"][0]+=1
    bind(packet,monkeypatch)
    analysis,diagnostics=actual(monkeypatch,packet)
    from app_core.slate_coverage import build_coverage, native_ncaaf
    run="20261009T120002.000000Z"
    candidates=analysis.assign(export_run_id=run).to_dict("records")
    inventory=native_ncaaf(diagnostics["ncaaf_schedule"],"2026-10-10")
    coverage=build_coverage([inventory],selected_date="2026-10-10",as_of=NOW.isoformat(),run_id=run,candidates=candidates,
        provider_health={"sports":{"americanfootball_ncaaf":{"outcome":"SUCCESS","processing":"SUCCESS"}}})
    assert coverage["counts"]["scheduled_events"]==len(coverage["decisions"])==2
    assert all(d["coverage_decision_state"]=="UNVERIFIED" for d in coverage["decisions"])
    first=next(d for d in coverage["decisions"] if "Alabama" in d["home_team"])
    assert "NCAAF_COMPAT_FEATURE_DERIVATION_CONFLICT" in first["blocker_codes"]
    candidate=first["market_results"][0]["candidates"][0]
    assert candidate["original_model_blocker"]=="NCAAF_COMPAT_FEATURE_DERIVATION_CONFLICT"


@pytest.mark.parametrize("change",["bytes","receipt","refresh","runtime"])
def test_saved_integrity_and_runtime_conflicts_remain_unavailable(synthetic,monkeypatch,change):
    packet,row=fixture(synthetic,monkeypatch)
    with adapter.selected([packet]):out=adapter.predict(row)
    source=dict(row,**out);item=json.loads(source["ml_estimate_metadata"])
    p=item["ncaaf_inputs"]["payload"]
    if change=="bytes":p["original_packet"]["payload"]["dependency_objects"][0]["bytes_b64"]="bad"
    elif change=="receipt":p["computation"]["payload"]["feature_derivation_verified"]=False
    elif change=="refresh":p["inference_time"]=(NOW+timedelta(seconds=1)).isoformat()
    else:p["consumed_reader"]["compatible_caller_sha256"]="0"*64
    item["ncaaf_inputs"]["sha256"]=model.digest(p)
    assert adapter.diagnose(dict(source,ml_estimate_metadata=encode(item)))["status"]=="REJECTED"
