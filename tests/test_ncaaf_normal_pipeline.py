"""SYNTHETIC native capture -> normal caller -> private export -> actual browser."""
from copy import deepcopy
from datetime import timedelta
import json
from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
import pytest

from app_core import ncaaf_pipeline_evidence as adapter, ncaaf_research_contract as contract
from app_core import market_probability_model, research_replay
from app_core.research_estimate_trace import encode
from core import streamlit_pipeline as pipeline
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_research_contract import captured, selection, review
from test_ncaaf_prospective import NOW
from test_source_contract_pipeline import exported
from test_research_probability_browser import inspect_browser

AT = NOW + timedelta(seconds=2)


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def fixture_packet(tmp_path, monkeypatch, kind="spread_home", *, half=True, accept=True):
    records, cid, _ = captured(tmp_path, monkeypatch, half=half)
    chosen = selection(records, cid, kind)
    packet = contract.export_observation(records, source_review=review(records, chosen),
        evidence_label="SYNTHETIC", **chosen)
    monkeypatch.setattr(adapter, "generated_time", lambda: AT.isoformat())
    catalog = {packet["sha256"]: dict(source_review_sha256=adapter.digest(packet["payload"]["source_review"]),
                                   public_derived_output="permitted")} if accept else {}
    monkeypatch.setattr(adapter, "ACCEPTED_PACKETS", catalog)
    q = packet["payload"]["original_quote"]
    row = dict(league="NCAAF", League="NCAAF", home_team=q["event_home_team"], away_team=q["event_away_team"],
        provider_namespace=q["provider_namespace"], provider_event_id=q["provider_event_id"],
        matchup_id="ALABAMA|GEORGIA|2026-10-30", game_start_utc=q["event_start_utc"],
        market_type=kind, odds_american=q["price"], odds_source=q["book"],
        spread_line=q["point"] if kind.startswith("spread") else None,
        total_line=q["point"] if kind.startswith("total") else None, provider_quotes=json.dumps([q]),
        best_pick=(q["event_home_team"] if kind=="spread_home" else q["event_away_team"] if kind=="spread_away" else kind.split("_")[1].title()) + " " + (f"{q['point']:+.1f}" if kind.startswith("spread") else str(q["point"])))
    return packet, row


def actual(monkeypatch, packet):
    from app_core import odds_api, college_novig, ncaaf_schedule as ns, football_identity_capture
    from test_ncaaf_schedule_coverage import event as schedule_event
    q = packet["payload"]["original_quote"]
    p=packet["payload"]
    records=p["records"]
    capture=next(r for r in records if r["id"]==p["selection"]["capture_id"])
    event=next(e for e in capture["data"]["events"] if e["event_id"]==p["selection"]["event_id"])
    offers=[c for c in event["models"][p["selection"]["model_name"]]["candidates"]
        if c["book"]==q["book"] and c["market_type"].split("_")[0]==q["market_type"].split("_")[0]]
    packets=[]
    for c in offers:
        chosen=selection(records,p["selection"]["capture_id"],c["market_type"])
        item=contract.export_observation(records,source_review=review(records,chosen),evidence_label="SYNTHETIC",**chosen)
        packets.append(item)
        adapter.ACCEPTED_PACKETS[item["sha256"]]=dict(source_review_sha256=adapter.digest(item["payload"]["source_review"]),public_derived_output="permitted")
    game = dict(id=q["provider_event_id"], home_team=q["event_home_team"], away_team=q["event_away_team"],
        sport_key="americanfootball_ncaaf", commence_time=q["event_start_utc"],
        bookmakers=[dict(key=q["book"],last_update=q["recorded_at"],markets=[
            dict(key="spreads" if q["market_type"].startswith("spread") else "totals",period=q["period"],
                settlement_rules=q["rules"],outcomes=[dict(name=(q["event_home_team"] if c["market_type"]=="spread_home" else
                    q["event_away_team"] if c["market_type"]=="spread_away" else c["market_type"].split("_")[1].title()),
                    point=c["point"],price=c["price"]) for c in offers])])])
    # Existing football board guard requires an independently corroborated
    # signed pair when moneyline orientation is absent. Synthetic only.
    second=deepcopy(game["bookmakers"][0]);second["key"]="betmgm"
    game["bookmakers"].append(second)
    class Client:
        def __init__(self, **kwargs): pass
        def get_odds(self, sport, date=None):
            return [deepcopy(game)] if sport=="americanfootball_ncaaf" else []
    inventory=ns.inventory_from_events([("FBS",[schedule_event("9",q["event_start_utc"],q["event_home_team"],q["event_away_team"])]),
        ("FCS",[schedule_event("10",q["event_start_utc"],"McNeese Cowboys","Other FCS")])],
        "2026-10-30","2026-10-30",complete=True)
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
    monkeypatch.setattr("app_core.candidate_chronology.now_utc",lambda:pd.Timestamp(AT))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats",lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration",lambda:None)
    with adapter.selected(packets):
        analysis, _, diagnostics = pipeline.run_analysis_pipeline(sports=["NCAAF"],use_ml=True,max_rows=20,
            schedule_start="2026-10-30",schedule_end="2026-10-30")
    assert len(analysis)>=1
    return analysis, diagnostics


@pytest.mark.parametrize("kind",sorted(contract.KINDS))
def test_native_half_point_normal_pipeline_capture_export_and_browser(tmp_path,monkeypatch,kind):
    packet, _ = fixture_packet(tmp_path,monkeypatch,kind)
    original=deepcopy(packet)
    analysis, diagnostics=actual(monkeypatch,packet)
    chosen=analysis.loc[analysis.market_type.eq(kind)]
    assert len(chosen)==1
    row=chosen.iloc[0].to_dict()
    matches=adapter.producer._matches(row)
    expected=packet["payload"]["original_quote"]
    differences={k:(matches[0].get(k),expected.get(k)) for k in ("book","market_type","point","price","recorded_at","provider_event_id","provider_namespace","event_home_team","event_away_team","event_start_utc","period","period_source","rules","rules_source") if matches and matches[0].get(k)!=expected.get(k)}
    assert row["ml_inference_status"]=="success",(row.get("ml_unavailable_reason"),len(matches),differences,{k:row.get(k) for k in ("provider_event_id","odds_american","odds_source","spread_line","total_line")})
    assert row["ml_probability"]==pytest.approx(packet["payload"]["probabilities"]["win"])
    assert row["ml_probability"]!=.99  # Parallel home-win output was retired.
    origin=json.loads(row["ml_estimate_metadata"]);origin.pop("ncaaf_inputs")
    from app_core.research_estimate_trace import origin_rejection
    assert adapter.diagnose(row)==dict(status="COMPLETE",reason="AVAILABLE"),(origin_rejection(dict(row,ml_estimate_metadata=encode(origin))),adapter.producer.diagnose(row,origin),{k:row.get(k) for k in ("matchup_id","historical_matchup_id","schedule_event_id","schedule_match_status","schedule_inventory_key","opposing_odds_source","opposing_odds_american")})
    assert len(diagnostics["ncaaf_schedule"]["events"])==2
    import test_source_contract_pipeline as export_fixture
    monkeypatch.setattr(export_fixture,"NOW",AT)
    monkeypatch.setattr(export_fixture,"CAPTURE",(AT+timedelta(seconds=1)).isoformat())
    result=exported(monkeypatch,tmp_path/"export",analysis)
    from app_core.per_game_boards import per_game_board
    from app_core.public_board import build_package,validate_package
    from scripts.publish_board import render,assets_from_html
    # Same existing NCAAF supported-book fallback enabled by the normal UI.
    result["frames"]=[per_game_board(result["card"],result["captured"],family=f,
        novig_only=True,college_fallback=True) for f in ("overall","sides","totals")]
    result["html"]=render(build_package(*result["frames"]))
    result["package"]=json.loads(assets_from_html(result["html"])["board-data.json"])
    validate_package(result["package"])
    family="sides" if kind.startswith("spread") else "totals"
    shown=result["package"]["games"][family][0]["research_display"]
    assert shown["availability_reason"]=="AVAILABLE",[(name,adapter.diagnose(f.loc[f.market_type.eq(kind)].iloc[0].to_dict()),{k:f.loc[f.market_type.eq(kind)].iloc[0].get(k) for k in ("schedule_event_id","schedule_inventory_key","schedule_match_status","historical_matchup_id")}) for name,f in (("authority",result["authority"]),("capture",result["captured"]))]+[shown]
    shown_row=analysis.loc[analysis.market_type.eq(shown["identity"]["market"])].iloc[0]
    assert shown["source_field"]=="ml_probability" and shown["probability"]==shown_row.ml_probability
    assert shown["push_probability"]==0 and shown["ev"] is None
    assert "SYNTHETIC" in shown["basis"] and "uncalibrated" in shown["basis"]
    receipt=research_replay.retain_export(result["frames"],result["package"],result["card"],result["captured"],path=result["db"])
    _, sources=research_replay.read_export(receipt["export_id"],path=result["db"])
    producers=research_replay.frame_from_payload(next(iter(sources.values()))["original"]["producer"])
    retained=producers.loc[producers.market_type.eq(kind)].iloc[0]
    p=json.loads(retained.ml_estimate_metadata)["ncaaf_inputs"]["payload"]
    assert p["original_packet"]==packet and p["inference_time"]==AT.isoformat()
    assert p["original_packet"]["payload"]["original_inference_time"] is None
    schedule=p["board_schedule"]
    assert schedule["scope"]=="selected_event_only" and schedule["sha256"]==adapter.digest(schedule["payload"])
    assert schedule["payload"]["identity_events"][0]["id"]=="9"
    assert schedule["payload"]["observed_at"]==diagnostics["ncaaf_schedule"]["observed_at"]
    assert p["original_blend"]["probability"]["value"]==retained.calibrated_probability
    assert packet==original
    public=json.dumps(result["package"])
    assert all(secret not in public for secret in ("ncaaf_inputs","records","dependency_receipts","SYNTHETIC_UNUSED_SECRET_CANARY"))
    assert all(c["production_bet_amount"]==0 for c in result["card"].wager_contract)
    browser=inspect_browser(result["package"],tmp_path/"browser",AT)
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert browser["initial"]["shown"][0]["probability"]==shown["probability"]


@pytest.mark.parametrize("change,reason",[
    ("integer","NCAAF_INTEGER_PUSH_MODEL_UNVALIDATED"),
    ("review","NCAAF_SOURCE_REVIEW_NOT_ACCEPTED"),
    ("rights","NCAAF_PUBLIC_DERIVED_RIGHTS_UNAVAILABLE"),
    ("stale","NCAAF_QUOTE_INFERENCE_START_CLOCK_CONFLICT"),
    ("quote","NCAAF_EXACT_OFFER_NOT_SELECTED"),
    ("other_league","NCAAF_EXACT_OFFER_NOT_SELECTED"),
    ("duplicate","NCAAF_PACKET_AMBIGUOUS"),
])
def test_fail_closed_before_inference(tmp_path,monkeypatch,change,reason):
    packet,row=fixture_packet(tmp_path,monkeypatch,half=change!="integer",accept=change!="review")
    packets=[packet]
    if change=="rights":adapter.ACCEPTED_PACKETS[packet["sha256"]]["public_derived_output"]="unknown"
    elif change=="stale":monkeypatch.setattr(adapter,"generated_time",lambda:(AT+timedelta(days=40)).isoformat())
    elif change=="quote":row["odds_american"]=-115
    elif change=="other_league":row["league"]=row["League"]="NFL"
    elif change=="duplicate":packets.append(packet)
    monkeypatch.setattr(research:=contract.research,"centers",Mock(side_effect=AssertionError("No numerical inference permitted")))
    with adapter.selected(packets):out=adapter.predict(row)
    assert out["ml_inference_status"]=="unavailable" and out["ml_unavailable_reason"]==reason
    research.centers.assert_not_called()
    assert not __import__('math').isfinite(out["ml_probability"])
    assert "ncaaf_inputs" in json.loads(out["ml_estimate_metadata"])


def test_corrupt_owner_upload_and_resource_bounds(tmp_path,monkeypatch):
    packet,_=fixture_packet(tmp_path,monkeypatch)
    with pytest.raises(ValueError):adapter.load(encode(packet).encode(),owner_upload=True)
    broken=deepcopy(packet);broken["payload"]["probabilities"]["win"]=.99
    with pytest.raises(ValueError,match="NCAAF_PACKET_INTEGRITY"):adapter.load(encode(broken).encode())
    with pytest.raises(ValueError):
        with adapter.selected([packet]*5):pass
    assert not adapter.selection_requested()


def test_static_diagnosis_does_not_rebuild_features_or_inference(tmp_path,monkeypatch):
    packet,row=fixture_packet(tmp_path,monkeypatch)
    with adapter.selected([packet]):out=adapter.predict(row)
    source=dict(row,**out)
    monkeypatch.setattr(contract.history,"build_dataset",Mock(side_effect=AssertionError("Historical reconstruction forbidden")))
    monkeypatch.setattr(contract.research,"centers",Mock(side_effect=AssertionError("Historical inference forbidden")))
    assert adapter.diagnose(source)==dict(status="COMPLETE",reason="AVAILABLE")
    changed=deepcopy(source);item=json.loads(changed["ml_estimate_metadata"])
    item["ncaaf_inputs"]["payload"]["raw_probability"]["value"] = .99
    item["ncaaf_inputs"]["sha256"]=adapter.digest(item["ncaaf_inputs"]["payload"])
    changed["ml_estimate_metadata"]=encode(item)
    assert adapter.diagnose(changed)["reason"]=="NCAAF_PROBABILITY_BINDING_CONFLICT"


def test_default_ncaaf_stays_unavailable_without_explicit_packet():
    out=market_probability_model.predict_market_probabilities(pd.DataFrame([dict(league="NCAAF",market_type="spread_home",spread_line=-10.5)]))
    assert out.iloc[0].ml_inference_status=="unavailable"
    assert "No market-specific model configured for NCAAF" in out.iloc[0].ml_unavailable_reason


@pytest.mark.parametrize("change",["feature","future","regulation","runtime"])
def test_rehashed_native_fact_changes_stay_rejected(tmp_path,monkeypatch,change):
    packet,row=fixture_packet(tmp_path,monkeypatch)
    p=packet["payload"]
    if change=="feature":p["ordered_features"][0] += 1
    elif change=="future":monkeypatch.setattr(adapter,"generated_time",lambda:(NOW-timedelta(seconds=1)).isoformat())
    elif change=="regulation":p["target_contract"]["period"]="regulation"
    elif change=="runtime":p["reader_binding"]["prospective_runtime"]="0"*64
    packet["sha256"]=adapter.digest(p)
    adapter.ACCEPTED_PACKETS[packet["sha256"]]=dict(source_review_sha256=adapter.digest(p["source_review"]),public_derived_output="permitted")
    with adapter.selected([packet]):out=adapter.predict(row)
    assert out["ml_inference_status"]=="unavailable" and not __import__('math').isfinite(out["ml_probability"])


def test_failed_native_inference_stays_precise(tmp_path,monkeypatch):
    packet,row=fixture_packet(tmp_path,monkeypatch)
    monkeypatch.setattr(contract.research,"centers",Mock(side_effect=RuntimeError("private failure details must not be public")))
    with adapter.selected([packet]):out=adapter.predict(row)
    assert out["ml_inference_status"]=="unavailable" and out["ml_unavailable_reason"]=="NCAAF_INFERENCE_FAILED"
    assert "private failure details" not in out["ml_estimate_metadata"]


@pytest.mark.parametrize("field,value",[("home_team","Georgia"),("provider_event_id","other"),
    ("matchup_id","espn:college-football:10"),("schedule_match_status","AMBIGUOUS"),
    ("schedule_inventory_key","different"),("odds_american",-115)])
def test_captured_schedule_and_offer_conflicts_reject(tmp_path,monkeypatch,field,value):
    packet,_=fixture_packet(tmp_path,monkeypatch)
    analysis,_=actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq("spread_home")].iloc[0].to_dict()
    assert adapter.diagnose(row)["status"]=="COMPLETE"
    row[field]=value
    assert adapter.diagnose(row)["reason"]==("NCAAF_SCHEDULE_RECEIPT_CONFLICT" if field=="matchup_id" else "NCAAF_EVENT_OFFER_CONFLICT")


def test_schedule_dependency_clock_and_integrity_reject(tmp_path,monkeypatch):
    packet,_=fixture_packet(tmp_path,monkeypatch)
    analysis,_=actual(monkeypatch,packet)
    row=analysis.loc[analysis.market_type.eq("spread_home")].iloc[0].to_dict()
    for rehash in (False,True):
        item=json.loads(row["ml_estimate_metadata"])
        schedule=item["ncaaf_inputs"]["payload"]["board_schedule"]
        schedule["payload"]["observed_at"]=(AT+timedelta(seconds=1)).isoformat()
        if rehash:schedule["sha256"]=adapter.digest(schedule["payload"])
        item["ncaaf_inputs"]["sha256"]=adapter.digest(item["ncaaf_inputs"]["payload"])
        assert adapter.diagnose(dict(row,ml_estimate_metadata=encode(item)))["reason"]=="NCAAF_SCHEDULE_RECEIPT_CONFLICT"


def owner_app():
    from app.ui.publish_panel import render_publish_panel
    render_publish_panel(None,None,lazy_history=True)


def test_owner_gate_and_selection_does_not_run_analysis(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from app.ui import ncaaf_pipeline_research,source_evidence_panel,activation_panel,public_results
    monkeypatch.setenv("PARLAYPICKER_PUBLISH_TOKEN","synthetic-owner-token")
    for module,name in ((source_evidence_panel,"render"),(activation_panel,"render"),(public_results,"render_history")):
        monkeypatch.setattr(module,name,lambda *a,**k:[])
    calls=[]
    monkeypatch.setattr(ncaaf_pipeline_research,"render",lambda:calls.append("owner"))
    at=AppTest.from_function(owner_app).run()
    at.text_input(key="publication_token").set_value("wrong").run()
    assert not at.exception and not calls
    at.text_input(key="publication_token").set_value("synthetic-owner-token").run()
    assert not at.exception and calls==["owner"]
    import streamlit_app
    state=dict(last_processed_run_counter=1)
    assert not streamlit_app._should_run_pipeline(state,1,dict(ncaaf_research_packets=[{"sha256":"new-selection"}]))
