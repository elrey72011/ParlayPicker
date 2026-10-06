"""Synthetic training and native receipts only; actual pipeline/capture/export."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import pandas as pd
import pytest
from app_core import prospective_evidence as pe, prospective_research_models as models
from app_core import nhl_puck_line_evidence as nhl, prediction_evidence as evidence, research_replay
from core import streamlit_pipeline as sp
from app_core.candidate_evidence_schema import project
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from core.live_wager_contract import finalize_live_wagers
from scripts.benchmark_drive_history_loading import blocked_network
from test_prospective_research_models import _event, _at

AT=_at(72,1)+timedelta(minutes=5)
class FrozenDateTime(datetime):
    @classmethod
    def now(cls,tz=None):return AT.astimezone(tz or timezone.utc)


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():yield


def make_store(path,home_line):
    clock=[_at(0)]
    patch=pytest.MonkeyPatch()
    patch.setattr(pe,"_clock",lambda:clock[0]);patch.setattr(models,"_utcnow",lambda:clock[0])
    original_event,original_quote=pe.insert_event,pe.insert_quote
    def event(path,data):
        data=deepcopy(data);data["raw_source"]["bookmakers"][0]["key"]="novig"
        outcomes=data["raw_source"]["bookmakers"][0]["markets"][0]["outcomes"]
        outcomes[0]["point"]=home_line;outcomes[1]["point"]=-home_line
        return original_event(path,data)
    def quote(path,data):
        data=deepcopy(data);data["sportsbook"]="novig";data["source_id"]=data["source_id"].replace(":book:",":novig:")
        if data["market_family"]=="PUCK_LINE":
            data["line"]=home_line
            raw=data["raw_source"]
            if isinstance(raw,dict):raw["outcomes"][0]["point"]=home_line;raw["outcomes"][1]["point"]=-home_line
        return original_quote(path,data)
    patch.setattr(pe,"insert_event",event);patch.setattr(pe,"insert_quote",quote)
    try:
        for index in range(70):_event(path,clock,"NHL",index)
        clock[0]=_at(71)
        fitted=models.train_scope(path,"NHL","PUCK_LINE")
        assert fitted["status"]=="FITTED_RESEARCH_ONLY"
        _event(path,clock,"NHL",72)
        e=pe.load_record(path,"prospective_event","event-NHL-72")
        q=pe.load_record(path,"prospective_quote","quote-event-NHL-72-PUCK_LINE")
        data={k:v for k,v in q.items() if k not in {"payload","payload_hash","source_hash"}}
        data.update(quote_id="quote-event-NHL-72-PUCK_LINE-away",selection=e["away_team"],line=-home_line,quote_verified=True)
        data["source_id"]=f"{e['provider_event_id']}:book:spreads:{e['away_team']}:{q['quote_timestamp']}"
        clock[0]=models._time(q["quote_timestamp"])
        data["source_id"]=data["source_id"].replace(":book:",":novig:")
        original_quote(path,data)
    finally:patch.undo()
    return path


@pytest.fixture(scope="module")
def synthetic_store(tmp_path_factory):
    root=tmp_path_factory.mktemp("nhl-synthetic")
    return {line:make_store(root/f"canonical-{line}.sqlite3",line) for line in (-1.5,1.5)}


def pipeline(monkeypatch,tmp_path,synthetic_store,*,change=None,reviews=True,home_line=-1.5):
    path=tmp_path/"canonical.sqlite3";shutil.copyfile(synthetic_store[home_line],path)
    events,quotes,scores=models._read_scope(path,"NHL","PUCK_LINE")
    e=events["event-NHL-72"];game=json.loads(e["raw_source"])
    game["matchup_id"]=sp._matchup_id(pd.DataFrame([dict(league="NHL",home_team=e["home_team"],away_team=e["away_team"],game_date="2030-03-13")])).iloc[0]
    market=game["bookmakers"][0]["markets"][0]
    game["bookmakers"][0]["key"]="novig"
    market.update(period="full_game",settlement_rules="SYNTHETIC NHL full-game OT shootout official final")
    game["bookmakers"][0]["markets"]=[market]
    m,artifact=models._latest_model(path,"NHL","PUCK_LINE",AT)
    vector,lineage=models._features(e,AT,events,scores,"PUCK_LINE")
    dependencies=[s for s in scores if (s["result_id"],s["source_hash"]) in lineage]
    review={}
    for q in quotes:
        if q["event_id"]!=e["event_id"]:continue
        binding=dict(event_hash=e["source_hash"],quote_hash=q["source_hash"],artifact_hash=m["artifact_hash"],target=nhl.TARGET,
            rules=market["settlement_rules"],period="full_game",overtime=True,shootout=True,dependency_hashes=sorted(s["source_hash"] for s in dependencies))
        review[nhl.review_key(e,q,artifact)]=dict(binding=binding,rights="ACCEPTED_FOR_PRIVATE_RESEARCH",review_id="SYNTHETIC independent review",
            operator="SYNTHETIC",product="SYNTHETIC",score_semantics="full_game_official_final_including_ot_shootout",
            effective_from=_at(70).isoformat(),effective_until=_at(74).isoformat(),reviewed_at=_at(70).isoformat())
    if change:change(game,review,path)
    class Client:
        def __init__(self,**kwargs):pass
        def get_odds(self,sport,date=None):return [deepcopy(game)] if sport=="icehockey_nhl" else []
    from app_core import odds_api
    monkeypatch.setattr(odds_api,"TheOddsAPIClient",Client);monkeypatch.setattr(odds_api,"datetime",FrozenDateTime)
    monkeypatch.setattr(sp,"_game_date_fallback",lambda:pd.Timestamp(AT.date()))
    monkeypatch.setattr(sp,"_get_odds_api_key",lambda:"offline-unused")
    monkeypatch.setattr(sp,"load_base_data",lambda:pd.DataFrame())
    def features(frame,*a):
        out=frame.copy();out["League"]="NHL";out["ml_feature_eligible"]=True;out["stats_resolution_status"]="resolved"
        return out
    monkeypatch.setattr("app_core.feature_processing.enrich_with_model_features",features)
    monkeypatch.setattr("app_core.external_data_fetcher.enrich_with_external_data",lambda f:f)
    monkeypatch.setattr(sp,"ML_AVAILABLE",True);monkeypatch.setattr(sp,"PredictionEngine",object)
    monkeypatch.setattr(sp,"get_cached_prediction_engine",lambda:SimpleNamespace(use_fallback=False,predict_batch=lambda f:[.8]*len(f)))
    monkeypatch.setattr("app_core.research_estimate_trace.generated_time",lambda:AT.isoformat())
    monkeypatch.setattr(nhl,"generated_time",lambda:AT.isoformat())
    monkeypatch.setattr("app_core.candidate_chronology.now_utc",lambda:pd.Timestamp(AT))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats",lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration",lambda:None)
    with nhl.selected(path,source_reviews=review if reviews else {}):
        analysis,_,diagnostics=sp.run_analysis_pipeline(sports=["NHL"],use_ml=True,max_rows=20)
    assert len(analysis)==2
    return analysis,path,diagnostics


def exported(monkeypatch,tmp_path,analysis):
    diagnostics={};best=sp.build_best_picks_df(analysis,diagnostics_out=diagnostics)
    authority=diagnostics["candidate_authority_df"]
    final,_=finalize_live_wagers(project(authority),best,1000,now=AT,policies={},config={})
    monkeypatch.setattr(evidence,"now_utc",lambda:(AT+timedelta(seconds=5)).isoformat())
    monkeypatch.setattr("app_core.public_board.datetime",FrozenDateTime)
    path=tmp_path/"capture.sqlite3";root=tmp_path/"root";root.mkdir()
    context=evidence.begin_run({},path=path,root=root)
    captured,card=evidence.capture_run(context,project(authority),final,analysis,path=path,authoritative_candidates=True)
    frames=[per_game_board(card,captured,family=f,novig_only=True) for f in ("overall","sides","totals")]
    package=build_package(*frames)
    receipt=research_replay.retain_export(frames,package,card,captured,path=path)
    return package,receipt,path,card,frames


@pytest.mark.parametrize("home_line",[-1.5,1.5])
def test_actual_two_sides_pipeline_capture_export_replay(monkeypatch,tmp_path,synthetic_store,home_line):
    analysis,path,diagnostics=pipeline(monkeypatch,tmp_path,synthetic_store,home_line=home_line)
    assert diagnostics["market_specific_ml_predictions"]==2,analysis[["ml_unavailable_reason","provider_namespace","market_period","settlement_rules"]].to_dict("records")
    assert set(analysis.ml_target)=={"spread_cover"}
    for row in analysis.to_dict("records"):
        result=nhl.diagnose(row);assert result["status"]=="COMPLETE",result
        replay=nhl.replay(row);assert replay["probabilities"]["win"]==row["ml_probability"]
        assert replay["probabilities"]["push"]==0 and not replay["wagering_authority"]
        p=json.loads(row["ml_estimate_metadata"])["nhl_inputs"]["payload"]
        assert p["missingness"]["home_goalie"]=="UNKNOWN"
        assert p["inference_time"]==AT.isoformat() and p["feature_order"]==list(nhl.FEATURE_ORDER)
        assert p["probabilities"]["win"]!=.8 # Parallel game-winner output is retired.
    assert analysis.iloc[0].ml_probability+analysis.iloc[1].ml_probability==pytest.approx(1)
    package,receipt,db,card,frames=exported(monkeypatch,tmp_path,analysis)
    retained,sources=research_replay.read_export(receipt["export_id"],path=db)
    original=research_replay.frame_from_payload(next(iter(sources.values()))["original"]["producer"])
    for row in original.to_dict("records"):assert nhl.replay(row)["probabilities"]["win"]==row["ml_probability"]
    public=json.dumps(package)
    assert not any(s in public for s in ("nhl_inputs","source_dependencies","model_record","coefficients"))
    assert all(r["production_bet_amount"]==0 for r in card.wager_contract)
    assert all(r["status"]=="PASS" for r in package["games"]["overall"])
    assert all(r["research_display"]["availability_reason"]=="AVAILABLE" for r in package["games"]["overall"]),json.dumps(dict(display=[r["research_display"] for r in package["games"]["overall"]],card=card[["matchup_id","best_pick","display_pick","quote_id","quote_bookmaker","quote_time","odds_american","game_start_utc","prediction_generated_at"]].to_dict("records")),default=str)
    assert all(r["research_display"]["ev"] is None for r in package["games"]["overall"])
    assert set(analysis.spread_line)=={-1.5,1.5}
    assert analysis.loc[analysis.market_type.eq("spread_home"),"spread_line"].iloc[0]==home_line


def test_explicit_selection_does_not_activate_subsequent_default_call(monkeypatch,tmp_path,synthetic_store):
    analysis,_,_=pipeline(monkeypatch,tmp_path,synthetic_store)
    assert analysis.ml_probability.notna().all()
    from app_core.market_probability_model import predict_market_probabilities
    later=predict_market_probabilities(analysis)
    assert later.ml_probability.isna().all()
    assert all("No market-specific model" in reason for reason in later.ml_unavailable_reason)
    assert all("nhl_inputs" not in json.loads(value) for value in later.ml_estimate_metadata)


@pytest.mark.parametrize("change,reason",[
    (lambda g,r,p:g["bookmakers"][0]["markets"][0].update(period="regulation"),"NHL_FULL_GAME_RULES_UNAVAILABLE"),
    (lambda g,r,p:g["bookmakers"][0]["markets"][0]["outcomes"][0].update(price=125),"NHL_ORIGINAL_EXACT_QUOTE_UNAVAILABLE"),
    (lambda g,r,p:g["bookmakers"][0]["markets"][0].pop("settlement_rules"),"NHL_FULL_GAME_RULES_UNAVAILABLE"),
    (lambda g,r,p:r.clear(),"NHL_SOURCE_TARGET_REVIEW_UNAVAILABLE"),
    (lambda g,r,p:[x["binding"].update(shootout=False) for x in r.values()],"NHL_SOURCE_TARGET_REVIEW_CONFLICT"),
])
def test_unsupported_native_facts_remain_pass(monkeypatch,tmp_path,synthetic_store,change,reason):
    analysis,_,_=pipeline(monkeypatch,tmp_path,synthetic_store,change=change)
    assert reason in set(analysis.ml_unavailable_reason)
    rejected=analysis.loc[analysis.ml_unavailable_reason.eq(reason)]
    assert rejected.ml_probability.isna().all()
    package,_,_,card,_=exported(monkeypatch,tmp_path,analysis)
    assert all(r["production_bet_amount"]==0 for r in card.wager_contract)
    assert all(r["status"]=="PASS" for r in package["games"]["overall"])


@pytest.mark.parametrize("mutation,reason",[
    (lambda p:p["features"].__setitem__(1,p["features"][1]+1),"NHL_ALTERED_ORDERED_FEATURES"),
    (lambda p:p["feature_order"].reverse(),"NHL_FEATURE_CONTRACT_INVALID"),
    (lambda p:p["artifact"]["model"]["coefficients"].__setitem__(0,100),"NHL_MODEL_ARTIFACT_CORRUPT"),
    (lambda p:p["source_dependencies"][0].update(available_at=_at(75).isoformat()),"NHL_CANONICAL_PAYLOAD_CORRUPT"),
    (lambda p:p["contract"]["offer"].update(line=1.5),"NHL_EVENT_OFFER_CONFLICT"),
    (lambda p:p["source_review"]["binding"].update(overtime=False),"NHL_SOURCE_TARGET_REVIEW_CONFLICT"),
])
def test_rehashed_corrupt_packets_reject_replay_and_display(monkeypatch,tmp_path,synthetic_store,mutation,reason):
    analysis,_,_=pipeline(monkeypatch,tmp_path,synthetic_store)
    row=analysis.iloc[0].to_dict();item=json.loads(row["ml_estimate_metadata"])
    mutation(item["nhl_inputs"]["payload"]);item["nhl_inputs"]["sha256"]=nhl.digest(item["nhl_inputs"]["payload"])
    row["ml_estimate_metadata"]=json.dumps(item)
    assert nhl.diagnose(row)["reason"]==reason
    with pytest.raises(ValueError):nhl.replay(row)
    analysis.at[analysis.index[0],"ml_estimate_metadata"]=row["ml_estimate_metadata"]
    package,_,_,card,_=exported(monkeypatch,tmp_path,analysis)
    assert all(r["production_bet_amount"]==0 for r in card.wager_contract)


@pytest.mark.parametrize("fault,reason",[
    ("ambiguous_offer","NHL_EVENT_OR_OFFER_AMBIGUOUS"),
    ("ambiguous_event","NHL_EVENT_OR_OFFER_AMBIGUOUS"),
    ("conflicting_event_alias","NHL_EVENT_OR_OFFER_AMBIGUOUS"),
    ("conflicting_start_alias","NHL_EVENT_OR_OFFER_AMBIGUOUS"),
    ("missing_features","NHL_ORIGINAL_TEAM_HISTORY_UNAVAILABLE"),
    ("missing_model","NHL_EXACT_PUCK_LINE_MODEL_UNAVAILABLE"),
    ("failed_inference","NHL_INFERENCE_FAILED"),
    ("totals","NHL_TOTAL_REQUIRES_SEPARATE_TARGET_CONTRACT"),
    ("unselected_store","NHL_LOCAL_RESEARCH_STORE_UNAVAILABLE"),
])
def test_explicit_unavailability_and_failure(monkeypatch,tmp_path,synthetic_store,fault,reason):
    analysis,path,_=pipeline(monkeypatch,tmp_path,synthetic_store)
    row=analysis.iloc[0].to_dict();p=json.loads(row["ml_estimate_metadata"])["nhl_inputs"]["payload"]
    reviews={nhl.review_key(p["event"],p["quote"],p["artifact"]):p["source_review"]}
    if fault=="ambiguous_offer":
        qs=json.loads(row["provider_quotes"]);row["provider_quotes"]=json.dumps(qs+qs)
    if fault=="ambiguous_event":row["away_team"]=row["home_team"]
    if fault=="conflicting_event_alias":row["provider_event_id"]="SYNTHETIC other event"
    if fault=="conflicting_start_alias":row["game_start_utc"]=_at(73).isoformat()
    if fault=="missing_features":monkeypatch.setattr(models,"_features",lambda *a:None)
    if fault=="missing_model":monkeypatch.setattr(models,"_latest_model",lambda *a:(None,None))
    if fault=="failed_inference":
        def failed(*a):raise ZeroDivisionError("SYNTHETIC inference failure")
        monkeypatch.setattr(models,"_expected",failed)
    if fault=="totals":row["market_type"]="total_over"
    if fault=="unselected_store":result=nhl.predict(row)
    else:
        with nhl.selected(path,source_reviews=reviews):result=nhl.predict(row)
    assert result["ml_unavailable_reason"]==reason and pd.isna(result["ml_probability"])
    assert result["ml_inference_status"]==("failed" if fault=="failed_inference" else "unavailable")


def test_unrehashed_corruption_and_future_canonical_dependency(monkeypatch,tmp_path,synthetic_store):
    analysis,_,_=pipeline(monkeypatch,tmp_path,synthetic_store)
    row=analysis.iloc[0].to_dict();item=json.loads(row["ml_estimate_metadata"])
    item["nhl_inputs"]["sha256"]="0"*64;row["ml_estimate_metadata"]=json.dumps(item)
    assert nhl.diagnose(row)["reason"]=="NHL_PACKET_INTEGRITY"
    with pytest.raises(ValueError):nhl.replay(row)
    item=json.loads(analysis.iloc[0].ml_estimate_metadata);p=item["nhl_inputs"]["payload"]
    dep=p["source_dependencies"][0];payload=json.loads(dep["payload"])
    payload["available_at"]=_at(75).isoformat();dep["available_at"]=payload["available_at"]
    dep["payload"]=nhl.encode(payload);dep["payload_hash"]=nhl.digest(payload)
    item["nhl_inputs"]["sha256"]=nhl.digest(p);row["ml_estimate_metadata"]=json.dumps(item)
    assert nhl.diagnose(row)["reason"]=="NHL_FUTURE_OR_INVALID_FEATURE_DEPENDENCY"
