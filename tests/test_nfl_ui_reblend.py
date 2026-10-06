"""Synthetic actual native adapter/pipeline/UI refresh/private export; no transport."""
import ast
import base64
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app_core import nfl_inference_evidence as packet, research_replay
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from scripts.benchmark_drive_history_loading import blocked_network
from test_nfl_native_provenance import native_pipeline
from test_source_contract_pipeline import exported, INFERENCE, NOW
from test_research_probability_browser import inspect_browser

ROOT=Path(__file__).resolve().parents[1]
STAGE=(pd.Timestamp(INFERENCE)+pd.Timedelta(seconds=5)).isoformat()


def ui_caller():
    # Load the exact UI functions without executing Streamlit's top-level application.
    names={"_safe_numeric_series","_safe_str_series","_ml_eligible_market_mask","_recompute_consensus_from_kalshi"}
    tree=ast.parse((ROOT/"streamlit_app.py").read_text(encoding="utf-8-sig"))
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(functions)==len(names)
    context={"pd":pd,"np":np,"logger":logging.getLogger("offline-ui-reblend")}
    exec(compile(ast.Module(body=functions,type_ignores=[]),"streamlit_app.py","exec"),context)
    return context["_recompute_consensus_from_kalshi"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(packet,"ui_reblend_time",lambda:STAGE)
    with blocked_network():yield


@pytest.fixture
def refreshed(monkeypatch):
    producer=native_pipeline(monkeypatch)
    result=ui_caller()(producer)
    return producer,result


def test_actual_ui_refresh_retains_both_stages_through_capture_export_replay_browser(monkeypatch,tmp_path,refreshed):
    producer,frame=refreshed
    for before,after in zip(producer.to_dict("records"),frame.to_dict("records")):
        old=json.loads(before["ml_estimate_metadata"]);new=json.loads(after["ml_estimate_metadata"])
        assert old["nfl_inputs"]==new["nfl_inputs"]
        assert old["nfl_inputs"]["payload"]["blend"]["estimated_ev"]!=packet.fact(after["expected_value"])
        assert before["ml_probability"]==after["ml_probability"]
        assert before["calibrated_probability"]==after["calibrated_probability"]
        assert after["prediction_generated_at"]==INFERENCE
        assert new["nfl_ui_reblends"][0]["payload"]["generated_at"]==STAGE!=INFERENCE
        assert packet.diagnose(after)==dict(status="COMPLETE",errors=[],unknown=[])
        replay=packet.replay(after)
        assert replay["raw_probability"]==after["ml_probability"]
        assert replay["blended_probability"]==after["calibrated_probability"]
        assert replay["estimated_ev"]==after["expected_value"]
        assert replay["scientific_acceptance"] is False and replay["wagering_authority"] is False
    saved=exported(monkeypatch,tmp_path,frame)
    receipt=research_replay.retain_export(saved["frames"],saved["package"],saved["card"],saved["captured"],path=saved["db"])
    export,sources=research_replay.read_export(receipt["export_id"],path=saved["db"])
    source=next(iter(sources.values()))
    retained=research_replay.frame_from_payload(source["original"]["producer"])
    for row in retained.to_dict("records"):
        assert packet.replay(row)["estimated_ev"]==row["expected_value"]
    frames=[per_game_board(research_replay.frame_from_payload(source["captured_card"]),research_replay.frame_from_payload(source["captured_candidates"]),family=f,novig_only=True) for f in ("overall","sides","totals")]
    assert build_package(*frames)==export["package"]
    public=json.dumps(export["package"])
    assert all(k not in public for k in ("nfl_ui_reblends","bytes_base64","source_dependencies"))
    browser=inspect_browser(export["package"],tmp_path/"browser",NOW)
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert all(s["status"]=="PASS" and s["stake"]==0 for s in browser["initial"]["saved"])
    assert browser["initial"]["shown"][0]["probability"]==export["package"]["games"]["overall"][0]["research_display"]["probability"]
    assert all(s["ev"] is None and s["edge"] is None and s["breakEven"] is None for s in browser["initial"]["shown"])


def test_retention_hooks_do_not_change_actual_ui_numbers(monkeypatch):
    producer=native_pipeline(monkeypatch)
    with monkeypatch.context() as off:
        off.setattr(packet,"retain_ui_reblend",lambda before,after,inputs:after)
        baseline=ui_caller()(producer)
    recorded=ui_caller()(producer)
    for field in ("ml_probability","calibrated_probability","expected_value","edge"):
        assert baseline[field].tolist()==recorded[field].tolist()
    assert all("blend.original_output_conflict" in packet.diagnose(r)["errors"] for r in baseline.to_dict("records"))
    assert all(packet.diagnose(r)["status"]=="COMPLETE" for r in recorded.to_dict("records"))


def alter(row, change, *, rehash=True):
    row=deepcopy(row);md=json.loads(row["ml_estimate_metadata"])
    receipt=md["nfl_ui_reblends"][0];change(receipt["payload"])
    if rehash:receipt["sha256"]=packet.digest(receipt["payload"])
    row["ml_estimate_metadata"]=packet.encode(md)
    return row


@pytest.mark.parametrize("change,reason",[
    (lambda q:q.update(event_offer=dict(q["event_offer"],event=dict(q["event_offer"]["event"],provider_event_id="unrelated-event"))),"event_offer"),
    (lambda q:q.update(previous_sha256="0"*64),"original_binding"),
    (lambda q:q.update(input_ev=packet.fact(.2)),"previous_output"),
    (lambda q:q.update(probability=packet.fact(.99)),"recorded_output"),
    (lambda q:q.update(estimated_ev=packet.fact(.99)),"recorded_output"),
    (lambda q:q.update(generated_at="invalid-clock"),"clock"),
    (lambda q:q.update(generated_at="2026-10-06T19:20:00Z"),"clock"),
    (lambda q:q.update(generated_at="2026-10-07T01:00:00Z"),"clock"),
    (lambda q:q.update(scientific_acceptance=True),"semantics_authority"),
    (lambda q:q.update(wagering_authority=True),"semantics_authority"),
])
def test_consistently_rehashed_refresh_contradictions_reject(refreshed,change,reason):
    row=alter(refreshed[1].iloc[0].to_dict(),change)
    assessment=packet.diagnose(row)
    assert assessment["status"]=="REJECTED" and any(reason in e for e in assessment["errors"])
    with pytest.raises(ValueError):packet.replay(row)


@pytest.mark.parametrize("field,value",[("odds_american",-110),("spread_line",4.5),("provider_event_id","other"),("ml_feature_eligible",False),("stats_resolution_status","unresolved")])
def test_changed_original_offer_or_eligibility_cannot_be_repaired_by_ui_receipt(refreshed,field,value):
    row=refreshed[1].iloc[0].to_dict();row[field]=value
    assert packet.diagnose(row)["status"]=="REJECTED"
    with pytest.raises(ValueError):packet.replay(row)


def test_corrupted_or_consistently_rehashed_wrong_ui_artifact_rejects(refreshed):
    row=refreshed[1].iloc[0].to_dict()
    for rehash_artifact in (False,True):
        def change(q):
            raw=b"# different consumed UI source\n"
            q["artifact"]["bytes_base64"]=base64.b64encode(raw).decode()
            if rehash_artifact:q["artifact"]["sha256"]=hashlib.sha256(raw).hexdigest()
        bad=alter(row,change)
        assert packet.diagnose(bad)["status"]=="REJECTED"
        with pytest.raises(ValueError):packet.replay(bad)


def test_external_numbers_alone_stay_unknown(monkeypatch):
    producer=native_pipeline(monkeypatch);producer["kalshi_probability"]=[.6,.4]
    refreshed=ui_caller()(producer)
    for row in refreshed.to_dict("records"):
        d=packet.diagnose(row)
        assert d["status"]=="INCOMPLETE" and any("external_source:p_kalshi" in k for k in d["unknown"])
        with pytest.raises(ValueError):packet.replay(row)


def test_original_missing_features_source_or_rules_never_filled(monkeypatch,refreshed):
    before,after=refreshed
    row=after.iloc[0].to_dict();row.pop("football_feature_receipt")
    assert packet.diagnose(row)["status"]=="INCOMPLETE"
    game_unknown=before.copy()
    from app_core import source_contract as adapter
    adapter_catalog=adapter.ACCEPTED_LISTINGS
    monkeypatch.setattr(adapter,"ACCEPTED_LISTINGS",{})
    result=ui_caller()(game_unknown)
    assert all(packet.diagnose(r)["status"]!="COMPLETE" for r in result.to_dict("records"))
    assert adapter_catalog


def test_repeated_ui_refresh_chain_and_missing_transition(refreshed):
    _,frame=refreshed;again=ui_caller()(frame)
    for row in again.to_dict("records"):
        assert len(json.loads(row["ml_estimate_metadata"])["nfl_ui_reblends"])==2
        # UI market_probability is independently rederived after its blend. A changed
        # market input without an original dependency binding must stay incomplete.
        d=packet.diagnose(row)
        assert d["status"] in {"COMPLETE","INCOMPLETE"}
        bad=deepcopy(row);item=json.loads(bad["ml_estimate_metadata"]);item["nfl_ui_reblends"].pop(0)
        bad["ml_estimate_metadata"]=packet.encode(item)
        assert packet.diagnose(bad)["status"]=="REJECTED"


def test_legacy_packet_rejection_is_not_repaired(refreshed):
    before,after=refreshed
    row=deepcopy(after.iloc[0].to_dict());item=json.loads(row["ml_estimate_metadata"]);item.pop("nfl_ui_reblends")
    row["ml_estimate_metadata"]=packet.encode(item)
    assert packet.diagnose(row)["errors"]==["blend.original_output_conflict"]
    assert json.loads(before.iloc[0].ml_estimate_metadata)["nfl_inputs"]==item["nfl_inputs"]


@pytest.mark.parametrize("missing", ["generated_at","artifact","consumed"])
def test_missing_refresh_facts_remain_incomplete(refreshed,missing):
    row=alter(refreshed[1].iloc[0].to_dict(),lambda q:q.pop(missing))
    d=packet.diagnose(row)
    assert d["status"]=="INCOMPLETE" and any("missing:"+missing in k for k in d["unknown"])
    with pytest.raises(ValueError):packet.replay(row)


def test_missing_refresh_clock_is_not_replaced_by_original_inference(refreshed):
    row=alter(refreshed[1].iloc[0].to_dict(),lambda q:q.update(generated_at=None))
    d=packet.diagnose(row)
    assert d["status"]=="INCOMPLETE" and "ui_reblend.0.clock" in d["unknown"]


def test_false_input_state_cannot_hide_a_numeric_external_source(refreshed):
    row=alter(refreshed[1].iloc[0].to_dict(),lambda q:q["inputs"].update(p_kalshi={"state":"INVALID","value":.6}))
    d=packet.diagnose(row)
    assert d["status"]=="REJECTED" and "ui_reblend.0.input_fact:p_kalshi" in d["errors"]
    with pytest.raises(ValueError):packet.replay(row)


def test_source_file_read_failure_rejects_without_changing_numbers(monkeypatch,refreshed):
    _,frame=refreshed;row=frame.iloc[0].to_dict();original=deepcopy(row)
    def unavailable():raise OSError("synthetic read failure")
    monkeypatch.setattr(packet,"_ui_artifact",unavailable)
    assert packet.diagnose(row)["status"]=="REJECTED"
    assert row["ml_probability"]==original["ml_probability"] and row["calibrated_probability"]==original["calibrated_probability"] and row["expected_value"]==original["expected_value"]
    with pytest.raises(ValueError):packet.replay(row)


def test_missing_native_dependencies_cannot_complete_after_refresh(monkeypatch):
    frame=native_pipeline(monkeypatch,retain=False);result=ui_caller()(frame)
    for row in result.to_dict("records"):
        d=packet.diagnose(row)
        assert d["status"]=="INCOMPLETE" and "features.source_dependencies_and_availability" in d["unknown"]
        with pytest.raises(ValueError):packet.replay(row)


def test_empty_observed_features_cannot_complete_after_refresh(monkeypatch):
    frame=native_pipeline(monkeypatch)
    for index,row in frame.iterrows():
        item=json.loads(row.ml_estimate_metadata);p=item["nfl_inputs"]["payload"]
        observation=json.loads(p["observation_receipt"]);observation["payload"]["features"]={}
        observation["sha256"]=packet.digest(observation["payload"])
        encoded=packet.encode(observation);p["observation_receipt"]=encoded
        item["nfl_inputs"]["sha256"]=packet.digest(p)
        frame.at[index,"football_feature_receipt"]=encoded
        frame.at[index,"ml_estimate_metadata"]=packet.encode(item)
    result=ui_caller()(frame)
    for row in result.to_dict("records"):
        d=packet.diagnose(row)
        assert d["status"]=="INCOMPLETE" and any("observation_missing:" in k for k in d["unknown"])
        with pytest.raises(ValueError):packet.replay(row)
