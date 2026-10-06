"""Synthetic owner intake -> native adapters -> actual pipeline -> private replay."""
import ast
import base64
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from zipfile import ZipFile
import pandas as pd
import pytest
from app_core import source_contract as source, source_evidence_intake as intake
from app_core import research_replay, nfl_inference_evidence as nfl, producer_provenance
from core import streamlit_pipeline as sp
from scripts.benchmark_drive_history_loading import blocked_network
import test_nfl_native_provenance as native_tests
from test_source_contract_pipeline import fixture, exported, INFERENCE, NOW
from test_nfl_ui_reblend import ui_caller, STAGE
from test_research_probability_browser import inspect_browser


@pytest.fixture(autouse=True)
def offline():
    with blocked_network(): yield


def reseal(packet):
    e=packet["evidence"];r=packet.get("review")
    if r is None:return packet
    r.update(evidence_sha256=intake.digest(e),identity_sha256=intake.digest(e.get("identity") or {}),product_sha256=intake.digest(e.get("product") or {}),rights_sha256=intake.digest(e.get("rights") or {}),rule_version=(e.get("terms") or {}).get("rule_version"))
    return packet


def synthetic(identity):
    product=dict(operator="SYNTHETIC legal operator",product="SYNTHETIC NFL event contract",jurisdiction="SYNTHETIC jurisdiction",bookmaker="novig")
    e=dict(identity=deepcopy(identity),inference_time=INFERENCE,product=product,
        listing=dict(id="SYNTHETIC-listing-"+identity["selection"],reference_team=identity["selection"],count=-identity["line"],comparison="above",position="YES",price_units="american_odds_from_bound_yes_offer"),
        terms=dict(period="full_game",overtime=True,rule_version=source.RULES,effective_from="2026-10-05T00:00:00Z",effective_until="2026-10-07T00:00:00Z",scheduling="SYNTHETIC original schedule terms",cancellation="SYNTHETIC cancellation rule",void="SYNTHETIC FVS contingencies",payoff="FVS"),
        clock=dict(field="market.last_update",meaning="SYNTHETIC original upstream offer revision; executability unclaimed",timezone="UTC"),
        rights=dict(owner="SYNTHETIC owner",product=deepcopy(product),storage="permitted",research="permitted",derived_output="permitted",effective_from="2026-10-05T00:00:00Z",effective_until="2026-10-07T00:00:00Z",limitations="SYNTHETIC private research only; no raw redistribution"),references={},documents=[])
    for purpose in sorted(intake.PURPOSES):
        raw=json.dumps(dict(SYNTHETIC=True,purpose=purpose,identity=identity,product=product,provision="SYNTHETIC supplied exact facts"),sort_keys=True).encode()
        e["documents"].append(dict(id=purpose,sha256=hashlib.sha256(raw).hexdigest(),bytes_base64=base64.b64encode(raw).decode(),media_type="text/plain",source="synthetic:"+purpose))
        e["references"][purpose]=dict(document_id=purpose,locator="SYNTHETIC provision 1")
    review=dict(version=intake.REVIEW_VERSION,id="SYNTHETIC independent review",reviewer="SYNTHETIC reviewer",decision="ACCEPTED_FOR_PRIVATE_RESEARCH",evidence_sha256="",identity_sha256="",product_sha256="",rights_sha256="",rule_version=source.RULES,reviewed_at="2026-10-05T01:00:00Z",expires_at="2026-10-07T00:00:00Z")
    return reseal(dict(version=intake.VERSION,evidence=e,review=review))


def accepted(packet):
    return dict(source_evidence_sha256=intake.digest(packet),source_review_sha256=intake.digest(packet.get("review") or {}))


def offer(monkeypatch):
    game,_=fixture(monkeypatch)
    b=game["bookmakers"][0];m=b["markets"][0]
    return source.identity(game,b,m,m["outcomes"][0])


def through_native(monkeypatch,tmp_path,*,approved=True,change=None,accept_changed=False,explicit_reference=None,declared_inference=True):
    original_fixture=native_tests.fixture
    state={}
    def input_fixture(*a,**kw):
        game,catalog=original_fixture(*a,**kw);state.update(game=game,catalog=catalog)
        return game,catalog
    monkeypatch.setattr(native_tests,"fixture",input_fixture)
    real_pipeline=sp.run_analysis_pipeline
    def run(*a,**kw):
        game=state["game"];b=game["bookmakers"][0];m=b["markets"][0]
        packets=[];refs=[]
        for outcome in m["outcomes"]:
            packet=synthetic(source.identity(game,b,m,outcome))
            if not declared_inference:packet["evidence"].pop("inference_time");reseal(packet)
            reference=intake.ref(packet)
            if approved:state["catalog"][reference]=accepted(packet)
            raw=(json.dumps(packet,indent=2)+"\n").encode()
            intake.intake(raw,path=tmp_path/"private.sqlite3")
            state.setdefault("original",[]).append(raw)
            if change is not None:
                change(packet);reseal(packet)
                if accept_changed:state["catalog"][intake.ref(packet)]=accepted(packet)
            reference=intake.ref(packet)
            outcome.pop("source_contract_ref",None)
            if explicit_reference is True or (explicit_reference is None and change is not None):outcome["source_evidence_ref"]=reference
            packets.append(packet);refs.append(reference)
        # Fault injection at the local-read boundary is deliberately untrusted.
        # Independent tests below verify original-byte/database integrity.
        if change is not None:
            monkeypatch.setattr(intake,"read",lambda reference,**kw:deepcopy(next(p for p in packets if intake.ref(p)==reference)))
        with intake.selected(refs,path=tmp_path/"private.sqlite3"):
            result=real_pipeline(*a,**kw)
        state.update(packets=packets,refs=refs)
        return result
    monkeypatch.setattr(sp,"run_analysis_pipeline",run)
    analysis=native_tests.native_pipeline(monkeypatch)
    return analysis,state


def test_owner_intake_retains_original_bytes_without_accepting_or_registering(monkeypatch,tmp_path):
    identity=offer(monkeypatch);packet=synthetic(identity)
    before=deepcopy(source.ACCEPTED_LISTINGS)
    raw=(json.dumps(packet,indent=3)+"\n").encode();db=tmp_path/"private.sqlite3"
    receipt=intake.intake(raw,path=db)
    assert receipt["status"]=="UNKNOWN" and "SOURCE_ADMISSIBILITY_REVIEW_NOT_ACCEPTED" in receipt["diagnostics"]
    assert source.ACCEPTED_LISTINGS==before
    assert intake.read(receipt["reference"],path=db)==packet
    assert intake.download(receipt["reference"],path=db)==raw
    assert intake.intake(raw,path=db)==receipt
    with sqlite3.connect(db) as connection:
        for action in ("UPDATE research_source_intakes SET payload='{}'","DELETE FROM research_source_intakes"):
            with pytest.raises(sqlite3.IntegrityError,match="append-only"):connection.execute(action)
    with pytest.raises(ValueError,match="ORIGINAL_BYTES_CONFLICT"):
        intake.intake(json.dumps(packet).encode(),path=db)
    assert not receipt["scientific_acceptance"] and not receipt["wagering_authority"]


def test_actual_native_pipeline_ui_capture_export_replay_browser(monkeypatch,tmp_path):
    analysis,state=through_native(monkeypatch,tmp_path)
    raw=analysis.ml_probability.tolist();blend=analysis.calibrated_probability.tolist()
    monkeypatch.setattr(nfl,"ui_reblend_time",lambda:STAGE)
    refreshed=ui_caller()(analysis)
    for before,row in zip(analysis.to_dict("records"),refreshed.to_dict("records")):
        old=json.loads(before["ml_estimate_metadata"]);meta=json.loads(row["ml_estimate_metadata"])
        assert old["nfl_inputs"]==meta["nfl_inputs"]
        contract=meta["producer_contract"];bound=contract["source_contract"]
        assert bound["status"]=="VERIFIED" and not source.replay(bound,contract["inference_time"])
        assert contract["inference_time"]==INFERENCE and meta["nfl_ui_reblends"][0]["payload"]["generated_at"]==STAGE
        assert nfl.diagnose(row)["status"]=="COMPLETE"
        replay=nfl.replay(row)
        assert replay["raw_probability"]==row["ml_probability"] and replay["blended_probability"]==row["calibrated_probability"]
        assert replay["scientific_acceptance"] is False and replay["wagering_authority"] is False
    assert refreshed.ml_probability.tolist()==raw and refreshed.calibrated_probability.tolist()==blend and raw!=blend
    output=exported(monkeypatch,tmp_path,refreshed)
    receipt=research_replay.retain_export(output["frames"],output["package"],output["card"],output["captured"],path=output["db"])
    saved,sources=research_replay.read_export(receipt["export_id"],path=output["db"])
    retained=research_replay.frame_from_payload(next(iter(sources.values()))["original"]["producer"])
    assert retained.ml_probability.tolist()==raw and retained.calibrated_probability.tolist()==blend
    for row in retained.to_dict("records"):
        md=json.loads(row["ml_estimate_metadata"]);contract=md["producer_contract"]
        assert not source.replay(contract["source_contract"],contract["inference_time"])
        packet=contract["source_contract"]["receipt"]
        assert packet in state["packets"]
        for d in packet["evidence"]["documents"]:assert hashlib.sha256(base64.b64decode(d["bytes_base64"])).hexdigest()==d["sha256"]
    bundle,verified=research_replay.download_bundle(receipt,expected_package_hash=receipt["package_hash"],path=output["db"])
    with ZipFile(BytesIO(bundle)) as archive:
        for name,sha in verified["files"].items():assert hashlib.sha256(archive.read(name)).hexdigest()==sha
        assert b"SYNTHETIC independent review" in b"".join(archive.read(n) for n in archive.namelist() if n.startswith("sources/"))
    public=json.dumps(saved["package"])
    assert all(k not in public for k in ("bytes_base64","SYNTHETIC legal operator","SYNTHETIC independent review","source_evidence_sha256","nfl_inputs","nfl_ui_reblends"))
    display=saved["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"]=="AVAILABLE" and display["probability"] in blend
    assert display["ev"] is None and display["edge"] is None and display["break_even_probability"] is None
    browser=inspect_browser(saved["package"],tmp_path/"browser",NOW)
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert all(x["status"]=="PASS" and x["stake"]==0 for x in browser["initial"]["saved"])
    assert all(x["ev"] is None and x["edge"] is None and x["breakEven"] is None for x in browser["initial"]["shown"])
    assert not output["captured"].production_eligible.fillna(False).any()


def test_source_intake_preserves_actual_predictor_and_blend_numbers(monkeypatch,tmp_path):
    with monkeypatch.context() as before: baseline=native_tests.native_pipeline(before)
    with monkeypatch.context() as after: actual,_=through_native(after,tmp_path)
    for key in ("ml_probability","calibrated_probability","expected_value","edge"):
        assert baseline[key].tolist()==actual[key].tolist()


@pytest.mark.parametrize("field,value",[("provider_event_id","unrelated"),("bookmaker","draftkings"),("selection","unrelated"),("line",-3.5),("line",4.5),("price",105),("provider_quote_id","forged-quote"),("source_time","2026-10-06T19:24:01Z"),("market","totals"),("home","swapped")])
def test_consistently_rehashed_offer_conflicts_fail_actual_pipeline(monkeypatch,tmp_path,field,value):
    def change(p):
        old=p["evidence"]["identity"][field]
        p["evidence"]["identity"][field]=(-value if field=="line" and value==old else value)
    analysis,_=through_native(monkeypatch,tmp_path,change=change)
    for row in analysis.to_dict("records"):
        d=producer_provenance.diagnose(row,json.loads(row["ml_estimate_metadata"]))
        assert d["reason"] and any(field.upper() in x or "SCOPE" in x or "ACCEPT" in x for x in d["source_contract_diagnostics"])
    result=exported(monkeypatch,tmp_path,analysis)
    assert all(json.loads(r)["availability_reason"]!="AVAILABLE" for r in result["frames"][0].research_display)
    assert all(c["production_bet_amount"]==0 for c in result["card"].wager_contract)


@pytest.mark.parametrize("change,expected",[
 (lambda p:p["evidence"]["terms"].update(period="first_half"),"SOURCE_PERIOD_CONFLICT"),
 (lambda p:p["evidence"]["terms"].update(overtime=False),"SOURCE_OVERTIME_CONFLICT"),
 (lambda p:p["evidence"]["terms"].update(effective_until="2026-10-06T19:00:00Z"),"SOURCE_CONTRACT_STALE_OR_NOT_YET_EFFECTIVE"),
 (lambda p:p["evidence"]["terms"].update(rule_version="another-market-rule"),"SOURCE_RULE_VERSION_UNSUPPORTED"),
 (lambda p:p["evidence"]["listing"].update(position="NO"),"SOURCE_SELECTED_SIDE_CONFLICT_POSITION"),
 (lambda p:p["evidence"]["listing"].update(count=99),"SOURCE_SELECTED_SIDE_CONFLICT_COUNT"),
 (lambda p:p["evidence"]["product"].update(operator="unrelated operator"),"SOURCE_RIGHTS_PRODUCT_CONFLICT"),
 (lambda p:p["evidence"]["product"].update(product="DFS"),"SOURCE_RIGHTS_PRODUCT_CONFLICT"),
 (lambda p:p["evidence"].update(inference_time="2026-10-06T19:25:51Z"),"SOURCE_CONFLICT_INFERENCE_TIME"),
 (lambda p:p["evidence"]["clock"].update(field="bookmaker.last_update"),"SOURCE_CONFLICT_QUOTE_CLOCK_FIELD"),
])
def test_source_contradictions_fail_actual_pipeline_and_display(monkeypatch,tmp_path,change,expected):
    analysis,_=through_native(monkeypatch,tmp_path,change=change)
    for row in analysis.to_dict("records"):
        assert expected in producer_provenance.diagnose(row,json.loads(row["ml_estimate_metadata"]))["source_contract_diagnostics"]
    result=exported(monkeypatch,tmp_path,analysis)
    assert all(json.loads(d)["availability_reason"]=="SOURCE_EVIDENCE_CONFLICT" for d in result["frames"][0].research_display)


@pytest.mark.parametrize("path",[("review",),("evidence","rights"),("evidence","terms","effective_until"),("evidence","references","bridge"),("evidence","product","jurisdiction")])
def test_missing_facts_remain_unknown_and_unavailable(monkeypatch,tmp_path,path):
    identity=offer(monkeypatch);p=synthetic(identity);target=p
    for key in path[:-1]:target=target[key]
    target.pop(path[-1]);reseal(p)
    source.ACCEPTED_LISTINGS[intake.ref(p)]=accepted(p)
    result=intake.assess(p,identity,inference_time=INFERENCE)
    assert result["status"]=="UNKNOWN" and result["diagnostics"]
    intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")


def test_unaccepted_complete_packet_has_specific_unavailable_reason(monkeypatch,tmp_path):
    analysis,_=through_native(monkeypatch,tmp_path,approved=False)
    output=exported(monkeypatch,tmp_path,analysis)
    assert all(json.loads(d)["availability_reason"]=="SOURCE_ADMISSIBILITY_REVIEW_NOT_ACCEPTED" for d in output["frames"][0].research_display)
    browser=inspect_browser(output["package"],tmp_path/"browser",NOW)
    assert all(x["probability"] is None and x["ev"] is None for x in browser["initial"]["shown"])


def test_rehashed_unrelated_documents_and_review_do_not_establish_acceptance(monkeypatch):
    identity=offer(monkeypatch);p=synthetic(identity);old=deepcopy(p)
    source.ACCEPTED_LISTINGS[intake.ref(old)]=accepted(old)
    raw=b"SYNTHETIC unrelated event and market"
    p["evidence"]["documents"][0].update(bytes_base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest());reseal(p)
    assert intake.assess(p,identity)["status"]=="UNKNOWN"
    source.ACCEPTED_LISTINGS[intake.ref(p)]=accepted(old)
    result=intake.assess(p,identity)
    assert result["status"]=="REJECTED" and "SOURCE_ACCEPTANCE_BINDING_CONFLICT" in result["diagnostics"]
    p["VERIFIED"]=True
    assert intake.assess(p,identity)["status"]=="REJECTED"


def test_corrupt_documents_and_original_retention_reject(monkeypatch,tmp_path):
    p=synthetic(offer(monkeypatch));p["evidence"]["documents"][0]["bytes_base64"]=base64.b64encode(b"corrupt").decode();reseal(p)
    with pytest.raises(ValueError,match="DOCUMENT_INTEGRITY"):intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")
    p=synthetic(offer(monkeypatch));receipt=intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")
    with sqlite3.connect(tmp_path/"private.sqlite3") as db:
        db.execute("DROP TRIGGER immutable_research_source_intakes_UPDATE")
        db.execute("UPDATE research_source_intakes SET payload='{}'")
    with pytest.raises(ValueError,match="INTEGRITY"):intake.download(receipt["reference"],path=tmp_path/"private.sqlite3")


def test_owner_selection_is_scoped_and_missing_or_cross_market_refs_fail_closed(monkeypatch,tmp_path):
    identity=offer(monkeypatch);p=synthetic(identity);r=intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")
    assert intake.for_offer(identity) is None
    with intake.selected([r["reference"]],path=tmp_path/"private.sqlite3"):
        assert intake.for_offer(identity)["status"]=="UNKNOWN"
        assert intake.for_offer(dict(identity,market="totals"),reference=r["reference"])["status"]=="REJECTED"
        with intake.selected([]):assert intake.for_offer(identity) is None
    assert intake.for_offer(identity) is None
    with pytest.raises(ValueError,match="SELECTED_INTAKE_UNAVAILABLE"):
        with intake.selected(["missing"],path=tmp_path/"private.sqlite3"):pass


def test_historical_evidence_and_original_stage_are_not_backfilled(monkeypatch,tmp_path):
    baseline=native_tests.native_pipeline(monkeypatch);before=baseline.copy(deep=True)
    metadata=baseline.ml_estimate_metadata.tolist()
    p=synthetic(json.loads(metadata[0])["producer_contract"]["source_contract"]["identity"])
    intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")
    pd.testing.assert_frame_equal(baseline,before)
    assert baseline.ml_estimate_metadata.tolist()==metadata
    assert all(json.loads(x)["producer_contract"]["source_contract"]["version"]==source.VERSION for x in metadata)


def test_private_storage_duplicate_keys_and_forged_schema_reject(monkeypatch,tmp_path):
    p=synthetic(offer(monkeypatch))
    with pytest.raises(ValueError,match="OUTSIDE_REPOSITORY"):intake.intake(json.dumps(p).encode(),path=__import__("app_core.prediction_evidence",fromlist=["ROOT"]).ROOT/"never-create.sqlite3")
    with pytest.raises(ValueError,match="DUPLICATE_JSON"):intake.intake(b'{"version":1,"version":2}',path=tmp_path/"private.sqlite3")
    with pytest.raises(ValueError,match="NONFINITE_JSON"):intake.intake(b'{"x":NaN}',path=tmp_path/"private.sqlite3")
    p["review"]["VERIFIED"]=True
    assert intake.assess(p,p["evidence"]["identity"])["status"]=="REJECTED"


def test_owner_intake_ui_is_after_auth_and_pipeline_has_run_local_scope():
    root=Path(__file__).resolve().parents[1]
    panel=ast.parse((root/"app/ui/publish_panel.py").read_text(encoding="utf-8-sig"))
    fn=next(n for n in panel.body if isinstance(n,ast.FunctionDef) and n.name=="render_publish_panel")
    auth=next(n for n in fn.body if isinstance(n,ast.If) and "compare_digest" in ast.unparse(n.test))
    call=next(n for n in fn.body if isinstance(n,ast.Expr) and "render_source_evidence()"==ast.unparse(n.value))
    assert auth.lineno<call.lineno and any(isinstance(n,ast.Return) for n in auth.body)
    ui=ast.parse((root/"streamlit_app.py").read_text(encoding="utf-8-sig"))
    pipeline=next(n for n in ui.body if isinstance(n,ast.FunctionDef) and n.name=="_run_pipeline")
    selected=next(n for n in ast.walk(pipeline) if isinstance(n,ast.With) and "selected_source_evidence" in ast.unparse(n.items[0].context_expr))
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="run_analysis_pipeline" for n in ast.walk(selected))


def test_source_packet_cannot_invent_future_inference_clock(monkeypatch):
    identity=offer(monkeypatch);p=synthetic(identity);p["evidence"].pop("inference_time");reseal(p)
    source.ACCEPTED_LISTINGS[intake.ref(p)]=accepted(p)
    result=intake.assess(p,identity,inference_time=INFERENCE)
    assert result["status"]=="VERIFIED"
    contract=dict(result,identity=identity)
    assert not intake.replay(contract,INFERENCE)
    assert intake.replay(contract,None)==["SOURCE_CONSUMED_INFERENCE_CLOCK_UNKNOWN"]
    assert "SOURCE_CONTRACT_STALE_OR_NOT_YET_EFFECTIVE" in intake.replay(contract,"2026-10-06T23:30:00Z")


@pytest.mark.parametrize("field,reason",[("rights","SOURCE_RIGHTS_NOT_VERIFIED"),("review","SOURCE_ADMISSIBILITY_REVIEW_NOT_ACCEPTED"),("period","SOURCE_EVIDENCE_INCOMPLETE"),("effective_until","SOURCE_EVIDENCE_INCOMPLETE")])
def test_missing_source_facts_fail_actual_capture_export_browser(monkeypatch,tmp_path,field,reason):
    def change(p):
        if field=="review":p.pop("review")
        elif field=="rights":p["evidence"].pop("rights")
        else:p["evidence"]["terms"].pop(field)
    analysis,_=through_native(monkeypatch,tmp_path,change=change,accept_changed=True)
    output=exported(monkeypatch,tmp_path,analysis)
    assert all(json.loads(d)["availability_reason"]==reason for d in output["frames"][0].research_display)
    assert all(c["production_bet_amount"]==0 for c in output["card"].wager_contract)
    browser=inspect_browser(output["package"],tmp_path/"browser",NOW)
    assert all(x["probability"] is None and x["ev"] is None for x in browser["initial"]["shown"])


def test_reproduce_predecessor_ui_projection_rejection_without_rewriting_stages(monkeypatch,tmp_path):
    import subprocess
    from app_core import research_estimate_trace as trace
    analysis,_=through_native(monkeypatch,tmp_path)
    monkeypatch.setattr(nfl,"ui_reblend_time",lambda:STAGE)
    row=ui_caller()(analysis).iloc[0].to_dict();original=row["ml_estimate_metadata"]
    from scripts import check_launch_change_scope as guard
    current=Path(trace.__file__).read_bytes().replace(b"\r\n",b"\n")
    # CI can have no ancestor objects. Reconstruct the exact reviewed bytes,
    # hash-bound to actual main; never fetch history or alter workflow checkout.
    with monkeypatch.context() as no_history:
        def forbid_git(*a,**kw): raise AssertionError("Predecessor proof must work without Git history")
        no_history.setattr(subprocess,"check_output",forbid_git)
        previous_bytes=guard._source_intake_previous_main_source("app_core/research_estimate_trace.py",current)
    assert hashlib.sha256(previous_bytes).hexdigest()=="8d5639f20af0fd7e6ec6dfead1a24e0e3fd51a46e5f7ef436299e55611086bdb"
    previous=previous_bytes.decode()
    fn=next(n for n in ast.parse(previous).body if isinstance(n,ast.FunctionDef) and n.name=="origin_rejection")
    namespace=dict(json=json,encode=trace.encode,_legacy_origin_rejection=trace._legacy_origin_rejection)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),"frozen_predecessor_projection","exec"),namespace)
    assert namespace["origin_rejection"](row)=="ESTIMATE_PROVENANCE_NOT_RECORDED"
    assert trace.origin_rejection(row) is None
    assert row["ml_estimate_metadata"]==original
    damaged=deepcopy(row);meta=json.loads(original);meta["nfl_ui_reblends"][0]["payload"]["probability"]={"state":"VALUE","value":.99}
    meta["nfl_ui_reblends"][0]["sha256"]=nfl.digest(meta["nfl_ui_reblends"][0]["payload"])
    damaged["ml_estimate_metadata"]=json.dumps(meta)
    assert trace.origin_rejection(damaged) is not None


def test_owner_token_guards_actual_upload_panel(monkeypatch):
    from types import SimpleNamespace
    from app.ui import publish_panel, source_evidence_panel
    calls=[]
    class StopAfterAuthorizedIntake(Exception):pass
    def entered():calls.append("intake");raise StopAfterAuthorizedIntake()
    monkeypatch.setattr(source_evidence_panel,"render",entered)
    monkeypatch.setattr(publish_panel,"setting",lambda name,default="":"x"*20)
    mock=SimpleNamespace(subheader=lambda *a,**k:None,caption=lambda *a,**k:None,info=lambda *a,**k:None,text_input=lambda *a,**k:"wrong-token")
    monkeypatch.setattr(publish_panel,"st",mock)
    publish_panel.render_publish_panel(None,None)
    assert calls==[]
    mock.text_input=lambda *a,**k:"x"*20
    with pytest.raises(StopAfterAuthorizedIntake):publish_panel.render_publish_panel(None,None)
    assert calls==["intake"]


def test_packet_selection_marks_results_stale_without_triggering_analysis():
    from typing import Any
    root=Path(__file__).resolve().parents[1]
    tree=ast.parse((root/"streamlit_app.py").read_text(encoding="utf-8-sig"))
    names={"_analysis_input_signature","_upload_fingerprint","_should_run_pipeline","_analysis_inputs_stale"}
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace=dict(pd=pd,hashlib=hashlib,Any=Any,PIPELINE_BUILD="offline-synthetic")
    exec(compile(ast.Module(body=functions,type_ignores=[]),"actual_analysis_click_controls","exec"),namespace)
    controls=dict(source_evidence_refs=[]);changed=dict(source_evidence_refs=["synthetic-ref"])
    state=dict(last_processed_run_counter=5,analysis_df=pd.DataFrame([dict(sport="NFL")]),last_successful_pipeline_signature=namespace["_analysis_input_signature"](controls))
    assert namespace["_analysis_inputs_stale"](state,changed)
    assert namespace["_should_run_pipeline"](state,5,changed) is False
    assert namespace["_should_run_pipeline"](state,6,changed) is True
    assert namespace["_should_run_pipeline"](state,6,changed) is False


@pytest.mark.parametrize("section,field,value",[("product","operator",True),("clock","meaning",7),("rights","owner",True),("listing","id",True),("terms","cancellation",True)])
def test_numeric_or_verified_boolean_cannot_be_a_source_fact(monkeypatch,section,field,value):
    identity=offer(monkeypatch);p=synthetic(identity);p["evidence"][section][field]=value;reseal(p)
    source.ACCEPTED_LISTINGS[intake.ref(p)]=accepted(p)
    assert intake.assess(p,identity)["status"]=="REJECTED"


@pytest.mark.parametrize("field,value",[("line",4.5),("price",105),("source_time","2026-10-06T19:24:01Z")])
def test_changed_consumed_offer_diagnosed_from_owner_selection_without_provider_ref(monkeypatch,tmp_path,field,value):
    analysis,state=through_native(monkeypatch,tmp_path,change=lambda p:p["evidence"]["identity"].update({field:value}),explicit_reference=False)
    assert all("source_evidence_ref" not in o for o in state["game"]["bookmakers"][0]["markets"][0]["outcomes"])
    for row in analysis.to_dict("records"):
        assert "SOURCE_CONFLICT_"+field.upper() in producer_provenance.diagnose(row,json.loads(row["ml_estimate_metadata"]))["source_contract_diagnostics"]
    output=exported(monkeypatch,tmp_path,analysis)
    assert all(json.loads(d)["availability_reason"]=="SOURCE_EVIDENCE_CONFLICT" for d in output["frames"][0].research_display)


def test_original_inference_is_produced_without_uploaded_future_clock(monkeypatch,tmp_path):
    analysis,state=through_native(monkeypatch,tmp_path,declared_inference=False)
    assert all("inference_time" not in p["evidence"] for p in state["packets"])
    for row in analysis.to_dict("records"):
        c=json.loads(row["ml_estimate_metadata"])["producer_contract"]
        assert c["inference_time"]==INFERENCE and not source.replay(c["source_contract"],INFERENCE)
    output=exported(monkeypatch,tmp_path,analysis)
    assert all(json.loads(d)["availability_reason"]=="AVAILABLE" for d in output["frames"][0].research_display)


def test_private_intake_cannot_land_in_another_checkout(monkeypatch,tmp_path):
    other=tmp_path/"other-public-checkout";other.mkdir();(other/".git").mkdir()
    p=synthetic(offer(monkeypatch))
    with pytest.raises(ValueError,match="OUTSIDE_REPOSITORY"):
        intake.intake(json.dumps(p).encode(),path=other/"private.sqlite3")
    assert not (other/"private.sqlite3").exists()


@pytest.mark.parametrize("field",sorted(intake.IDENTITY_FIELDS))
def test_missing_original_offer_fact_is_retained_unknown_not_invented(monkeypatch,tmp_path,field):
    identity=offer(monkeypatch);p=synthetic(identity);p["evidence"]["identity"].pop(field);reseal(p)
    source.ACCEPTED_LISTINGS[intake.ref(p)]=accepted(p)
    assert intake.assess(p,identity,inference_time=INFERENCE)["status"]=="UNKNOWN"
    receipt=intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")
    assert receipt["status"]=="UNKNOWN"
    assert field not in intake.read(receipt["reference"],path=tmp_path/"private.sqlite3")["evidence"]["identity"]


@pytest.mark.parametrize("section,field",[("product","jurisdiction"),("terms","period"),("terms","overtime"),("terms","effective_until"),("terms","rule_version"),("rights","owner"),("rights","research"),("clock","meaning"),("clock","field")])
def test_explicit_unknown_bindings_cannot_become_verified(monkeypatch,tmp_path,section,field):
    identity=offer(monkeypatch);p=synthetic(identity);p["evidence"][section][field]="UNKNOWN";reseal(p)
    source.ACCEPTED_LISTINGS[intake.ref(p)]=accepted(p)
    result=intake.assess(p,identity,inference_time=INFERENCE,quote_clock_field="market.last_update")
    assert result["status"]=="UNKNOWN",result
    receipt=intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")
    assert receipt["status"]=="UNKNOWN"
    assert intake.read(receipt["reference"],path=tmp_path/"private.sqlite3")["evidence"][section][field]=="UNKNOWN"


@pytest.mark.parametrize("identity",[True,["invalid"]])
def test_malformed_identity_intake_has_specific_rejection(monkeypatch,tmp_path,identity):
    p=synthetic(offer(monkeypatch));p["evidence"]["identity"]=identity;reseal(p)
    with pytest.raises(ValueError,match="SOURCE_IDENTITY_SCHEMA_UNSUPPORTED"):
        intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")


def test_missing_original_document_bytes_remain_unknown(monkeypatch,tmp_path):
    identity=offer(monkeypatch);p=synthetic(identity);p["evidence"]["documents"][0].pop("bytes_base64");reseal(p)
    source.ACCEPTED_LISTINGS[intake.ref(p)]=accepted(p)
    assert intake.assess(p,identity)["status"]=="UNKNOWN"
    assert intake.intake(json.dumps(p).encode(),path=tmp_path/"private.sqlite3")["status"]=="UNKNOWN"
