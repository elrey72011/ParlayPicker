"""Actual native intake -> inference -> refresh -> capture -> export -> dataset."""
from copy import deepcopy
import json
import hashlib
import sqlite3
import pytest
from app_core import nfl_calibration_evidence as builder, nfl_inference_evidence as inputs, research_replay
from scripts.benchmark_drive_history_loading import blocked_network
from test_source_evidence_intake import through_native
from test_source_contract_pipeline import exported
from test_nfl_ui_reblend import ui_caller, STAGE


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def prepared(monkeypatch, tmp_path, stage="raw_model"):
    analysis, _ = through_native(monkeypatch, tmp_path)
    monkeypatch.setattr(inputs, "ui_reblend_time", lambda:STAGE)
    analysis = ui_caller()(analysis)
    output = exported(monkeypatch, tmp_path, analysis)
    receipt = research_replay.retain_export(output["frames"],output["package"],output["card"],output["captured"],path=output["db"])
    meta = json.loads(analysis.iloc[0].ml_estimate_metadata)
    p = meta["nfl_inputs"]["payload"]
    refresh = meta["nfl_ui_reblends"][-1]["payload"]
    reviews, settlements = {}, {}
    for row in analysis.to_dict("records"):
        packet = json.loads(row["ml_estimate_metadata"])["nfl_inputs"]["payload"]
        c = packet["event_offer"];h = inputs.digest(c)
        original = dict(event=c["event"],offer=c["offer"],outcome="WIN",available_at="2026-10-07T03:00:00Z",source_id="SYNTHETIC settlement")
        payload = dict(contract_sha256=h,outcome="WIN",available_at=original["available_at"],original=original)
        settlements[h] = dict(payload=payload,sha256=inputs.digest(payload))
        reviews[h] = dict(contract_sha256=h,review_id="SYNTHETIC independent decision",operator="SYNTHETIC",product="SYNTHETIC",
            canonical_game=dict(event=c["event"],game_id="SYNTHETIC canonical game"),
            settlement_sha256=settlements[h]["sha256"],outcomes_previously_inspected=True,
            out_of_sample=dict(packet_sha256=inputs.digest(packet),game_excluded_from_development=True,predictor_available_at="2026-10-05T00:00:00Z"),
            **{name:"ACCEPTED" for name in ("exact_offer","product_rules","quote_clock","feature_rights","quote_rights","settlement_rights")})
    c = p["event_offer"];game=c["event"]["provider_namespace"]+":"+c["event"]["provider_event_id"]
    spec = dict(export_ids=[receipt["export_id"]],path=output["db"],stage=stage,
        expected_binding=builder.binding(p,stage,refresh),reviews=reviews,settlements=settlements,
        proposed_assignments={game:dict(role="calibration",selected_contract_sha256=inputs.digest(c),groups=dict(season=2026,week=5))})
    return spec, output, analysis


@pytest.mark.parametrize("stage",builder.STAGES)
def test_accepted_software_path_preserves_all_stages_without_fitting(monkeypatch,tmp_path,stage):
    spec, output, analysis = prepared(monkeypatch,tmp_path,stage)
    before = output["db"].read_bytes()
    dataset = builder.build_dataset(**spec)
    report=dataset["report"]
    assert report["observations"]==2 and report["independent_games"]==1 and report["duplicate_offers"]==1
    assert report["proposed_role_counts"]==dict(development=0,calibration=1,validation=0,holdout=0)
    admitted=next(r for r in dataset["observations"] if r["admissibility"]=="SOFTWARE_ADMISSIBLE_PROPOSAL")
    assert admitted["raw_model_probability"]!=admitted["original_blended_probability"]
    assert admitted["validated_ui_refresh_probability"] is not None
    assert admitted["packet"]["feature_order"]==list(inputs.FEATURES)
    assert admitted["settlement"]["payload"]["available_at"]=="2026-10-07T03:00:00Z"
    assert report["approved_assignments"]=={} and not report["fitted"] and not report["scientific_acceptance"]
    assert output["db"].read_bytes()==before
    raw, digest=builder.export_private(dataset)
    assert hashlib.sha256(raw).hexdigest()==digest
    assert b"source_dependencies" in raw and "source_dependencies" not in json.dumps(output["package"])
    assert all(row["production_bet_amount"]==0 for row in output["card"].wager_contract)


@pytest.mark.parametrize("fault,code",[
    ("settlement","VERIFIED_SETTLEMENT_MISSING"),("review","INDEPENDENT_SOURCE_ADMISSIBILITY_REVIEW_MISSING"),
    ("lineage","OUT_OF_SAMPLE_LINEAGE_NOT_DEMONSTRATED"),("future","FUTURE_PREDICTOR"),
    ("mismatch","PREDICTOR_PIPELINE_CONFIGURATION_MISMATCH"),("holdout","EVALUATION_ROLE_CONTAMINATION"),
    ("approved","APPROVED_ROLE_CONFLICT"),("wrong_settlement","SETTLEMENT_EXACT_OFFER_MISMATCH"),
    ("features","PREDICTOR_PIPELINE_CONFIGURATION_MISMATCH"),("rights","SOURCE_ADMISSIBILITY:feature_rights")])
def test_failures_remain_exclusions(monkeypatch,tmp_path,fault,code):
    spec,_,_=prepared(monkeypatch,tmp_path)
    if fault=="settlement":spec["settlements"]={}
    if fault=="review":spec["reviews"]={}
    if fault=="lineage":
        for r in spec["reviews"].values():r["out_of_sample"]={}
    if fault=="future":
        for r in spec["reviews"].values():r["out_of_sample"]["predictor_available_at"]="2026-10-07T00:00:00Z"
    if fault=="mismatch":spec["expected_binding"]["predictor_id"]="OTHER"
    if fault=="features":spec["expected_binding"]["feature_order"].reverse()
    if fault=="holdout":
        for r in spec["proposed_assignments"].values():r.update(role="holdout",outcomes_uninspected=True,custodian_seal="SYNTHETIC",sealed_at="2026-10-05T00:00:00Z")
    if fault=="approved":spec["approved_assignments"]={k:dict(role="development",original_manifest="SYNTHETIC") for k in spec["proposed_assignments"]}
    if fault=="wrong_settlement":
        for r in spec["settlements"].values():r["payload"]["contract_sha256"]="wrong";r["sha256"]=inputs.digest(r["payload"])
    if fault=="rights":
        for r in spec["reviews"].values():r["feature_rights"]="UNKNOWN"
    result=builder.build_dataset(**spec)
    assert code in result["report"]["exclusion_counts"]
    assert not any(result["report"]["proposed_role_counts"].values())


def test_corrupt_private_export_cannot_be_dataset(monkeypatch,tmp_path):
    spec, output,_=prepared(monkeypatch,tmp_path)
    with sqlite3.connect(output["db"]) as db:
        db.execute("DROP TRIGGER immutable_research_replay_exports_UPDATE")
        db.execute("UPDATE research_replay_exports SET payload='{}'")
    with pytest.raises((ValueError,KeyError)):
        builder.build_dataset(**spec)


def test_review_file_is_pinned_and_cli_never_overwrites(monkeypatch,tmp_path):
    spec,_,_=prepared(monkeypatch,tmp_path)
    from scripts.prepare_nfl_calibration_evidence import main
    path=spec.pop("path")
    document=tmp_path/"SYNTHETIC-spec.json";document.write_text(json.dumps(spec))
    digest=hashlib.sha256(document.read_bytes()).hexdigest()
    args=["--database",str(path),"--specification",str(document),"--specification-sha256",digest,"--output",str(tmp_path/"private.json")]
    assert main(args)==0
    with pytest.raises(FileExistsError):main(args)
    with pytest.raises(ValueError,match="REVIEW_ARTIFACT_INTEGRITY"):builder.read_review(document,"0"*64)


@pytest.mark.parametrize("end,quote,blocked",[
    ("2026-10-06T23:00:00Z","2026-10-07T00:00:00Z",False),
    ("2026-10-07T00:00:00Z","2026-10-07T00:00:00Z",True),
    ("2026-10-07T01:00:00Z","2026-10-07T00:00:00Z",True),
    (None,"2026-10-07T00:00:00Z",True),
])
def test_skipped_calibration_and_validation_cannot_hide_chronology(end,quote,blocked):
    rows=[dict(proposed_role="development",settlement=dict(payload=dict(available_at=end)),errors=[]),
          dict(proposed_role="holdout",contract=dict(offer=dict(source_time=quote)),errors=[])]
    builder.partition_barriers(rows)
    assert all(bool(r["errors"]) is blocked for r in rows)
