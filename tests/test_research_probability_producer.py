"""Synthetic retained forecasts through real producer/capture/export projections."""
import json
import socket
from copy import deepcopy
from io import StringIO

import pandas as pd
import pytest

from app_core import prediction_evidence as evidence
from app_core.candidate_evidence_schema import project
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package, validate_package
from core.live_wager_contract import finalize_live_wagers
from core.streamlit_pipeline import build_best_picks_df, _coerce_export_to_canonical
from scripts.publish_board import render, assets_from_html
from test_research_probability_display import NOW, ANALYSIS, QUOTE, START, FrozenDateTime
from test_candidate_chronology_hotfix import quote_row


def forecast(**changes):
    # Authentic-shaped fake transport record; metadata is supplied before exports.
    row=quote_row("Home","Away",START,QUOTE,"MLB",odds=-110,line=-1.5,
                  source="score-distribution-v1:mlb")
    row.update(game_date="2026-10-01",game_time_est=START,candidate_id="source-candidate",
        quote_id="source-quote",market_period="full_game",settlement_rules="includes_overtime",
        ml_target="spread_cover",probability_semantics="win_unconditional_with_push",
        push_probability=0.0,market_push_probability=0.0,model_status="success",
        inference_status="success",prediction_generated_at=ANALYSIS,calibrated_probability=.6,
        model_probability=.6,ml_probability=.6,expected_value=.6*(1+100/110)-1,
        secret_unrelated_payload="MUST-NOT-LEAK")
    row.update(changes)
    return row


def real_path(monkeypatch,tmp_path,raw):
    def blocked(*args,**kwargs):
        raise AssertionError("Real transport is prohibited in producer regression")
    monkeypatch.setattr(socket.socket,"connect",blocked)
    monkeypatch.setattr("app_core.candidate_chronology.now_utc",lambda:pd.Timestamp(NOW))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats",lambda:{})
    monkeypatch.setattr("core.probability_calibration.load_calibration",lambda:None)
    monkeypatch.setattr(evidence,"now_utc",lambda:ANALYSIS)
    monkeypatch.setattr("app_core.public_board.datetime",FrozenDateTime)
    diagnostics={}
    best=build_best_picks_df(pd.DataFrame([raw]),diagnostics_out=diagnostics)
    authority=diagnostics["candidate_authority_df"]
    # No policy or exposure authority: real terminal gate supplies the null PASS
    # contract. No fixture is injected after the producer export boundary.
    final,_=finalize_live_wagers(project(authority),best,1000,now=NOW,policies={},config={})
    db=tmp_path/"isolated-evidence.sqlite3"
    root=tmp_path/"isolated-root";root.mkdir()
    context=evidence.begin_run({},path=db,root=root)
    captured,card=evidence.capture_run(context,project(authority),final,pd.DataFrame([raw]),
        path=db,authoritative_candidates=True)
    # Match the application's preferred captured-authority publication frame.
    frames=[per_game_board(card,captured,family=f,novig_only=True) for f in ("overall","sides","totals")]
    package=build_package(*frames);validate_package(package)
    html=render(package)
    serialized=json.loads(assets_from_html(html)["board-data.json"])
    validate_package(serialized)
    return dict(best=best,reporting=diagnostics["candidate_audit_df"],authority=authority,
        captured=captured,card=card,frames=frames,package=serialized,html=html)


def assert_pass(result):
    for view in ("overall","sides"):
        row=result["package"]["games"][view][0]
        assert row["status"]=="PASS"
        contract=row.get("wager_contract")
        if contract is not None:
            assert row["win_estimate"] is None and row["ev"] is None
            assert contract["conservative_probability"] is None
            assert contract["conservative_ev"] is None
            assert contract["production_bet_amount"]==0
        else:
            # A contradictory line prevents exact contract attachment. Existing
            # legacy fields confer no authority and the research display rejects it.
            assert result["card"].iloc[0].wager_contract["production_bet_amount"]==0


def test_normal_producer_capture_exports_preserve_genuine_provenance(monkeypatch,tmp_path):
    raw=forecast(); original=deepcopy(raw)
    result=real_path(monkeypatch,tmp_path,raw)
    for stage in ("best","reporting","authority","captured","card"):
        frame=result[stage]
        assert frame.columns.is_unique,stage
        row=frame.iloc[0]
        for field in ("quote_id","market_period","settlement_rules","spread_line",
                      "push_probability","inference_status","model_status","ml_target"):
            assert row[field]==raw[field],(stage,field)
        assert "secret_unrelated_payload" not in frame
        assert "MUST-NOT-LEAK" not in row["research_source_semantics"]
    for frame in result["frames"][:2]:
        exported=pd.read_csv(StringIO(frame.to_csv(index=False)))
        assert exported.iloc[0].quote_id=="source-quote"
        assert exported.iloc[0].line==-1.5
    assert_pass(result)
    row=result["package"]["games"]["overall"][0]
    assert row["research_display"]["probability"]==.6
    assert row["research_display"]["ev"]==pytest.approx(.6*(1+100/110)-1)
    assert row["research_display"]["label"]=="Research estimate"
    assert raw==original
    from test_research_probability_browser import inspect_browser
    browser=inspect_browser(result["package"],tmp_path/"browser",NOW,rendered_html=result["html"])
    assert browser["initial"]["shown"][0]["probability"]==.6
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert browser["initial"]["saved"]==[dict(probability=None,ev=None,status="PASS",stake=0)]
    assert browser["expired"]["current"]==browser["started"]["current"]==0


@pytest.mark.parametrize("changes,reason",[
    ({"quote_id":""},"ESTIMATE_PROVENANCE_NOT_RECORDED"),
    ({"market_period":"","period":""},"ESTIMATE_PROVENANCE_NOT_RECORDED"),
    ({"settlement_rules":""},"ESTIMATE_PROVENANCE_NOT_RECORDED"),
    ({"market_period":"full_game","period":"first_half"},"ESTIMATE_IDENTITY_MISMATCH"),
    ({"quote_id":"source-quote","prospective_quote_id":"unrelated-quote"},"ESTIMATE_IDENTITY_MISMATCH"),
    ({"market_line_used":-2.5},"ESTIMATE_IDENTITY_MISMATCH"),
    ({"inference_status":"FAILED"},"INFERENCE_FAILED"),
    ({"inference_status":"success","model_status":"FAILED"},"INFERENCE_FAILED"),
    ({"inference_status":"UNAVAILABLE"},"INFERENCE_UNAVAILABLE"),
    ({"probability_semantics":"unsupported","push_probability":0.0},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability":.1},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability":float("nan")},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability":float("inf")},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability":True},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability":False},"UNSUPPORTED_PROBABILITY_SEMANTICS"),
])
def test_real_projection_never_repairs_missing_conflicting_or_failed_source(monkeypatch,tmp_path,changes,reason):
    result=real_path(monkeypatch,tmp_path,forecast(**changes))
    assert_pass(result)
    display=result["package"]["games"]["overall"][0]["research_display"]
    assert display["probability"] is None and display["ev"] is None
    assert display["availability_reason"]==reason
    if changes.get("probability_semantics")=="unsupported":
        # The protected capture's authority canonicalization is intentionally
        # untouched; its output cannot erase evidence of unsupported input.
        assert result["captured"].iloc[0].probability_semantics!="unsupported"
        assert "unsupported" in result["captured"].iloc[0].research_source_semantics


def test_canonical_retained_input_projection_preserves_only_supplied_metadata(monkeypatch):
    raw=forecast(push_probability=float("nan"),inference_status="FAILED")
    # Exercise the real upload/retained-export normalization, not a selected list.
    frame=_coerce_export_to_canonical(pd.DataFrame([raw]),["MLB"])
    for field in ("quote_id","market_period","settlement_rules","inference_status",
                  "prediction_generated_at","game_start_utc","provider_quotes"):
        assert frame.iloc[0][field]==raw[field]
    assert "secret_unrelated_payload" not in frame
    saved=json.loads(frame.iloc[0].research_source_semantics)
    assert saved["fields"]["push_probability"]=={"state":"INVALID"}
    assert saved["fields"]["inference_status"]=={"state":"VALUE","value":"FAILED"}
    absent=forecast()
    for field in ("quote_id","market_period","settlement_rules"):
        absent.pop(field)
    frame=_coerce_export_to_canonical(pd.DataFrame([absent]),["MLB"])
    for field in ("quote_id","market_period","settlement_rules"):
        assert pd.isna(frame.iloc[0][field])


@pytest.mark.parametrize("push,semantics,available",[
    (0.0,"win_unconditional_with_push",True),
    (None,"",True),
    (.1,"win_unconditional_with_push",False),
    (None,"win_unconditional_with_push",False),
    (0.0,"unsupported",False),
    (float("nan"),"",False),
    (float("inf"),"",False),
    (True,"",False),
    (False,"",False),
])
def test_non_probability_first_and_direct_exports_validate_original_semantics(monkeypatch,tmp_path,push,semantics,available):
    from test_research_probability_display import source,package_for
    # The non-probability-first exporter does not retain push semantics in its
    # traditional fields. The additive capture must independently inspect source.
    raw=source(best_available_selection_policy="",production_win_probability=.6,
        production_expected_value=.6*(1+100/110)-1,push_probability=push,
        probability_semantics=semantics)
    frames,package=package_for(monkeypatch,raw)
    display=package["games"]["overall"][0]["research_display"]
    assert (display["probability"] is not None)==available
    assert (display["ev"] is not None)==available
    assert package["games"]["overall"][0]["win_estimate"] is None
    assert package["games"]["overall"][0]["status"]=="PASS"
    # A direct legacy export carries its own original probability contract.
    # This is input to the real builder/validator/serialization/browser route.
    direct=frames[0].copy()
    direct=direct.drop(columns=["research_display"])
    direct["push_probability"]=push
    direct["probability_semantics"]=semantics
    direct_package=build_package(direct,frames[1],frames[2])
    if push is True:
        # Existing strict price-display metadata rejects boolean push contracts.
        # Do not weaken that validator just to render an invalid direct package.
        with pytest.raises(ValueError,match="Invalid saved display push probability"):
            validate_package(direct_package)
        return
    validate_package(direct_package)
    html=render(direct_package)
    serialized=json.loads(assets_from_html(html)["board-data.json"])
    validate_package(serialized)
    display=serialized["games"]["overall"][0]["research_display"]
    assert (display["probability"] is not None)==available
    assert (display["ev"] is not None)==available
    if not available:
        assert display["availability_reason"]=="UNSUPPORTED_PROBABILITY_SEMANTICS"
    from test_research_probability_browser import inspect_browser
    browser=inspect_browser(serialized,tmp_path/"direct-browser",NOW,rendered_html=html)
    assert (browser["initial"]["shown"][0]["probability"] is not None)==available
    assert browser["initial"]["current"]==browser["initial"]["top"]==0
    assert browser["initial"]["saved"]==[dict(probability=None,ev=None,status="PASS",stake=0)]


def test_original_carrier_cannot_override_a_later_explicit_push_contradiction(monkeypatch):
    from app_core.research_display import preserve_source_semantics
    from test_research_probability_display import source,package_for
    raw=source()
    captured=preserve_source_semantics(pd.DataFrame([raw])).iloc[0].to_dict()
    captured["push_probability"]=.1
    _,package=package_for(monkeypatch,captured)
    display=package["games"]["overall"][0]["research_display"]
    assert display["probability"] is None and display["ev"] is None
    assert display["availability_reason"]=="UNSUPPORTED_PROBABILITY_SEMANTICS"
