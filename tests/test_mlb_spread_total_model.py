import json
from copy import deepcopy
from datetime import timedelta
import pandas as pd
import pytest
from app_core import mlb_spread_total_model as m
from test_mlb_spread_total_training import trained, receipt


def future_receipt(bundle, market="spread_home"):
    r = receipt(2,market); p=r["payload"]
    now=m.timestamp(bundle["manifest"]["model_available_at"])+timedelta(minutes=1)
    p.update(captured_at=now.isoformat(),prediction_cutoff=now.isoformat(),
             game_start_utc=(now+timedelta(hours=1)).isoformat(),season=now.year)
    p["quote"]["observed_at"]=now.isoformat()
    for game in p["prior_games"]: game["season"]=now.year
    r["sha256"]=m.digest(p)
    return r,now


@pytest.mark.parametrize("market", m.TARGETS)
def test_inference_provenance_and_no_promotion(trained,market):
    bundle=m.load_mlb_spread_total_model(trained)
    r,now=future_receipt(bundle,market)
    result=m.predict_mlb_spread_total(bundle,r,prediction_generated_at=now.isoformat())
    for k in ("model_version","model_trained_through","model_available_at","training_cutoff_basis"):
        assert result[k]==bundle["manifest"][k]
    assert 0<result["probability"]<1
    assert result["production_eligible"] is False and result["production_bet_amount"]==0
    assert result["probability_semantics"]=="win_conditional_on_decision"
    with pytest.raises(ValueError):
        m.predict_mlb_spread_total(bundle,r,prediction_generated_at=(now-timedelta(days=1)).isoformat())
    with pytest.raises(ValueError):
        m.predict_mlb_spread_total(bundle,r,prediction_generated_at=r["payload"]["game_start_utc"])
    with pytest.raises(ValueError):
        m.predict_mlb_spread_total(bundle,r,prediction_generated_at=now.isoformat(),calibration={"calibration_probability_field":"selection_probability_used"})


@pytest.mark.parametrize("field", ["bytes","feature_schema_version","probability_semantics","model_available_at","production_eligible"])
def test_loader_tamper(trained,field):
    if field=="bytes":
        with (trained/"estimators.json").open("ab") as f:f.write(b" ")
    else:
        p=trained/"manifest.json";v=json.loads(p.read_text());v[field]="wrong";p.write_text(json.dumps(v))
    with pytest.raises(ValueError):m.load_mlb_spread_total_model(trained)


def test_missing_and_mismatched_receipt_never_supply_model_provenance(trained,tmp_path):
    frame=pd.DataFrame([{"league":"MLB","market_type":"spread_home","model_probability":.61}])
    out=m.attach_challenger(frame,model_path=tmp_path/"missing")
    assert out.iloc[0].mlb_challenger_status=="ARTIFACT_UNVERIFIED"
    out=m.attach_challenger(frame,model_path=trained)
    assert out.iloc[0].mlb_challenger_status=="PREGAME_EVIDENCE_UNAVAILABLE"
    assert "model_version" not in out
    assert out.iloc[0].model_probability==.61


def test_challenger_real_private_projection_snapshot(trained,tmp_path,monkeypatch):
    from test_candidate_authority_projection import source,build
    from app_core import prediction_evidence as pe
    from core.prospective_uncertainty import prepare_live
    from core.live_wager_contract import finalize_live_wagers
    bundle=m.load_mlb_spread_total_model(trained)
    r,now=future_receipt(bundle)
    row,_,_=source(tmp_path)
    for field in ("model_version","model_available_at","model_trained_through","calibration_version","calibration_available_at"):
        row.pop(field,None)
    p=r["payload"]
    row.update({k:p[k] for k in ("provider_namespace","provider_event_id","home_team_id","away_team_id","game_start_utc")})
    row.update(sport="MLB",league="MLB",line=-1.5,spread_line=-1.5,market_line_used=-1.5,
        start=p["game_start_utc"],mlb_pregame_receipt=r,game_date=p["game_start_utc"][:10],model_validated=False,
        calibration_validated=False,production_eligible=False,production_bet_amount=0,
        live_spread_line=-1.5,quote_time=now.isoformat(),
        selection='Indianapolis Colts -1.5',best_pick='Indianapolis Colts -1.5',
        provider_quotes=json.dumps([dict(book='draftkings',market_type='spread_home',point=-1.5,
            price=-110,recorded_at=now.isoformat(),provider_namespace=p['provider_namespace'],
            provider_event_id=p['provider_event_id'])]))
    monkeypatch.setenv("PARLAYPICKER_MLB_CHALLENGER_MODEL",str(trained))
    monkeypatch.setattr(m,"utcnow",lambda:now)
    monkeypatch.setattr('app_core.candidate_chronology.now_utc',lambda:pd.Timestamp(now))
    best,diag=build([row],monkeypatch)
    private=diag["candidate_authority_df"]
    result=json.loads(private.iloc[0].mlb_challenger_result)
    assert result["model_version"]==bundle["manifest"]["model_version"]
    assert pd.isna(private.iloc[0].get("model_version"))
    prepared=prepare_live(private,database=tmp_path/"prior.db",plan_dir=tmp_path/"plans",now=now)
    final,_=finalize_live_wagers(prepared,best,1000,now=now,policies={},reviews=prepared)
    assert final.production_bet_amount.eq(0).all()
    root=tmp_path/"root";root.mkdir();db=tmp_path/"snapshot.db"
    monkeypatch.setattr(pe,"now_utc",lambda:now.isoformat())
    context=pe.begin_run({},path=db,root=root)
    pe.capture_run(context,prepared,final,pd.DataFrame([row]),path=db,authoritative_candidates=True)
    _,saved,_=pe.load_snapshots(db)[0]
    assert json.loads(saved.iloc[0].mlb_challenger_result)==result
    assert pd.isna(saved.iloc[0].model_version)


def test_exact_complementary_targets(trained):
    bundle=m.load_mlb_spread_total_model(trained)
    values={}
    for market in m.TARGETS:
        r,now=future_receipt(bundle,market)
        values[market]=m.predict_mlb_spread_total(bundle,r,prediction_generated_at=now.isoformat())["probability"]
    assert values["spread_home"]+values["spread_away"]==pytest.approx(1)
    assert values["total_over"]+values["total_under"]==pytest.approx(1)


@pytest.mark.parametrize("change", ["schema", "semantics", "cutoff"])
def test_rehashed_invalid_manifest_still_fails(trained,change):
    bundle=m.load_mlb_spread_total_model(trained)
    manifest=bundle["manifest"]
    if change=="schema":manifest["feature_schema_version"]="unknown"
    if change=="semantics":manifest["probability_semantics"]="moneyline_win"
    if change=="cutoff":manifest["model_trained_through"]="2099-01-01T00:00:00Z"
    manifest["model_version"]=m.model_version(manifest)
    with pytest.raises(ValueError):m.validate_manifest(manifest)


def test_model_bytes_not_changed_by_inference(trained):
    before={p.name:p.read_bytes() for p in trained.iterdir()}
    bundle=m.load_mlb_spread_total_model(trained)
    r,now=future_receipt(bundle)
    m.predict_mlb_spread_total(bundle,r,prediction_generated_at=now.isoformat())
    assert before=={p.name:p.read_bytes() for p in trained.iterdir()}


def test_malformed_manifest_does_not_crash_analysis(trained):
    (trained/"manifest.json").write_text("[]")
    result=m.attach_challenger(pd.DataFrame([{"league":"MLB"}]),model_path=trained)
    assert result.iloc[0].mlb_challenger_status=="ARTIFACT_UNVERIFIED"


@pytest.mark.parametrize("receipt", [None, [], {"payload": []}])
def test_malformed_receipt_rejected(receipt):
    with pytest.raises(ValueError):m.receipt_features(receipt)


@pytest.mark.parametrize("market", m.TARGETS)
def test_tied_final_is_not_a_betting_push(market):
    with pytest.raises(ValueError):m.label(market,0,3,3,"FINAL")
    assert m.label(market,0,None,None,"VOID")=="VOID"
    assert m.label("spread_home",-2,5,3)=="PUSH"
    assert m.label("total_over",8,5,3)=="PUSH"
    assert m.label("spread_home",-1.5,5,3)=="WIN"
