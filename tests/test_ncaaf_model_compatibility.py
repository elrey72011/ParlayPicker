"""All numerical fixtures/artifacts are SYNTHETIC. Authentic receipts live privately."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from unittest.mock import Mock

import pytest

from app_core import ncaaf_model_compatibility as model
from app_core import ncaaf_compatible_observation as observation
from app_core import ncaaf_research as research, ncaaf_research_contract as v1
from scripts.benchmark_drive_history_loading import blocked_network

AT = "2026-10-09T12:00:00+00:00"


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


@pytest.fixture
def synthetic(monkeypatch):
    binding = deepcopy(model._binding())
    fit = dict(kind="ridge", intercept=7., bias=0., sigma=10., x_mean=[0.]*7,
               x_scale=[1.]*7, coefficients=[0.]*7)
    artifact = dict(protocol=deepcopy(research.PROTOCOL), source_hash=binding["current_components"][
        "app_core/ncaaf_research.py"]["git_blob_sha256"], train_hash=model.digest("SYNTHETIC train"),
        calibration_hash=model.digest("SYNTHETIC calibration"),
        models=dict(ridge=dict(margin=fit, total=dict(fit,intercept=50.))))
    predecessor = dict(schema=1, kind="model", created_at="2026-10-01T00:00:00Z",
        data=dict(artifact=artifact,artifact_hash=model.digest(artifact),
            runtime_hash=binding["historical_runtime"]["source"]["sha256"],policy="SYNTHETIC; no authority"))
    old = model.encode(predecessor)
    binding.update(predecessor_record_sha256=hashlib.sha256(old).hexdigest(),
        artifact_sha256=model.digest(artifact),parameters_sha256=model.digest(artifact["models"]),
        training_sha256=artifact["train_hash"],calibration_sha256=artifact["calibration_hash"])
    binding["recovery"]["source_model_id"] = binding["predecessor_record_sha256"]
    original = dict(schema=1,kind="model",created_at="2026-10-02T00:00:00Z",data=dict(
        predecessor["data"],runtime_hash=binding["historical_runtime"]["recovered"]["sha256"],
        recovery=deepcopy(binding["recovery"])))
    raw = model.encode(original)
    binding["original_record_sha256"] = hashlib.sha256(raw).hexdigest()
    # A private synthetic catalog fixture; production binding is never modified.
    monkeypatch.setattr(model,"_binding",lambda:deepcopy(binding))
    return raw,old,binding


def make_packet(synthetic,monkeypatch,kind="spread_home",line=-3.5,home="LSU",schedule_home="Louisiana State"):
    raw,old,_ = synthetic
    family = "total" if kind.startswith("total") else "spread"
    checked=model.read_model(raw,old,model.envelope(family=family))
    event=dict(sport="NCAAF",provider_namespace="odds_api",provider_event_id="SYNTHETIC-event-9",
        canonical_event_id="cfbd:9",schedule_game_id=9,home_team=home,away_team="McNeese",
        start_utc="2026-10-10T12:00:00Z",neutral_site=False)
    schedule=[dict(id=9,homeId=1,awayId=2,homeTeam=schedule_home,awayTeam="McNeese State",
        startDate=event["start_utc"],season=2026,completed=False,startTimeTBD=False,neutralSite=False)]
    crosswalk=[dict(provider_name=home,schedule_name=schedule_home,team_id=1),
               dict(provider_name="McNeese",schedule_name="McNeese State",team_id=2)]
    review=dict(review_id="SYNTHETIC mapping",reviewed_at="2026-10-09T11:00:00Z",
                binding=observation.mapping_binding(event,schedule,crosswalk))
    monkeypatch.setattr(observation,"ACCEPTED_EVENT_MAPPINGS",{review["review_id"]:model.digest(review)})
    quote=dict(provider_namespace=event["provider_namespace"],provider_event_id=event["provider_event_id"],
        event_home_team=home,event_away_team="McNeese",event_start_utc=event["start_utc"],
        market_type=kind,point=line,price=-110,book="SYNTHETIC binary operator",
        operator="SYNTHETIC binary operator",product="SYNTHETIC sportsbook",listing_id="SYNTHETIC listing",
        recorded_at=AT,period="full_game",period_source="SYNTHETIC period receipt",
        rules="full_game_including_overtime_binary_win_push_loss",rules_source="SYNTHETIC terms")
    source_review=dict(review_id="SYNTHETIC source review",quote_sha256=model.digest(quote),
        operator=quote["book"],product="SYNTHETIC sportsbook",listing_id="SYNTHETIC listing",
        rights_document="SYNTHETIC permissions",settlement=quote["rules"],
        reviewed_at="2026-10-09T11:00:00Z",effective_from="2026-10-01T00:00:00Z",effective_until="2026-11-01T00:00:00Z")
    monkeypatch.setattr(observation,"ACCEPTED_SOURCE_REVIEWS",{source_review["review_id"]:model.digest(source_review)})
    dependencies=[dict(game_id=team*100+i,team_id=team,season=2026,start_utc="2026-09-20T12:00:00Z",
        available_at=AT,kind=k,source_sha256=model.digest(["SYNTHETIC",team,i,k]))
        for team in (1,2) for i in range(3) for k in ("scoring","yardage")]
    p=dict(version=observation.VERSION,evidence_label="SYNTHETIC",model=model.export_model(checked),
        event=event,schedule=schedule,crosswalk=crosswalk,mapping_review=review,quote=quote,as_of=AT,
        features=dict(order=list(research.FEATURES),values=[30.,20.,17.,23.,400.,330.,0.],available_at=AT),
        feature_dependencies=dependencies,source_review=source_review,original_inference_time=None,
        source_acceptance=False,scientific_qualification=False,probability_calibration=False,
        wagering_authority=False,wager_action="PASS",live_stake=0)
    return dict(payload=p,sha256=model.digest(p))


def rehash(p):
    p["sha256"]=model.digest(p["payload"])
    return p


def test_exact_static_reader_and_private_export_keep_bytes(synthetic,monkeypatch):
    raw,old,_=synthetic
    monkeypatch.setattr(research,"centers",Mock(side_effect=AssertionError("No static inference")))
    checked=model.read_model(raw,old,model.envelope(family="spread"))
    exported=model.export_model(checked)
    restored=model.load_model(json.loads(model.encode(exported)))
    assert restored["original_bytes"]==raw and restored["predecessor_bytes"]==old
    assert restored["original_record"]["data"]["recovery"]["production_eligible"] is False
    assert restored["compatibility"]["historical_runtime"]["python"]=="UNKNOWN"
    assert restored["compatibility"]["consumed_reader"]["python"]
    research.centers.assert_not_called()


@pytest.mark.parametrize("change",["recovery","artifact","protocol","training","calibration","policy","runtime","created_at","remove_recovery","other_model"])
def test_mutated_or_borrowed_original_rejected(synthetic,change):
    raw,old,_=synthetic
    altered=json.loads(raw)
    if change=="remove_recovery":del altered["data"]["recovery"]
    elif change=="created_at":altered["created_at"]="2026-10-03T00:00:00Z"
    elif change in {"training","calibration"}:altered["data"]["artifact"][change+"_hash" if change=="calibration" else "train_hash"]="0"*64
    elif change=="protocol":altered["data"]["artifact"]["protocol"]["minimum_prior_games"]=0
    elif change=="artifact":altered["data"]["artifact"]["models"]["ridge"]["margin"]["intercept"]=99.
    elif change=="recovery":altered["data"]["recovery"]["production_eligible"]=True
    elif change=="other_model":altered["data"]["recovery"]["source_model_id"]="1"*64
    elif change=="runtime":altered["data"]["runtime_hash"]="0"*64
    else:altered["data"]["policy"]="altered"
    with pytest.raises(ValueError,match="NCAAF_COMPAT_RECORD_HASH"):
        model.read_model(model.encode(altered),old,model.envelope(family="spread"))


def test_predecessor_cannot_be_replaced(synthetic):
    raw,old,_=synthetic
    with pytest.raises(ValueError,match="RECORD_HASH"):
        model.read_model(raw,old+b" ",model.envelope(family="spread"))


@pytest.mark.parametrize("change",["historical","consumed","calibrated","family","model"])
def test_envelope_is_exact_and_non_authoritative(synthetic,change):
    raw,old,_=synthetic
    e=model.envelope(family="spread")
    if change=="historical":e["historical_runtime"]["python"]="invented"
    elif change=="consumed":e["consumed_reader"]["python"]="invented"
    elif change=="calibrated":e["probability_calibration"]=True
    elif change=="family":e["family"]="winner"
    else:e["model_name"]="constant"
    with pytest.raises(ValueError):model.read_model(raw,old,e)


def test_changed_runtime_and_in_memory_alias_table_rejected(synthetic,monkeypatch):
    raw,old,binding=synthetic
    binding["current_components"]["core/team_mapper.py"]["installed_sha256"]=["0"*64]
    with pytest.raises(ValueError,match="RUNTIME_COMPONENT_CHANGED"):model.envelope(family="spread")
    binding["current_components"]["core/team_mapper.py"]["installed_sha256"]=[hashlib.sha256(
        (model.BINDING_PATH.parents[2]/"core/team_mapper.py").read_bytes()).hexdigest()]
    monkeypatch.setattr(model.identity,"ALIASES",dict(model.identity.ALIASES,lsu="mcneese"))
    with pytest.raises(ValueError,match="ALIAS_TABLE_CHANGED"):model.envelope(family="spread")


def test_original_v1_still_rejects_recovered_schema(synthetic):
    raw,_,_=synthetic
    record=dict(id=hashlib.sha256(raw).hexdigest(),**json.loads(raw))
    with pytest.raises(ValueError,match="NCAAF_MODEL_SCHEMA"):
        v1._artifact(record,datetime(2026,10,9,tzinfo=timezone.utc))


def test_original_v1_runtime_equality_remains_strict(synthetic):
    _,old,_=synthetic
    record=dict(id=hashlib.sha256(old).hexdigest(),**json.loads(old))
    with pytest.raises(ValueError,match="NCAAF_FROZEN_RUNTIME_MISMATCH"):
        v1._artifact(record,datetime(2026,10,9,tzinfo=timezone.utc))


def test_current_unreviewed_dynamic_file_is_unavailable(synthetic,monkeypatch):
    from pathlib import Path
    original=Path.exists
    monkeypatch.setattr(Path,"exists",lambda path: True if path.name=="dynamic_aliases.json" else original(path))
    with pytest.raises(ValueError,match="DYNAMIC_ALIASES_UNREVIEWED"):
        model.envelope(family="spread")


@pytest.mark.parametrize("target",["app_core/ncaaf_research.py","app_core/ncaaf_history.py","app_core/ncaaf_identity.py"])
def test_modified_installed_component_is_not_blanket_compatible(synthetic,monkeypatch,target):
    from pathlib import Path
    original=Path.read_bytes
    monkeypatch.setattr(Path,"read_bytes",lambda path: original(path)+b"\n# unreviewed change\n"
        if path.as_posix().endswith(target) else original(path))
    with pytest.raises(ValueError,match="RUNTIME_COMPONENT_CHANGED"):
        model.envelope(family="spread")


@pytest.mark.parametrize("kind,line",[("spread_home",-3.5),("spread_away",3.5),("total_over",50.5),("total_under",50.5)])
def test_same_synthetic_math_separate_target_and_no_authority(synthetic,monkeypatch,kind,line):
    packet=make_packet(synthetic,monkeypatch,kind,line)
    checked=observation.read_observation(packet)
    feature=dict(zip(research.FEATURES,checked["ordered_features"]))
    total=kind.startswith("total")
    threshold=line if total or kind=="spread_away" else -line
    baseline=json.loads(synthetic[0])["data"]["artifact"]["models"]["ridge"]["total" if total else "margin"]
    expected=research.probabilities(float(research.centers(baseline,[feature],"total" if total else "margin")[0]),baseline["sigma"],threshold,total=total)
    actual=research.probabilities(float(research.centers(checked["fit"],[feature],"total" if total else "margin")[0]),checked["fit"]["sigma"],threshold,total=total)
    assert actual==expected and actual["push"]==0
    assert checked["probability"] is None and checked["original_inference_time"] is None
    assert checked["feature_derivation_verified"] is False and checked["dependency_objects_verified"] is False
    assert checked["wager_action"]=="PASS" and checked["live_stake"]==0
    assert not any(checked[k] for k in ("source_acceptance","scientific_qualification","probability_calibration","wagering_authority"))
    assert checked["original_packet"]==packet


@pytest.mark.parametrize("home,native",[("LSU","Louisiana State"),("McNeese Cowboys","McNeese State Cowboys"),("GardnerWebb","Gardner-Webb")])
def test_reviewed_aliases_need_exact_event_review(synthetic,monkeypatch,home,native):
    # Keep a different opponent to avoid an alias collision in the McNeese case.
    packet=make_packet(synthetic,monkeypatch,home=home,schedule_home=native)
    if home.startswith("McNeese"):
        p=packet["payload"];p["event"]["away_team"]="LSU";p["schedule"][0]["awayTeam"]="Louisiana State"
        p["crosswalk"][1]=dict(provider_name="LSU",schedule_name="Louisiana State",team_id=2)
        p["quote"]["event_away_team"]="LSU"
        p["mapping_review"]["binding"]=observation.mapping_binding(p["event"],p["schedule"],p["crosswalk"])
        p["source_review"]["quote_sha256"]=model.digest(p["quote"])
        observation.ACCEPTED_EVENT_MAPPINGS[p["mapping_review"]["review_id"]]=model.digest(p["mapping_review"])
        observation.ACCEPTED_SOURCE_REVIEWS[p["source_review"]["review_id"]]=model.digest(p["source_review"])
        rehash(packet)
    assert observation.read_observation(packet)["status"]=="COMPATIBLE_INPUT_READER"
    monkeypatch.setattr(observation,"ACCEPTED_EVENT_MAPPINGS",{})
    with pytest.raises(ValueError,match="EVENT_REVIEW_NOT_ACCEPTED"):observation.read_observation(packet)


@pytest.mark.parametrize("change,reason",[
    ("collision","ALIAS_COLLISION"),("ambiguous","EVENT_AMBIGUOUS"),("orientation","EVENT_AMBIGUOUS"),
    ("neutral","EVENT_FACT_CONFLICT"),("event", "EVENT_MISSING"),("quote","QUOTE_MISSING"),
    ("feature_clock","FEATURE_CLOCK"),("future_feature","FEATURE_CLOCK"),
    ("dependency","DEPENDENCIES_MISSING"),("lag","DEPENDENCY_CLOCK"),("minimum","MINIMUM_HISTORY"),
    ("source","SOURCE_REVIEW_MISSING"),("not_accepted","SOURCE_REVIEW_NOT_ACCEPTED"),
    ("integer","INTEGER_PUSH_MODEL_UNVALIDATED"),("authority","AUTHORITY_FORBIDDEN"),
    ("quote_clock","QUOTE_CLOCK"),("period","PERIOD_OR_SETTLEMENT"),("unreviewed_alias","ALIAS_UNREVIEWED")])
def test_unresolved_facts_never_infer_or_promote(synthetic,monkeypatch,change,reason):
    packet=make_packet(synthetic,monkeypatch);p=packet["payload"]
    if change=="collision":p["crosswalk"].append(dict(provider_name="LSU Tigers",schedule_name="Louisiana State Tigers",team_id=99))
    elif change=="ambiguous":p["schedule"].append(deepcopy(p["schedule"][0]))
    elif change=="orientation":p["event"]["home_team"],p["event"]["away_team"]=p["event"]["away_team"],p["event"]["home_team"]
    elif change=="neutral":p["event"]["neutral_site"]=True
    elif change=="event":p["event"]=None
    elif change=="quote":p["quote"]=None
    elif change=="feature_clock":p["features"]["available_at"]=None
    elif change=="future_feature":p["features"]["available_at"]="2026-10-09T12:00:01Z"
    elif change=="dependency":p["feature_dependencies"]=[]
    elif change=="minimum":p["feature_dependencies"]=p["feature_dependencies"][:2]
    elif change=="lag":p["feature_dependencies"][0]["start_utc"]="2026-10-03T12:00:00Z"
    elif change=="source":p["source_review"]=None
    elif change=="not_accepted":monkeypatch.setattr(observation,"ACCEPTED_SOURCE_REVIEWS",{})
    elif change=="integer":p["quote"]["point"]=10.
    elif change=="authority":p["live_stake"]=1
    elif change=="quote_clock":p["quote"]["recorded_at"]=None
    elif change=="period":p["quote"]["period"]=None
    elif change=="unreviewed_alias":p["crosswalk"][0]["provider_name"]="Unreviewed LSU shortcut"
    monkeypatch.setattr(research,"centers",Mock(side_effect=AssertionError("Static reader must never infer")))
    with pytest.raises(ValueError,match=reason):observation.read_observation(rehash(packet))
    research.centers.assert_not_called()


def test_distinct_rangers_islanders_and_no_dynamic_fallback(synthetic):
    from core.team_mapper import normalize_team_name
    assert normalize_team_name("New York Rangers")=="New York Rangers"
    assert normalize_team_name("New York Islanders")=="New York Islanders"
    assert model.current_runtime()["dynamic_aliases"]==dict(status="ABSENT",consumed=False)


def test_reader_never_enrolls_source_or_mapping_review(synthetic,monkeypatch):
    raw,old,_=synthetic
    monkeypatch.setattr(observation,"ACCEPTED_EVENT_MAPPINGS",{})
    monkeypatch.setattr(observation,"ACCEPTED_SOURCE_REVIEWS",{})
    from app_core import ncaaf_pipeline_evidence
    before=deepcopy(ncaaf_pipeline_evidence.ACCEPTED_PACKETS)
    model.read_model(raw,old,model.envelope(family="total"))
    assert observation.ACCEPTED_EVENT_MAPPINGS=={} and observation.ACCEPTED_SOURCE_REVIEWS=={}
    assert ncaaf_pipeline_evidence.ACCEPTED_PACKETS==before
