"""Offline adapter from existing native research records to exact NHL inference.

Reads explicit local evidence only. No source catalog, acquisition, fitting,
calibration, registration, activation or wager consumer is called here.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from app_core import prospective_evidence as evidence, prospective_research_models as model
from app_core import producer_provenance as producer
from app_core.research_estimate_trace import encode, fact, origin_metadata, generated_time
from app_core.odds_research_adapter import identity as native_identity

VERSION = "nhl-selected-puck-line-inputs-v1"
TARGET = dict(version="nhl-selected-full-game-half-point-cover-v1", sport="NHL",
    market_family="PUCK_LINE", target="selected_side_margin_plus_signed_handicap_gt_zero",
    lines=[-1.5,1.5], orientation="named_home_away", period="full_game",
    overtime="included", shootout="official_final_score_included",
    push_probability=0.0, goalie_policy="explicitly_missing_not_consumed_by_score_form_v1")
FEATURE_ORDER = ("intercept", "home_margin", "away_margin", "home_for_minus_away_against",
                 "away_for_minus_home_against", "home_rest_days_div7", "away_rest_days_div7")
_SELECTED = ContextVar("nhl_local_research_selection", default=None)
PUBLIC_REASONS = frozenset("""NHL_LOCAL_RESEARCH_STORE_UNAVAILABLE NHL_TOTAL_REQUIRES_SEPARATE_TARGET_CONTRACT
NHL_EVENT_OR_OFFER_AMBIGUOUS NHL_HALF_POINT_1_5_REQUIRED NHL_FULL_GAME_RULES_UNAVAILABLE
NHL_QUOTE_INFERENCE_START_CLOCK_CONFLICT NHL_CANONICAL_EVENT_AMBIGUOUS NHL_ORIGINAL_EXACT_QUOTE_UNAVAILABLE
NHL_EXACT_PUCK_LINE_MODEL_UNAVAILABLE NHL_ORIGINAL_TEAM_HISTORY_UNAVAILABLE NHL_FEATURE_DEPENDENCY_MISSING
NHL_SOURCE_TARGET_REVIEW_UNAVAILABLE NHL_SOURCE_TARGET_REVIEW_CONFLICT NHL_SOURCE_RIGHTS_OR_SCORE_SEMANTICS_UNAVAILABLE
NHL_SOURCE_REVIEW_CLOCK_CONFLICT NHL_INFERENCE_FAILED NHL_PACKET_INTEGRITY NHL_TARGET_CONTRACT_CHANGED
NHL_MODEL_RUNTIME_MISMATCH NHL_RUNTIME_ARTIFACT_MISMATCH NHL_EVIDENCE_SCHEMA_INVALID NHL_ALTERED_ORDERED_FEATURES
NHL_REPLAY_PROBABILITY_CONFLICT NHL_EVENT_OFFER_CONFLICT NHL_NATIVE_QUOTE_CONFLICT NHL_MODEL_ARTIFACT_CORRUPT
NHL_CANONICAL_PAYLOAD_CORRUPT NHL_SOURCE_DEPENDENCY_CORRUPT NHL_FUTURE_OR_INVALID_FEATURE_DEPENDENCY
NHL_BLEND_CONFIGURATION_CONFLICT NHL_BLEND_OUTPUT_CONFLICT NHL_MODEL_TARGET_CONFLICT
NHL_FEATURE_CONTRACT_INVALID NHL_MODEL_FEATURE_WIDTH_MISMATCH NHL_SELECTED_SIDE_AMBIGUOUS
NHL_AUTHORITY_FORBIDDEN NHL_UNAVAILABLE_NUMERIC_CONFLICT NHL_FULL_GAME_RULES_CONFLICT
NHL_INFERENCE_CLOCK_CONFLICT NHL_FUTURE_MODEL NHL_GOALIE_MISSINGNESS_CONFLICT""".split())


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


@contextmanager
def selected(path, *, source_reviews):
    """An explicit offline caller supplies separate review decisions, never feed flags.

    No real source reviews are bundled or registered. The default path is absent.
    Review keys bind exact canonical event/quote/model hashes and target contract.
    """
    context = dict(path=Path(path), source_reviews=deepcopy(source_reviews))
    token = _SELECTED.set(context)
    try:
        yield
    finally:
        _SELECTED.reset(token)


def selection_requested():
    """The adapter runs only inside an explicitly selected offline context."""
    return _SELECTED.get() is not None


def _runtime():
    from app_core.nfl_inference_evidence import runtime
    return runtime()


def _code():
    root=Path(__file__).resolve().parents[1]
    paths=("app_core/prospective_research_models.py", "app_core/nhl_puck_line_evidence.py",
           "app_core/market_probability_model.py", "core/streamlit_pipeline.py")
    return {p:hashlib.sha256((root/p).read_bytes().replace(b"\r\n",b"\n")).hexdigest() for p in paths}


def review_key(event, quote, artifact):
    return digest(dict(event_hash=event["source_hash"],quote_hash=quote["source_hash"],
                       artifact_hash=digest(artifact),target=TARGET))


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _retained(record):
    result=dict(record)
    if isinstance(result.get("raw_source"),bytes):result["raw_source"]=result["raw_source"].decode("utf-8")
    return result


def _calculate(packet):
    """Replay the installed existing score-distribution reader, never retained code."""
    artifact=packet["artifact"]
    _require(artifact.get("sport")=="NHL" and artifact.get("market_family")=="PUCK_LINE"
             and artifact.get("target")=="final_score_home_margin"
             and artifact.get("protocol")==model.MODEL_VERSION
             and artifact.get("feature_version")==model.FEATURE_VERSION
             and artifact.get("training_target_kind")==model.TARGET_KIND,
             "NHL_MODEL_TARGET_CONFLICT")
    _require(artifact.get("runtime_hash")==model.runtime_hash(), "NHL_MODEL_RUNTIME_MISMATCH")
    vector=packet["features"]
    _require(packet.get("feature_order")==list(FEATURE_ORDER) and len(vector)==len(FEATURE_ORDER)
             and all(type(v) in (int,float) and math.isfinite(v) for v in vector),"NHL_FEATURE_CONTRACT_INVALID")
    fit=artifact["model"]
    _require(all(len(fit[k])==len(vector) for k in ("means","scales","coefficients")),"NHL_MODEL_FEATURE_WIDTH_MISMATCH")
    mean=model._expected(fit,vector)
    side=packet["contract"]["offer"]["side"]
    _require(side in {"home","away"},"NHL_SELECTED_SIDE_AMBIGUOUS")
    # Native model is oriented home-minus-away. Reverse the margin for the away
    # selection, while preserving that selection's own signed handicap.
    selected_mean=mean if side=="home" else -mean
    line=packet["contract"]["offer"]["line"]
    probabilities=model.score_probabilities(selected_mean,fit["residual_sd"],"PUCK_LINE",line)
    return probabilities, selected_mean


def _inputs(source, at):
    context=_SELECTED.get()
    _require(context is not None and context["path"].is_file(),"NHL_LOCAL_RESEARCH_STORE_UNAVAILABLE")
    kind=str(source.get("market_type", ""))
    _require(kind in {"spread_home","spread_away"},"NHL_TOTAL_REQUIRES_SEPARATE_TARGET_CONTRACT")
    contract=producer._offer(source,at)
    event,offer=contract["event"],contract["offer"]
    _require(contract["matched_offer_count"]==1 and event["home"] and event["away"]
             and event["home"]!=event["away"] and event["provider_namespace"]=="odds_api"
             and producer.team(source.get("home_team"),"NHL")==event["home"]
             and producer.team(source.get("away_team"),"NHL")==event["away"],
             "NHL_EVENT_OR_OFFER_AMBIGUOUS")
    # Sticky supplied aliases must agree before any numeric inference. An
    # unordered slate merge may preserve one event's IDs and another's quotes.
    _require(all(not producer.text(source.get(k)) or producer.text(source[k])==v
        for k,v in (("provider_namespace",event["provider_namespace"]),
                    ("provider_event_id",event["provider_event_id"]))),"NHL_EVENT_OR_OFFER_AMBIGUOUS")
    _require(all(not producer.text(source.get(k)) or producer.clock(source[k])==event["start"]
        for k in ("game_start_utc","commence_time_raw")),"NHL_EVENT_OR_OFFER_AMBIGUOUS")
    _require(offer["line"] in TARGET["lines"],"NHL_HALF_POINT_1_5_REQUIRED")
    _require(offer["period"]=="full_game" and bool(offer["rules"]),"NHL_FULL_GAME_RULES_UNAVAILABLE")
    times=[producer.clock(offer["source_time"]),producer.clock(at),producer.clock(event["start"])]
    _require(all(times) and times[0]<=times[1]<times[2],"NHL_QUOTE_INFERENCE_START_CLOCK_CONFLICT")
    events,quotes,scores=model._read_scope(context["path"],"NHL","PUCK_LINE")
    found=[]
    for e in events.values():
        if e["provider_event_id"]!=event["provider_event_id"]:
            continue
        raw=json.loads(e["raw_source"])
        native=native_identity("NHL",raw)
        if (producer.team(native["home"],"NHL")==event["home"]
            and producer.team(native["away"],"NHL")==event["away"]
            and producer.clock(native["start"])==event["start"]
            and e["provider_namespace"]=="THE_ODDS_API"):
            found.append(e)
    _require(len(found)==1,"NHL_CANONICAL_EVENT_AMBIGUOUS")
    e=found[0]
    selection=e["home_team"] if offer["side"]=="home" else e["away_team"]
    found=[q for q in quotes if q["event_id"]==e["event_id"] and q["selection"]==selection
        and q["line"]==offer["line"] and q["american_odds"]==offer["price"]
        and q["sportsbook"]==offer["book"] and producer.clock(q["quote_timestamp"])==offer["source_time"]
        and q["quote_verified"]==1 and evidence.verify_provider_offer(e,q)]
    _require(len(found)==1,"NHL_ORIGINAL_EXACT_QUOTE_UNAVAILABLE")
    q=found[0]
    when=model._time(at)
    m,artifact=model._latest_model(context["path"],"NHL","PUCK_LINE",when)
    _require(m is not None,"NHL_EXACT_PUCK_LINE_MODEL_UNAVAILABLE")
    result=model._features(e,when,events,scores,"PUCK_LINE")
    _require(result is not None,"NHL_ORIGINAL_TEAM_HISTORY_UNAVAILABLE")
    features,lineage=result
    dependencies=[s for s in scores if (s["result_id"],s["source_hash"]) in lineage]
    _require(len(dependencies)==len(lineage),"NHL_FEATURE_DEPENDENCY_MISSING")
    review=context["source_reviews"].get(review_key(e,q,artifact))
    _require(isinstance(review,dict),"NHL_SOURCE_TARGET_REVIEW_UNAVAILABLE")
    expected=dict(event_hash=e["source_hash"],quote_hash=q["source_hash"],artifact_hash=m["artifact_hash"],
        target=TARGET, rules=offer["rules"], period="full_game", overtime=True, shootout=True,
        dependency_hashes=sorted(s["source_hash"] for s in dependencies))
    _require(review.get("binding")==expected,"NHL_SOURCE_TARGET_REVIEW_CONFLICT")
    _require(review.get("rights")=="ACCEPTED_FOR_PRIVATE_RESEARCH" and bool(review.get("review_id"))
        and bool(review.get("operator")) and bool(review.get("product"))
        and review.get("score_semantics")=="full_game_official_final_including_ot_shootout",
        "NHL_SOURCE_RIGHTS_OR_SCORE_SEMANTICS_UNAVAILABLE")
    windows=[producer.clock(review.get(k)) for k in ("effective_from","effective_until","reviewed_at")]
    _require(all(windows) and windows[0]<=times[0] and times[1]<windows[1] and windows[2]<=times[0],
             "NHL_SOURCE_REVIEW_CLOCK_CONFLICT")
    packet=dict(version=VERSION,target_contract=TARGET,contract=contract,event=e,quote=q,
        inference_time=at,feature_observed_at=at,feature_order=list(FEATURE_ORDER),features=features,
        source_dependencies=[_retained(r) for r in dependencies],model_record=_retained(m),artifact=artifact,
        dependency_events=[_retained(events[k]) for k in sorted({s["event_id"] for s in dependencies})],blend=None,
        missingness=dict(home_goalie="UNKNOWN",away_goalie="UNKNOWN",reason="NO_AUTHENTIC_GOALIE_STATUS_FEATURE"),
        source_review=review,runtime=_runtime(),code_hashes=_code(),probabilities=None,
        scientific_acceptance=False,wagering_authority=False)
    packet["event"]=_retained(e);packet["quote"]=_retained(q)
    probabilities,projection=_calculate(packet)
    packet["probabilities"]=probabilities
    return packet,projection


def predict(source):
    """Called at the existing market-probability boundary; unavailable is precise."""
    at=generated_time()
    result=dict(ml_probability=float("nan"),ml_probability_source="",ml_target="spread_cover",
        ml_projection=float("nan"),ml_residual_scale=float("nan"),ml_feature_quality="unavailable",
        ml_inference_status="unavailable",ml_unavailable_reason="",ml_estimate_metadata="")
    packet=None
    try:
        packet,projection=_inputs(source,at)
        result.update(ml_probability=packet["probabilities"]["win"],
            ml_probability_source=model.MODEL_VERSION+":nhl:puck_line:"+packet["model_record"]["artifact_hash"],
            ml_projection=projection,ml_residual_scale=packet["artifact"]["model"]["residual_sd"],
            ml_feature_quality=model.FEATURE_VERSION,ml_inference_status="success")
    except (ValueError,KeyError,TypeError,OverflowError,ZeroDivisionError,OSError) as exc:
        result["ml_unavailable_reason"]=str(exc) if str(exc) in PUBLIC_REASONS else "NHL_INFERENCE_FAILED"
        if result["ml_unavailable_reason"]=="NHL_INFERENCE_FAILED":result["ml_inference_status"]="failed"
    line=producer._line(source)
    raw=origin_metadata(source,result,line,generated_at=at)
    raw,fields=producer.record(source,result,raw,at)
    item=json.loads(raw)
    if packet is None:
        packet=dict(version=VERSION,capture_status="UNAVAILABLE",reason=result["ml_unavailable_reason"],
                    inference_time=at,target_contract=TARGET,scientific_acceptance=False,wagering_authority=False)
    item["nhl_inputs"]={"payload":packet,"sha256":digest(packet)}
    result.update(fields,ml_estimate_metadata=encode(item))
    return result


def diagnose(source, item=None):
    """Validate original facts without consulting a current store or source catalog."""
    errors=[]
    try:
        item=item if item is not None else json.loads(source.get("ml_estimate_metadata") or "{}")
        saved=item["nhl_inputs"];p=saved["payload"]
        _require(set(saved)=={"payload","sha256"} and digest(p)==saved["sha256"],"NHL_PACKET_INTEGRITY")
        _require(p["version"]==VERSION and p["target_contract"]==TARGET,"NHL_TARGET_CONTRACT_CHANGED")
        _require(p["scientific_acceptance"] is False and p["wagering_authority"] is False,"NHL_AUTHORITY_FORBIDDEN")
        if p.get("capture_status")=="UNAVAILABLE":
            _require(item["inference_status"] in {"unavailable","failed"} and fact(source.get("ml_probability"))["state"]!="VALUE",
                     "NHL_UNAVAILABLE_NUMERIC_CONFLICT")
            return dict(status="UNAVAILABLE",errors=[],reason=p["reason"])
        _require(p["contract"]==item.get("producer_contract"),"NHL_EVENT_OFFER_CONFLICT")
        diagnostic=producer.diagnose(source,item)
        _require(not diagnostic["reason"],"NHL_EVENT_OFFER_CONFLICT")
        _require(p["runtime"]==_runtime() and p["code_hashes"]==_code(),"NHL_RUNTIME_ARTIFACT_MISMATCH")
        e,q,m=p["event"],p["quote"],p["model_record"]
        for record in [e,q,m]+p["source_dependencies"]+p["dependency_events"]:
            payload=json.loads(record["payload"])
            _require(digest(payload)==record["payload_hash"] and all(record.get(k)==v for k,v in payload.items() if k in record),"NHL_CANONICAL_PAYLOAD_CORRUPT")
            if "raw_source" in record:
                _require(hashlib.sha256(record["raw_source"].encode()).hexdigest()==record["source_hash"],"NHL_SOURCE_DEPENDENCY_CORRUPT")
        _require(digest(p["artifact"])==m["artifact_hash"] and json.loads(m["payload"])["model_artifact"]==p["artifact"],"NHL_MODEL_ARTIFACT_CORRUPT")
        _require(e["event_id"]==q["event_id"] and evidence.verify_provider_offer(e,q),"NHL_NATIVE_QUOTE_CONFLICT")
        event,offer=p["contract"]["event"],p["contract"]["offer"]
        _require(e["sport"]=="NHL" and e["provider_namespace"]=="THE_ODDS_API"
            and e["provider_event_id"]==event["provider_event_id"]
            and producer.team(e["home_team"],"NHL")==event["home"]
            and producer.team(e["away_team"],"NHL")==event["away"]
            and producer.clock(e["scheduled_start"])==event["start"]
            and q["selection"]==e[offer["side"]+"_team"] and q["market_family"]=="PUCK_LINE"
            and q["line"]==offer["line"] and q["american_odds"]==offer["price"]
            and q["sportsbook"]==offer["book"] and q["quote_verified"]==1
            and producer.clock(q["quote_timestamp"])==offer["source_time"],"NHL_NATIVE_QUOTE_CONFLICT")
        _require(offer["line"] in TARGET["lines"] and offer["period"]=="full_game" and bool(offer["rules"]),"NHL_FULL_GAME_RULES_CONFLICT")
        _require(producer.clock(p["inference_time"])==producer.clock(item["generated_at"]),"NHL_INFERENCE_CLOCK_CONFLICT")
        when=producer.clock(p["inference_time"])
        _require(offer["source_time"]<=producer.clock(e["observed_at"])<=when<event["start"],"NHL_INFERENCE_CLOCK_CONFLICT")
        _require(all(producer.clock(m[k]) is not None and producer.clock(m[k])<=when
            for k in ("training_cutoff","created_at","available_at")),"NHL_FUTURE_MODEL")
        _require(p["missingness"]==dict(home_goalie="UNKNOWN",away_goalie="UNKNOWN",reason="NO_AUTHENTIC_GOALIE_STATUS_FEATURE"),"NHL_GOALIE_MISSINGNESS_CONFLICT")
        history={e["event_id"]:e}
        # Dependencies carry original past event receipts, added at capture below.
        history.update({r["event_id"]:r for r in p["dependency_events"]})
        _require(len(history)==len(p["dependency_events"])+1,"NHL_FUTURE_OR_INVALID_FEATURE_DEPENDENCY")
        for score in p["source_dependencies"]:
            _require(score["event_id"]!=e["event_id"] and score["event_id"] in history
                and evidence.verify_provider_score(history[score["event_id"]],score)
                and producer.clock(score["available_at"])<=when
                and producer.clock(score["observed_at"])<=when
                and producer.clock(history[score["event_id"]]["observed_at"])<=when,"NHL_FUTURE_OR_INVALID_FEATURE_DEPENDENCY")
        rebuilt=model._features(e,model._time(p["inference_time"]),history,p["source_dependencies"],"PUCK_LINE")
        _require(rebuilt is not None and rebuilt[0]==p["features"],"NHL_ALTERED_ORDERED_FEATURES")
        review=p["source_review"]
        expected=dict(event_hash=e["source_hash"],quote_hash=q["source_hash"],artifact_hash=m["artifact_hash"],
            target=TARGET,rules=offer["rules"],period="full_game",overtime=True,shootout=True,
            dependency_hashes=sorted(s["source_hash"] for s in p["source_dependencies"]))
        _require(review.get("binding")==expected,"NHL_SOURCE_TARGET_REVIEW_CONFLICT")
        _require(review.get("rights")=="ACCEPTED_FOR_PRIVATE_RESEARCH" and review.get("review_id")
            and review.get("operator") and review.get("product")
            and review.get("score_semantics")=="full_game_official_final_including_ot_shootout",
            "NHL_SOURCE_RIGHTS_OR_SCORE_SEMANTICS_UNAVAILABLE")
        windows=[producer.clock(review.get(k)) for k in ("effective_from","effective_until","reviewed_at")]
        qt=producer.clock(q["quote_timestamp"])
        _require(all(windows) and windows[0]<=qt and producer.clock(p["inference_time"])<windows[1] and windows[2]<=qt,"NHL_SOURCE_REVIEW_CLOCK_CONFLICT")
        probabilities,_=_calculate(p)
        _require(probabilities==p["probabilities"] and fact(probabilities["win"])==item["probability"]
            and item["probability"]==fact(source.get("ml_probability")) and item["inference_status"]=="success",
            "NHL_REPLAY_PROBABILITY_CONFLICT")
        if p["blend"] is not None:
            from app_core.nfl_inference_evidence import consumed_blend
            from core.streamlit_pipeline import compute_blended_probability
            import pandas as pd
            b=p["blend"]
            _require(b["consumed"]==consumed_blend(),"NHL_BLEND_CONFIGURATION_CONFLICT")
            # Existing pipeline bounds are preserved separately from raw inference.
            # Unsupported injury/goalie/context transforms cannot be authenticated
            # by this score-form contract and therefore fail this replay check.
            _require(b["preblend_transform"]=="existing_pipeline_probability_bounds_0_01_0_99"
                and b["inputs"]["p_ml"]==fact(min(.99,max(.01,probabilities["win"]))),"NHL_BLEND_OUTPUT_CONFLICT")
            probability=compute_blended_probability(**{k:pd.Series([v.get("value")],dtype="float64") for k,v in b["inputs"].items()},
                league=pd.Series(["NHL"]),market_type=pd.Series([offer["market"]])).iloc[0]
            _require(fact(probability)==b["probability"] and b["probability"]==fact(source.get("calibrated_probability")),"NHL_BLEND_OUTPUT_CONFLICT")
    except (ValueError,KeyError,TypeError,AttributeError,OSError,OverflowError,ZeroDivisionError) as exc:
        errors.append(str(exc) if isinstance(exc,ValueError) else "NHL_EVIDENCE_SCHEMA_INVALID")
    return dict(status="REJECTED" if errors else "COMPLETE",errors=errors,reason=errors[0] if errors else "")


def replay(source):
    assessment=diagnose(source)
    if assessment["status"]!="COMPLETE":
        raise ValueError(encode(assessment))
    packet=json.loads(source["ml_estimate_metadata"])["nhl_inputs"]["payload"]
    probabilities,projection=_calculate(packet)
    return dict(probabilities=probabilities,selected_margin=projection,scientific_acceptance=False,wagering_authority=False)


def blend_inputs(frame, inputs):
    from app_core.nfl_inference_evidence import consumed_blend
    for index,row in frame.iterrows():
        try:item=json.loads(row.get("ml_estimate_metadata") or "{}")
        except (ValueError,TypeError):continue
        if "nhl_inputs" not in item:continue
        packet=item["nhl_inputs"]["payload"]
        if packet.get("capture_status")=="UNAVAILABLE":continue
        packet["blend"]=dict(consumed=consumed_blend(),inputs={k:fact(v.loc[index]) for k,v in inputs.items()},
            probability=None,semantics="market_context_research_blend_not_scoped_calibration",
            preblend_transform="existing_pipeline_probability_bounds_0_01_0_99")
        item["nhl_inputs"]["sha256"]=digest(packet)
        frame.at[index,"ml_estimate_metadata"]=encode(item)


def finish(frame):
    for index,row in frame.iterrows():
        try:item=json.loads(row.get("ml_estimate_metadata") or "{}")
        except (ValueError,TypeError):continue
        if "nhl_inputs" not in item:continue
        packet=item["nhl_inputs"]["payload"]
        if packet.get("blend") is None:continue
        packet["blend"]["probability"]=fact(row.get("calibrated_probability"))
        item["nhl_inputs"]["sha256"]=digest(packet)
        frame.at[index,"ml_estimate_metadata"]=encode(item)
    return frame
