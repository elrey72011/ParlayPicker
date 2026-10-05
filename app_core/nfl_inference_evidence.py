"""Private extension of the existing inference metadata/replay; never authority."""
from __future__ import annotations
import base64
import hashlib
import json
import marshal
import types
import platform
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from app_core.research_estimate_trace import fact, encode
from app_core.producer_provenance import clock

VERSION = "nfl-inference-inputs-v1"
FEATURES = ("feature_home_ppg", "feature_away_ppg", "feature_home_oppg", "feature_away_oppg",
    "feature_home_games_played", "feature_away_games_played", "feature_home_win_pct",
    "feature_away_win_pct", "feature_diff_last5", "feature_home_recent_point_margin",
    "feature_away_recent_point_margin")
REQUIRED = FEATURES[:6]
ROOT = Path(__file__).resolve().parents[1]
BLEND_WEIGHTS = ("KALSHI_WEIGHT", "MARKET_WEIGHT", "ML_MODEL_WEIGHT", "THEOVER_WEIGHT", "SENTIMENT_WEIGHT",
                "FALLBACK_MARKET_WEIGHT", "FALLBACK_ML_WEIGHT", "FALLBACK_THEOVER_WEIGHT", "FALLBACK_SENTIMENT_WEIGHT")
ARTIFACTS = ("app_core/market_probability_model.py", "app_core/weights_config.py")


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def artifacts():
    return {name: {"sha256": hashlib.sha256(raw).hexdigest(),
                   "bytes_base64": base64.b64encode(raw).decode()}
            for name in ARTIFACTS for raw in [(ROOT/name).read_bytes().replace(b"\r\n", b"\n")]}


def callable_digest(fn):
    def logical(code):
        return code.replace(co_filename="retained-logical-module",co_consts=tuple(logical(x) if isinstance(x,types.CodeType) else x for x in code.co_consts))
    return hashlib.sha256(marshal.dumps(logical(fn.__code__))).hexdigest()


def predictor_callables():
    from app_core import market_probability_model as model
    return {name:callable_digest(getattr(model,name)) for name in
            ("predict_market_probabilities","_numeric","_text","_normal_cdf","_unscaled_scoring_stat")}


def consumed_blend():
    from core import streamlit_pipeline as pipeline
    return dict(callable_sha256=callable_digest(pipeline.compute_blended_probability),
                weights={name:getattr(pipeline,name) for name in BLEND_WEIGHTS})


def runtime():
    return dict(python=platform.python_version(), implementation=sys.implementation.name,
                numpy=np.__version__, pandas=pd.__version__, platform=platform.platform())


def configuration():
    from app_core.market_probability_model import _LEAGUE_PARAMS
    from app_core import weights_config
    return dict(score_parameters=dict(_LEAGUE_PARAMS["NFL"]),
        blend_weights={k:v for k,v in vars(weights_config).items()
                      if k.isupper() and type(v) in (str, int, float, bool, type(None))})


def _begin(source, result, metadata):
    """Record actual input states at the consumed score predictor boundary."""
    from app_core.research_replay import cell
    item=json.loads(metadata)
    line=item.get("line", {}).get("value")
    if str(source.get("league", source.get("League", ""))).upper() != "NFL" or str(source.get("market_type")) not in {"spread_home", "spread_away"} or line is None or abs(line % 1) != .5:
        return metadata
    try:
        dependencies=json.loads(source.get("nfl_feature_dependencies") or "null")
        encode(dependencies)  # Nonfinite JSON facts must not abort numerical inference.
    except (ValueError, TypeError):
        dependencies={"invalid": True}
    contract=item.get("producer_contract")
    packet=dict(version=VERSION, feature_order=list(FEATURES),
        features={k:fact(source.get(k)) for k in FEATURES},
        eligibility={k:cell(source.get(k)) for k in ("ml_feature_eligible", "stats_resolution_status")},
        eligibility_present=[k for k in ("ml_feature_eligible", "stats_resolution_status") if k in source],
        # Observation and upstream availability are different facts. Never invent the latter.
        observation_receipt=source.get("football_feature_receipt") if isinstance(source.get("football_feature_receipt"),str) else None,
        source_dependencies=dependencies,
        event_offer=contract if contract is not None else None,
        inference_time=item["generated_at"], inference_status=result.get("ml_inference_status"),
        predictor_id=result.get("ml_probability_source"), target=result.get("ml_target"),
        raw_probability=fact(result.get("ml_probability")),
        raw_semantics=item["probability_semantics"], push_probability=item["push_probability"],
        runtime=runtime(), predictor_callables=predictor_callables(), configuration=configuration(), artifacts=artifacts(),
        fitted_artifact="NOT_APPLICABLE: consumed fixed score-distribution code, not the parallel home-win classifier",
        blend=None, scientific_acceptance=False, wagering_authority=False)
    item["nfl_inputs"]={"payload":packet,"sha256":digest(packet)}
    return encode(item)


def begin(source, result, metadata):
    try:
        return _begin(source, result, metadata)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        # Retention failure is a display/replay rejection, never a numerical fallback.
        item=json.loads(metadata)
        payload=dict(version=VERSION,capture_status="FAILED",capture_errors=["inference_inputs:"+type(exc).__name__])
        item["nfl_inputs"]={"payload":payload,"sha256":digest(payload)}
        return encode(item)


def _blend_inputs(frame, inputs):
    """Record exactly the arguments passed to the existing blend, without editing them."""
    for index,row in frame.iterrows():
        try:
            item=json.loads(row.get("ml_estimate_metadata", ""))
        except (ValueError, TypeError):
            continue
        if "nfl_inputs" not in item: continue
        packet=item["nfl_inputs"]["payload"]
        if packet.get("capture_status")=="FAILED":continue
        packet["blend"]={"consumed":consumed_blend(),"inputs":{k:fact(v.loc[index]) for k,v in inputs.items()},
            "probability":None, "estimated_ev":None,
            "probability_semantics":"market_context_research_blend_not_scoped_calibration",
            "ev_semantics":"recorded_binary_price_estimate_not_certified_operator_payoff",
            "pipeline_sha256":hashlib.sha256((ROOT/"core/streamlit_pipeline.py").read_bytes().replace(b"\r\n",b"\n")).hexdigest()}
        item["nfl_inputs"]["sha256"]=digest(packet)
        frame.at[index,"ml_estimate_metadata"]=encode(item)


def blend_inputs(frame, inputs):
    try:
        _blend_inputs(frame, inputs)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        for index,row in frame.iterrows():
            try:item=json.loads(row.get("ml_estimate_metadata", ""))
            except (ValueError,TypeError):continue
            if "nfl_inputs" not in item:continue
            p=dict(version=VERSION,capture_status="FAILED",capture_errors=["blend_inputs:"+type(exc).__name__])
            item["nfl_inputs"]={"payload":p,"sha256":digest(p)}
            frame.at[index,"ml_estimate_metadata"]=encode(item)


def finish(frame):
    for index,row in frame.iterrows():
        try:
            item=json.loads(row.get("ml_estimate_metadata", ""))
        except (ValueError, TypeError):
            continue
        if "nfl_inputs" not in item: continue
        packet=item["nfl_inputs"]["payload"]
        if packet.get("capture_status")=="FAILED":continue
        if packet["blend"] is not None:
            packet["blend"].update(probability=fact(row.get("calibrated_probability")),
                                    estimated_ev=fact(row.get("expected_value")))
        item["nfl_inputs"]["sha256"]=digest(packet)
        frame.at[index,"ml_estimate_metadata"]=encode(item)
    return frame


ELIGIBILITY = ("ml_feature_eligible", "stats_resolution_status")
FEATURE_SCOPE = "nfl-score-feature-scope-v1"


def _eligibility_binding(source, packet, errors, unknown):
    """Compare original consumed cells, not truthy replacements or defaults."""
    from app_core.research_replay import cell, value
    retained = packet["eligibility"]
    present = packet["eligibility_present"]
    if set(retained) != set(ELIGIBILITY) or len(present) != len(set(present)):
        errors.append("eligibility.schema")
        return
    for name in ELIGIBILITY:
        original = retained[name]
        if name not in present or fact(value(original))["state"] != "VALUE":
            unknown.append("eligibility.original:" + name)
        current = cell(source.get(name))
        if name not in source or fact(source.get(name))["state"] != "VALUE":
            unknown.append("eligibility.current:" + name)
        elif name not in present or encode(original) != encode(current):
            errors.append("eligibility.source_conflict:" + name)
    status = packet.get("inference_status")
    if fact(status)["state"] != "VALUE" or status == "unknown":
        unknown.append("origin.inference_status")
    elif status != "success" and packet["raw_probability"].get("state") == "VALUE":
        errors.append("origin.non_success_numeric_output")
    if status == "success":
        # The existing scoring gate's exact normalization; no prediction change.
        frame = pd.DataFrame([{k: value(retained[k]) for k in present}])
        if "ml_feature_eligible" in frame and not frame["ml_feature_eligible"].astype("string").str.lower().str.strip().isin({"true", "1"}).iloc[0]:
            errors.append("eligibility.success_conflict:ml_feature_eligible")
        if "stats_resolution_status" in frame:
            from app_core.market_probability_model import _text
            if not _text(frame, "stats_resolution_status").str.lower().isin({"resolved", "live", "cached"}).iloc[0]:
                errors.append("eligibility.success_conflict:stats_resolution_status")


def _event_binding(event, expected, prefix, errors, unknown, fields=("provider_namespace", "provider_event_id", "sport", "home", "away", "start")):
    """Named orientation and namespace are facts; unordered keys are opaque."""
    from app_core.producer_provenance import team
    if not isinstance(event, dict):
        unknown.append(prefix)
        return
    for key in fields:
        supplied = event.get(key)
        if not supplied:
            unknown.append(prefix + ":" + key)
            continue
        actual = clock(supplied) if key == "start" else team(supplied, "NFL") if key in {"home", "away"} else supplied
        if expected is None or not expected.get(key):
            unknown.append(prefix + ":expected_" + key)
        elif actual != expected[key]:
            errors.append(prefix + ":" + key)


def _observation_binding(observed, packet, item, errors, unknown):
    payload = observed["payload"]
    if payload.get("schema") != "football-feature-observation-v1":
        errors.append("features.observation_schema")
    contract = packet["event_offer"]
    expected = contract["event"] if contract else None
    event = {"sport": payload.get("sport"), "home": payload.get("home_team"),
             "away": payload.get("away_team"), "start": payload.get("game_start_utc")}
    # This existing receipt predates provider IDs; bind its named event fields to
    # the original quote event, without pretending it recorded a provider ID.
    _event_binding(event, expected, "features.observation_event", errors, unknown,
                   fields=("sport", "home", "away", "start"))
    matchup = item.get("identity", {}).get("matchup_id", {})
    if matchup.get("state") != "VALUE" or not payload.get("matchup_id"):
        unknown.append("features.observation_matchup")
    elif payload["matchup_id"] != matchup["value"]:
        errors.append("features.observation_matchup")
    resolution = packet["eligibility"]["stats_resolution_status"]
    from app_core.research_replay import value
    if not payload.get("stats_resolution_status"):
        unknown.append("features.observation_resolution")
    elif fact(value(resolution))["state"] == "VALUE" and payload["stats_resolution_status"] != value(resolution):
        errors.append("features.observation_resolution")
    values = payload.get("features")
    if not isinstance(values, dict):
        unknown.append("features.observation_feature_set")
        return
    for name, consumed in packet["features"].items():
        if consumed.get("state") != "VALUE":
            continue
        if name not in values:
            unknown.append("features.observation_missing:" + name)
        elif fact(values[name]) != consumed:
            errors.append("features.observation_value:" + name)


def _dependency_scope(dependency, at, packet, name, errors, unknown):
    """Original bytes must carry applicable feature/event/availability facts."""
    scope = dependency.get("scope")
    if scope is None:
        unknown.append("features.dependency_scope:" + name)
        return
    if not isinstance(scope, dict):
        errors.append("features.dependency_scope_schema:" + name)
        return
    if "scope_path" not in dependency:
        unknown.append("features.original_scope:" + name)
    elif encode(at(dependency["scope_path"])) != encode(scope):
        errors.append("features.original_scope_conflict:" + name)
    for key, expected in (("contract", FEATURE_SCOPE), ("feature", name)):
        if key not in scope:
            unknown.append("features.dependency_scope_" + key + ":" + name)
        elif scope[key] != expected:
            errors.append("features.dependency_scope_" + key + ":" + name)
    event = packet["event_offer"]["event"] if packet["event_offer"] else None
    _event_binding(scope.get("event"), event, "features.dependency_scope_event:" + name, errors, unknown)
    for key in ("available_at", "observed_at"):
        if key not in scope:
            unknown.append("features.dependency_scope_" + key + ":" + name)
        elif not scope[key]:
            unknown.append("features.dependency_scope_" + key + ":" + name)
        elif clock(scope[key]) is None or clock(scope[key]) != clock(dependency.get(key)):
            errors.append("features.dependency_scope_" + key + ":" + name)


def diagnose(source, item=None):
    """Fail closed for false/missing consumed facts; separate unknown upstream bindings."""
    errors,unknown=[],[]
    try:
        item=item if item is not None else json.loads(source.get("ml_estimate_metadata", ""))
        if "nfl_inputs" not in item:
            return dict(status="UNKNOWN", errors=[], unknown=["original_nfl_inputs_not_retained"])
        retained=item["nfl_inputs"];p=retained["payload"]
        if p.get("capture_status")=="FAILED":
            return dict(status="REJECTED",errors=p.get("capture_errors",["inference_inputs.failed"]),unknown=[])
        if set(retained)!={"payload","sha256"} or digest(p)!=retained["sha256"]:errors.append("packet.integrity")
        if p["version"]!=VERSION or p["feature_order"]!=list(FEATURES) or set(p["features"])!=set(FEATURES):errors.append("features.contract_order")
        for k in REQUIRED:
            if p["features"][k].get("state")!="VALUE" or not isinstance(p["features"][k].get("value"),(float,int)) or isinstance(p["features"][k].get("value"),bool) or p["features"][k]["value"]<=0:errors.append("features."+k)
        if not isinstance(p.get("configuration"),dict) or set(p["configuration"])!={"score_parameters","blend_weights"}:errors.append("configuration.missing")
        if not isinstance(p.get("eligibility_present"),list) or set(p["eligibility_present"])-set(p["eligibility"]):errors.append("eligibility.schema")
        _eligibility_binding(source, p, errors, unknown)
        if not isinstance(p.get("predictor_callables"),dict) or set(p["predictor_callables"])!=set(predictor_callables()):errors.append("predictor.callables_missing")
        if p["scientific_acceptance"] is not False or p["wagering_authority"] is not False:errors.append("packet.authority_forbidden")
        for k,expected in {"inference_time":item["generated_at"],"inference_status":item["inference_status"],"raw_probability":item["probability"],"raw_semantics":item["probability_semantics"],"push_probability":item["push_probability"]}.items():
            if p[k]!=expected:errors.append("origin."+k)
        if p["raw_probability"]!=fact(source.get("ml_probability")) or p["predictor_id"]!=source.get("ml_probability_source") or p["target"]!=source.get("ml_target"):errors.append("origin.predictor_probability_target")
        if p["event_offer"]!=item.get("producer_contract"):errors.append("event_offer.conflict")
        if p["event_offer"] is None:unknown.append("event_offer.original_contract")
        else:
            from app_core.producer_provenance import diagnose as offer_diagnose
            # This reader has no dependency on nfl_inputs, avoiding recursive validation.
            d=offer_diagnose(source,item)
            if d["conflicting_fields"]:errors.extend("offer."+x for x in d["conflicting_fields"])
            unknown.extend("offer."+x for x in d["missing_fields"])
            unknown.extend(d.get("source_contract_diagnostics",[]))
            event,offer=p["event_offer"]["event"],p["event_offer"]["offer"]
            times=[clock(offer["source_time"]),clock(p["inference_time"]),clock(event["start"])]
            if any(t is None for t in times) or not(times[0]<=times[1]<times[2]):errors.append("clocks.quote_inference_kickoff")
        for name in ARTIFACTS:
            a=p["artifacts"][name];raw=base64.b64decode(a["bytes_base64"],validate=True)
            if hashlib.sha256(raw).hexdigest()!=a["sha256"]:errors.append("artifact.integrity:"+name)
        if set(p["artifacts"])!=set(ARTIFACTS):errors.append("artifacts.contract")
        if any(not isinstance(p["runtime"].get(k),str) or not p["runtime"][k] for k in runtime()):errors.append("runtime.missing")
        current_receipt = source.get("football_feature_receipt")
        if fact(current_receipt)["state"] != "VALUE":
            unknown.append("features.current_observation_receipt")
        elif current_receipt != p["observation_receipt"]:
            errors.append("features.observation_source_conflict")
        if p["observation_receipt"] is None:unknown.append("features.observation_receipt")
        else:
            observed=json.loads(p["observation_receipt"])
            if digest(observed["payload"])!=observed["sha256"]:errors.append("features.observation_integrity")
            observed_time=clock(observed["payload"].get("observed_at"));inf=clock(p["inference_time"])
            if observed_time is None or inf is None or observed_time>inf:errors.append("features.observation_clock")
            _observation_binding(observed, p, item, errors, unknown)
        deps=p["source_dependencies"]
        if deps is None:unknown.append("features.source_dependencies_and_availability")
        elif not isinstance(deps,dict):errors.append("features.dependencies.schema")
        else:
            if set(deps)-set(FEATURES):errors.append("features.dependencies.schema")
            for k in FEATURES:
                if p["features"][k].get("state")=="MISSING":continue
                if k not in deps:unknown.append("features.dependency:"+k);continue
                receipt=deps[k];dependency=receipt["payload"]
                if set(receipt)!={"payload","sha256"} or digest(dependency)!=receipt["sha256"]:errors.append("features.dependency_integrity:"+k)
                if dependency["feature"]!=k or fact(dependency["value"])!=p["features"][k]:errors.append("features.dependency_value:"+k)
                if not dependency.get("source_id"):unknown.append("features.source_id:"+k)
                if p["event_offer"] and dependency.get("provider_event_id")!=p["event_offer"]["event"]["provider_event_id"]:errors.append("features.dependency_event:"+k)
                artifact=dependency.get("source_artifact")
                if artifact is None:unknown.append("features.source_artifact:"+k)
                else:
                    raw=base64.b64decode(artifact["bytes_base64"],validate=True)
                    if hashlib.sha256(raw).hexdigest()!=artifact["sha256"]:errors.append("features.source_artifact_integrity:"+k)
                    original=json.loads(raw)
                    def at(path):
                        value=original
                        if not isinstance(path,list) or not path:raise ValueError("source field path required")
                        for part in path:value=value[part]
                        return value
                    if fact(at(dependency["value_path"]))!=p["features"][k]:errors.append("features.original_source_value:"+k)
                    if at(dependency["event_path"])!=dependency["provider_event_id"]:errors.append("features.original_source_event:"+k)
                    _dependency_scope(dependency, at, p, k, errors, unknown)
                av,ob,inf=clock(dependency.get("available_at")),clock(dependency.get("observed_at")),clock(p["inference_time"])
                if (dependency.get("available_at") and av is None) or (dependency.get("observed_at") and ob is None):errors.append("features.invalid_clock:"+k)
                elif av is None or ob is None:unknown.append("features.availability:"+k)
                elif inf is None or not av<=ob<=inf:errors.append("features.availability_clock:"+k)
                elif p["observation_receipt"]:
                    observation = clock(json.loads(p["observation_receipt"])["payload"].get("observed_at"))
                    if observation is None or not ob <= observation <= inf:
                        errors.append("features.dependency_observation_window:" + k)
        if p["blend"] is None or p["blend"]["probability"] is None:unknown.append("blend.original_output")
        else:
            if p["blend"]["probability"]!=fact(source.get("calibrated_probability")) or p["blend"]["estimated_ev"]!=fact(source.get("expected_value")):errors.append("blend.original_output_conflict")
            if p["blend"]["probability_semantics"]!="market_context_research_blend_not_scoped_calibration" or p["blend"]["ev_semantics"]!="recorded_binary_price_estimate_not_certified_operator_payoff":errors.append("blend.semantics")
    except (ValueError,TypeError,KeyError,IndexError,AttributeError,OverflowError):errors.append("packet.schema")
    return dict(status="REJECTED" if errors else "INCOMPLETE" if unknown else "COMPLETE",errors=sorted(set(errors)),unknown=sorted(set(unknown)))


def replay(source):
    """Offline numeric verification only, using installed exact consumed code, never exec saved bytes."""
    item=json.loads(source.get("ml_estimate_metadata", ""));assessment=diagnose(source,item)
    if assessment["status"]!="COMPLETE":raise ValueError(encode(assessment))
    p=item["nfl_inputs"]["payload"]
    if p["predictor_callables"]!=predictor_callables():raise ValueError("Replay consumed predictor callable mismatch")
    if p["artifacts"]!=artifacts() or p["configuration"]!=configuration() or p["runtime"]!=runtime():raise ValueError("Replay consumed artifact/configuration/runtime mismatch")
    from app_core.market_probability_model import predict_market_probabilities
    from app_core.research_replay import value
    row={k:v.get("value") for k,v in p["features"].items()}
    row.update({k:value(v) for k,v in p["eligibility"].items() if k in p["eligibility_present"]})
    row.update(league="NFL",market_type=p["event_offer"]["offer"]["market"],spread_line=p["event_offer"]["offer"]["line"])
    # No provider quotes: numerical check produces no replacement retained offer or old packet.
    result=predict_market_probabilities(pd.DataFrame([row])).iloc[0]
    if fact(result.ml_probability)!=p["raw_probability"]:raise ValueError("Replay probability mismatch")
    if p["blend"]["consumed"]!=consumed_blend():raise ValueError("Replay consumed blend configuration mismatch")
    from core.streamlit_pipeline import compute_blended_probability
    if p["blend"]["pipeline_sha256"] != hashlib.sha256((ROOT/"core/streamlit_pipeline.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest():
        raise ValueError("Replay consumed pipeline mismatch")
    blend = compute_blended_probability(**{k:pd.Series([v.get("value")],dtype="float64") for k,v in p["blend"]["inputs"].items()},
        league=pd.Series(["NFL"]),market_type=pd.Series([row["market_type"]])).iloc[0]
    if fact(blend)!=p["blend"]["probability"]:raise ValueError("Replay blended probability mismatch")
    return dict(raw_probability=result.ml_probability, blended_probability=blend,
        estimated_ev=p["blend"]["estimated_ev"].get("value"), scientific_acceptance=False,wagering_authority=False)
