"""Additive, price-bound research display. Never read by wagering consumers."""
from __future__ import annotations
from copy import deepcopy
import math
import json
import re
import pandas as pd
from core.price_value import price_value
from core.wager_decisions import decimal_price

VERSION = "research-display-v1"
FIELDS = frozenset("""version label source_field basis identity probability push_probability
probability_semantics ev break_even_probability edge availability_reason value_reason inference_status""".split())
IDENTITY_FIELDS = frozenset("""event_id candidate_id export_run_id sport market selection line period
rules model_target sportsbook odds quote_id quote_time analysis_time start""".split())
REASONS = frozenset("""AVAILABLE ESTIMATE_NOT_RECORDED INVALID_PROBABILITY NONFINITE_PROBABILITY
ESTIMATE_PROVENANCE_NOT_RECORDED ESTIMATE_IDENTITY_MISMATCH TARGET_MISMATCH MODEL_TARGET_NOT_RECORDED
INFERENCE_FAILED INFERENCE_UNAVAILABLE UNSUPPORTED_PROBABILITY_SEMANTICS""".split())
VALUE_REASONS = frozenset("""RECORDED_PRICE_VALUE VALUE_NOT_RECORDED PRICE_VALUE_MISMATCH
PUSH_PROBABILITY_NOT_RECORDED ESTIMATE_UNAVAILABLE""".split())
SOURCE_FIELDS = {"best_available_probability", "calibrated_probability",
                 "production_win_probability", "win_probability"}

def _text(value):
    return value.strip() if isinstance(value, str) else ""

def _number(value):
    # bool and numpy.bool_ are not probabilities, prices or zero estimates.
    if isinstance(value, (bool,)) or type(value).__name__ == "bool_":
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None

def _time(value):
    from app_core.public_board import timestamp
    try:
        return timestamp(_text(value))
    except (ValueError, TypeError):
        return None

def _identity(row):
    return dict(event_id=_text(row.get("matchup_id")), candidate_id=_text(row.get("candidate_id")),
        export_run_id=_text(row.get("export_run_id")), sport=_text(row.get("league")),
        market=_text(row.get("market_type")), selection=_text(row.get("pick")),
        line=_number(row.get("line")), period=_text(row.get("market_period")),
        rules=_text(row.get("settlement_rules")), model_target=_text(row.get("ml_target")).casefold(), sportsbook=_text(row.get("quote_source")),
        odds=_number(row.get("odds")), quote_id=_text(row.get("quote_id")),
        quote_time=_time(row.get("quote_time")),
        analysis_time=_time(_text(row.get("prediction_generated_at")) or row.get("export_run_id")),
        start=_time(_text(row.get("game_start_utc")) or row.get("start")))

def _empty(identity, field="", basis="", reason="ESTIMATE_NOT_RECORDED", inference="UNKNOWN"):
    return dict(version=VERSION, label="Research estimate", source_field=field, basis=basis,
        identity=identity, probability=None, push_probability=None, probability_semantics="",
        ev=None, break_even_probability=None, edge=None, availability_reason=reason,
        value_reason="ESTIMATE_UNAVAILABLE", inference_status=inference)

def _complete(identity):
    # Names alone, a best-pick score, or a price are never sufficient provenance.
    required=("event_id","candidate_id","export_run_id","sport","market","selection",
              "sportsbook","quote_id","quote_time","analysis_time","start","period","rules")
    if not all(identity.get(key) for key in required) or identity["line"] is None:
        return False
    match=re.search(r"(?:^|\s)([+-]?\d+(?:\.\d+)?)$", identity["selection"])
    allowed={identity["market"], "spread_cover" if identity["market"].startswith("spread") else "total"}
    return bool(identity["model_target"] in allowed and match and float(match.group(1)) == identity["line"]
                and identity["market"] in {"spread_home","spread_away","total_over","total_under"})

def from_export(row, *, source=None, source_field="win_probability"):
    """Capture the actual export estimate before contract authorization replaces it."""
    identity=_identity(row)
    basis=_text(row.get("probability_basis"))
    source=source if source is not None else row
    status=_text(_text(source.get("inference_status")) or source.get("model_status")).casefold()
    inference="FAILED" if status in {"failed","error","inference_failed"} else (
        "UNAVAILABLE" if status in {"missing","unavailable","inference_unavailable"} else
        "RECORDED" if status in {"ok","success","complete"} else "UNKNOWN")
    result=_empty(identity, source_field, basis, inference=inference)
    if inference in {"FAILED","UNAVAILABLE"}:
        result["availability_reason"]="INFERENCE_"+inference
        return result
    target=_text(source.get("ml_target")).casefold()
    allowed={identity["market"], "spread_cover" if identity["market"].startswith("spread") else "total"}
    if not target:
        result["availability_reason"]="MODEL_TARGET_NOT_RECORDED"
        return result
    if target not in allowed:
        result["availability_reason"]="TARGET_MISMATCH"
        return result
    source_push=source.get("push_probability")
    if isinstance(source_push,bool) or type(source_push).__name__=="bool_":
        result["availability_reason"]="UNSUPPORTED_PROBABILITY_SEMANTICS"
        return result
    raw=source.get(source_field)
    probability=_number(row.get("win_probability"))
    if raw is None or raw is pd.NA or (isinstance(raw,float) and math.isnan(raw)):
        result["availability_reason"]="ESTIMATE_NOT_RECORDED"
        return result
    if _number(raw) is None:
        result["availability_reason"]="NONFINITE_PROBABILITY" if not isinstance(raw,(bool,)) and type(raw).__name__!="bool_" else "INVALID_PROBABILITY"
        return result
    if not 0 <= _number(raw) <= 1:
        result["availability_reason"]="INVALID_PROBABILITY"
        return result
    if probability is None:
        result["availability_reason"]="UNSUPPORTED_PROBABILITY_SEMANTICS" if _text(source.get("probability_semantics")) else "ESTIMATE_NOT_RECORDED"
        return result
    if not _complete(identity) or not basis or basis=="Unavailable":
        result["availability_reason"]="ESTIMATE_PROVENANCE_NOT_RECORDED"
        return result
    if not 0 <= probability <= 1:
        result["availability_reason"]="INVALID_PROBABILITY"
        return result
    result.update(probability=probability, availability_reason="AVAILABLE",
                  value_reason="PUSH_PROBABILITY_NOT_RECORDED")
    push=_number(row.get("push_probability"))
    line=identity["line"]
    half_point=abs(line*2-round(line*2))<=1e-9 and abs(line-round(line))>1e-9
    if push is None and half_point:
        push=0.0  # Same bounded no-push compatibility used by the real exporter.
    if push is None:
        return result
    priced=price_value(probability,push,decimal_price(identity["odds"]))
    if priced is None or (half_point and push>1e-9):
        return _empty(identity,source_field,basis,reason="UNSUPPORTED_PROBABILITY_SEMANTICS",inference=inference)
    result.update(push_probability=push,probability_semantics="win_unconditional_with_push")
    saved_ev=_number(row.get("ev"))
    if saved_ev is None:
        result["value_reason"]="VALUE_NOT_RECORDED"
    elif not math.isclose(saved_ev,priced["expected_value"],rel_tol=0,abs_tol=1e-9):
        result["value_reason"]="PRICE_VALUE_MISMATCH"
    else:
        # Preserve recorded EV. Edge and break-even are diagnostic price math
        # for this same probability/price/push basis, never qualification input.
        result.update(ev=saved_ev,break_even_probability=priced["break_even"],edge=priced["edge"],
                      value_reason="RECORDED_PRICE_VALUE")
    return result

def _matches(display, row):
    identity=display["identity"]
    mappings={"sport":"sport","market":"market","selection":"pick","odds":"odds",
              "sportsbook":"quote_source","quote_time":"quote_time",
              "analysis_time":"as_of","start":"start"}
    for key, public in mappings.items():
        if identity[key] != row.get(public):
            return False
    contract=row.get("controlled_trial_contract") or row.get("wager_contract")
    if isinstance(contract,dict):
        for key, fields in {
            "event_id":("game_id","matchup_id"),"sport":("sport",),"market":("market_type",),
            "selection":("selection",),"line":("line",),"odds":("odds",),
            "sportsbook":("sportsbook",),"quote_time":("quote_timestamp",),"start":("start",),
        }.items():
            for field in fields:
                value=contract.get(field)
                if value is None or value=="":
                    continue
                if key in {"quote_time","start"}: value=_time(value)
                if key=="sportsbook":
                    if str(value).casefold()!=identity[key].casefold(): return False
                elif value!=identity[key]:
                    return False
    return True

def public_display(export, row):
    saved=export.get("research_display")
    if isinstance(saved,str):
        try:
            saved=json.loads(saved)
        except ValueError:
            return _empty(_identity(export),reason="ESTIMATE_PROVENANCE_NOT_RECORDED")
    if isinstance(saved,dict):
        result=deepcopy(saved)
        validate(result)
        if result["identity"] != _identity(export):
            return _empty(_identity(export),reason="ESTIMATE_IDENTITY_MISMATCH")
    else:
        result=from_export(export)
    if not _matches(result,row):
        return _empty(result["identity"],result["source_field"],result["basis"],
                      reason="ESTIMATE_IDENTITY_MISMATCH",inference=result["inference_status"])
    return result

def validate(display, row=None):
    if not isinstance(display,dict) or set(display)!=FIELDS or display["version"]!=VERSION:
        raise ValueError("Invalid research display schema")
    identity=display["identity"]
    if not isinstance(identity,dict) or set(identity)!=IDENTITY_FIELDS:
        raise ValueError("Invalid research display identity")
    for key,value in identity.items():
        if key in {"line","odds"}:
            if value is not None and (_number(value) is None or isinstance(value,bool)):
                raise ValueError("Invalid research display identity metric")
        elif value is not None and not isinstance(value,str):
            raise ValueError("Invalid research display identity label")
    if display["label"]!="Research estimate" or display["source_field"] not in SOURCE_FIELDS | {""}:
        raise ValueError("Invalid research display source")
    if not isinstance(display["basis"],str) or display["inference_status"] not in {"UNKNOWN","FAILED","UNAVAILABLE","RECORDED"}:
        raise ValueError("Invalid research display provenance")
    if display["availability_reason"] not in REASONS or display["value_reason"] not in VALUE_REASONS:
        raise ValueError("Invalid research display reason")
    for key in ("probability","push_probability","ev","break_even_probability","edge"):
        value=display[key]
        if value is not None and (not isinstance(value,(int,float)) or isinstance(value,bool) or _number(value) is None):
            raise ValueError("Invalid research display metric")
    if display["availability_reason"]!="AVAILABLE":
        if any(display[key] is not None for key in ("probability","push_probability","ev","break_even_probability","edge")) or display["probability_semantics"]!="":
            raise ValueError("Unavailable research display contains estimates")
        return
    if display["inference_status"] in {"FAILED","UNAVAILABLE"}:
        raise ValueError("Failed inference cannot claim an available research estimate")
    if not _complete(identity) or display["probability"] is None or not 0<=display["probability"]<=1 or not display["basis"]:
        raise ValueError("Research estimate requires exact exported provenance")
    if row is not None and not _matches(display,row):
        raise ValueError("Research display does not match public selection")
    push=display["push_probability"]
    if push is None:
        if display["probability_semantics"]!="" or display["ev"] is not None:
            raise ValueError("Unknown push semantics cannot price research value")
    elif display["probability_semantics"]!="win_unconditional_with_push":
        raise ValueError("Invalid research display semantics")
    if push is not None:
        line=identity["line"]
        half_point=abs(line*2-round(line*2))<=1e-9 and abs(line-round(line))>1e-9
        if price_value(display["probability"],push,decimal_price(identity["odds"])) is None or (half_point and push>1e-9):
            raise ValueError("Invalid research probability/push mass")
    if (display["value_reason"]=="RECORDED_PRICE_VALUE") != (display["ev"] is not None):
        raise ValueError("Research value reason must match recorded EV availability")
    if display["ev"] is not None:
        priced=price_value(display["probability"],push,decimal_price(identity["odds"]))
        if priced is None or any(not math.isclose(display[key],priced[other],rel_tol=0,abs_tol=1e-9)
                                 for key,other in (("ev","expected_value"),("edge","edge"),("break_even_probability","break_even"))):
            raise ValueError("Research price value does not match displayed basis")
    elif display["edge"] is not None or display["break_even_probability"] is not None:
        raise ValueError("Research value unavailable without compatible recorded EV")
