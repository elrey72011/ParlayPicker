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
PUSH_PROBABILITY_NOT_RECORDED INVALID_RECORDED_EV ESTIMATE_UNAVAILABLE""".split())
# Explicit public-research provenance only; never an arbitrary source-column copy.
EXPORT_PROVENANCE_COLUMNS = ["quote_id", "prospective_quote_id", "market_period", "period",
    "settlement_rules", "inference_status", "model_status", "spread_line", "total_line",
    "market_line_used", "push_probability", "probability_semantics", "research_source_semantics"]
SEMANTIC_FIELDS = ("probability_semantics", "push_probability", "inference_status", "model_status")
SEMANTICS = frozenset({"win_conditional_on_decision","win_unconditional_with_push",
                      "unconditional_win_push_loss","unconditional"})
EV_SOURCE_FIELDS = frozenset({"production_expected_value", "expected_value"})


def _numeric_rejections(row):
    """Keep invalid-type facts only; never copy a private payload or infer a value."""
    rejected={}
    for field in EV_SOURCE_FIELDS:
        value=row.get(field)
        if not _absent(value) and _number(value) is None:
            rejected[field]="INVALID_RECORDED_EV"
    for field in SOURCE_FIELDS:
        value=row.get(field)
        if _absent(value): continue
        if isinstance(value,bool) or type(value).__name__=="bool_":
            rejected[field]="INVALID_PROBABILITY"
        elif isinstance(value,float) and math.isnan(value):
            rejected[field]="ESTIMATE_NOT_RECORDED"  # Existing NaN probability semantics.
        elif _number(value) is None:
            rejected[field]="NONFINITE_PROBABILITY"
        elif not 0<=_number(value)<=1:
            rejected[field]="INVALID_PROBABILITY"
    return rejected


def _absent(value):
    # NaN/inf/booleans are explicit invalid input, not a missing push contract.
    return value is None or value is pd.NA or value is pd.NaT or (isinstance(value,str) and not value.strip())


def _fact(value):
    if _absent(value):
        return {"state":"MISSING"}
    if isinstance(value,str):
        return {"state":"VALUE","value":value}
    if isinstance(value,bool) or type(value).__name__=="bool_":
        return {"state":"VALUE","value":bool(value)}
    if _number(value) is not None:
        return {"state":"VALUE","value":_number(value)}
    return {"state":"INVALID"}


def _valid_push(value):
    number=_number(value)
    return number is not None and 0<=number<=1


def _status(value, *, field=None):
    if _absent(value):
        return "UNKNOWN"
    name=_text(value).casefold()
    if name in {"failed","error","inference_failed"}:
        return "FAILED"
    if name in {"ok","success","complete"}:
        return "RECORDED"
    if field=="model_status" and name=="market score model":
        # The existing producer records its model type here, not run success.
        # Only a separately recorded inference status can establish success.
        return "UNKNOWN"
    # Explicit unknown/missing/unavailable, invalid types and unsupported declarations
    # cannot become a successful retained model run.
    return "UNAVAILABLE"


def preserve_source_semantics(frame):
    """Keep pre-capture facts for display only, including invalid numeric types.

    Immutable evidence capture legitimately canonicalizes its authority semantics.
    This additive allowlisted carrier must not turn that derivation into proof that
    an unsupported original research contract was valid. It is never read by gates.
    """
    out=frame.copy()
    if out.empty:
        return out
    def encode(row):
        saved=row.get("research_source_semantics")
        if not _absent(saved):
            original=_source_semantics(row)
            if original is None:
                return saved  # Malformed carriers remain invalid and unchanged.
            decoded=json.loads(saved)
            changed=False
            numeric=decoded.get("invalid_numeric_fields",{})
            merged={**_numeric_rejections(row),**numeric}  # Original invalid facts remain sticky.
            if merged!=numeric:
                decoded.update(version=2,invalid_numeric_fields=merged)
                changed=True
            for field in SEMANTIC_FIELDS:
                previous=original.get(field)
                current=row.get(field)
                if _absent(current):
                    continue  # A projection may retain a fact only in the carrier.
                # An original invalid/failing fact is sticky; current success cannot
                # repair it or erase the retained source's recorded rejection.
                if field=="probability_semantics":
                    if not _absent(previous) and _text(previous) not in SEMANTICS:
                        continue
                    reject=not _current_semantics_compatible(previous,current,push=original.get("push_probability"))
                    conflict=_text(current) in SEMANTICS
                elif field=="push_probability":
                    if not _absent(previous) and not _valid_push(previous):
                        continue
                    reject=(not _valid_push(current) or
                            not math.isclose(_number(current),_number(previous) if not _absent(previous) else 0.0,
                                             rel_tol=0,abs_tol=1e-9))
                    conflict=_valid_push(current)
                else:
                    if _status(previous,field=field) in {"FAILED","UNAVAILABLE"}:
                        continue
                    reject=_status(current,field=field) in {"FAILED","UNAVAILABLE"}
                    conflict=False
                if reject:
                    decoded["fields"][field]={"state":"INVALID"} if conflict else _fact(current)
                    changed=True
            return json.dumps(decoded,sort_keys=True,separators=(",",":"),allow_nan=False) if changed else saved
        values={field:_fact(row.get(field)) for field in SEMANTIC_FIELDS}
        decoded={"version":1,"fields":values}
        numeric=_numeric_rejections(row)
        if numeric: decoded.update(version=2,invalid_numeric_fields=numeric)
        return json.dumps(decoded,sort_keys=True,separators=(",",":"),allow_nan=False)
    out["research_source_semantics"]=out.apply(encode,axis=1)
    return out


def _source_semantics(source):
    saved=source.get("research_source_semantics")
    if _absent(saved): return source
    try:
        decoded=json.loads(saved)
        version=decoded.get("version")
        keys={"version","fields"} if version==1 else {"version","fields","invalid_numeric_fields"}
        if set(decoded)!=keys or type(version) is not int or version not in {1,2} or set(decoded["fields"])!=set(SEMANTIC_FIELDS):
            return None
        numeric=decoded.get("invalid_numeric_fields",{})
        if not isinstance(numeric,dict) or (version==2 and not numeric): return None
        for field,reason in numeric.items():
            allowed={"INVALID_RECORDED_EV"} if field in EV_SOURCE_FIELDS else (
                {"INVALID_PROBABILITY","NONFINITE_PROBABILITY","ESTIMATE_NOT_RECORDED"} if field in SOURCE_FIELDS else set())
            if not isinstance(reason,str) or reason not in allowed: return None
        out=dict(source)
        out["_research_invalid_numeric_fields"]=numeric
        for field,item in decoded["fields"].items():
            if item=={"state":"MISSING"}: out[field]=None
            elif item=={"state":"INVALID"}: out[field]=float("nan")
            elif set(item)=={"state","value"} and item["state"]=="VALUE" and isinstance(item["value"],(str,bool,int,float)):
                if isinstance(item["value"],(int,float)) and not isinstance(item["value"],bool) and _number(item["value"]) is None:
                    return None  # Python accepts nonfinite JSON tokens; the carrier is malformed.
                out[field]=item["value"]
            else: return None
        return out
    except (ValueError,TypeError,AttributeError): return None


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

def _current_semantics_compatible(original,current,*,push=None):
    if not _absent(original) and _text(original) not in SEMANTICS:
        return False  # Missing target metadata cannot excuse explicit rejection.
    if _absent(current):
        return True  # Canonical projection may retain semantics only in the carrier.
    unconditional={"win_unconditional_with_push","unconditional_win_push_loss","unconditional"}
    current_name=_text(current)
    if current_name not in unconditional | {"win_conditional_on_decision"}:
        return False
    # Capture can express a no-push market conditionally without changing mass.
    # Nonzero-push reversal cannot reinterpret an unconditional source.
    return not (_text(original) in unconditional and current_name=="win_conditional_on_decision"
                and not (_valid_push(push) and math.isclose(_number(push),0.0,rel_tol=0,abs_tol=1e-9)))


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

def _export_price_mass(row,source,source_field,*,direct=False,current_push=None):
    """Check retained source mass against the export using existing price math."""
    raw_probability=_number(source.get(source_field))
    probability=_number(row.get("win_probability"))
    source_push=source.get("push_probability")
    semantics=source.get("probability_semantics")
    line=_number(row.get("line"))
    half_point=(line is not None and abs(line*2-round(line*2))<=1e-9
                and abs(line-round(line))>1e-9)
    push=_number(source_push)
    semantic_name=_text(semantics)
    mass=None
    if raw_probability is None or probability is None:
        return None,None
    if _absent(semantics) and _absent(source_push) and half_point:
        mass={"p_win":raw_probability,"p_push":0.0}
    elif push is not None and not _absent(semantics):
        if semantic_name=="win_conditional_on_decision":
            from core.probability_semantics import unconditional_from_conditional
            mass=unconditional_from_conditional(raw_probability,push)
        elif semantic_name in {"win_unconditional_with_push","unconditional_win_push_loss","unconditional"}:
            mass={"p_win":raw_probability,"p_push":push}
    if (mass is None or (half_point and mass["p_push"]>1e-9)
        or (not _absent(current_push) and (_number(current_push) is None
            or not math.isclose(_number(current_push),mass["p_push"],rel_tol=0,abs_tol=1e-9)))):
        return None,None
    priced=price_value(mass["p_win"],mass["p_push"],decimal_price(_number(row.get("odds"))))
    exported_push=row.get("push_probability")
    exported_semantics=row.get("probability_semantics")
    # Direct legacy input can contain conditional mass; a per-game export must
    # contain the normalized mass, never merely a relabeled source probability.
    expected=raw_probability if direct and semantic_name=="win_conditional_on_decision" else mass["p_win"]
    if (priced is None or not math.isclose(probability,expected,rel_tol=0,abs_tol=1e-9)
        or (not _absent(exported_push) and (_number(exported_push) is None or not math.isclose(_number(exported_push),mass["p_push"],rel_tol=0,abs_tol=1e-9)))
        or (not _absent(exported_semantics) and _text(exported_semantics) not in
            ({"win_conditional_on_decision"} if direct and semantic_name=="win_conditional_on_decision" else
             {"win_unconditional_with_push","unconditional_win_push_loss","unconditional"}))):
        return None,None
    return mass,priced


def from_export(row, *, source=None, source_field="win_probability"):
    """Capture the actual export estimate before contract authorization replaces it."""
    identity=_identity(row)
    basis=_text(row.get("probability_basis"))
    direct=source is None
    source=source if source is not None else row
    current_statuses=[_status(source.get(field),field=field) for field in ("inference_status","model_status")]
    current_push=source.get("push_probability")
    current_semantics=source.get("probability_semantics")
    source=_source_semantics(source)
    if source is None or not _current_semantics_compatible(
            source.get("probability_semantics"),current_semantics,push=source.get("push_probability")):
        return _empty(identity,source_field,basis,reason="UNSUPPORTED_PROBABILITY_SEMANTICS")
    statuses=current_statuses+[_status(source.get(field),field=field) for field in ("inference_status","model_status")]
    inference=next((state for state in ("FAILED","UNAVAILABLE","RECORDED") if state in statuses),"UNKNOWN")
    result=_empty(identity, source_field, basis, inference=inference)
    if inference in {"FAILED","UNAVAILABLE"}:
        result["availability_reason"]="INFERENCE_"+inference
        return result
    raw=source.get(source_field)
    numeric=source.get("_research_invalid_numeric_fields",{})
    if source_field in numeric:
        return _empty(identity,source_field,basis,reason=numeric[source_field],inference=inference)
    if raw is None or raw is pd.NA or (isinstance(raw,float) and math.isnan(raw)):
        result["availability_reason"]="ESTIMATE_NOT_RECORDED"
        return result
    if isinstance(raw,bool) or type(raw).__name__=="bool_":
        return _empty(identity,source_field,basis,reason="INVALID_PROBABILITY",inference=inference)
    if _number(raw) is None:
        result["availability_reason"]="NONFINITE_PROBABILITY"
        return result
    if not 0 <= _number(raw) <= 1:
        result["availability_reason"]="INVALID_PROBABILITY"
        return result
    ev_field={"production_win_probability":"production_expected_value","calibrated_probability":"expected_value"}.get(source_field)
    raw_ev=source.get(ev_field) if ev_field else None
    invalid_ev=ev_field in numeric or (not _absent(raw_ev) and _number(raw_ev) is None)
    if invalid_ev:
        result["value_reason"]="INVALID_RECORDED_EV"
    # Check explicit source rejection before missing target/provenance can
    # replace its reason and the export projection drops the original facts.
    source_push=source.get("push_probability")
    if not _absent(source_push) and not _valid_push(source_push):
        return _empty(identity,source_field,basis,reason="UNSUPPORTED_PROBABILITY_SEMANTICS",inference=inference)
    raw_probability=_number(source.get(source_field))
    if (raw_probability is not None and 0<=raw_probability<=1
            and (identity["line"] is not None or not _absent(source.get("probability_semantics"))
                 or not _absent(source_push))):
        mass,priced=_export_price_mass(row,source,source_field,direct=direct,current_push=current_push)
        if mass is None:
            return _empty(identity,source_field,basis,reason="UNSUPPORTED_PROBABILITY_SEMANTICS",inference=inference)
        explicit_contract=not _absent(source.get("probability_semantics")) or not _absent(source_push)
        saved_ev=_number(row.get("ev"))
        if (explicit_contract and not invalid_ev and saved_ev is not None
                and not math.isclose(saved_ev,priced["expected_value"],rel_tol=0,abs_tol=1e-9)):
            result["value_reason"]="PRICE_VALUE_MISMATCH"
    # Explicit aliases cannot disagree about the exact displayed target or quote.
    source_line=_number(source.get("total_line" if identity["market"].startswith("total") else "spread_line"))
    market_line=_number(source.get("market_line_used"))
    selection_line=re.search(r"(?:^|\s)([+-]?\d+(?:\.\d+)?)$",identity["selection"])
    conflicts=(identity["line"] is not None and selection_line is not None and float(selection_line.group(1))!=identity["line"])
    conflicts=conflicts or (source_line is not None and source_line!=identity["line"]) or (market_line is not None and market_line!=identity["line"])
    for first,second in (("market_period","period"),("quote_id","prospective_quote_id")):
        if _text(source.get(first)) and _text(source.get(second)) and _text(source[first])!=_text(source[second]): conflicts=True
    if conflicts:
        return _empty(identity,source_field,basis,reason="ESTIMATE_IDENTITY_MISMATCH",inference=inference)
    target=_text(source.get("ml_target")).casefold()
    allowed={identity["market"], "spread_cover" if identity["market"].startswith("spread") else "total"}
    if not target:
        result["availability_reason"]="MODEL_TARGET_NOT_RECORDED"
        return result
    if target not in allowed:
        result["availability_reason"]="TARGET_MISMATCH"
        return result
    probability=_number(row.get("win_probability"))
    if probability is None:
        result["availability_reason"]="UNSUPPORTED_PROBABILITY_SEMANTICS"
        return result
    if not _complete(identity) or not basis or basis=="Unavailable":
        result["availability_reason"]="ESTIMATE_PROVENANCE_NOT_RECORDED"
        return result
    if not 0 <= probability <= 1:
        result["availability_reason"]="INVALID_PROBABILITY"
        return result
    mass,priced=_export_price_mass(row,source,source_field,direct=direct,current_push=current_push)
    if mass is None:
        return _empty(identity,source_field,basis,reason="UNSUPPORTED_PROBABILITY_SEMANTICS",inference=inference)
    result.update(probability=mass["p_win"],push_probability=mass["p_push"],
                  probability_semantics="win_unconditional_with_push",availability_reason="AVAILABLE",
                  value_reason="VALUE_NOT_RECORDED")
    saved_ev=_number(row.get("ev"))
    if invalid_ev:
        result["value_reason"]="INVALID_RECORDED_EV"
    elif saved_ev is None:
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

def _legacy_value_rejection(export):
    """Validate recorded export EV types and explicitly declared price basis."""
    ev=export.get("ev")
    if _absent(ev):
        return None
    if _number(ev) is None:
        return "INVALID_RECORDED_EV"
    if _absent(export.get("probability_semantics")) and _absent(export.get("push_probability")):
        return None  # An old undeclared basis remains unknown; do not invent one.
    mass,priced=_export_price_mass(export,export,"win_probability",current_push=export.get("push_probability"))
    if mass is None or not math.isclose(_number(ev),priced["expected_value"],rel_tol=0,abs_tol=1e-9):
        return "PRICE_VALUE_MISMATCH"
    return None


def legacy_unrecorded_display(export):
    """Identify legacy missing evidence without waiving explicit source rejection.

    Public identity checks can replace a saved missing-target reason. Inspect the
    validated saved reason before that replacement; never fill the separate
    research object or treat legacy metrics as conservative authorization.
    """
    saved=export.get("research_display")
    try:
        if isinstance(saved,str): saved=json.loads(saved)
        validate(saved)
    except (ValueError,TypeError):
        return False
    export_identity=_identity(export)
    if saved["identity"]["start"] is None:
        # Existing legacy adapters can add an unrecorded schedule after export.
        # This preserves historical metrics only; public_display still rejects
        # the changed identity and cannot create provenance or wager authority.
        export_identity["start"]=None
    if saved["identity"] != export_identity:
        return False  # A copied display cannot authorize another row's legacy values.
    if saved["availability_reason"] not in {
            "MODEL_TARGET_NOT_RECORDED","ESTIMATE_PROVENANCE_NOT_RECORDED","ESTIMATE_NOT_RECORDED"}:
        return False
    if saved["value_reason"] in {"INVALID_RECORDED_EV","PRICE_VALUE_MISMATCH"} or _legacy_value_rejection(export):
        return False  # Missing metadata cannot hide an explicit invalid EV type/basis.
    if _number(export.get("win_probability")) is None:
        return False  # A missing estimate cannot retain an orphaned legacy EV.
    source=_source_semantics(export)
    if source is None or not _current_semantics_compatible(
            source.get("probability_semantics"),export.get("probability_semantics"),
            push=source.get("push_probability")):
        return False
    numeric=source.get("_research_invalid_numeric_fields",{})
    ev_field={"production_win_probability":"production_expected_value","calibrated_probability":"expected_value"}.get(saved["source_field"])
    if saved["source_field"] in numeric or ev_field in numeric:
        return False
    for row in (source,export):
        if any(_status(row.get(field),field=field) in {"FAILED","UNAVAILABLE"}
               for field in ("inference_status","model_status")):
            return False
        push=row.get("push_probability")
        if not _absent(push) and not _valid_push(push):
            return False
    original_push=source.get("push_probability")
    current_push=export.get("push_probability")
    if (not _absent(original_push) and not _absent(current_push)
            and not math.isclose(_number(original_push),_number(current_push),rel_tol=0,abs_tol=1e-9)):
        return False
    line=_number(export.get("line"))
    if (line is not None and not _absent(current_push) and _number(current_push)>1e-9
            and abs(line*2-round(line*2))<=1e-9 and abs(line-round(line))>1e-9):
        return False  # Missing provenance cannot authorize impossible half-point push mass.
    field=saved["source_field"]
    retained_source=(not _absent(export.get("research_source_semantics")) or field in source)
    explicit_contract=(not _absent(source.get("probability_semantics")) or not _absent(original_push))
    if retained_source and explicit_contract:
        # A carrier records labels, not the original probability. Without a
        # retained raw value, it cannot prove any source/export mass match.
        if (_number(source.get(field)) is None or (field=="win_probability"
                and _text(source.get("probability_semantics"))=="win_conditional_on_decision"
                and not _absent(original_push) and _number(original_push)>1e-9)):
            return False
        mass,priced=_export_price_mass(export,source,field,current_push=current_push)
        if mass is None:
            return False
        ev=export.get("ev")
        if (not _absent(ev) and (_number(ev) is None or not math.isclose(
                _number(ev),priced["expected_value"],rel_tol=0,abs_tol=1e-9))):
            return False  # Converted probability cannot retain a differently based EV.
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
    if result["availability_reason"] in {
            "MODEL_TARGET_NOT_RECORDED","ESTIMATE_PROVENANCE_NOT_RECORDED","ESTIMATE_NOT_RECORDED"}:
        rejection=_legacy_value_rejection(export)
        if rejection and result["value_reason"] not in {"INVALID_RECORDED_EV","PRICE_VALUE_MISMATCH"}:
            result["value_reason"]=rejection
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
