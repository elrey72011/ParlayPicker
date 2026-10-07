"""Allowlisted owner diagnostics; no qualification or public authority consumer."""
from datetime import datetime, timezone
import json
import math

from app_core.producer_provenance import TRANSFER_FIELDS
ORIGIN_COLUMNS = ("ml_inference_status", "ml_estimate_metadata") + TRANSFER_FIELDS


def carry_origin_columns(frame, predictions):
    """Copy supplied producer facts; absent rows never erase older facts."""
    for column in ORIGIN_COLUMNS:
        if column in predictions:
            supplied = predictions[column].notna()
            frame.loc[predictions.index[supplied], column] = predictions.loc[supplied, column]
SOURCE_FIELDS = """snapshot_id candidate_id matchup_id export_run_id league market_type
best_pick display_pick spread_line total_line market_line_used provider_event_id
provider_namespace quote_id prospective_quote_id quote_bookmaker quote_source
quote_time odds_recorded_at odds_american selected_quote_recorded_at market_period
period settlement_rules prediction_generated_at game_start_utc inference_status
model_status probability_semantics push_probability research_source_semantics
ml_probability ml_probability_source ml_target ml_inference_status ml_estimate_metadata
calibrated_probability best_available_probability best_available_probability_source
selection_probability_source expected_value estimated_expected_value""".split()
EXPORT_FIELDS = """candidate_id matchup_id export_run_id league market_type pick line
odds quote_id quote_source quote_time market_period settlement_rules
prediction_generated_at start win_probability probability_basis ev
probability_semantics push_probability status Play_Stake ml_target""".split()


def fact(value):
    # Preserve absence/type rejection, never stringify arbitrary source objects.
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return {"state":"MISSING"}
    if isinstance(value,datetime):
        return {"state":"VALUE","value":value.isoformat()}
    if isinstance(value,str):
        return {"state":"VALUE","value":value} if value.strip() else {"state":"MISSING"}
    if isinstance(value,bool) or type(value).__name__=="bool_":
        return {"state":"VALUE","value":bool(value)}
    try:
        number=float(value)
        return {"state":"VALUE","value":number} if math.isfinite(number) else {"state":"INVALID"}
    except (ValueError,TypeError,OverflowError):
        return {"state":"INVALID"}


def encode(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


def origin_metadata(source, result, line, *, generated_at):
    """Called only by the actual predictor after its explicit control-flow outcome.

    The continuous approximation has no integer-line push model. Half-point
    integer-score targets have no push state; no probability or parameter changes.
    """
    success=result["ml_inference_status"]=="success"
    half=(line is not None and math.isfinite(line) and abs(line*2-round(line*2))<=1e-9
          and abs(line-round(line))>1e-9)
    return encode(dict(version=1,generated_at=generated_at,
        inference_status=result["ml_inference_status"],
        probability_field="ml_probability",probability=fact(result.get("ml_probability")),
        predictor_id=fact(result.get("ml_probability_source")),target=fact(result.get("ml_target")),
        market_type=fact(source.get("market_type")),line=fact(line),
        probability_semantics="win_unconditional_with_push" if success and half else "UNDECLARED_PUSH_MODEL",
        push_probability=0.0 if success and half else None,
        reason=result.get("ml_unavailable_reason",""),
        identity={k:fact(source.get(k)) for k in ("candidate_id","matchup_id","export_run_id",
            "provider_event_id","provider_namespace","prediction_generated_at","game_start_utc",
            "market_period","period","settlement_rules")}))


def generated_time():
    return datetime.now(timezone.utc).isoformat()


def boundary_trace(source, export, display):
    """Saved only in the owner CSV; original missing facts remain missing.

    Raw model and blended research estimates retain separate fields. Provider
    payloads, features, secrets, review prose and private configuration are omitted.
    """
    from app_core.research_display import missing_identity_fields
    trace = dict(version=2,
        missing_identity_fields=missing_identity_fields(display["identity"]),
        source={k:fact(source.get(k)) for k in SOURCE_FIELDS} if source is not None else None,
        export={k:fact(export.get(k)) for k in EXPORT_FIELDS},
        display={k:display[k] for k in ("source_field","basis","identity","inference_status",
            "availability_reason","value_reason","probability","push_probability","ev")})
    try:
        origin = json.loads(source.get("ml_estimate_metadata", "")) if source is not None else {}
        if "nfl_inputs" in origin:
            from app_core.nfl_inference_evidence import diagnose as diagnose_nfl
            assessment = diagnose_nfl(source, origin)
            trace["nfl_evidence"] = assessment
            if assessment["status"] == "REJECTED":
                trace["first_rejection_stage"] = "producer.nfl_inputs"
        if origin.get("version") == 2:
            from app_core.producer_provenance import diagnose
            diagnostic = diagnose(source, origin)
            trace.update(version=3, origin=diagnostic, first_rejection_stage=(
                None if display["availability_reason"] == "AVAILABLE" else
                "producer.inference" if origin.get("inference_status") != "success" else
                "per_game_export.research_display"))
            if diagnostic.get("first_source_rejection_stage"):
                trace["first_rejection_stage"] = diagnostic["first_source_rejection_stage"]
    except (ValueError, TypeError, AttributeError):
        pass
    if trace.get("nfl_evidence", {}).get("status") == "REJECTED":
        trace["first_rejection_stage"] = "producer.nfl_inputs"
    return encode(trace)


ORIGIN_IDENTITY_FIELDS = frozenset("""candidate_id matchup_id export_run_id provider_event_id
provider_namespace prediction_generated_at game_start_utc market_period period settlement_rules""".split())


def _legacy_origin_rejection(source):
    """Validate supplied producer diagnostics without promoting a blend's inference."""
    from app_core.research_display import _absent, _number, _text, _time
    raw=source.get("ml_estimate_metadata")
    status=source.get("ml_inference_status")
    if _absent(raw) or (isinstance(raw,float) and math.isnan(raw)):
        return None if _absent(status) or (isinstance(status,float) and math.isnan(status)) else "ESTIMATE_PROVENANCE_NOT_RECORDED"
    try:
        item=json.loads(raw)
        keys={"version","generated_at","identity","inference_status","line","market_type","predictor_id",
              "probability","probability_field","probability_semantics","push_probability","reason","target"}
        if not isinstance(item,dict) or set(item)!=keys or type(item["version"]) is not int or item["version"]!=1:
            raise ValueError("Invalid origin schema")
        if item["inference_status"] not in {"success","failed","unavailable"} or item["inference_status"]!=_text(status):
            raise ValueError("Contradictory origin outcome")
        if item["probability_field"]!="ml_probability" or _time(item["generated_at"]) is None or not isinstance(item["reason"],str):
            raise ValueError("Invalid origin provenance")
        if not isinstance(item["identity"],dict) or set(item["identity"])!=ORIGIN_IDENTITY_FIELDS:
            raise ValueError("Invalid origin identity")
        for field in ("line","market_type","predictor_id","probability","target"):
            value=item[field]
            if value not in ({"state":"MISSING"},{"state":"INVALID"}):
                if not isinstance(value,dict) or set(value)!={"state","value"} or value["state"]!="VALUE" or fact(value["value"])!=value:
                    raise ValueError("Invalid origin fact")
        for value in item["identity"].values():
            if value not in ({"state":"MISSING"},{"state":"INVALID"}):
                if not isinstance(value,dict) or set(value)!={"state","value"} or value["state"]!="VALUE" or not isinstance(value["value"],str):
                    raise ValueError("Invalid origin identity fact")
        # Capture assigns a new run ID; it does not change these recorded
        # event/target facts. Compare clocks by instant without overwriting them.
        def event(value):
            import re
            from core.team_mapper import normalize_team_name
            parts=value.split("|")
            if len(parts)!=3: return value
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}",parts[0]): day,home,away=parts
            elif re.fullmatch(r"\d{4}-\d{2}-\d{2}",parts[2]): home,away,day=parts
            else: return value
            return (day,normalize_team_name(home),normalize_team_name(away))
        for field in ORIGIN_IDENTITY_FIELDS-{"export_run_id"}:
            original=item["identity"][field]
            if original.get("state")!="VALUE": continue
            current=source.get(field)
            if _absent(current) or (isinstance(current,float) and math.isnan(current)):
                return "ESTIMATE_PROVENANCE_NOT_RECORDED"
            if field in {"prediction_generated_at","game_start_utc"}:
                if _time(original["value"]) is None or _time(original["value"])!=_time(current):
                    return "ESTIMATE_IDENTITY_MISMATCH"
            elif field=="matchup_id":
                if event(original["value"])!=event(_text(current)): return "ESTIMATE_IDENTITY_MISMATCH"
            elif original["value"]!=_text(current): return "ESTIMATE_IDENTITY_MISMATCH"
        if item["inference_status"]=="failed": return "INFERENCE_FAILED"
        if item["inference_status"]=="unavailable":
            return "INFERENCE_UNAVAILABLE" if _number(source.get("ml_probability")) is not None else None
        line=_number(source.get("total_line" if _text(source.get("market_type")).startswith("total") else "spread_line"))
        expected={"probability":fact(source.get("ml_probability")),"predictor_id":fact(source.get("ml_probability_source")),
                  "target":fact(source.get("ml_target")),"market_type":fact(source.get("market_type")),"line":fact(line)}
        if any(item[k]!=v for k,v in expected.items()): return "ESTIMATE_IDENTITY_MISMATCH"
        probability=_number(source.get("ml_probability"))
        if probability is None or not 0<=probability<=1: return "INVALID_PROBABILITY"
        half=line is not None and abs(line*2-round(line*2))<=1e-9 and abs(line-round(line))>1e-9
        if (item["probability_semantics"]!=("win_unconditional_with_push" if half else "UNDECLARED_PUSH_MODEL")
                or item["push_probability"]!=(0.0 if half else None) or isinstance(item["push_probability"],bool)):
            return "UNSUPPORTED_PROBABILITY_SEMANTICS"
        return None
    except (ValueError,TypeError,KeyError,AttributeError):
        return "ESTIMATE_PROVENANCE_NOT_RECORDED"


def origin_rejection(source):
    """V1 stays frozen; V2 proves orientation using independently named facts."""
    try:
        item = json.loads(source.get("ml_estimate_metadata", ""))
        if isinstance(item, dict) and "ncaaf_inputs" in item:
            from app_core.ncaaf_pipeline_evidence import diagnose as diagnose_ncaaf
            assessment = diagnose_ncaaf(source, item)
            if assessment["status"] != "COMPLETE":
                return assessment["reason"]
            item = dict(item)
            item.pop("ncaaf_inputs")
            source = dict(source, ml_estimate_metadata=encode(item))
        if isinstance(item, dict) and "nhl_inputs" in item:
            from app_core.nhl_puck_line_evidence import diagnose as diagnose_nhl
            assessment = diagnose_nhl(source, item)
            if assessment["status"] != "COMPLETE":
                from app_core.nhl_puck_line_evidence import PUBLIC_REASONS
                return assessment["reason"] if assessment["reason"] in PUBLIC_REASONS else "ESTIMATE_PROVENANCE_NOT_RECORDED"
            item = dict(item)
            item.pop("nhl_inputs")
            source = dict(source, ml_estimate_metadata=encode(item))
        if isinstance(item, dict) and "nfl_inputs" in item:
            from app_core.nfl_inference_evidence import diagnose as diagnose_nfl
            assessment = diagnose_nfl(source, item)
            if assessment["status"] == "REJECTED":
                original = dict(item)
                original.pop("nfl_inputs")
                return origin_rejection(dict(source, ml_estimate_metadata=encode(original))) or "ESTIMATE_PROVENANCE_NOT_RECORDED"
            item = dict(item)
            item.pop("nfl_inputs")
            if assessment["status"] == "COMPLETE":
                # The validated additive refresh is private, not a V1 field.
                # Keep both original stages in retention; remove it only from
                # the temporary legacy-reader projection after full validation.
                item.pop("nfl_ui_reblends", None)
            source = dict(source, ml_estimate_metadata=encode(item))
        if isinstance(item, dict) and item.get("version") == 2:
            from app_core.producer_provenance import diagnose
            diagnostic = diagnose(source, item)
            if diagnostic["reason"]:
                return diagnostic["reason"]
            legacy = dict(item)
            legacy.pop("producer_contract")
            legacy["version"] = 1
            legacy["identity"] = dict(item["identity"], matchup_id={"state":"MISSING"})
            return _legacy_origin_rejection(dict(source, ml_estimate_metadata=encode(legacy)))
    except (ValueError, TypeError, KeyError):
        pass
    return _legacy_origin_rejection(source)
