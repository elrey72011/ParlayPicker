"""Allowlisted owner diagnostics; no qualification or public authority consumer."""
from datetime import datetime, timezone
import json
import math

ORIGIN_COLUMNS = ("ml_inference_status", "ml_estimate_metadata")
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
probability_semantics push_probability status Play_Stake""".split()


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
    return encode(dict(version=1,
        source={k:fact(source.get(k)) for k in SOURCE_FIELDS} if source is not None else None,
        export={k:fact(export.get(k)) for k in EXPORT_FIELDS},
        display={k:display[k] for k in ("source_field","basis","identity","inference_status",
            "availability_reason","value_reason","probability","push_probability","ev")}))
