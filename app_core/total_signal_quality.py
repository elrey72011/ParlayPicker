"""Versioned prediction-time diagnostics only; never selection or wager authority."""
from core.wager_decisions import finite

VERSION = "mlb-total-inputs-v1"
FIELDS = ("total_input_version", "total_input_status", "total_input_complete",
          "total_input_reason_codes", "total_input_signal_count")
REASONS = {"missing_theover", "stale_empirical_evidence", "missing_target_model",
           "degraded_feature_subset"}


def probability(value):
    value = finite(value)
    return value is not None and 0 <= value <= 1


def assess(row):
    """Presence is not independence, calibration, or proof of input quality.

    COMPLETE means no observed deficiency among these instrumented fields, not
    complete weather/pitcher/lineup coverage. Unknown context is never fabricated.
    """
    market = str(row.get("market_type", ""))
    if str(row.get("league", row.get("sport", ""))).upper() != "MLB" or market not in {"total_over", "total_under"}:
        return {}
    reasons = []
    if not probability(row.get("theover_probability")):
        reasons.append("missing_theover")
    if row.get("recent_regime_penalty_reason") in {"stale_empirical_history", "stale_recent_regime_history"}:
        reasons.append("stale_empirical_evidence")
    target = str(row.get("ml_target", "")) == market and probability(row.get("ml_probability"))
    if not target:
        reasons.append("missing_target_model")
    if str(row.get("degraded_feature_subset_flag", "")).lower() in {"true", "1"}:
        reasons.append("degraded_feature_subset")
    count = sum(probability(row.get(k)) for k in ("theover_probability", "market_probability", "kalshi_probability")) + int(target)
    status = "INCOMPLETE" if not target or count < 2 else "DEGRADED" if reasons else "COMPLETE"
    return dict(total_input_version=VERSION, total_input_status=status,
                total_input_complete=status == "COMPLETE", total_input_reason_codes="|".join(sorted(reasons)),
                total_input_signal_count=count)


def attach(frame):
    """Call only in live candidate generation, never during outcome reporting."""
    out = frame.copy()
    labels = [assess(row) for row in frame.to_dict("records")]
    for field in FIELDS:
        out[field] = [label.get(field) for label in labels]
    return out


def counters(frame):
    rows = [r for r in frame.to_dict("records") if r.get("total_input_version") == VERSION]
    result = {"mlb_total_candidate_count": len(rows)}
    result.update({"mlb_total_" + s.lower() + "_count": sum(r["total_input_status"] == s for r in rows)
                   for s in ("COMPLETE", "DEGRADED", "INCOMPLETE")})
    result.update({"mlb_total_" + reason + "_count": sum(reason in str(r["total_input_reason_codes"]).split("|") for r in rows)
                   for reason in sorted(REASONS)})
    return result


def public_fields(row):
    """Copy saved labels only; legacy predictions remain explicitly unrecorded."""
    if not isinstance(row.get("total_input_version"), str) or row.get("total_input_version") != VERSION:
        return {}
    from app_core.candidate_evidence_schema import evidence_value
    fields = {k: evidence_value(row.get(k)) for k in FIELDS}
    validate(fields)
    return fields


def validate(fields):
    if set(fields) != set(FIELDS) or fields["total_input_version"] != VERSION:
        raise ValueError("Invalid totals diagnostic version/fields")
    if fields["total_input_status"] not in {"COMPLETE", "DEGRADED", "INCOMPLETE"}:
        raise ValueError("Invalid totals diagnostic status")
    if type(fields["total_input_complete"]) is not bool or fields["total_input_complete"] != (fields["total_input_status"] == "COMPLETE"):
        raise ValueError("Invalid totals diagnostic completeness")
    reasons = fields["total_input_reason_codes"]
    if not isinstance(reasons, str) or not set(filter(None, reasons.split("|"))) <= REASONS:
        raise ValueError("Invalid totals diagnostic reasons")
    count = fields["total_input_signal_count"]
    if isinstance(count, bool) or not isinstance(count, (int, float)) or count not in range(5):
        raise ValueError("Invalid totals diagnostic signal count")
