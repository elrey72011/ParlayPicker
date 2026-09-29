from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

# Default location of the recap-fitted isotonic calibration table written by
# scripts/fit_calibration.py. Callers (e.g. generate_parlays) load this and pass
# it explicitly — nothing in core auto-loads it, so unit tests and callers that
# want raw probabilities are unaffected.
DEFAULT_CALIBRATION_PATH = Path("data/calibration/effective_prob_calibration.json")
CALIBRATION_SCHEMA_VERSION = 3
FITTING_IMPLEMENTATION_VERSION = "weighted-pav-unique-x-v2"
LEGACY_READABLE_SCHEMA_VERSIONS = frozenset({1, 2})
CONDITIONAL_PROBABILITY_SEMANTICS = "win_conditional_on_decision"
CONDITIONAL_FIT_TARGET = "win_given_decided"
PER_CANDIDATE_PUSH_CONVERSION = "per_candidate_supported_push_v1"
SUPPORTED_CALIBRATION_METHODS = {
    "isotonic": FITTING_IMPLEMENTATION_VERSION,
}


def _is_missing(value: object) -> bool:
    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def validate_calibration_knots(table: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return normalized knots or reject an ambiguous calibration mapping."""
    if (
        isinstance(table, (str, bytes))
        or not isinstance(table, Sequence)
        or len(table) == 0
    ):
        raise ValueError("calibration table must contain at least one knot")
    normalized: list[list[float]] = []
    for index, knot in enumerate(table):
        if isinstance(knot, (str, bytes)) or not isinstance(knot, Sequence) or len(knot) != 2:
            raise ValueError(f"calibration knot {index} must be an [x, y] pair")
        x = _finite_number(knot[0], name=f"calibration knot {index} x")
        y = _finite_number(knot[1], name=f"calibration knot {index} y")
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("calibration knots must lie in [0, 1]")
        if normalized and x <= normalized[-1][0]:
            raise ValueError("calibration knot x values must be strictly increasing")
        if normalized and y < normalized[-1][1]:
            raise ValueError("calibration knot y values must be nondecreasing")
        normalized.append([x, y])
    return normalized


def fit_isotonic_calibration(
    probs: Iterable[float],
    outcomes: Iterable[int],
    *,
    sample_weights: Iterable[float] | None = None,
    missing: str = "reject",
) -> list[list[float]]:
    """Fit a monotone predicted->realized probability mapping via pooled adjacent
    violators (PAV). Dependency-free on purpose: sklearn is not guaranteed in the
    runtime environments this repo deploys to.

    Duplicate predictor values are grouped before fitting and every unique
    predictor coordinate is retained in the returned knots. This makes the
    piecewise-linear representation agree with isotonic regression at the
    observed inputs, including the endpoints of pooled plateaus.

    Outcomes are exactly 1 for WIN or 0 for LOSS; pushes/voids must be excluded
    by the caller. ``missing`` is explicit: ``"reject"`` is the production-safe
    default and ``"drop"`` is available to research callers that deliberately
    choose complete-case fitting.
    """
    if missing not in {"reject", "drop"}:
        raise ValueError("missing must be 'reject' or 'drop'")
    probability_values = list(probs)
    outcome_values = list(outcomes)
    weight_values = [1.0] * len(probability_values) if sample_weights is None else list(sample_weights)
    if len(probability_values) != len(outcome_values):
        raise ValueError("probabilities and outcomes must have equal lengths")
    if len(weight_values) != len(probability_values):
        raise ValueError("sample_weights must match probabilities and outcomes")

    observations: list[tuple[float, float, float]] = []
    for index, (raw_probability, raw_outcome, raw_weight) in enumerate(
        zip(probability_values, outcome_values, weight_values)
    ):
        if any(_is_missing(value) for value in (raw_probability, raw_outcome, raw_weight)):
            if missing == "drop":
                continue
            raise ValueError(f"missing calibration observation at index {index}")
        probability = _finite_number(raw_probability, name=f"probability at index {index}")
        outcome = _finite_number(raw_outcome, name=f"outcome at index {index}")
        weight = _finite_number(raw_weight, name=f"sample weight at index {index}")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("calibration probabilities must lie in [0, 1]")
        if outcome not in {0.0, 1.0}:
            raise ValueError("calibration outcomes must be binary 0/1")
        if weight <= 0.0:
            raise ValueError("calibration sample weights must be positive")
        observations.append((probability, outcome, weight))
    if len(observations) < 2:
        raise ValueError("need at least 2 graded picks to fit calibration")

    # Isotonic regression is a function of x. Collapse duplicate predictor
    # coordinates first so their aggregate weight and label mean are inseparable.
    grouped: dict[float, list[float]] = {}
    for probability, outcome, weight in sorted(observations):
        aggregate = grouped.setdefault(probability, [0.0, 0.0])
        aggregate[0] += outcome * weight
        aggregate[1] += weight
    xs = sorted(grouped)

    # Blocks contain [start_unique_index, end_unique_index, weighted_y, weight].
    blocks: list[list[float]] = []
    for index, x in enumerate(xs):
        weighted_y, weight = grouped[x]
        blocks.append([float(index), float(index), weighted_y, weight])
        while len(blocks) >= 2 and blocks[-2][2] / blocks[-2][3] > blocks[-1][2] / blocks[-1][3]:
            right = blocks.pop()
            blocks[-1][1] = right[1]
            blocks[-1][2] += right[2]
            blocks[-1][3] += right[3]

    fitted = [0.0] * len(xs)
    for start, end, weighted_y, weight in blocks:
        mean = weighted_y / weight
        for index in range(int(start), int(end) + 1):
            fitted[index] = mean
    return validate_calibration_knots([[x, fitted[index]] for index, x in enumerate(xs)])


def apply_calibration(probs: pd.Series, table: list[list[float]] | None) -> pd.Series:
    """Map predicted probabilities through the fitted knots with linear
    interpolation; values outside the fitted range clamp to the end knots.
    Returns ``probs`` unchanged when no table is available."""
    if table is None or len(table) == 0:
        return probs
    if isinstance(table, CalibrationTable) and not table.trusted_snapshot_valid():
        return pd.Series(float("nan"), index=probs.index, dtype=float)
    normalized = validate_calibration_knots(table)
    xs = [k[0] for k in normalized]
    ys = [k[1] for k in normalized]

    def _interp(p: float) -> float:
        if pd.isna(p):
            return p
        if not math.isfinite(float(p)) or not 0.0 <= float(p) <= 1.0:
            return float("nan")
        if p <= xs[0]:
            return ys[0]
        if p >= xs[-1]:
            return ys[-1]
        for i in range(1, len(xs)):
            if p <= xs[i]:
                span = xs[i] - xs[i - 1]
                frac = (p - xs[i - 1]) / span if span > 0 else 0.0
                return ys[i - 1] + frac * (ys[i] - ys[i - 1])
        return ys[-1]

    return pd.to_numeric(probs, errors="coerce").map(_interp).clip(0.0, 1.0)


def calibrated_unconditional_mass(
    conditional_probability: object,
    push_probability: object,
    table: list[list[float]] | None,
) -> dict[str, float] | None:
    """Calibrate ``P(win | decided)`` and attach supported candidate push mass.

    The calibration result remains conditional until this explicit conversion.
    A loaded artifact that has been mutated after inspection is never reused.
    """
    from core.probability_semantics import unconditional_from_conditional

    if isinstance(table, CalibrationTable) and not table.trusted_snapshot_valid():
        return None
    try:
        calibrated = float(
            apply_calibration(pd.Series([conditional_probability]), table).iloc[0]
        )
    except (TypeError, ValueError, IndexError):
        return None
    return unconditional_from_conditional(calibrated, push_probability)


def apply_bucket_calibration(
    probs: pd.Series,
    buckets,
    table: list[list[float]] | None,
    bucket_stats: dict | None,
    shrink_n: int = 50,
) -> pd.Series:
    """Bucket-CONDITIONAL calibration: the global isotonic curve, then a per-bucket tilt.

    The global ``table`` fixes the pooled predicted->realized mapping but averages away the
    fact that accuracy varies a lot by bucket (e.g. MLB under:Agrees ~61% vs over:Neutral
    ~49%). This adds the bucket's realized delta from the overall rate, Laplace-smoothed and
    shrunk by sample size (``n/(n+shrink_n)``), so proven buckets aren't crushed by the pooled
    curve and thin buckets barely move. Identical tilt math to
    ``empirical_tiers.empirical_win_probability`` (pinned by test) — this is just the
    vectorized form so the gates and the lean view share one bucket-aware number.

    Falls back to the plain global calibration when ``bucket_stats`` is missing.
    """
    cal = apply_calibration(probs, table)
    if not bucket_stats or not bucket_stats.get("buckets"):
        return cal
    overall = float(bucket_stats["overall"]["win_rate"])
    bmap = bucket_stats["buckets"]
    cal = pd.to_numeric(cal, errors="coerce")
    bucket_list = list(buckets)

    def _tilt(p, b):
        if pd.isna(p):
            return p
        rec = bmap.get(b)
        if not rec or int(rec.get("n", 0)) <= 0:
            return float(p)
        n = int(rec["n"])
        smoothed = (int(rec["wins"]) + overall * 10.0) / (n + 10.0)
        weight = n / (n + float(shrink_n))
        return float(min(0.95, max(0.05, float(p) + weight * (smoothed - overall))))

    return pd.Series([_tilt(p, b) for p, b in zip(cal, bucket_list)], index=cal.index)


def calibration_digest(payload: dict) -> str:
    """Content identity excluding only the self-referential version field."""
    meta = dict(payload.get("meta") or {})
    meta.pop("calibration_version", None)
    body = dict(payload, meta=meta)
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


class CalibrationTable(list):
    """Loaded artifact whose authority is bound to one immutable byte snapshot.

    ``list`` compatibility is retained for the existing interpolation callers.
    The private binding detects mutation of either exposed knots, payload metadata,
    or acceptance facts before those facts can be reused as trusted provenance.
    """

    def __init__(self, payload, *, acceptance=None, raw_sha256: str | None = None):
        from copy import deepcopy

        copied = deepcopy(payload)
        super().__init__(deepcopy(copied["knots"]))
        self.payload = copied
        self.acceptance = deepcopy(acceptance or {})
        self.raw_sha256 = raw_sha256
        self._binding = self._current_binding()

    def _current_binding(self) -> str:
        body = {
            "knots": list(self),
            "payload": self.payload,
            "acceptance": self.acceptance,
            "raw_sha256": self.raw_sha256,
        }
        try:
            encoded = json.dumps(
                body, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            return "INVALID"
        return hashlib.sha256(encoded).hexdigest()

    def trusted_snapshot_valid(self) -> bool:
        return self._binding == self._current_binding()

    def __bool__(self) -> bool:
        """Only a verified production snapshot is truthy to legacy consumers."""
        return (
            self.trusted_snapshot_valid()
            and self.acceptance.get("acceptance_state") == "PRODUCTION_ACCEPTED"
        )


def calibration_provenance(table, *, now=None) -> dict:
    """Return verified artifact facts only, never model or validation authority."""
    from core.wager_decisions import aware
    if not isinstance(table, CalibrationTable):
        return {}
    if not table.trusted_snapshot_valid():
        return {}
    payload = table.payload
    meta = payload.get("meta") or {}
    validation = meta.get("validation") or {}
    try:
        if list(table) != payload["knots"] or meta.get("calibration_version") != calibration_digest(payload):
            return {}
        train = aware(meta.get("calibration_trained_through"))
        available = aware(meta.get("calibration_available_at"))
        train_end = aware(validation.get("train_end"))
        test_start = aware(validation.get("test_start"))
        clock = aware(now) if now is not None else pd.Timestamp.now(tz="UTC").to_pydatetime()
        if (validation.get("promotable") is not True or not meta.get("source")
                or None in (train, available, train_end, test_start, clock)
                or not train_end.date() < test_start.date()
                or not test_start <= train <= available <= clock):
            return {}
    except (TypeError, ValueError):
        return {}
    return {
        **{
            k: meta[k]
            for k in (
                "calibration_version",
                "calibration_trained_through",
                "calibration_available_at",
            )
        },
        "artifact_raw_sha256": table.raw_sha256,
        "acceptance_state": table.acceptance.get("acceptance_state", "REJECTED"),
        "rejection_reasons": list(table.acceptance.get("rejection_reasons", [])),
    }


def save_calibration(table: list[list[float]], path: Path | str, meta: dict | None = None) -> None:
    payload = {"knots": validate_calibration_knots(table), "meta": meta or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _scope_dict(value: object) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    sport = str(value.get("exact_sport", "")).strip().upper()
    market = str(value.get("exact_market_family", "")).strip().upper()
    return {"exact_sport": sport, "exact_market_family": market} if sport and market else None


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    body = dict(manifest)
    body.pop("manifest_hash", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _inspect_calibration_payload(
    payload: object,
    *,
    path: str,
    requested_mode: str,
    now: object | None = None,
    raw_sha256: str | None = None,
    expected_predictor_version: str | None = None,
    expected_training_scope: Mapping[str, str] | None = None,
) -> dict:
    """Inspect one already-read payload; this function performs no filesystem I/O."""
    result = {
        "path": path,
        "requested_mode": requested_mode,
        "acceptance_state": "REJECTED",
        "rejection_reasons": [],
        "artifact_raw_sha256": raw_sha256,
    }
    if not isinstance(payload, dict):
        result["rejection_reasons"].append("ARTIFACT_NOT_OBJECT")
        return result
    try:
        knots = validate_calibration_knots(payload.get("knots"))
    except (TypeError, ValueError) as exc:
        result["rejection_reasons"].append(f"KNOTS_INVALID:{exc}")
        return result
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        result["rejection_reasons"].append("METADATA_INVALID")
        return result
    raw_version = meta.get("schema_version")
    if raw_version is None:
        schema_version = None
        result["rejection_reasons"].append("SCHEMA_VERSION_MISSING")
    elif isinstance(raw_version, bool) or not isinstance(raw_version, int):
        schema_version = None
        result["rejection_reasons"].append("SCHEMA_VERSION_MALFORMED")
    else:
        schema_version = raw_version
        if schema_version not in LEGACY_READABLE_SCHEMA_VERSIONS | {CALIBRATION_SCHEMA_VERSION}:
            result["rejection_reasons"].append(f"SCHEMA_VERSION_UNSUPPORTED:{schema_version}")
    result.update(
        knot_count=len(knots),
        calibration_version=meta.get("calibration_version"),
        schema_version=schema_version,
        fitting_implementation_version=meta.get("fitting_implementation_version"),
        canonical_artifact_digest=calibration_digest(payload),
    )
    if requested_mode != "production_default":
        result["acceptance_state"] = "RESEARCH_ONLY"
        return result

    validation = meta.get("validation") or {}
    if validation.get("promotable") is not True:
        result["rejection_reasons"].append("CHRONOLOGICAL_VALIDATION_NOT_PROMOTABLE")
    train_end = pd.to_datetime(validation.get("train_end"), errors="coerce", utc=True)
    test_start = pd.to_datetime(validation.get("test_start"), errors="coerce", utc=True)
    if pd.isna(train_end) or pd.isna(test_start):
        result["rejection_reasons"].append("CHRONOLOGICAL_BOUNDARY_MISSING")
    elif train_end.normalize() >= test_start.normalize():
        result["rejection_reasons"].append("HOLDOUT_NOT_STRICTLY_FUTURE")

    if schema_version in LEGACY_READABLE_SCHEMA_VERSIONS or schema_version is None:
        result["rejection_reasons"].append("LEGACY_SCHEMA_NOT_PRODUCTION_ELIGIBLE")
    if schema_version != CALIBRATION_SCHEMA_VERSION:
        # A numerically and chronologically valid legacy/unknown artifact can
        # still be inspected as research. Invalid historical inputs stay rejected.
        only_authority_reasons = {
            "SCHEMA_VERSION_MISSING",
            "SCHEMA_VERSION_MALFORMED",
            "LEGACY_SCHEMA_NOT_PRODUCTION_ELIGIBLE",
        }
        if any(reason.startswith("SCHEMA_VERSION_UNSUPPORTED:") for reason in result["rejection_reasons"]):
            only_authority_reasons.update(
                reason for reason in result["rejection_reasons"]
                if reason.startswith("SCHEMA_VERSION_UNSUPPORTED:")
            )
        if set(result["rejection_reasons"]).issubset(only_authority_reasons):
            result["acceptance_state"] = "RESEARCH_ONLY"
        return result

    required = (
        "calibration_version", "calibration_trained_through",
        "calibration_available_at", "outcome_available_through", "source", "probability_semantics",
        "fit_target", "output_semantics", "push_conversion",
        "fitting_implementation_version", "calibration_method", "artifact_status",
        "source_predictor_version", "training_scope", "fit_manifest",
    )
    missing = [field for field in required if not meta.get(field)]
    if missing:
        result["rejection_reasons"].append("REQUIRED_METADATA_MISSING:" + ",".join(missing))
    method = meta.get("calibration_method")
    expected_implementation = SUPPORTED_CALIBRATION_METHODS.get(method)
    if expected_implementation is None:
        result["rejection_reasons"].append("CALIBRATION_METHOD_UNSUPPORTED")
    elif meta.get("fitting_implementation_version") != expected_implementation:
        result["rejection_reasons"].append("FITTING_IMPLEMENTATION_INCOMPATIBLE")
    if meta.get("artifact_status") != "PRODUCTION_CANDIDATE":
        result["rejection_reasons"].append("ARTIFACT_NOT_PRODUCTION_CANDIDATE")
    if meta.get("calibration_version") != calibration_digest(payload):
        result["rejection_reasons"].append("CALIBRATION_DIGEST_MISMATCH")
    if (
        meta.get("probability_semantics") != CONDITIONAL_PROBABILITY_SEMANTICS
        or meta.get("fit_target") != CONDITIONAL_FIT_TARGET
        or meta.get("output_semantics") != CONDITIONAL_PROBABILITY_SEMANTICS
        or meta.get("push_conversion") != PER_CANDIDATE_PUSH_CONVERSION
    ):
        result["rejection_reasons"].append("PROBABILITY_SEMANTICS_INCOMPATIBLE")

    manifest = meta.get("fit_manifest")
    if not isinstance(manifest, dict):
        result["rejection_reasons"].append("FIT_MANIFEST_INVALID")
    else:
        if manifest.get("manifest_hash") != _manifest_digest(manifest):
            result["rejection_reasons"].append("FIT_MANIFEST_DIGEST_MISMATCH")
        manifest_predictor = manifest.get("source_predictor_version")
        manifest_scope = _scope_dict(manifest.get("training_scope"))
        if manifest_predictor != meta.get("source_predictor_version"):
            result["rejection_reasons"].append("MANIFEST_PREDICTOR_MISMATCH")
        if manifest_scope != _scope_dict(meta.get("training_scope")):
            result["rejection_reasons"].append("MANIFEST_SCOPE_MISMATCH")
        if manifest.get("probability_semantics") != CONDITIONAL_PROBABILITY_SEMANTICS:
            result["rejection_reasons"].append("MANIFEST_SEMANTICS_MISMATCH")
        rows = manifest.get("rows")
        if not isinstance(rows, list):
            result["rejection_reasons"].append("FIT_MANIFEST_ROWS_INVALID")
        else:
            included = [row for row in rows if isinstance(row, dict) and row.get("fit_included") is True]
            if manifest.get("fitted_row_count") != len(included):
                result["rejection_reasons"].append("FIT_MANIFEST_ROW_COUNT_MISMATCH")
            identities = {
                str(row.get("canonical_event_id"))
                for row in included if row.get("canonical_event_id")
            }
            if not included or manifest.get("independent_event_count") != len(identities):
                result["rejection_reasons"].append("FIT_MANIFEST_EVENT_COUNT_MISMATCH")
            required_row_fields = (
                "source_record_id", "source_hash", "canonical_event_id",
                "exact_sport", "exact_market_family", "source_predictor_version",
                "input_probability", "probability_semantics", "outcome_class",
                "event_time", "outcome_available_at", "sample_weight",
            )
            if any(any(row.get(field) is None for field in required_row_fields) for row in included):
                result["rejection_reasons"].append("FIT_MANIFEST_INCLUDED_ROW_INCOMPLETE")
            row_scope = _scope_dict(meta.get("training_scope"))
            if any(
                str(row.get("source_predictor_version")) != str(meta.get("source_predictor_version"))
                or _scope_dict(row) != row_scope
                or row.get("probability_semantics") != CONDITIONAL_PROBABILITY_SEMANTICS
                for row in included
            ):
                result["rejection_reasons"].append("FIT_MANIFEST_ROW_LINEAGE_MISMATCH")
            event_times = pd.to_datetime(
                [row.get("event_time") for row in included], errors="coerce", utc=True
            )
            availability_times = pd.to_datetime(
                [row.get("outcome_available_at") for row in included], errors="coerce", utc=True
            )
            if (
                len(event_times) == 0
                or event_times.isna().any()
                or availability_times.isna().any()
            ):
                result["rejection_reasons"].append("FIT_MANIFEST_CUTOFF_EVIDENCE_MISSING")
            else:
                recorded_train = pd.to_datetime(
                    meta.get("calibration_trained_through"), errors="coerce", utc=True
                )
                recorded_available = pd.to_datetime(
                    meta.get("outcome_available_through"), errors="coerce", utc=True
                )
                if recorded_train != event_times.max():
                    result["rejection_reasons"].append("TRAINING_CUTOFF_MANIFEST_MISMATCH")
                if recorded_available != availability_times.max():
                    result["rejection_reasons"].append("AVAILABILITY_CUTOFF_MANIFEST_MISMATCH")
    expected_scope = _scope_dict(expected_training_scope)
    if not expected_predictor_version or expected_scope is None:
        result["rejection_reasons"].append("CONSUMER_EXPECTATION_MISSING")
    else:
        if meta.get("source_predictor_version") != expected_predictor_version:
            result["rejection_reasons"].append("SOURCE_PREDICTOR_MISMATCH")
        if _scope_dict(meta.get("training_scope")) != expected_scope:
            result["rejection_reasons"].append("TRAINING_SCOPE_MISMATCH")

    trained_through = pd.to_datetime(
        meta.get("calibration_trained_through"), errors="coerce", utc=True
    )
    available_at = pd.to_datetime(
        meta.get("calibration_available_at"), errors="coerce", utc=True
    )
    clock = pd.to_datetime(now, errors="coerce", utc=True) if now is not None else pd.Timestamp.now(tz="UTC")
    if any(pd.isna(value) for value in (trained_through, available_at, clock)):
        result["rejection_reasons"].append("ARTIFACT_CHRONOLOGY_INVALID")
    elif not test_start <= trained_through <= available_at <= clock:
        result["rejection_reasons"].append("ARTIFACT_CHRONOLOGY_INVALID")
    if not result["rejection_reasons"]:
        result["acceptance_state"] = "PRODUCTION_ACCEPTED"
    return result


def _read_artifact_snapshot(path: Path) -> tuple[object, str]:
    raw = path.read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8")), raw_sha256


def inspect_calibration_artifact(
    path: Path | str | None = None,
    *,
    now: object | None = None,
    expected_predictor_version: str | None = None,
    expected_training_scope: Mapping[str, str] | None = None,
) -> dict:
    """Return acceptance diagnostics bound to exactly one byte snapshot."""
    production_default = path is None
    calibration_path = DEFAULT_CALIBRATION_PATH if production_default else Path(path)
    requested_mode = "production_default" if production_default else "explicit_research"
    try:
        payload, raw_sha256 = _read_artifact_snapshot(calibration_path)
    except (OSError, ValueError, UnicodeDecodeError, TypeError) as exc:
        return {
            "path": str(calibration_path),
            "requested_mode": requested_mode,
            "acceptance_state": "REJECTED",
            "rejection_reasons": [f"ARTIFACT_UNREADABLE:{type(exc).__name__}"],
            "artifact_raw_sha256": None,
        }
    return _inspect_calibration_payload(
        payload,
        path=str(calibration_path),
        requested_mode=requested_mode,
        now=now,
        raw_sha256=raw_sha256,
        expected_predictor_version=expected_predictor_version,
        expected_training_scope=expected_training_scope,
    )


def load_calibration(
    path: Path | str | None = None,
    *,
    now: object | None = None,
    expected_predictor_version: str | None = None,
    expected_training_scope: Mapping[str, str] | None = None,
) -> list[list[float]] | None:
    """Load a fitted calibration table, failing closed for production.

    The default artifact drives live selection, tiers, recovery, Kelly sizing,
    and parlays, so it is eligible only after the chronological promotion test
    in :mod:`scripts.fit_calibration` records ``validation.promotable=true``.
    Explicit paths remain permissive for tests, research, and legacy imports.
    Missing, unreadable, or unapproved production artifacts return ``None`` so
    callers fall back to the upstream effective probability.
    """
    production_default = path is None
    calibration_path = DEFAULT_CALIBRATION_PATH if production_default else Path(path)
    try:
        payload, raw_sha256 = _read_artifact_snapshot(calibration_path)
        acceptance = _inspect_calibration_payload(
            payload,
            path=str(calibration_path),
            requested_mode="production_default" if production_default else "explicit_research",
            now=now,
            raw_sha256=raw_sha256,
            expected_predictor_version=expected_predictor_version,
            expected_training_scope=expected_training_scope,
        )
        if acceptance["acceptance_state"] == "REJECTED":
            return None
        payload["knots"] = validate_calibration_knots(payload.get("knots"))
        return CalibrationTable(payload, acceptance=acceptance, raw_sha256=raw_sha256)
    except (OSError, UnicodeDecodeError, TypeError, ValueError):
        return None


def calibrate_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Blend model and market probabilities to reduce overconfidence."""
    if df is None or df.empty or "model_probability" not in df.columns:
        return df

    calibrated = df.copy()
    calibrated["calibrated_probability"] = (
        pd.to_numeric(calibrated["model_probability"], errors="coerce").fillna(0.5) * 0.3
        + pd.to_numeric(calibrated.get("market_probability", 0.5), errors="coerce").fillna(0.5) * 0.7
    ).clip(0.0, 1.0)

    return calibrated
