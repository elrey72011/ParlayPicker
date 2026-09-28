from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# Default location of the recap-fitted isotonic calibration table written by
# scripts/fit_calibration.py. Callers (e.g. generate_parlays) load this and pass
# it explicitly — nothing in core auto-loads it, so unit tests and callers that
# want raw probabilities are unaffected.
DEFAULT_CALIBRATION_PATH = Path("data/calibration/effective_prob_calibration.json")
CALIBRATION_SCHEMA_VERSION = 2
FITTING_IMPLEMENTATION_VERSION = "weighted-pav-unique-x-v2"


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
    if isinstance(table, (str, bytes)) or not isinstance(table, Sequence) or not table:
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
    if not table:
        return probs
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
    import hashlib
    meta = dict(payload.get("meta") or {})
    meta.pop("calibration_version", None)
    body = dict(payload, meta=meta)
    return hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


class CalibrationTable(list):
    """Keep the exact loaded artifact with its knots; no second metadata read."""
    def __init__(self, payload, *, acceptance=None):
        from copy import deepcopy
        super().__init__(payload["knots"])
        self.payload = deepcopy(payload)
        self.acceptance = deepcopy(acceptance or {})


def calibration_provenance(table, *, now=None) -> dict:
    """Return verified artifact facts only, never model or validation authority."""
    from core.wager_decisions import aware
    if not isinstance(table, CalibrationTable):
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
    return {k: meta[k] for k in ("calibration_version", "calibration_trained_through", "calibration_available_at")}


def save_calibration(table: list[list[float]], path: Path | str, meta: dict | None = None) -> None:
    payload = {"knots": validate_calibration_knots(table), "meta": meta or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def inspect_calibration_artifact(
    path: Path | str | None = None, *, now: object | None = None
) -> dict:
    """Return structured, non-authorizing acceptance diagnostics for one artifact."""
    production_default = path is None
    calibration_path = DEFAULT_CALIBRATION_PATH if production_default else Path(path)
    result = {
        "path": str(calibration_path),
        "requested_mode": "production_default" if production_default else "explicit_research",
        "acceptance_state": "REJECTED",
        "rejection_reasons": [],
    }
    try:
        payload = json.loads(calibration_path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        result["rejection_reasons"].append(f"ARTIFACT_UNREADABLE:{type(exc).__name__}")
        return result
    if not isinstance(payload, dict):
        result["rejection_reasons"].append("ARTIFACT_NOT_OBJECT")
        return result
    try:
        knots = validate_calibration_knots(payload.get("knots"))
    except (TypeError, ValueError) as exc:
        result["rejection_reasons"].append(f"KNOTS_INVALID:{exc}")
        return result
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict):
        result["rejection_reasons"].append("METADATA_INVALID")
        return result
    result.update(
        knot_count=len(knots),
        calibration_version=meta.get("calibration_version"),
        schema_version=meta.get("schema_version", 1),
        fitting_implementation_version=meta.get("fitting_implementation_version"),
    )
    if not production_default:
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

    # Versioned artifacts use the strict v2 provenance contract. Legacy artifacts
    # retain their previous acceptance rules, which avoids silently reinterpreting
    # historical releases while all newly fitted candidates are held to v2.
    if meta.get("schema_version") == CALIBRATION_SCHEMA_VERSION:
        required = (
            "calibration_version", "calibration_trained_through",
            "calibration_available_at", "source", "probability_semantics",
            "fitting_implementation_version", "artifact_status",
            "source_predictor_version", "training_scope",
        )
        missing = [field for field in required if not meta.get(field)]
        if missing:
            result["rejection_reasons"].append("REQUIRED_METADATA_MISSING:" + ",".join(missing))
        if meta.get("fitting_implementation_version") != FITTING_IMPLEMENTATION_VERSION:
            result["rejection_reasons"].append("FITTING_IMPLEMENTATION_INCOMPATIBLE")
        if meta.get("artifact_status") != "PRODUCTION_CANDIDATE":
            result["rejection_reasons"].append("ARTIFACT_NOT_PRODUCTION_CANDIDATE")
        if meta.get("calibration_version") != calibration_digest(payload):
            result["rejection_reasons"].append("CALIBRATION_DIGEST_MISMATCH")
        if meta.get("probability_semantics") != "win_unconditional_with_push; pushes excluded from binary fit":
            result["rejection_reasons"].append("PROBABILITY_SEMANTICS_INCOMPATIBLE")
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


def load_calibration(path: Path | str | None = None) -> list[list[float]] | None:
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
        payload = json.loads(calibration_path.read_text())
        acceptance = inspect_calibration_artifact(None if production_default else calibration_path)
        if acceptance["acceptance_state"] == "REJECTED":
            return None
        payload["knots"] = validate_calibration_knots(payload.get("knots"))
        return CalibrationTable(payload, acceptance=acceptance)
    except (OSError, TypeError, ValueError):
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
