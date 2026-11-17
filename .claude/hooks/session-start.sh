#!/bin/bash
# ParlayPicker SessionStart hook: keep the calibration feedback loop healthy.
#
#   (a) In a fresh Claude-Code-on-the-web container, make sure the Python deps + pytest
#       are present so scripts/refresh_calibration.py and the test suite actually run.
#   (b) Say what production code does with the stored calibration files, by asking the
#       loaders it uses. A stored table that load_calibration() refuses shapes no win
#       estimate, so it gets one factual line and no refit prompt. The LOUD reminder is
#       kept for a table that is accepted AND stale: the grade->refit loop is manual, and
#       a silently week-stale calibration is exactly what produced the bad 19 Jun card.
#
# Non-blocking: never fails the session (always exits 0).
set -uo pipefail
cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0

# (a) Dependencies â€” remote web container only; cached after first run; idempotent.
# Install into a project venv. Installing into the distro interpreter can collide
# with system packages (notably cryptography) and leave the session half-broken.
PYTHON_BIN="python3"
if [ "${CLAUDE_CODE_REMOTE:-}" = "true" ] && [ -f requirements.txt ]; then
  VENV_DIR=".claude/.venv"
  if [ ! -x "${VENV_DIR}/bin/python" ]; then
    python3 -m venv "${VENV_DIR}" \
      || echo "[session-start] WARNING: could not create ${VENV_DIR}; dependency install skipped."
  fi
  if [ -x "${VENV_DIR}/bin/python" ]; then
    PYTHON_BIN="${VENV_DIR}/bin/python"
    export PATH="$(pwd)/${VENV_DIR}/bin:${PATH}"
    if [ -n "${CLAUDE_ENV_FILE:-}" ]; then
      printf 'export PATH="%s/.claude/.venv/bin:$PATH"\n' "$(pwd)" >> "${CLAUDE_ENV_FILE}"
    fi
    if ! "${PYTHON_BIN}" -c "import pandas, numpy" >/dev/null 2>&1; then
      "${PYTHON_BIN}" -m pip install --quiet --disable-pip-version-check -r requirements.txt \
        || echo "[session-start] WARNING: venv pip install -r requirements.txt failed; tests may not run."
    fi
    "${PYTHON_BIN}" -c "import pytest" >/dev/null 2>&1 \
      || "${PYTHON_BIN}" -m pip install --quiet --disable-pip-version-check pytest >/dev/null 2>&1 || true
  fi
fi

# (b) Calibration status - always; non-blocking. Each line reports what the loaders
# used by production code say, so a file that is old but not in use is not called stale.
"${PYTHON_BIN}" - <<'PY' || true
import json, datetime, glob, os, re
CAL = "data/calibration/effective_prob_calibration.json"
STALE_DAYS = 2


def production_acceptance(meta):
    """How production code treats the stored calibration: (accepted, scope, verdict).

    Callers apply a loaded table only when it is truthy, which CalibrationTable
    reserves for PRODUCTION_ACCEPTED. Two questions are asked of the loader: the
    plain load_calibration() the pipeline makes, and the scoped one a consumer makes
    for the sport, market family and predictor version the artifact itself declares.
    accepted is True when either returns a usable table; scope is None for the plain
    call and names the declared scope when only the scoped call is accepted.
    Otherwise verdict says what the loader did, with inspect_calibration_artifact()'s
    reasons.
    """
    from core.probability_calibration import inspect_calibration_artifact, load_calibration
    asks = [{}]
    predictor, scope = meta.get("source_predictor_version"), meta.get("training_scope")
    if predictor and isinstance(scope, dict):
        asks.append({"expected_predictor_version": predictor, "expected_training_scope": scope})
    tables = [load_calibration(**ask) for ask in asks]
    for ask, table in zip(asks, tables):
        if table:
            named = "{} {}, predictor {}".format(
                scope.get("exact_sport"), scope.get("exact_market_family"), predictor) if ask else None
            return True, named, ""
    report = inspect_calibration_artifact(**asks[-1])
    reasons = ", ".join(str(r) for r in report.get("rejection_reasons") or []) or "no reason given"
    how = ("refuses it" if all(table is None for table in tables)
           else "loads it as {}, which production code does not apply".format(report.get("acceptance_state")))
    return False, None, f"load_calibration() {how} ({reasons})"


def bucket_statistics():
    """(used, sentence) for bucket_stats.json, from the gate every consumer checks first."""
    import pandas as pd
    from core.empirical_tiers import (
        BUCKET_STATS_MAX_AGE_DAYS as LIMIT, DEFAULT_BUCKET_STATS_PATH as PATH,
        bucket_stats_are_fresh, load_bucket_stats)
    stats = load_bucket_stats()
    if stats is None:
        return False, f"Bucket statistics are NOT used: {PATH} is missing, unreadable or has no buckets."
    meta = stats.get("meta") or {}
    # Same evidence date and day count as bucket_stats_are_fresh, which gives the verdict.
    dated = meta.get("recency_anchor") or meta.get("fitted_on")
    fresh = bucket_stats_are_fresh(stats)
    if not dated:
        return fresh, "Bucket statistics carry no date, so the freshness check passes and they can be used."
    anchor = pd.to_datetime(dated, errors="coerce", utc=True)
    age = None if pd.isna(anchor) else int((pd.Timestamp.now(tz="UTC").normalize() - anchor.normalize()).days)
    label = "newest graded slate" if meta.get("recency_anchor") else "fitted"
    described = f"Bucket statistics ({label} {dated}, age {age}d, limit {LIMIT}d)"
    if fresh:
        return True, f"{described} are fresh enough to be used."
    if age is not None and age > LIMIT:
        return False, f"{described} are stale and NOT used."
    return False, f"Bucket statistics (dated {dated}) fail the freshness check and are NOT used."


# Checked first: whether the bucket statistics are used decides what the calibration
# line may say about the win estimates.
try:
    buckets_used, bucket_line = bucket_statistics()
except Exception as e:
    buckets_used = None
    bucket_line = f"Bucket statistics: could not check whether they are used ({type(e).__name__}: {e})."

try:
    meta = json.load(open(CAL)).get("meta", {})
    fitted, n = meta.get("fitted_on"), meta.get("n_graded")
    stored = f"Stored calibration (fitted {fitted or 'date unknown'}, n={n})"
    try:
        accepted, scope, verdict = production_acceptance(meta)
    except Exception as e:
        accepted, scope, verdict = None, None, f"{type(e).__name__}: {e}"
    if accepted is None:
        print(f"[calibration] {stored}: could not check whether it is applied ({verdict}).")
    elif not accepted:
        raw = " Win estimates are the blend's raw output." if buckets_used is False else ""
        print(f"[calibration] {stored} is NOT applied: {verdict}.{raw}")
    else:
        today = datetime.date.today()
        fitted_date = datetime.date.fromisoformat(fitted) if fitted else None
        age = (today - fitted_date).days if fitted_date else None
        dates = [m.group(1) for f in glob.glob("data/backtest_exports/*.csv")
                 if (m := re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(f)))]
        latest = max(dates) if dates else "n/a"
        if age is None:
            print("[calibration] freshness unknown (no fitted_on in calibration meta).")
        elif age > STALE_DAYS:
            bar = "=" * 74
            print(bar)
            print(f"[calibration] STALE: last refit {fitted} ({age} days ago), n_graded={n}.")
            print(f"[calibration] Latest graded slate on disk: {latest}.")
            print("[calibration] After grading new slates, refresh so the model keeps learning:")
            print("    python3 scripts/refresh_calibration.py <dir-with-export+recap-csvs>")
            print("    then commit data/calibration/*.json and data/backtest_exports/*.csv")
            print(bar)
        else:
            print(f"[calibration] OK: refit {fitted} ({age}d ago), n_graded={n}, latest slate {latest}.")
        if scope:
            print(f"[calibration] Accepted only for {scope}: callers that do not name that scope get no calibration.")
except FileNotFoundError:
    print(f"[calibration] freshness check skipped: {CAL} not found.")
except Exception as e:
    print(f"[calibration] freshness check skipped: {e}")
print(f"[calibration] {bucket_line}")
PY

exit 0

