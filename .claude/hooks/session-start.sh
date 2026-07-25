#!/bin/bash
# ParlayPicker SessionStart hook: keep the calibration feedback loop healthy.
#
#   (a) In a fresh Claude-Code-on-the-web container, make sure the Python deps + pytest
#       are present so scripts/refresh_calibration.py and the test suite actually run.
#   (b) Surface a LOUD reminder when the calibration has gone stale. The grade->refit
#       loop is manual, and a silently week-stale calibration is exactly what produced
#       the bad 19 Jun card. This turns that silent drift into an unmissable prompt.
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

# (b) Calibration freshness guard â€” always; non-blocking.
"${PYTHON_BIN}" - <<'PY' || true
import json, datetime, glob, os, re
CAL = "data/calibration/effective_prob_calibration.json"
STALE_DAYS = 2
try:
    meta = json.load(open(CAL)).get("meta", {})
    fitted, n = meta.get("fitted_on"), meta.get("n_graded")
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
except FileNotFoundError:
    print(f"[calibration] freshness check skipped: {CAL} not found.")
except Exception as e:
    print(f"[calibration] freshness check skipped: {e}")
PY

exit 0

