"""Go-live checklist for one sport and market (roadmap C4 prep). Read-only; grants nothing.

A qualified market reaches a staked pick only when every link of the existing chain
holds. This lists each link, in order, with its status and the next step:

* ``PASS``    the requirement holds now;
* ``ACTION``  the owner can do the next step now (the command or setting is named);
* ``WAIT``    it needs more games or an earlier item first.

It reads the canonical store, the exposure ledger, the market activation directory and
the app's settings (only whether a key is present, never its value). It writes nothing
and never stakes: the live gate (``core.sport_market_gate``) still decides every pick.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

PASS, ACTION, WAIT = "PASS", "ACTION", "WAIT"
DECISION_RECENT = timedelta(hours=36)


def _item(key, title, status, detail, next_step=None):
    return {"key": key, "title": title, "status": status, "detail": detail, "next_step": next_step}


def _present(name, secret):
    try:
        value = secret(name)
    except Exception:
        value = None
    return bool(str(value or "").strip()) and str(value).strip() != "None"


def checklist(sport, family, *, path=None, now=None, setting=None, secret=None,
              ledger_path=None, activations_dir=None):
    """[item, ...] for ``sport`` / ``family`` (for example "NCAAB", "SPREAD")."""
    from app_core import prospective_calibration as calibration
    from app_core import prospective_evidence as evidence
    from core.market_policy import SPORT_MARKET_FAMILIES
    if family not in SPORT_MARKET_FAMILIES.get(sport, ()):
        raise ValueError("unknown sport or market family")
    now = now or datetime.now(timezone.utc)
    setting = setting or os.environ.get
    secret = secret or setting
    name = f"{sport}:{family}"
    items = []

    # 1. Evidence: a bound plan, then validation and holdout passed and reviewed.
    plan = calibration.bound_plan(path, sport, family)
    items.append(_item("bound_plan", "Model and calibration bound to a frozen plan",
                       PASS if plan else ACTION,
                       f"plan {plan['validation_plan_id']} v{plan['version']}" if plan else "no bound plan",
                       None if plan else "Freeze the reviewed draft hash with the Bound plan workflow"))
    readiness = evidence.market_readiness(path, sport, family)
    state = evidence.deployment_state(path, sport, family)
    validated = state["deployment_state"] in {"PROVISIONAL_VALIDATED", "STANDARD_VALIDATED", "PREMIUM_VALIDATED"}
    counts = (f"validation {readiness.get('validation_count', 0)}, holdout {readiness.get('holdout_count', 0)} "
              f"settled games; next blocker {readiness.get('next_blocker')}")
    items.append(_item("validated", "Validation and holdout passed, deployment review recorded",
                       PASS if validated else WAIT,
                       f"{state['deployment_state']} ({state['validation_id']})" if validated else counts,
                       None if validated else
                       "Wait for the windows to fill; when the plan passes, run the Deployment review workflow (D11)"))

    # 2. Owner authority: bankroll and limits, then the market activation.
    ledger_path = ledger_path or setting("PARLAYPICKER_EXPOSURE_LEDGER", None) or "data/exposure/exposure.sqlite3"
    snapshot = None
    try:
        from core.exposure_ledger import snapshot as ledger_snapshot
        snapshot = ledger_snapshot(ledger_path, now=now)
        items.append(_item("exposure_ledger", "Bankroll and exposure limits configured", PASS,
                           f"bankroll {snapshot['bankroll']} {snapshot['currency']}, unit {snapshot['unit_value']}"))
    except Exception as exc:
        items.append(_item("exposure_ledger", "Bankroll and exposure limits configured", ACTION,
                           str(exc) or type(exc).__name__,
                           "python scripts/exposure_ledger.py configure --input <bankroll-and-limits.json> --confirm"))
    directory = Path(activations_dir or setting("PARLAYPICKER_MARKET_ACTIVATIONS_DIR", None)
                     or "data/policies/active_markets")
    record_path = directory / f"{sport}-{family}.json"
    activation = _item("activation", "Owner market activation", WAIT, "needs a validated deployment first",
                       "After the review passes: python scripts/activate_sport_market.py --sport "
                       f"{sport} --market-family {family} --evidence-db <store> --ledger <ledger> "
                       "--policy <reviewed-policy.json> --owner-id <you> --expires-at <date> --confirm. "
                       "Stakes also need the live wager policy (PARLAYPICKER_WAGER_POLICY_PATH, activation-v1) "
                       "to carry this sport's validated exposure limits; without it every stake is $0")
    if validated and snapshot is not None:
        try:
            from core.sport_market_activation import verify_market_activation
            verify_market_activation(json.loads(record_path.read_text(encoding="utf-8")), state, snapshot, now=now)
            activation.update(status=PASS, detail=f"{record_path} verifies against the deployment", next_step=None)
        except FileNotFoundError:
            activation.update(status=ACTION, detail=f"no record at {record_path}")
        except Exception as exc:
            activation.update(status=ACTION, detail=f"{record_path}: {exc}")
    elif validated:
        activation.update(detail="needs the exposure ledger first")
    items.append(activation)

    # 3. Fresh picks: decision capture running, and the app's bridge and its inputs.
    decision = [p for p in evidence.read_records(path, "prospective_prediction", sport=sport, market_family=family)
                if str(p["observation_id"]).startswith("decision-")]
    latest = max((p["prediction_timestamp"] for p in decision), default=None)
    recent = latest is not None and now - evidence._time(latest, "prediction") <= DECISION_RECENT
    items.append(_item("decision_capture", "Fresh decision quotes and predictions (every 15 minutes in slates)",
                       PASS if recent else ACTION if validated else WAIT,
                       f"latest {latest}" if latest else "none recorded",
                       None if recent else
                       f"Set the repository variable PARLAYPICKER_DECISION_MARKETS to include {name} "
                       "(it runs only while the market is qualified)"))
    bridge = str(setting("PARLAYPICKER_QUALIFIED_BRIDGE", "") or "").strip().lower() == "true"
    items.append(_item("bridge", "App bridge on (qualified picks replace the app's own for this market)",
                       PASS if bridge else ACTION, "PARLAYPICKER_QUALIFIED_BRIDGE=" + ("true" if bridge else "unset/false"),
                       None if bridge else "Set PARLAYPICKER_QUALIFIED_BRIDGE=true in the app's settings"))
    drive = _present("PARLAYPICKER_DRIVE_FOLDER_ID", secret) and _present("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", secret)
    items.append(_item("drive", "App can refresh its evidence store from Drive", PASS if drive else ACTION,
                       "Drive credentials present" if drive else "Drive credentials missing",
                       None if drive else
                       "Add PARLAYPICKER_DRIVE_FOLDER_ID and PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT to the app's secrets"))
    gemini = _present("GEMINI_API_KEY", secret) or _present("GOOGLE_API_KEY", secret)
    items.append(_item("gemini", "Gemini secondary review available", PASS if gemini else ACTION,
                       "key present" if gemini else "no GEMINI_API_KEY or GOOGLE_API_KEY",
                       None if gemini else "Add GEMINI_API_KEY to the app's secrets (without it, a staked pick is "
                       "held at $0 unless an outage policy allows a capped stake)"))
    return items


def summary(items):
    """'READY' only when every item passes; otherwise the first item that is not PASS."""
    first = next((item for item in items if item["status"] != PASS), None)
    return {"ready": first is None, "next": first}


def markdown(sport, family, items):
    lines = [f"## Go-live checklist: {sport} {family}", "",
             "| | requirement | status | detail | next step |", "|---|---|---|---|---|"]
    for i, item in enumerate(items, 1):
        lines.append(f"| {i} | {item['title']} | {item['status']} | {item['detail']} | {item['next_step'] or ''} |")
    head = summary(items)
    lines += ["", "Ready to stake: **yes**." if head["ready"] else
              f"Next: **{head['next']['title']}** ({head['next']['status']})."]
    return "\n".join(lines) + "\n"
