"""Deployment audit and the hash-bound review record (D11 option B). SYNTHETIC store."""
import importlib.util
from pathlib import Path

import pytest

from app_core import deployment_audit as audit
from app_core import prospective_evidence as pe
from tests.test_prospective_evidence import (  # noqa: F401  (clock is a fixture)
    _trained_market, clock, event, plan, prediction, quote, result)


def _script():
    spec = importlib.util.spec_from_file_location("deployment_review_script",
                                                  Path(__file__).resolve().parents[1] / "scripts" / "deployment_review.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _passed(path, clock):  # noqa: F811
    _trained_market(path, clock)
    clock(hours=8)
    plan(path)
    for name, hour in (("validation", 9), ("holdout", 18)):
        clock(hours=hour)
        event(path, event_id=name, start=hour + 2, clock_hour=hour)
        quote(path, event_id=name, quote_id=f"{name}-quote", stamp=hour)
        prediction(path, event_id=name, quote_id=f"{name}-quote", observation_id=f"{name}-pred", stamp=hour)
        clock(hours=hour + 3)
        result(path, event_id=name, result_id=f"{name}-result", observed=hour + 3, available=hour + 3)
    clock(hours=26)
    assert pe.evaluate_validation_plan(path, "plan")["status"] == "VALIDATION_PASSED"


def test_audit_passes_hard_checks_and_is_reproducible(tmp_path, clock):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _passed(path, clock)
    first = audit.audit(path, "plan")
    checks = {c["check"]: c["status"] for c in first["checks"]}
    assert checks["VALIDATION_PASSED"] == checks["CHRONOLOGY"] == checks["NO_REFERENCE_BOOK_ENTRY"] == "PASS"
    assert first["status"] in ("PASS", "WARN") and len(first["audit_hash"]) == 64
    assert audit.audit(path, "plan")["audit_hash"] == first["audit_hash"]
    assert "Deployment audit" in audit.markdown(first)


def test_reference_book_entry_fails_the_audit(tmp_path, clock, monkeypatch):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _passed(path, clock)
    monkeypatch.setattr(audit, "REFERENCE_BOOKS", frozenset({"TestBook"}))
    assert audit.audit(path, "plan")["status"] == "FAIL"


def test_record_needs_the_reviewed_hash_and_binds_the_reviewer_to_it(tmp_path, clock):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _passed(path, clock)
    script = _script()
    reviewed = audit.audit(path, "plan")
    with pytest.raises(SystemExit, match="changed since it was reviewed"):
        script.record(path, "plan", "0" * 64, "owner")
    out = script.record(path, "plan", reviewed["audit_hash"], "owner")
    review = pe.load_record(path, "prospective_deployment_review", out["deployment_id"])
    assert review["reviewer_id"] == f"owner;{audit.VERSION};sha256={reviewed['audit_hash']}"
    state = pe.deployment_state(path, "NBA", "SPREAD")
    assert state["deployment_state"] != "UNVALIDATED"
    assert state["owner_authorized"] is False and state["recommended_stake"] == 0   # activation is separate


def test_record_refuses_a_failing_audit(tmp_path, clock, monkeypatch):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _passed(path, clock)
    monkeypatch.setattr(audit, "REFERENCE_BOOKS", frozenset({"TestBook"}))
    failing = audit.audit(path, "plan")
    with pytest.raises(SystemExit, match="FAIL"):
        _script().record(path, "plan", failing["audit_hash"], "owner")
    assert pe.read_records(path, "prospective_deployment_review") == []


def test_reviewer_label_rejects_names_that_could_forge_the_binding():
    with pytest.raises(ValueError):
        audit.reviewer_label("owner;sha256=x", "a" * 64)
    with pytest.raises(ValueError):
        audit.reviewer_label("  ", "a" * 64)
