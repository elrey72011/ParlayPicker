"""Policy freezing tests; fixture databases are never live evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3

import pytest

from app_core import prospective_evidence as evidence
from app_core import prospective_validation_plans as plans


SOURCE_COMMIT = "e3c30108827df6da1f7626fa04a552caa8b2c46c"


@pytest.fixture(autouse=True)
def before_holdout(monkeypatch):
    monkeypatch.setattr(evidence, "_clock", lambda: datetime(2026, 9, 23, tzinfo=timezone.utc))


def test_exact_twelve_predeclared_market_plans_remain_research_only(tmp_path):
    path = tmp_path / "canonical.sqlite3"
    frozen = plans.freeze_current_validation_plans(path, source_commit=SOURCE_COMMIT)
    assert len(frozen) == 12
    assert {(r["sport"], r["market_family"]) for r in frozen} == set(plans.EXPECTED_MARKETS)
    assert all(r["version"] == 1 and r["artifact_hash"] and
               r["model_id"] is None and r["calibration_id"] is None and
               r["validation_state"] == "UNVALIDATED" and
               r["production_eligible"] is False and r["recommended_stake"] == 0
               for r in frozen)
    rows = evidence.read_records(path, "prospective_validation_plan")
    assert len(rows) == 12
    for row in rows:
        policy = json.loads(row["payload"])
        assert policy["source_commit"] == SOURCE_COMMIT
        assert policy["policy_source_hash"]
        assert policy["model_scope"]["sport"] == row["sport"]
        assert policy["model_scope"]["market_family"] == row["market_family"]
        assert policy["calibration_requirements"]["required"] is True
        assert policy["clv_policy"]["required"] is True
        assert policy["clv_policy"]["certified_replayable_close_only"] is True
        assert policy["independence_policy"]["duplicate_provider_adds_unit"] is False
        assert policy["push_void_policy"]["pending_or_needs_review"] == "BLOCK_VALIDATION"
    readiness = evidence.all_market_readiness(path)
    assert len(readiness) == 12
    assert all(row["validation_plan_status"] == "FROZEN" and
               row["deployment_state"] == "UNVALIDATED" and
               row["production_eligible"] is False and row["recommended_stake"] == 0
               for row in readiness)


def test_repeated_install_is_idempotent_even_after_source_commit_changes(tmp_path):
    path = tmp_path / "canonical.sqlite3"
    first = plans.freeze_current_validation_plans(path, source_commit=SOURCE_COMMIT)
    second = plans.freeze_current_validation_plans(path, source_commit="a" * 40)
    assert first == second
    assert len(evidence.read_records(path, "prospective_validation_plan")) == 12


def test_material_policy_drift_refuses_existing_plan_without_partial_writes(tmp_path, monkeypatch):
    path = tmp_path / "canonical.sqlite3"
    frozen = plans.freeze_current_validation_plans(path, source_commit=SOURCE_COMMIT)
    monkeypatch.setattr(plans, "MINIMUM_INDEPENDENT_EVENTS", 201)
    with pytest.raises(evidence.EvidenceConflict, match="differs from frozen"):
        plans.freeze_current_validation_plans(path, source_commit=SOURCE_COMMIT)
    assert [row["artifact_hash"] for row in evidence.read_records(
        path, "prospective_validation_plan")] == [row["artifact_hash"] for row in frozen]


def test_explicit_new_version_supersedes_v1_without_rewriting_it(tmp_path):
    path = tmp_path / "canonical.sqlite3"
    initial = plans.freeze_current_validation_plans(path, source_commit=SOURCE_COMMIT)
    changed = plans.plan_specs(source_commit="a" * 40)[0]
    changed.update(validation_plan_id="prospective-nfl-spread-v2", version=2,
                   supersedes_plan_id=initial[0]["validation_plan_id"],
                   plan_policy_version="six-sport-market-v2",
                   policy_source_hash="f" * 64)
    changed["probability_thresholds"]["max_brier"] = 0.23
    evidence.freeze_validation_plan(path, changed)
    current = plans.freeze_current_validation_plans(path, source_commit="b" * 40)
    assert len(current) == 12
    assert current[0]["validation_plan_id"] == "prospective-nfl-spread-v2"
    assert current[0]["version"] == 2
    assert current[0]["artifact_hash"] != initial[0]["artifact_hash"]
    assert len(evidence.read_records(path, "prospective_validation_plan")) == 13
    assert evidence.load_record(path, "prospective_validation_plan",
                                initial[0]["validation_plan_id"])["artifact_hash"] == initial[0]["artifact_hash"]


def test_plan_rows_are_immutable_and_wrong_scope_metadata_rejected(tmp_path):
    path = tmp_path / "canonical.sqlite3"
    plans.freeze_current_validation_plans(path, source_commit=SOURCE_COMMIT)
    with sqlite3.connect(path) as db, pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute("UPDATE prospective_validation_plan SET version=2")
    with sqlite3.connect(path) as db, pytest.raises(sqlite3.IntegrityError, match="append-only"):
        db.execute("DELETE FROM prospective_validation_plan")
    wrong = plans.plan_specs(source_commit=SOURCE_COMMIT)[0]
    wrong["validation_plan_id"] = "wrong-scope"
    wrong["model_scope"]["market_family"] = "TOTAL"
    with pytest.raises(ValueError, match="model scope"):
        evidence.freeze_validation_plan(path, wrong)


def test_freezing_after_holdout_begins_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "_clock", lambda: datetime(2027, 11, 1, tzinfo=timezone.utc))
    path = tmp_path / "canonical.sqlite3"
    with pytest.raises(ValueError, match="before holdout"):
        plans.freeze_current_validation_plans(path, source_commit=SOURCE_COMMIT)
    assert evidence.read_records(path, "prospective_validation_plan") == []


@pytest.mark.parametrize("bad", ["", "not-a-commit", "ABCDEF" * 7, "f" * 64])
def test_source_commit_must_be_exact_git_sha(bad):
    with pytest.raises(ValueError, match="source_commit"):
        plans.plan_specs(source_commit=bad)
