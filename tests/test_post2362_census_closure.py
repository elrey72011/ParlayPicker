from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app_core.canonical_schema import (
    CANONICAL_PRIMARY_KEYS,
    RECONCILED_RESEARCH_TABLES,
)
from app_core.prospective_evidence import connect
from app_core.prospective_reconciliation import ensure_reconciliation_schema
from app_core.prospective_remote import _encode, _schema
from app_core.read_only_census import CensusIntegrityError, _canonical_fact
from scripts.validate_github_actions import validate_workflow


ROOT = Path(__file__).resolve().parents[1]


def test_census_workflow_initializes_runtime_directory_before_resume():
    path = ROOT / ".github" / "workflows" / "read-only-census.yml"
    workflow = path.read_text(encoding="utf-8")

    assert validate_workflow(path) == []
    assert workflow.index("name: Initialize census directory") < workflow.index(
        "name: Resume a verified prior checkpoint"
    )
    job_env = workflow.split("steps:", 1)[0]
    assert "CENSUS_DIR" not in job_env
    assert 'census_dir="$RUNNER_TEMP/read-only-census"' in workflow
    assert "python -u -m scripts.run_read_only_census" in workflow


def test_actions_validator_rejects_job_runner_context_but_allows_step_context(tmp_path):
    invalid = tmp_path / "invalid.yml"
    invalid.write_text(
        "name: invalid\non: workflow_dispatch\njobs:\n  census:\n"
        "    runs-on: ubuntu-latest\n    env:\n"
        "      STATE: ${{ runner.temp }}/state\n    steps:\n"
        "      - run: echo ${{ runner.temp }}\n",
        encoding="utf-8",
    )

    diagnostics = validate_workflow(invalid)

    assert len(diagnostics) == 1
    assert "jobs.census.env.STATE" in diagnostics[0]
    assert "context 'runner' is unavailable" in diagnostics[0]


def _writer_objects(tmp_path):
    database = tmp_path / "canonical.sqlite3"
    ensure_reconciliation_schema(database)
    with connect(database) as db:
        schema = _schema(db)
    objects = {}
    for table, (columns, primary) in schema.items():
        values = {column: f"{table}:{column}" for column in columns}
        for evidence_field, hash_field in (
            ("payload", "payload_hash"),
            ("raw_source", "source_hash"),
        ):
            if evidence_field in values and hash_field in values:
                values[evidence_field] = "{}"
                values[hash_field] = hashlib.sha256(b"{}").hexdigest()
        row = tuple(values[column] for column in columns)
        key, raw = _encode(table, columns, primary, row)
        objects[table] = (key, raw)
    return schema, objects


def test_actual_writer_schema_and_census_registry_are_exactly_compatible(tmp_path):
    schema, objects = _writer_objects(tmp_path)

    assert set(schema) == set(CANONICAL_PRIMARY_KEYS)
    assert {table: primary for table, (_, primary) in schema.items()} == dict(
        CANONICAL_PRIMARY_KEYS
    )
    for table, (key, raw) in objects.items():
        facts = _canonical_fact(key, raw)
        assert facts[0]["record_type"] == table


@pytest.mark.parametrize("table", sorted(RECONCILED_RESEARCH_TABLES))
def test_writer_generated_reconciled_rows_remain_research_only(tmp_path, table):
    _, objects = _writer_objects(tmp_path)

    fact = _canonical_fact(*objects[table])[0]

    assert fact["research_only"] is True
    assert fact["production_eligible"] is False
    assert fact["recommended_stake"] == 0.0


def test_unknown_writer_table_wrong_key_and_evidence_hash_remain_blocked(tmp_path):
    _, objects = _writer_objects(tmp_path)
    key, raw = objects["prospective_reconciled_fact"]

    unknown = json.loads(raw)
    unknown["table"] = "prospective_future_table"
    with pytest.raises(CensusIntegrityError, match="CANONICAL_SCHEMA_INVALID"):
        _canonical_fact(key, json.dumps(unknown, sort_keys=True, separators=(",", ":")).encode())

    with pytest.raises(CensusIntegrityError, match="CANONICAL_KEY_MISMATCH"):
        _canonical_fact(key.replace(".json", "-wrong.json"), raw)

    corrupt = json.loads(raw)
    payload_index = corrupt["columns"].index("payload")
    corrupt["row"][payload_index] = '{"changed":true}'
    corrupt_raw = json.dumps(corrupt, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(CensusIntegrityError, match="CANONICAL_EVIDENCE_HASH_MISMATCH"):
        _canonical_fact(key, corrupt_raw)
