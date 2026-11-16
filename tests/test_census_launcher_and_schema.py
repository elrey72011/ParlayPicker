"""Census launcher, census/writer schema agreement and the Actions validator's context check.

These tests are the ones from ``tests_archive/test_post2360_census_launcher.py`` and
``tests_archive/test_post2362_census_closure.py`` that do not read the census workflow file,
which no longer exists. The census code itself is still used by the qualification tooling.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

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
MODULE_COMMAND = [sys.executable, "-u", "-m", "scripts.run_read_only_census"]


def clean_environment():
    env = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "PARLAYPICKER_DRIVE_FOLDER_ID",
        "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT",
    ):
        env.pop(name, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def launch(*args):
    return subprocess.run(
        [*MODULE_COMMAND, *args], cwd=ROOT, env=clean_environment(),
        text=True, capture_output=True, check=False, timeout=30,
    )


def test_exact_workflow_launcher_help_succeeds_without_pythonpath():
    result = launch("--help")

    assert result.returncode == 0, result.stderr
    assert "usage: run_read_only_census.py" in result.stdout
    assert "ModuleNotFoundError" not in result.stderr


def test_exact_workflow_launcher_rejects_invalid_arguments_cleanly(tmp_path):
    result = launch(
        "--checkpoint", str(tmp_path / "checkpoint.json"),
        "--output", str(tmp_path / "report.json"),
        "--max-objects", "not-an-integer",
    )

    assert result.returncode == 2
    assert "invalid int value" in result.stderr
    assert not (tmp_path / "report.json").exists()


def test_missing_configuration_retains_sanitized_blocked_report(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    output = tmp_path / "report.json"

    result = launch(
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--source-revision", "a" * 40, "--max-objects", "1",
        "--max-bytes", "1", "--deadline-seconds", "1",
    )

    assert result.returncode == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "BLOCKED"
    assert report["sanitized_error"] == "EXTERNAL_CONFIGURATION_OR_RUNTIME_FAILURE"
    assert report["checkpoint_retained"] is False
    assert report["hard_kill_artifact_retention"] == "NOT_GUARANTEED"
    assert "service_account" not in output.read_text(encoding="utf-8").lower()
    assert not checkpoint.exists()


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
