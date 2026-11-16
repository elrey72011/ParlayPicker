from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


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


def test_workflow_uses_clean_module_launcher_from_repository_root():
    workflow = (ROOT / ".github" / "workflows" / "read-only-census.yml").read_text(
        encoding="utf-8")

    assert "python -u -m scripts.run_read_only_census" in workflow
    assert "working-directory: ${{ github.workspace }}" in workflow
    assert "python -u scripts/run_read_only_census.py" not in workflow


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
