"""CI shard reconciliation after "re-run failed jobs": the newest attempt's report decides. SYNTHETIC reports."""
import importlib
import json
from pathlib import Path

import pytest

runner = importlib.import_module("scripts.run_ci_tests")
NODE = "tests/test_sample.py::test_a"


def _report(root, shard, attempt, *, exit_code=0, directory=None):
    """One shard's uploaded report: its assignment manifest and JUnit file, as download-artifact lays them out."""
    folder = root / (directory or f"full-suite-test-results-{shard}-attempt-{attempt}")
    folder.mkdir(parents=True)
    rows = [{"nodeid": NODE, "path": "tests/test_sample.py", "shard": 1}]
    manifest = dict(schema=runner.MANIFEST_SCHEMA, shard=shard, shards=1, profile_sha256="a" * 64, profile_version=1,
                    collected_count=1, selected_count=1, deselected_count=0, collected_file_count=1,
                    selected_file_count=1, assignment_sha256=runner.digest(rows), assignments=rows, files=[],
                    estimated_shard_seconds=[0], collect_only=False, completed_count=1,
                    execution_complete=exit_code == 0, exit_code=exit_code, completed_nodeids=[NODE])
    (folder / f"full-suite-{shard}-assignment.json").write_text(json.dumps(manifest), encoding="utf-8")
    failure = "" if exit_code == 0 else "<failure message='boom'/>"
    (folder / f"full-suite-{shard}.xml").write_text(
        f"<testsuite><testcase classname='tests.test_sample' name='test_a'>{failure}</testcase></testsuite>",
        encoding="utf-8")
    return folder


def test_a_rerun_shard_is_judged_on_its_newest_attempt(tmp_path):
    _report(tmp_path, 1, 1, exit_code=1)          # the first attempt failed
    with pytest.raises(ValueError, match="Unsuccessful or missing shard execution"):
        runner.reconcile(tmp_path, 1)
    _report(tmp_path, 1, 2)                       # "re-run failed jobs": the shard now passes
    assert runner.reconcile(tmp_path, 1)["status"] == "PASS"


def test_a_newer_failed_attempt_is_not_hidden_by_an_older_pass(tmp_path):
    _report(tmp_path, 1, 1)
    _report(tmp_path, 1, 2, exit_code=1)
    with pytest.raises(ValueError, match="Unsuccessful or missing shard execution"):
        runner.reconcile(tmp_path, 1)


def test_reports_without_an_attempt_in_their_name_count_as_the_first(tmp_path):
    _report(tmp_path, 1, 1, directory="full-suite-test-results-1")
    assert runner.reconcile(tmp_path, 1)["status"] == "PASS"
    _report(tmp_path, 1, 2, exit_code=1)
    with pytest.raises(ValueError, match="Unsuccessful or missing shard execution"):
        runner.reconcile(tmp_path, 1)


def test_newest_attempt_is_chosen_per_shard_and_duplicates_are_still_rejected(tmp_path):
    def manifest(shard, attempt, copy=""):
        return Path(f"r/full-suite-test-results-{shard}-attempt-{attempt}{copy}/full-suite-{shard}-assignment.json")
    kept = runner.latest_attempts([manifest(9, 1), manifest(9, 2), manifest(11, 1), manifest(10, 12), manifest(10, 2)])
    assert kept == sorted([manifest(9, 2), manifest(11, 1), manifest(10, 12)])
    # Two reports for the same shard and attempt are both kept, so the count check fails the run.
    twice = runner.latest_attempts([manifest(9, 2), Path("other/full-suite-test-results-9-attempt-2/full-suite-9-assignment.json")])
    assert len(twice) == 2
    _report(tmp_path, 1, 1)
    _report(tmp_path / "copy", 1, 1)
    with pytest.raises(ValueError, match="Missing or duplicate shard manifests"):
        runner.reconcile(tmp_path, 1)


def test_the_workflow_names_each_report_by_attempt_and_downloads_every_attempt():
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/ci-fifteen.yml").read_text(encoding="utf-8")
    assert "name: full-suite-test-results-${{ matrix.shard }}-attempt-${{ github.run_attempt }}" in text
    assert "pattern: full-suite-test-results-*" in text
