"""Real Git/legacy-guard regressions for the unapproved exact-scope proposal."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import review_clock_scope_exception as proposal


SOURCE = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo).decode().strip()


def write(repo, path, raw):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def commit(repo, message):
    git(repo, "add", "--all")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def fixture(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Offline Fixture")
    git(repo, "config", "user.email", "offline@example.invalid")
    git(repo, "config", "core.autocrlf", "false")
    guard = (SOURCE / proposal.GUARD).read_bytes().replace(b"\r\n", b"\n")
    workflow = b"name: offline fixture\n"
    write(repo, "README.md", b"fixture\n")
    write(repo, proposal.GUARD, guard)
    write(repo, ".github/workflows/paid-launch.yml", workflow)
    write(repo, proposal.TEST, b"assert future.expired == 1\n")
    write(repo, "tests/test_unrelated.py", b"assert 1 == 1\n")
    write(repo, "core/protected.py", b"BOUNDARY = True\n")
    recorded = commit(repo, "recorded baseline")
    manifest = {"base_sha": recorded,
                "required_ancestry": {"pr_2349_merge_commit": recorded},
                "protected_files": ["core/protected.py"],
                "protected_git_blobs": {"core/protected.py": git(repo, "rev-parse", recorded + ":core/protected.py")},
                "tooling_sha256": {proposal.GUARD: hashlib.sha256(guard).hexdigest(),
                                   ".github/workflows/paid-launch.yml": hashlib.sha256(workflow).hexdigest()}}
    manifest_bytes = json.dumps(manifest, indent=2).encode() + b"\n"
    write(repo, proposal.MANIFEST, manifest_bytes)
    base = commit(repo, "integration baseline")
    before = git(repo, "rev-parse", base + ":" + proposal.TEST)
    write(repo, proposal.TEST, b"assert future.expired == 0\nassert future.unavailable == 1\n")
    head = commit(repo, "authorized clock correction")
    binding = proposal.Binding(base, head, before,
                               git(repo, "rev-parse", head + ":" + proposal.TEST),
                               hashlib.sha256(manifest_bytes).hexdigest(),
                               hashlib.sha256(guard).hexdigest())
    return repo, binding


def test_exact_correction_retains_original_failure(fixture):
    repo, binding = fixture
    result = proposal._assess(repo, binding)
    assert result["proposal_status"] == "MATCHED_FOR_REVIEW", result
    assert result["original_guard"]["status"] == "FAIL"
    assert result["original_guard"]["reason_codes"] == ["EXISTING_TEST_EXPECTATION_CHANGED"]
    assert result["approval"] == "REQUIRED" and result["gate_authority"] is False


def test_exact_ci_merge_identity(fixture):
    repo, binding = fixture
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    merge = git(repo, "commit-tree", tree, "-p", binding.base, "-p", binding.head, "-m", "CI merge")
    git(repo, "checkout", "-q", merge)
    assert proposal._assess(repo, binding)["proposal_status"] == "MATCHED_FOR_REVIEW"


def test_crlf_conversion_diagnosed_without_hash_reset(fixture):
    repo, binding = fixture
    git(repo, "config", "core.autocrlf", "true")
    for path in (proposal.GUARD, ".github/workflows/paid-launch.yml", proposal.MANIFEST):
        raw = (repo / path).read_bytes()
        write(repo, path, raw.replace(b"\n", b"\r\n"))
    result = proposal._assess(repo, binding)
    assert result["proposal_status"] == "MATCHED_FOR_REVIEW"
    assert result["original_guard"]["status"] == "FAIL"
    assert "SCOPE_GUARD_TOOLING_HASH_MISMATCH" in result["original_guard"]["reason_codes"]
    assert len(result["checkout_line_ending_conversions"]) == 2


@pytest.mark.parametrize("path,raw", [
    (proposal.TEST, b"assert future.expired == 1\n"),
    ("tests/test_unrelated.py", b"assert 0 == 0\n"),
    ("core/protected.py", b"BOUNDARY = False\n"),
    (proposal.GUARD, b"# changed guard\n"),
    (proposal.MANIFEST, b"{}\n"),
    (".github/workflows/paid-launch.yml", b"name: changed\n"),
])
def test_dirty_existing_files_never_excepted(fixture, path, raw):
    repo, binding = fixture
    write(repo, path, raw)
    assert proposal._assess(repo, binding)["proposal_status"] == "REJECTED"


@pytest.mark.parametrize("path", ["tests/test_unrelated.py", "core/protected.py", "README.md"])
def test_unrelated_committed_work_is_outside_exact_head(fixture, path):
    repo, binding = fixture
    write(repo, path, b"unauthorized change\n")
    commit(repo, "unrelated")
    assert proposal._assess(repo, binding)["proposal_status"] == "REJECTED"


def test_wrong_test_blob_binding(fixture):
    from dataclasses import replace
    repo, binding = fixture
    result = proposal._assess(repo, replace(binding, after_blob=binding.before_blob))
    assert result["proposal_status"] == "REJECTED"


def test_wrong_merge_parent_order(fixture):
    repo, binding = fixture
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    merge = git(repo, "commit-tree", tree, "-p", binding.head, "-p", binding.base, "-m", "wrong parents")
    git(repo, "checkout", "-q", merge)
    assert proposal._assess(repo, binding)["proposal_status"] == "REJECTED"


def test_shadowed_protected_dependency_is_rejected(fixture):
    repo, binding = fixture
    write(repo, "shadow/core/protected.py", b"BOUNDARY = False\n")
    result = proposal._assess(repo, binding)
    assert result["proposal_status"] == "REJECTED"
    assert "PROTECTED_RUNTIME_SHADOWING_RISK" in result["original_guard"]["reason_codes"]


def test_rebound_baseline_is_rejected(fixture):
    from dataclasses import replace
    repo, binding = fixture
    result = proposal._assess(repo, replace(binding, manifest_sha256="0" * 64))
    assert result["proposal_status"] == "REJECTED"


def test_cli_cannot_return_gate_success(fixture):
    repo, _ = fixture
    result = subprocess.run([sys.executable, str(SOURCE / "tools/review_clock_scope_exception.py"), "--repo", str(repo)], capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(result.stdout)["gate_authority"] is False


@pytest.mark.parametrize("path,reason", [
    ("tests/test_unrelated.py", "EXISTING_TEST_EXPECTATION_CHANGED"),
    ("core/protected.py", "PROTECTED_FILE_CHANGED"),
])
def test_other_guard_failures_cannot_be_hidden_by_rebinding(fixture, path, reason):
    from dataclasses import replace
    repo, binding = fixture
    write(repo, path, b"unrelated change\n")
    head = commit(repo, "additional failure")
    result = proposal._assess(repo, replace(binding, head=head))
    assert result["proposal_status"] == "REJECTED"
    assert reason in result["original_guard"]["reason_codes"]
    assert result["reasons"] == ["OTHER_GUARD_REJECTION_OR_TEST_CHANGE"]


def test_dirty_staged_test_is_rejected(fixture):
    repo, binding = fixture
    write(repo, "tests/test_unrelated.py", b"unrelated change\n")
    git(repo, "add", "tests/test_unrelated.py")
    assert proposal._assess(repo, binding)["proposal_status"] == "REJECTED"


def test_same_parents_with_changed_merge_tree_is_rejected(fixture):
    repo, binding = fixture
    write(repo, "README.md", b"different tree\n")
    git(repo, "add", "README.md")
    tree = git(repo, "write-tree")
    merge = git(repo, "commit-tree", tree, "-p", binding.base, "-p", binding.head, "-m", "wrong tree")
    git(repo, "checkout", "-q", merge)
    assert proposal._assess(repo, binding)["proposal_status"] == "REJECTED"
