"""Sealed integration and negative regressions using real Git and the original guard."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import check_launch_change_scope as guard

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


def make_policy(repo, binding, implementation):
    manifest = json.loads((repo / guard.MANIFEST_PATH).read_text())
    return {
        "schema_version": 2, "policy_version": "paid-launch-clock-exception-v2",
        "approval_reference": guard.APPROVAL_REFERENCE,
        "approved_proposal_head": binding["proposal"], "shipping_head": binding["shipping"],
        "base_sha": binding["base"], "original_manifest_sha256": binding["manifest_sha256"],
        "integration_commit": implementation, "integration_tree": git(repo, "rev-parse", implementation + "^{tree}"),
        "integration_changes": {path: {"before_blob": guard.blob(binding["shipping"], path),
                                      "after_blob": guard.blob(implementation, path)} for path in guard.INTEGRATION_PATHS},
        "tooling_sha256": {path: hashlib.sha256(subprocess.check_output(["git", "show", implementation + ":" + path], cwd=repo)).hexdigest()
                           for path in manifest["tooling_sha256"]},
        "clock_test_exception": {"path": guard.CLOCK_TEST, "before_blob": binding["before"], "after_blob": binding["after"]},
    }


@pytest.fixture
def fx(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "user.name", "Offline Test")
    git(repo, "config", "user.email", "offline@example.invalid")
    # Full-suite CI intentionally uses a shallow checkout. The unchanged
    # original run() is the prefix of the checked-out guard; fixture history
    # is generated locally and never depends on fetching production history.
    original = (SOURCE / guard.GUARD_PATH).read_bytes().replace(b"\r\n", b"\n").split(b"\nPOLICY_PATH =", 1)[0] + b"\n"
    write(repo, "README.md", b"offline fixture\n")
    write(repo, guard.GUARD_PATH, original)
    write(repo, ".github/workflows/paid-launch.yml", b"name: offline\n")
    write(repo, "core/protected.py", b"PROTECTED = True\n")
    write(repo, "core/probability_calibration.py", b"CALIBRATION = 'original'\n")
    write(repo, guard.CLOCK_TEST, b"assert future.expired == 1\n")
    write(repo, "tests/test_unrelated.py", b"assert 1 == 1\n")
    recorded = commit(repo, "original baseline files")
    manifest = {"base_sha": recorded, "required_ancestry": {"pr_2349_merge_commit": recorded},
                "protected_files": ["core/protected.py", "core/probability_calibration.py"],
                "protected_git_blobs": {path: git(repo, "rev-parse", recorded + ":" + path)
                                        for path in ("core/protected.py", "core/probability_calibration.py")},
                "tooling_sha256": {path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
                                   for path in (guard.GUARD_PATH, ".github/workflows/paid-launch.yml")}}
    raw = json.dumps(manifest, indent=2).encode() + b"\n"
    write(repo, guard.MANIFEST_PATH, raw)
    base = commit(repo, "immutable manifest")
    for path in guard.PROPOSAL_PATHS:
        write(repo, path, (SOURCE / path).read_bytes().replace(b"\r\n", b"\n"))
    proposal = commit(repo, "reviewed proposal")
    git(repo, "checkout", "-q", base)
    write(repo, guard.CLOCK_TEST, b"assert future.expired == 0\nassert future.unavailable == 1\n")
    shipping = commit(repo, "shipping correction")
    binding = {"base": base, "shipping": shipping, "proposal": proposal,
               "before": git(repo, "rev-parse", base + ":" + guard.CLOCK_TEST),
               "after": git(repo, "rev-parse", shipping + ":" + guard.CLOCK_TEST),
               "manifest_sha256": hashlib.sha256(raw).hexdigest(), "original_guard_sha256": hashlib.sha256(original).hexdigest()}
    for path in guard.INTEGRATION_PATHS:
        write(repo, path, (SOURCE / path).read_bytes().replace(b"\r\n", b"\n"))
    implementation = commit(repo, "review implementation")
    monkeypatch.setattr(guard, "ROOT", repo)
    policy = make_policy(repo, binding, implementation)
    write(repo, guard.POLICY_PATH, json.dumps(policy, indent=2).encode() + b"\n")
    seal = commit(repo, "seal exact implementation")
    return repo, binding, implementation, seal, policy


def assess(fx, base=None):
    repo, binding, *_ = fx
    return guard._run_integrated(repo / guard.MANIFEST_PATH, base, binding)


def test_exact_seal_passes_unchanged_subscriber_contract(fx):
    code, report = assess(fx)
    assert code == 0, report
    assert report["protected_changes"] == []
    assert report["existing_test_changes"] == []
    assert report["approved_exceptions"][0]["after_blob"] == fx[1]["after"]
    assert report["original_guard_report"]["status"] == "FAIL"
    assert "EXISTING_TEST_EXPECTATION_CHANGED" in report["original_guard_report"]["reason_codes"]


def test_exact_ci_merge_and_pr_base_pass(fx):
    repo, binding, _, seal, _ = fx
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    merge = git(repo, "commit-tree", tree, "-p", binding["base"], "-p", seal, "-m", "CI")
    git(repo, "checkout", "-q", merge)
    code, report = assess(fx, binding["base"])
    assert code == 0, report


def test_crlf_only_is_reported_without_baseline_reset(fx):
    repo, *_ = fx
    git(repo, "config", "core.autocrlf", "true")
    for path in (guard.GUARD_PATH, guard.MANIFEST_PATH, guard.POLICY_PATH, ".github/workflows/paid-launch.yml"):
        write(repo, path, (repo / path).read_bytes().replace(b"\n", b"\r\n"))
    code, report = assess(fx)
    assert code == 0, report
    assert len(report["checkout_line_ending_conversions"]) == 2


@pytest.mark.parametrize("path", [guard.CLOCK_TEST, "tests/test_unrelated.py", "core/protected.py",
                                  "core/probability_calibration.py", guard.GUARD_PATH,
                                  guard.MANIFEST_PATH, guard.POLICY_PATH, ".github/workflows/paid-launch.yml"])
def test_dirty_and_committed_mutations_rejected(fx, path):
    repo, *_ = fx
    write(repo, path, b"unauthorized\n")
    assert assess(fx)[0] != 0
    commit(repo, "unauthorized mutation")
    assert assess(fx)[0] != 0


def test_staged_existing_test_rejected(fx):
    repo, *_ = fx
    write(repo, "tests/test_unrelated.py", b"assert True\n")
    git(repo, "add", "tests/test_unrelated.py")
    assert assess(fx)[0] != 0


def test_unchanged_tree_extra_commit_is_not_a_seal(fx):
    repo, *_ = fx
    git(repo, "commit", "--allow-empty", "-qm", "unapproved successor")
    assert assess(fx)[0] != 0


@pytest.mark.parametrize("field,value", [("approval_reference", "generic publish approval"),
                                         ("tooling_sha256", {}), ("integration_tree", "0" * 40),
                                         ("clock_test_exception", {}), ("extra_allowlist", ["tests/"])])
def test_changed_policy_fields_rejected_even_when_sealed(fx, field, value):
    repo, _, implementation, _, policy = fx
    git(repo, "checkout", "-q", implementation)
    policy = dict(policy)
    policy[field] = value
    write(repo, guard.POLICY_PATH, json.dumps(policy).encode() + b"\n")
    commit(repo, "altered policy seal")
    assert assess(fx)[0] != 0


def test_unrelated_test_cannot_enter_rebound_implementation(fx):
    repo, binding, _, _, _ = fx
    git(repo, "checkout", "-q", binding["shipping"])
    for path in guard.INTEGRATION_PATHS:
        write(repo, path, (SOURCE / path).read_bytes().replace(b"\r\n", b"\n"))
    write(repo, "tests/test_unrelated.py", b"assert True\n")
    implementation = commit(repo, "malicious rebinding")
    policy = make_policy(repo, binding, implementation)
    write(repo, guard.POLICY_PATH, json.dumps(policy).encode() + b"\n")
    commit(repo, "rebound seal")
    code, report = assess(fx)
    assert code != 0
    assert report["reason_codes"] == ["INTEGRATION_CHANGE_SET_NOT_APPROVED"]


def test_shadow_remains_original_guard_failure(fx):
    repo, *_ = fx
    write(repo, "shadow/core/protected.py", b"PROTECTED = False\n")
    code, report = assess(fx)
    assert code != 0
    assert "PROTECTED_RUNTIME_SHADOWING_RISK" in report["reason_codes"]
    assert report["approved_exceptions"]  # Approved correction never masks shadow rejection.


def test_wrong_ci_parent_order_rejected(fx):
    repo, binding, _, seal, _ = fx
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    merge = git(repo, "commit-tree", tree, "-p", seal, "-p", binding["base"], "-m", "wrong parents")
    git(repo, "checkout", "-q", merge)
    assert assess(fx)[0] != 0


def test_wrong_ci_merge_tree_rejected(fx):
    repo, binding, _, seal, _ = fx
    write(repo, "README.md", b"wrong tree\n")
    git(repo, "add", "README.md")
    tree = git(repo, "write-tree")
    merge = git(repo, "commit-tree", tree, "-p", binding["base"], "-p", seal, "-m", "wrong tree")
    git(repo, "checkout", "-q", merge)
    assert assess(fx)[0] != 0


def test_alternate_comparison_base_and_manifest_rejected(fx):
    repo, binding, implementation, *_ = fx
    assert assess(fx, implementation)[0] != 0
    code, _ = guard._run_integrated(repo / "other.json", None, binding)
    assert code != 0
