"""Successor seals and adversarial candidates, using real offline Git history."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import check_launch_change_scope as guard

SOURCE = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo).decode().strip()


def raw_blob(repo, revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path], cwd=repo)


def write(repo, path, raw):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def commit(repo, message):
    git(repo, "add", "--all")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def make_policy(repo, binding, implementation):
    manifest = json.loads(raw_blob(repo, binding["base"], guard.MANIFEST_PATH))
    return {
        "schema_version": 3, "policy_version": guard.V3_POLICY_VERSION,
        "approval_reference": guard.V3_APPROVAL_REFERENCE,
        "base_sha": binding["base"], "base_tree": binding["base_tree"],
        "original_manifest_sha256": binding["manifest_sha256"],
        "previous_policy_blob": binding["previous_policy_blob"], "clock_test_blob": binding["clock_blob"],
        "implementation_commit": implementation,
        "implementation_tree": git(repo, "rev-parse", implementation + "^{tree}"),
        "implementation_changes": {path: {"before_blob": guard.blob(binding["base"], path),
                                          "after_blob": guard.blob(implementation, path)}
                                   for path in guard.PROVIDER_PATHS},
        "tooling_sha256": {path: hashlib.sha256(raw_blob(repo, implementation, path)).hexdigest()
                           for path in manifest["tooling_sha256"]},
        "unchanged_bindings": {path: guard.blob(binding["base"], path)
                               for path in guard.PROVIDER_UNCHANGED_PATHS},
    }


def implementation_files(repo):
    for path in guard.PROVIDER_PATHS:
        write(repo, path, (SOURCE / path).read_bytes().replace(b"\r\n", b"\n"))


@pytest.fixture
def fx(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "user.name", "Offline Test")
    git(repo, "config", "user.email", "offline@example.invalid")
    source_guard = (SOURCE / guard.GUARD_PATH).read_bytes().replace(b"\r\n", b"\n")
    original_guard = source_guard.split(b"\nPOLICY_PATH =", 1)[0] + b"\n"
    previous_guard = source_guard.split(b"\nV3_POLICY_PATH =", 1)[0] + b"\ndef main() -> int:\n    pass\n"
    write(repo, "README.md", b"offline fixture\n")
    write(repo, guard.GUARD_PATH, original_guard)
    write(repo, ".github/workflows/paid-launch.yml", b"name: unchanged offline workflow\n")
    for path in ("core/protected.py", "core/probability_calibration.py", "app_core/prediction_evidence.py"):
        write(repo, path, b"PROTECTED = True\n")
    write(repo, guard.CLOCK_TEST, b"assert future.expired == 1\n")
    write(repo, "tests/test_unrelated.py", b"assert 1 == 1\n")
    write(repo, "tests/paid_launch/case_isolation_and_scope.py", b"assert scope_status == 'PASS'\n")
    recorded = commit(repo, "original files")
    protected = ["core/protected.py", "core/probability_calibration.py", "app_core/prediction_evidence.py"]
    manifest = {"base_sha": recorded, "required_ancestry": {"pr_2349_merge_commit": recorded},
                "protected_files": protected,
                "protected_git_blobs": {path: git(repo, "rev-parse", recorded + ":" + path) for path in protected},
                "tooling_sha256": {path: hashlib.sha256((repo / path).read_bytes()).hexdigest()
                                   for path in (guard.GUARD_PATH, ".github/workflows/paid-launch.yml")}}
    raw_manifest = json.dumps(manifest, indent=2).encode() + b"\n"
    write(repo, guard.MANIFEST_PATH, raw_manifest)
    write(repo, guard.GUARD_PATH, previous_guard)
    write(repo, guard.CLOCK_TEST, b"assert future.expired == 0\nassert future.unavailable == 1\n")
    write(repo, guard.POLICY_PATH, b'{"retained": "previous sealed policy"}\n')
    # Reverse the reviewed replacements to reconstruct the immutable starting
    # sources without fetching production history from a shallow CI checkout.
    for path, edits in guard.PROVIDER_DIAGNOSTIC_EDITS.items():
        text = (SOURCE / path).read_bytes().replace(b"\r\n", b"\n").decode()
        for before, after in reversed(edits):
            assert text.count(after) == 1
            text = text.replace(after, before, 1)
        write(repo, path, text.encode())
    base = commit(repo, "approved starting tree with retained baseline")
    monkeypatch.setattr(guard, "ROOT", repo)
    binding = {"base": base, "base_tree": git(repo, "rev-parse", "HEAD^{tree}"),
               "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
               "previous_guard_sha256": hashlib.sha256(previous_guard).hexdigest(),
               "previous_policy_blob": guard.blob(base, guard.POLICY_PATH),
               "clock_blob": guard.blob(base, guard.CLOCK_TEST)}
    implementation_files(repo)
    implementation = commit(repo, "exact implementation")
    policy = make_policy(repo, binding, implementation)
    write(repo, guard.V3_POLICY_PATH, json.dumps(policy, indent=2).encode() + b"\n")
    seal = commit(repo, "policy seal only")
    return repo, binding, implementation, seal, policy


def assess(fx, base=None):
    repo, binding, *_ = fx
    return guard._run_provider_integrated(repo / guard.MANIFEST_PATH, base, binding)


def reseal(repo, binding):
    implementation = commit(repo, "rebound implementation")
    policy = make_policy(repo, binding, implementation)
    write(repo, guard.V3_POLICY_PATH, json.dumps(policy).encode() + b"\n")
    commit(repo, "rebound policy seal")


def test_exact_candidate_separates_retained_exception_and_integration(fx):
    code, report = assess(fx)
    assert code == 0, report
    assert report["approved_exceptions"][0]["retained_unchanged"] is True
    assert report["new_existing_test_exceptions"] == []
    assert set(report["approved_integration_changes"]) == set(guard.PROVIDER_PATHS)
    assert report["protected_changes"] == report["existing_test_changes"] == []
    assert report["original_guard_report"]["status"] == "FAIL"
    assert "EXISTING_TEST_EXPECTATION_CHANGED" in report["original_guard_report"]["reason_codes"]
    assert "SCOPE_GUARD_TOOLING_HASH_MISMATCH" in report["original_guard_report"]["reason_codes"]
    recorded = json.loads((fx[0] / guard.MANIFEST_PATH).read_text())["base_sha"]
    assert assess(fx, recorded)[0] == 0


def test_exact_ci_merge_passes(fx):
    repo, binding, _, seal, _ = fx
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    merge = git(repo, "commit-tree", tree, "-p", binding["base"], "-p", seal, "-m", "offline CI merge")
    git(repo, "checkout", "-q", merge)
    assert assess(fx, binding["base"])[0] == 0


def test_crlf_conversion_reported_without_changing_any_binding(fx):
    repo, binding, *_ = fx
    git(repo, "config", "core.autocrlf", "true")
    for path in (guard.GUARD_PATH, guard.MANIFEST_PATH, guard.POLICY_PATH, guard.V3_POLICY_PATH,
                 guard.CLOCK_TEST, ".github/workflows/paid-launch.yml", "tests/paid_launch/case_isolation_and_scope.py"):
        write(repo, path, (repo / path).read_bytes().replace(b"\n", b"\r\n"))
    code, report = assess(fx)
    assert code == 0, report
    assert set(report["checkout_line_ending_conversions"]) == {guard.GUARD_PATH, ".github/workflows/paid-launch.yml"}
    assert hashlib.sha256(raw_blob(repo, "HEAD", guard.MANIFEST_PATH)).hexdigest() == binding["manifest_sha256"]


@pytest.mark.parametrize("path", [guard.CLOCK_TEST, "tests/test_unrelated.py", "core/protected.py",
                                  "core/probability_calibration.py", "app_core/prediction_evidence.py",
                                  guard.GUARD_PATH, guard.MANIFEST_PATH, guard.POLICY_PATH,
                                  ".github/workflows/paid-launch.yml", "tests/paid_launch/case_isolation_and_scope.py"])
def test_dirty_staged_and_committed_changes_rejected(fx, path):
    repo, *_ = fx
    write(repo, path, b"unauthorized change\n")
    assert assess(fx)[0] != 0
    git(repo, "add", path)
    assert assess(fx)[0] != 0
    commit(repo, "unauthorized change")
    assert assess(fx)[0] != 0


@pytest.mark.parametrize("field,value", [("approval_reference", "blanket approval"),
                                         ("policy_version", "paid-launch-provider-health-v3"),
                                         ("base_sha", "0" * 40), ("base_tree", "0" * 40),
                                         ("tooling_sha256", {}), ("implementation_tree", "0" * 40),
                                         ("clock_test_blob", "0" * 40), ("unchanged_bindings", {}),
                                         ("implementation_changes", {}), ("extra_allowlist", ["tests/"])])
def test_changed_policy_fields_rejected_even_in_single_seal(fx, field, value):
    repo, _, implementation, _, policy = fx
    git(repo, "checkout", "-q", implementation)
    policy = dict(policy)
    policy[field] = value
    write(repo, guard.V3_POLICY_PATH, json.dumps(policy).encode() + b"\n")
    commit(repo, "altered policy seal")
    assert assess(fx)[0] != 0


@pytest.mark.parametrize("path", ["tests/test_unrelated.py", "core/probability_calibration.py",
                                  "app_core/prediction_evidence.py", guard.MANIFEST_PATH,
                                  guard.CLOCK_TEST, ".github/workflows/paid-launch.yml"])
def test_unapproved_paths_cannot_enter_rebound_implementation(fx, path):
    repo, binding, *_ = fx
    git(repo, "checkout", "-q", binding["base"])
    implementation_files(repo)
    write(repo, path, b"malicious rebound\n")
    reseal(repo, binding)
    code, report = assess(fx)
    assert code != 0
    assert report["reason_codes"] == ["IMPLEMENTATION_CHANGE_SET_NOT_APPROVED"]


@pytest.mark.parametrize("path,needle", [
    ("core/streamlit_pipeline.py", "    diagnostics[\"provider_health\"] = provider_health\n"),
    ("core/run_readiness.py", "    report[\"provider_health\"] = sanitized_health(diagnostics.get(\"provider_health\"))\n"),
    ("app/ui/readiness_dashboard.py", "        render_provider_health(diagnostics)\n"),
    ("app_core/espn_ncaaf_odds.py", "        detail = failure(exc)\n"),
])
def test_shared_file_authority_change_rejected_despite_rebound_hashes(fx, path, needle):
    repo, binding, *_ = fx
    git(repo, "checkout", "-q", binding["base"])
    implementation_files(repo)
    raw = (repo / path).read_text(encoding="utf-8")
    assert raw.count(needle) == 1
    # Simulate an authority/staking override hidden beside an approved hunk.
    write(repo, path, raw.replace(needle, needle + "    FORCE_WAGER_AUTHORITY = True\n", 1).encode())
    reseal(repo, binding)
    code, report = assess(fx)
    assert code != 0
    assert report["reason_codes"] == ["DIAGNOSTIC_SCOPE_NOT_APPROVED"]


def test_original_guard_logic_cannot_be_rebound(fx):
    repo, binding, *_ = fx
    git(repo, "checkout", "-q", binding["base"])
    implementation_files(repo)
    raw = (repo / guard.GUARD_PATH).read_bytes()
    write(repo, guard.GUARD_PATH, b"# unauthorized previous-guard alteration\n" + raw)
    reseal(repo, binding)
    code, report = assess(fx)
    assert code != 0
    assert report["reason_codes"] == ["PREVIOUS_GUARD_LOGIC_CHANGED"]


@pytest.mark.parametrize("shadow", ["shadow/core/streamlit_pipeline.py", "shadow/core/run_readiness.py",
                                   "shadow/app/ui/readiness_dashboard.py", "shadow/app_core/provider_health.py",
                                   "shadow/app_core/espn_ncaaf_odds.py",
                                   "sitecustomize.py", "usercustomize.py", "shadow/injected.pth",
                                   "shadow/core/protected.py"])
def test_runtime_shadows_and_hooks_rejected(fx, shadow):
    write(fx[0], shadow, b"unauthorized shadow\n")
    code, report = assess(fx)
    assert code != 0
    assert "PROTECTED_RUNTIME_SHADOWING_RISK" in report["reason_codes"]


@pytest.mark.parametrize("change", ["extra_commit", "seal_extra_file", "wrong_parent_order", "wrong_merge_tree"])
def test_candidate_identity_and_policy_only_seal_required(fx, change):
    repo, binding, implementation, seal, policy = fx
    if change == "extra_commit":
        git(repo, "commit", "--allow-empty", "-qm", "not the reviewed seal")
    elif change == "seal_extra_file":
        git(repo, "checkout", "-q", implementation)
        write(repo, guard.V3_POLICY_PATH, json.dumps(policy).encode() + b"\n")
        write(repo, "README.md", b"unapproved seal edit\n")
        commit(repo, "seal plus unrelated file")
    else:
        if change == "wrong_merge_tree":
            write(repo, "README.md", b"different merge tree\n")
            git(repo, "add", "README.md")
        tree = git(repo, "write-tree")
        parents = [seal, binding["base"]] if change == "wrong_parent_order" else [binding["base"], seal]
        merge = git(repo, "commit-tree", tree, "-p", parents[0], "-p", parents[1], "-m", "invalid CI merge")
        git(repo, "checkout", "-q", merge)
    assert assess(fx)[0] != 0


def test_alternate_manifest_comparison_or_starting_binding_rejected(fx):
    repo, binding, implementation, *_ = fx
    assert assess(fx, implementation)[0] != 0
    assert guard._run_provider_integrated(repo / "other.json", None, binding)[0] != 0
    wrong = dict(binding, base_tree="0" * 40)
    assert guard._run_provider_integrated(repo / guard.MANIFEST_PATH, None, wrong)[0] != 0
