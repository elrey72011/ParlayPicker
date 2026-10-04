"""Real offline Git fixtures for exact DFS successor seals and scope attacks."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import check_launch_change_scope as guard

SOURCE = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo).decode().strip()


def raw(repo, revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path], cwd=repo)


def write(repo, path, value):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value)


def commit(repo, message):
    git(repo, "add", "--all")
    git(repo, "commit", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def policy_for(repo, binding, implementation):
    manifest = json.loads(raw(repo, binding["base"], guard.MANIFEST_PATH))
    return {
        "schema_version": 4, "policy_version": guard.V4_POLICY_VERSION,
        "approval_reference": guard.V4_APPROVAL_REFERENCE,
        "base_sha": binding["base"], "base_tree": binding["base_tree"],
        "original_manifest_sha256": binding["manifest_sha256"],
        "previous_policy_blob": binding["previous_policy_blob"], "clock_test_blob": binding["clock_blob"],
        "implementation_commit": implementation,
        "implementation_tree": git(repo, "rev-parse", implementation + "^{tree}"),
        "implementation_changes": {p: {"before_blob": guard.blob(binding["base"], p),
                                        "after_blob": guard.blob(implementation, p)} for p in guard.DFS_PATHS},
        "tooling_sha256": {p: hashlib.sha256(raw(repo, implementation, p)).hexdigest()
                           for p in manifest["tooling_sha256"]},
        "unchanged_bindings": {p: guard.blob(binding["base"], p)
                               for p in sorted(set(manifest["protected_files"]) | set(guard.DFS_UNCHANGED_PATHS))},
    }


def seal(repo, binding):
    implementation = commit(repo, "bounded implementation")
    policy = policy_for(repo, binding, implementation)
    write(repo, guard.V4_POLICY_PATH, json.dumps(policy, indent=2).encode() + b"\n")
    candidate = commit(repo, "policy only seal")
    return implementation, candidate, policy


@pytest.fixture
def fx(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "core.autocrlf", "false")
    git(repo, "config", "user.name", "Offline Test")
    git(repo, "config", "user.email", "offline@example.invalid")
    source_guard = guard._drive_previous_guard_source((SOURCE / guard.GUARD_PATH).read_bytes().replace(b"\r\n", b"\n"))
    previous_guard = source_guard.split(b"\nV4_POLICY_PATH =", 1)[0] + guard.DFS_PREVIOUS_CLI
    original_guard = source_guard.split(b"\nPOLICY_PATH =", 1)[0] + b"\n"
    write(repo, "README.md", b"offline DFS fixture\n")
    write(repo, guard.GUARD_PATH, original_guard)
    write(repo, ".github/workflows/paid-launch.yml", b"name: unchanged offline workflow\n")
    write(repo, "core/protected.py", b"PROTECTED = True\n")
    write(repo, guard.CLOCK_TEST, b"assert future.expired == 1\n")
    recorded = commit(repo, "original baseline")
    manifest = {"base_sha": recorded, "required_ancestry": {"pr_2349_merge_commit": recorded},
                "protected_files": ["core/protected.py"],
                "protected_git_blobs": {"core/protected.py": git(repo, "rev-parse", recorded + ":core/protected.py")},
                "tooling_sha256": {p: hashlib.sha256((repo / p).read_bytes()).hexdigest()
                                   for p in (guard.GUARD_PATH, ".github/workflows/paid-launch.yml")}}
    manifest_bytes = json.dumps(manifest, indent=2).encode() + b"\n"
    for path in guard.DFS_UNCHANGED_PATHS:
        if not (repo / path).exists():
            write(repo, path, b"retained exact evidence\n")
    write(repo, guard.MANIFEST_PATH, manifest_bytes)
    write(repo, guard.CLOCK_TEST, b"assert future.expired == 0\nassert future.unavailable == 1\n")
    write(repo, guard.GUARD_PATH, previous_guard)
    write(repo, "tests/test_ncaaf_coverage_scope_policy.py",
          guard._dfs_restore_coverage_fixture((SOURCE / "tests/test_ncaaf_coverage_scope_policy.py").read_bytes().replace(b"\r\n", b"\n")))
    write(repo, "app_core/draftkings_classic.py", b"ORIGINAL_DFS = True\n")
    write(repo, "app/ui/draftkings.py", b"ORIGINAL_UI = True\n")
    base = commit(repo, "approved main with prior policies")
    monkeypatch.setattr(guard, "ROOT", repo)
    binding = {"base": base, "base_tree": git(repo, "rev-parse", "HEAD^{tree}"),
               "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
               "previous_guard_sha256": hashlib.sha256(previous_guard).hexdigest(),
               "previous_policy_blob": guard.blob(base, guard.COVERAGE_POLICY_PATH),
               "clock_blob": guard.blob(base, guard.CLOCK_TEST),
               "successor_guard_sha256": guard.DFS_BINDINGS["successor_guard_sha256"]}
    for path in guard.DFS_PATHS:
        write(repo, path, guard._drive_previous_main_source(path, (SOURCE / path).read_bytes().replace(b"\r\n", b"\n")))
    binding["reviewed_dfs_blobs"] = {p: git(repo, "hash-object", "--", p)
                                     for p in guard.DFS_BINDINGS["reviewed_dfs_blobs"]}
    implementation, candidate, policy = seal(repo, binding)
    return repo, binding, implementation, candidate, policy


def assess(fx, base=None):
    repo, binding, *_ = fx
    return guard._run_dfs_integrated(repo / guard.MANIFEST_PATH, base, binding)


def test_exact_candidate_and_ci_merge_pass_with_no_new_test_exception(fx):
    repo, binding, _, candidate, _ = fx
    code, report = assess(fx)
    assert code == 0, report
    assert report["policy_valid"]
    assert report["new_existing_test_exceptions"] == []
    assert report["protected_changes"] == report["existing_test_changes"] == []
    assert report["approved_exceptions"][0]["retained_unchanged"] is True
    assert set(report["approved_integration_changes"]) == set(guard.DFS_PATHS)
    assert report["original_guard_report"]["status"] == "FAIL"
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    merge = git(repo, "commit-tree", tree, "-p", binding["base"], "-p", candidate, "-m", "offline CI merge")
    git(repo, "checkout", "-q", merge)
    assert assess(fx, binding["base"])[0] == 0


@pytest.mark.parametrize("path", [guard.CLOCK_TEST, guard.MANIFEST_PATH, guard.POLICY_PATH,
    guard.V3_POLICY_PATH, guard.SCHEDULE_POLICY_PATH, guard.COVERAGE_POLICY_PATH,
    "tests/test_ncaaf_coverage_corrections.py", "tests/test_ncaaf_coverage_scope_policy.py",
    "tests/paid_launch/case_isolation_and_scope.py", "app_core/ncaaf_schedule.py",
    "tests/test_ncaaf_schedule_coverage.py", "parlaypicker/app/streamlit_app.py", guard.GUARD_PATH, "core/protected.py", "core/probability_calibration.py",
    "tests/test_draftkings_classic.py", ".github/workflows/paid-launch.yml", "app_core/draftkings_classic.py"])
def test_dirty_staged_and_committed_mutations_fail(fx, path):
    write(fx[0], path, b"unauthorized mutation\n")
    assert assess(fx)[0] != 0
    git(fx[0], "add", path)
    assert assess(fx)[0] != 0
    commit(fx[0], "unauthorized mutation")
    assert assess(fx)[0] != 0


@pytest.mark.parametrize("field,value", [("approval_reference", "blanket exception"),
    ("schema_version", 3), ("base_sha", "0" * 40), ("base_tree", "0" * 40),
    ("implementation_tree", "0" * 40), ("previous_policy_blob", "0" * 40),
    ("tooling_sha256", {}), ("implementation_changes", {}), ("unchanged_bindings", {}),
    ("extra_allowlist", ["tests/"])])
def test_policy_mutation_fails_even_in_single_seal(fx, field, value):
    repo, _, implementation, _, policy = fx
    git(repo, "checkout", "-q", implementation)
    policy = dict(policy, **{field: value})
    write(repo, guard.V4_POLICY_PATH, json.dumps(policy).encode() + b"\n")
    commit(repo, "mutated policy seal")
    assert assess(fx)[0] != 0


@pytest.mark.parametrize("path", ["app_core/draftkings_classic.py", "app/ui/draftkings.py",
    "tests/test_dfs_projection_identity.py", "tests/test_draftkings_classic.py", "core/protected.py",
    "app_core/ncaaf_schedule.py", "core/streamlit_pipeline.py", guard.SCHEDULE_POLICY_PATH,
    guard.COVERAGE_POLICY_PATH, "tests/test_ncaaf_coverage_corrections.py",
    "tests/test_ncaaf_coverage_scope_policy.py", "tests/paid_launch/case_isolation_and_scope.py",
    "parlaypicker/app/streamlit_app.py", guard.GUARD_PATH])
def test_resealed_implementation_cannot_change_reviewed_dfs_or_retained_logic(fx, path):
    repo, binding, implementation, *_ = fx
    git(repo, "checkout", "-q", implementation)
    # Rebuild directly from the approved base, then append a malicious hunk.
    git(repo, "reset", "--soft", binding["base"])
    target = repo / path
    write(repo, path, target.read_bytes() + b"\nFORCE_BYPASS = True\n")
    seal(repo, binding)
    assert assess(fx)[0] != 0


@pytest.mark.parametrize("shadow", ["shadow/app_core/draftkings_classic.py",
    "shadow/core/protected.py", "shadow/core/streamlit_pipeline.py",
    "shadow/app_core/ncaaf_schedule.py", "shadow/app_core/prediction_evidence.py",
    "shadow/tests/test_ncaaf_coverage_corrections.py", "shadow/streamlit_app.py", "sitecustomize.py", "shadow/injected.pth"])
def test_untracked_runtime_shadow_fails(fx, shadow):
    write(fx[0], shadow, b"untracked bypass\n")
    assert assess(fx)[1]["reason_codes"] == ["PROTECTED_RUNTIME_SHADOWING_RISK"]


def test_crlf_conversion_does_not_reseal_baseline(fx):
    repo, binding, *_ = fx
    git(repo, "config", "core.autocrlf", "true")
    for path in (guard.GUARD_PATH, guard.MANIFEST_PATH, guard.POLICY_PATH, guard.V3_POLICY_PATH,
                 guard.SCHEDULE_POLICY_PATH, guard.COVERAGE_POLICY_PATH, guard.V4_POLICY_PATH, guard.CLOCK_TEST, ".github/workflows/paid-launch.yml"):
        write(repo, path, (repo / path).read_bytes().replace(b"\n", b"\r\n"))
    code, report = assess(fx)
    assert code == 0, report
    assert set(report["checkout_line_ending_conversions"]) == {guard.GUARD_PATH, ".github/workflows/paid-launch.yml"}
    assert hashlib.sha256(raw(repo, "HEAD", guard.MANIFEST_PATH)).hexdigest() == binding["manifest_sha256"]


@pytest.mark.parametrize("attack", ["extra_commit", "extra_seal_path", "wrong_merge_parent", "wrong_merge_tree"])
def test_candidate_ancestry_and_policy_only_shape_are_exact(fx, attack):
    repo, binding, implementation, candidate, policy = fx
    if attack == "extra_commit":
        git(repo, "commit", "--allow-empty", "-qm", "extra commit")
    elif attack == "extra_seal_path":
        git(repo, "checkout", "-q", implementation)
        write(repo, guard.V4_POLICY_PATH, json.dumps(policy).encode() + b"\n")
        write(repo, "README.md", b"extra seal content\n")
        commit(repo, "policy plus extra path")
    else:
        if attack == "wrong_merge_tree":
            write(repo, "README.md", b"extra CI tree content\n")
            git(repo, "add", "README.md")
        tree = git(repo, "write-tree")
        parents = [candidate, binding["base"]] if attack == "wrong_merge_parent" else [binding["base"], candidate]
        merge = git(repo, "commit-tree", tree, "-p", parents[0], "-p", parents[1], "-m", "invalid CI merge")
        git(repo, "checkout", "-q", merge)
    assert assess(fx)[0] != 0


def test_prior_guard_and_fixture_reconstruct_exact_reviewed_bytes():
    source = guard._drive_previous_guard_source((SOURCE / guard.GUARD_PATH).read_bytes().replace(b"\r\n", b"\n"))
    previous = guard._dfs_previous_guard_source(source)
    assert hashlib.sha256(previous).hexdigest() == guard.DFS_BINDINGS["previous_guard_sha256"]
    fixture = (SOURCE / "tests/test_ncaaf_coverage_scope_policy.py").read_bytes().replace(b"\r\n", b"\n")
    restored = guard._dfs_restore_coverage_fixture(fixture)
    assert hashlib.sha256(restored).hexdigest() == guard.DFS_PRIOR_COVERAGE_FIXTURE_SHA256


@pytest.mark.parametrize("valid", [True, False])
def test_cli_selects_dfs_successor_and_never_falls_back_to_prior_coverage(fx, monkeypatch, capsys, valid):
    monkeypatch.setattr(guard, "DFS_BINDINGS", fx[1])
    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid DFS successor must not fall back to prior coverage approval")
    monkeypatch.setattr(guard, "_run_coverage_integrated", forbidden)
    monkeypatch.setattr("sys.argv", ["guard", "--manifest", str(fx[0] / guard.MANIFEST_PATH)])
    if not valid:
        policy = dict(fx[4], approval_reference="unauthorized")
        write(fx[0], guard.V4_POLICY_PATH, json.dumps(policy).encode() + b"\n")
    assert guard.main() == (0 if valid else 1)
    report = json.loads(capsys.readouterr().out)
    assert report["policy_valid"] is valid
