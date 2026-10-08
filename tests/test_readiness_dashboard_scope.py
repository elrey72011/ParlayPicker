"""Offline Git fixtures verify exact successor seals and hostile mutations."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from scripts import check_launch_change_scope as guard
from scripts import readiness_dashboard_scope as successor
from scripts.benchmark_drive_history_loading import blocked_network

SOURCE = Path(__file__).resolve().parents[1]


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo).decode().strip()


def raw(repo, revision, path):
    return subprocess.check_output(["git", "show", revision + ":" + path], cwd=repo)


def write(repo, path, data):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def commit(repo, message):
    git(repo, "add", "--all")
    git(repo, "commit", "--allow-empty", "-qm", message)
    return git(repo, "rev-parse", "HEAD")


def seal(repo, binding):
    implementation = commit(repo, "SYNTHETIC reviewed implementation")
    policy = successor.make_policy(guard, binding, implementation)
    write(repo, guard.READINESS_DASHBOARD_POLICY_PATH, json.dumps(policy, indent=2).encode() + b"\n")
    return implementation, commit(repo, "SYNTHETIC policy-only seal"), policy


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    repo = tmp_path_factory.mktemp("readiness-scope") / "r"
    repo.mkdir()
    original_root = guard.ROOT
    try:
        with blocked_network():
            git(repo, "init", "-q")
            git(repo, "config", "user.name", "Offline Test")
            git(repo, "config", "user.email", "offline@example.invalid")
            git(repo, "config", "core.autocrlf", "false")
            current = (SOURCE / guard.GUARD_PATH).read_bytes().replace(b"\r\n", b"\n")
            previous = guard._readiness_dashboard_previous_guard_source(current)
            original = current.split(b"\nPOLICY_PATH =", 1)[0] + b"\n"
            write(repo, guard.GUARD_PATH, original)
            write(repo, "core/protected.py", b"FROZEN = True\n")
            write(repo, "README.md", b"SYNTHETIC offline scope fixture\n")
            write(repo, ".github/workflows/paid-launch.yml", b"name: frozen offline fixture\n")
            write(repo, guard.CLOCK_TEST, b"assert future.expired == 1\n")
            recorded = commit(repo, "SYNTHETIC original frozen baseline")
            manifest = dict(base_sha=recorded, required_ancestry={"pr_2349_merge_commit": recorded},
                protected_files=["core/protected.py"],
                protected_git_blobs={"core/protected.py": git(repo, "rev-parse", recorded + ":core/protected.py")},
                tooling_sha256={p: hashlib.sha256((repo/p).read_bytes()).hexdigest()
                    for p in [guard.GUARD_PATH, ".github/workflows/paid-launch.yml"]})
            manifest_raw = json.dumps(manifest).encode() + b"\n"
            old_policy = json.loads((SOURCE/guard.NCAAF_COMPAT_POLICY_PATH).read_text())
            retained = (set(old_policy["unchanged_bindings"]) | set(old_policy["implementation_changes"])
                | {guard.NCAAF_COMPAT_POLICY_PATH} | set(guard.READINESS_DASHBOARD_FROZEN_PATHS))
            for path in retained:
                if path in {guard.GUARD_PATH, guard.MANIFEST_PATH, ".github/workflows/paid-launch.yml"}:
                    continue
                if path in guard.READINESS_DASHBOARD_PATHS:
                    # The dashboard already existed on main. Its synthetic base
                    # must contain the actual predecessor, not the corrected UI.
                    if guard.exists_at(guard.READINESS_DASHBOARD_BINDINGS["base"], path):
                        write(repo, path, raw(SOURCE, guard.READINESS_DASHBOARD_BINDINGS["base"], path))
                    continue
                if (SOURCE/path).exists():
                    write(repo, path, (SOURCE/path).read_bytes().replace(b"\r\n", b"\n"))
            # Include an assertion outside predecessor allowlists: all existing
            # tests must be frozen by the new policy, not merely protected ones.
            # Existing main also contains committed CRLF sources. Their exact
            # bytes are valid; normalization must not rewrite their identities.
            write(repo, "tests/test_synthetic_existing.py", b"assert 1 == 1\r\n")
            write(repo, guard.NCAAF_COMPAT_POLICY_PATH, json.dumps(old_policy).encode() + b"\n")
            write(repo, guard.MANIFEST_PATH, manifest_raw)
            write(repo, guard.GUARD_PATH, previous)
            base = commit(repo, "SYNTHETIC verified main")
            guard.ROOT = repo
            binding = dict(guard.READINESS_DASHBOARD_BINDINGS, base=base,
                base_tree=git(repo, "rev-parse", "HEAD^{tree}"),
                manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                previous_guard_sha256=hashlib.sha256(previous).hexdigest(),
                previous_policy_blob=guard.blob(base, guard.NCAAF_COMPAT_POLICY_PATH))
            for path in guard.READINESS_DASHBOARD_PATHS:
                write(repo, path, (SOURCE/path).read_bytes().replace(b"\r\n", b"\n"))
            binding["reviewed_blobs"] = {p: git(repo, "hash-object", p)
                for p in guard.READINESS_DASHBOARD_PATHS if p != guard.GUARD_PATH}
            implementation, candidate, policy = seal(repo, binding)
    finally:
        guard.ROOT = original_root
    return repo, binding, implementation, candidate, policy


@pytest.fixture
def fx(tmp_path, monkeypatch, prepared):
    template, binding, implementation, candidate, policy = prepared
    repo = tmp_path / "r"
    shutil.copytree(template, repo)
    monkeypatch.setattr(guard, "ROOT", repo)
    with blocked_network():
        yield repo, binding, implementation, candidate, policy


def assess(fx, base=None):
    try:
        return guard._run_readiness_dashboard_integrated(fx[0]/guard.MANIFEST_PATH, base, fx[1])
    except (OSError, ValueError) as exc:
        return 1, {"reason_codes": [str(exc)]}


def test_exact_seal_and_ci_merge_preserve_predecessor(fx):
    repo, binding, implementation, candidate, policy = fx
    code, report = assess(fx)
    assert code == 0, report
    assert report["policy_valid"] and not report["new_existing_test_exceptions"]
    assert policy["schema_version"] == 27
    assert guard._readiness_dashboard_previous_guard_source(raw(repo, "HEAD", guard.GUARD_PATH), binding) == raw(repo, binding["base"], guard.GUARD_PATH)
    assert git(repo, "diff", "--name-status", implementation, candidate).splitlines() == ["A\t" + guard.READINESS_DASHBOARD_POLICY_PATH]
    assert "tests/test_synthetic_existing.py" in policy["unchanged_bindings"]
    retained = (set(json.loads(raw(repo, binding["base"], guard.NCAAF_COMPAT_POLICY_PATH))["unchanged_bindings"])
        | set(guard.NCAAF_COMPAT_PATHS) | {guard.NCAAF_COMPAT_POLICY_PATH}) - set(guard.READINESS_DASHBOARD_PATHS)
    assert retained.issubset(policy["unchanged_bindings"])
    merge = git(repo, "commit-tree", git(repo, "rev-parse", "HEAD^{tree}"),
        "-p", binding["base"], "-p", candidate, "-m", "SYNTHETIC CI merge")
    git(repo, "checkout", "-q", merge)
    assert assess(fx, binding["base"])[0] == 0


@pytest.mark.parametrize("path", sorted(set(guard.READINESS_DASHBOARD_PATHS)
    | set(guard.READINESS_DASHBOARD_FROZEN_PATHS)
    | {guard.MANIFEST_PATH, guard.NCAAF_COMPAT_POLICY_PATH, "core/protected.py", "tests/test_synthetic_existing.py"}))
def test_dirty_staged_committed_resealed_changes_reject(fx, path):
    repo, binding, _, _, _ = fx
    write(repo, path, (repo/path).read_bytes() + b"\nFORCE_BYPASS = True\n")
    assert assess(fx)[0] != 0
    git(repo, "add", path)
    assert assess(fx)[0] != 0
    commit(repo, "SYNTHETIC unreviewed change")
    assert assess(fx)[0] != 0
    git(repo, "reset", "--soft", binding["base"])
    seal(repo, binding)
    assert assess(fx)[0] != 0


@pytest.mark.parametrize("field,value", [("implementation_changes", {}), ("unchanged_bindings", {}),
    ("tooling_sha256", {}), ("base_tree", "0"*40), ("schema_version", 25),
    ("approval_reference", "blanket approval"), ("implementation_tree", "0"*40), ("extra_allowlist", ["tests/"])])
def test_mutated_policy_rejects(fx, field, value):
    repo, _, implementation, _, policy = fx
    git(repo, "checkout", "-q", implementation)
    write(repo, guard.READINESS_DASHBOARD_POLICY_PATH, json.dumps(dict(policy, **{field: value})).encode() + b"\n")
    commit(repo, "SYNTHETIC mutated seal")
    assert assess(fx)[0] != 0


@pytest.mark.parametrize("attack", ["extra_commit", "extra_seal_path", "wrong_parent_order", "wrong_tree", "shadow", "hook"])
def test_ancestry_seal_shape_and_runtime_shadowing(fx, attack):
    repo, binding, implementation, candidate, _ = fx
    if attack == "extra_commit":
        commit(repo, "SYNTHETIC extra")
    elif attack == "extra_seal_path":
        git(repo, "checkout", "-q", implementation)
        write(repo, "extra.txt", b"extra")
        commit(repo, "SYNTHETIC bad seal")
    elif attack in {"shadow", "hook"}:
        path = "shadow/core/run_readiness.py" if attack == "shadow" else "sitecustomize.py"
        write(repo, path, b"# SYNTHETIC runtime mutation\n")
    else:
        tree = git(repo, "rev-parse", ("HEAD" if attack == "wrong_parent_order" else binding["base"]) + "^{tree}")
        parents = [candidate, binding["base"]] if attack == "wrong_parent_order" else [binding["base"], candidate]
        merge = git(repo, "commit-tree", tree, "-p", parents[0], "-p", parents[1], "-m", "SYNTHETIC hostile merge")
        git(repo, "checkout", "-q", merge)
    assert assess(fx)[0] != 0


def test_all_historical_source_readers_peel_only_reviewed_successor():
    current = (SOURCE/guard.GUARD_PATH).read_bytes().replace(b"\r\n", b"\n")
    previous = guard._readiness_dashboard_previous_guard_source(current)
    assert hashlib.sha256(previous).hexdigest() == guard.READINESS_DASHBOARD_BINDINGS["previous_guard_sha256"]
    for name in dir(guard):
        if name.endswith("_previous_guard_source") and not name.startswith("_readiness_dashboard_"):
            reader = getattr(guard, name)
            assert reader(current) == reader(previous), name
            with pytest.raises(ValueError, match="SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED"):
                reader(current + b"\nUNREVIEWED=True\n")
    for binding in (guard.READINESS_DASHBOARD_BINDINGS, guard.NCAAF_COMPAT_BINDINGS,
                    guard.CI_SCHEDULING_BINDINGS, guard.SLATE_AUDIT_BINDINGS, guard.FOOTBALL_RESEARCH_BINDINGS):
        assert guard._dfs_guard_matches(current, binding["successor_guard_sha256"])
        assert not guard._dfs_guard_matches(current + b"\nUNREVIEWED=True\n", binding["successor_guard_sha256"])
