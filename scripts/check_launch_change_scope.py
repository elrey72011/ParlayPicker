"""Fail closed when paid-launch work touches protected research dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import copy
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "paid-launch" / "launch-baseline-manifest.json"
SELF_PROTECTED = (
    "scripts/check_launch_change_scope.py",
    "docs/paid-launch/launch-baseline-manifest.json",
)


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if check and result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"git {' '.join(args)} failed")
    return result.stdout.strip()


def git_success(*args: str) -> bool:
    return subprocess.run(
        ["git", *args], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exists_at(revision: str, path: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{revision}:{path}"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def changed_entries(base: str) -> list[dict[str, Any]]:
    raw = git("diff", "--name-status", "-M", f"{base}...HEAD")
    entries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0]
        item: dict[str, Any] = {"status": status, "path": parts[-1].replace("\\", "/")}
        if status.startswith("R") and len(parts) == 3:
            item["old_path"] = parts[1].replace("\\", "/")
        entries.append(item)
    return entries


def run(manifest_path: Path, base: str | None) -> tuple[int, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded_base = manifest["base_sha"]
    comparison_base = base or recorded_base
    findings: list[str] = []
    tooling_mismatches: list[str] = []
    for path, expected_hash in manifest.get("tooling_sha256", {}).items():
        candidate = ROOT / path
        if not candidate.is_file() or hash_file(candidate) != expected_hash:
            tooling_mismatches.append(path)
    if tooling_mismatches:
        findings.append("SCOPE_GUARD_TOOLING_HASH_MISMATCH")
    if not git_success("merge-base", "--is-ancestor", manifest["required_ancestry"]["pr_2349_merge_commit"], "HEAD"):
        findings.append("PR_2349_MERGE_NOT_IN_ANCESTRY")
    if not exists_at(comparison_base, "README.md"):
        findings.append("COMPARISON_BASE_NOT_AVAILABLE")
    entries = changed_entries(comparison_base)
    changed_paths = {entry["path"] for entry in entries}
    changed_paths.update(entry.get("old_path", "") for entry in entries)
    protected = manifest["protected_files"]
    baseline_blobs = manifest["protected_git_blobs"]
    protected_changes: list[dict[str, Any]] = []
    for path in protected:
        expected = baseline_blobs[path]
        baseline_blob = git("rev-parse", f"{recorded_base}:{path}", check=False)
        current_blob = git("hash-object", "--", path, check=False)
        changed = path in changed_paths or baseline_blob != expected or current_blob != expected
        if changed:
            protected_changes.append(
                {"path": path, "expected_blob": expected, "baseline_blob": baseline_blob, "current_blob": current_blob}
            )
    if protected_changes:
        findings.append("PROTECTED_FILE_CHANGED")
    existing_test_changes = sorted(
        path for path in changed_paths
        if path.startswith("tests/") and exists_at(recorded_base, path)
    )
    if existing_test_changes:
        findings.append("EXISTING_TEST_EXPECTATION_CHANGED")
    local = {
        "staged": git("diff", "--name-only", "--cached").splitlines(),
        "unstaged": git("diff", "--name-only").splitlines(),
    }
    local_protected = sorted(set(local["staged"] + local["unstaged"]) & set(protected))
    if local_protected:
        findings.append("LOCAL_PROTECTED_CHANGE")
    self_edits: list[str] = []
    for path in SELF_PROTECTED:
        if exists_at(comparison_base, path):
            if git("rev-parse", f"{comparison_base}:{path}") != git("hash-object", "--", path):
                self_edits.append(path)
        elif comparison_base != recorded_base:
            self_edits.append(path)
    if self_edits:
        findings.append("SCOPE_GUARD_OR_BASELINE_CHANGED")
    shadowing: list[str] = []
    for protected_path in protected:
        suffix = Path(protected_path).parts
        for candidate in ROOT.rglob(Path(protected_path).name):
            relative = candidate.relative_to(ROOT).as_posix()
            if tuple(candidate.relative_to(ROOT).parts[-len(suffix):]) == tuple(suffix) and relative != protected_path:
                shadowing.append(relative)
    forbidden_runtime_hooks = sorted(
        path for path in changed_paths
        if Path(path).name in {"sitecustomize.py", "usercustomize.py"} or path.endswith(".pth")
    )
    if shadowing or forbidden_runtime_hooks:
        findings.append("PROTECTED_RUNTIME_SHADOWING_RISK")
    report = {
        "schema_version": 1,
        "base_sha": comparison_base,
        "recorded_base_sha": recorded_base,
        "head_sha": git("rev-parse", "HEAD"),
        "status": "PASS" if not findings else "FAIL",
        "reason_codes": findings,
        "changed_files": entries,
        "protected_changes": protected_changes,
        "existing_test_changes": existing_test_changes,
        "local_changes": local,
        "local_protected_changes": local_protected,
        "self_protected_changes": self_edits,
        "tooling_hash_mismatches": tooling_mismatches,
        "runtime_shadowing": sorted(set(shadowing + forbidden_runtime_hooks)),
    }
    return (0 if not findings else 1), report


POLICY_PATH = "docs/paid-launch/launch-scope-policy-v2.json"
GUARD_PATH = "scripts/check_launch_change_scope.py"
MANIFEST_PATH = "docs/paid-launch/launch-baseline-manifest.json"
CLOCK_TEST = "tests/test_board_diagnostics.py"
PROPOSAL_PATHS = (
    "tools/review_clock_scope_exception.py",
    "tests/test_clock_scope_exception_proposal.py",
    "docs/paid-launch/clock-test-exception-proposal.md",
)
INTEGRATION_PATHS = (*PROPOSAL_PATHS, GUARD_PATH,
                     "tests/test_launch_scope_integration.py",
                     "docs/paid-launch/clock-scope-integration.md")
APPROVAL_REFERENCE = "Owner approval of PR #2374 at its exact reviewed head for draft integration preparation"
PRODUCTION_BINDINGS = {
    "base": "d8f580734c28b712f93e0e4a647e9b21ab1f2928",
    "shipping": "dc211cc9438390c73848ce1a43d512ada00338d8",
    "proposal": "051baf59c07060576fbf7a169f8ef7984e7a7672",
    "before": "cfa07b6b6c622f083cb7d2d7e3a0713780758013",
    "after": "e610143aff5611235f9cfb44da13e2d54e1c6b48",
    "manifest_sha256": "2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343",
    "original_guard_sha256": "d9e4b9c3954d7b1803d23527a77b7e7c34ed5d73e5936d7fe07111da3c930ee8",
}


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT, stderr=subprocess.PIPE)


def blob(revision: str, path: str) -> str | None:
    return git("rev-parse", f"{revision}:{path}") if exists_at(revision, path) else None


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _validate_policy(manifest_path: Path, base: str | None, binding: dict) -> tuple[dict, list[str]]:
    _require(manifest_path.resolve() == (ROOT / MANIFEST_PATH).resolve(), "BASELINE_PATH_NOT_APPROVED")
    original_manifest = git_bytes("show", f"{binding['base']}:{MANIFEST_PATH}")
    _require(hashlib.sha256(original_manifest).hexdigest() == binding["manifest_sha256"], "ORIGINAL_BASELINE_IDENTITY_CHANGED")
    _require((ROOT / MANIFEST_PATH).read_bytes().replace(b"\r\n", b"\n") == original_manifest,
             "BASELINE_CHECKOUT_CHANGED")
    _require(git_bytes("show", f"HEAD:{MANIFEST_PATH}") == original_manifest, "BASELINE_COMMIT_CHANGED")
    original_guard = git_bytes("show", f"{binding['base']}:{GUARD_PATH}")
    _require(hashlib.sha256(original_guard).hexdigest() == binding["original_guard_sha256"], "ORIGINAL_GUARD_IDENTITY_CHANGED")
    manifest = json.loads(original_manifest)
    _require(base in (None, binding["base"], manifest["base_sha"]), "COMPARISON_BASE_NOT_APPROVED")
    _require(not git("diff", "--name-only") and not git("diff", "--cached", "--name-only"), "LOCAL_TRACKED_CHANGE")
    policy_bytes = git_bytes("show", f"HEAD:{POLICY_PATH}")
    _require((ROOT / POLICY_PATH).read_bytes().replace(b"\r\n", b"\n") == policy_bytes, "POLICY_CHECKOUT_CHANGED")
    policy = json.loads(policy_bytes)
    keys = {"schema_version", "policy_version", "approval_reference", "approved_proposal_head",
            "shipping_head", "base_sha", "original_manifest_sha256", "integration_commit",
            "integration_tree", "integration_changes", "tooling_sha256", "clock_test_exception"}
    _require(set(policy) == keys and policy["schema_version"] == 2 and
             policy["policy_version"] == "paid-launch-clock-exception-v2", "SUCCESSOR_POLICY_SCHEMA_INVALID")
    expected_clock = {"path": CLOCK_TEST, "before_blob": binding["before"], "after_blob": binding["after"]}
    _require(policy["approval_reference"] == APPROVAL_REFERENCE and
             policy["approved_proposal_head"] == binding["proposal"] and
             policy["shipping_head"] == binding["shipping"] and policy["base_sha"] == binding["base"] and
             policy["original_manifest_sha256"] == binding["manifest_sha256"] and
             policy["clock_test_exception"] == expected_clock, "APPROVAL_BINDING_CHANGED")
    implementation = policy["integration_commit"]
    _require(git("show", "-s", "--format=%P", implementation).split() == [binding["shipping"]], "INTEGRATION_PARENT_NOT_APPROVED")
    _require(git("rev-parse", f"{implementation}^{{tree}}") == policy["integration_tree"], "INTEGRATION_TREE_MISMATCH")
    _require(not exists_at(implementation, POLICY_PATH), "POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION")
    changed = git("diff", "--name-only", binding["shipping"], implementation).splitlines()
    _require(set(changed) == set(INTEGRATION_PATHS), "INTEGRATION_CHANGE_SET_NOT_APPROVED")
    expected_changes = {path: {"before_blob": blob(binding["shipping"], path), "after_blob": blob(implementation, path)}
                        for path in INTEGRATION_PATHS}
    _require(policy["integration_changes"] == expected_changes, "INTEGRATION_BLOB_BINDINGS_CHANGED")
    for path in PROPOSAL_PATHS:
        _require(blob(implementation, path) == blob(binding["proposal"], path), "REVIEWED_PROPOSAL_CHANGED")
    _require(blob(binding["base"], CLOCK_TEST) == binding["before"] and
             blob(binding["shipping"], CLOCK_TEST) == binding["after"] and
             blob("HEAD", CLOCK_TEST) == binding["after"], "CLOCK_TEST_BLOB_NOT_APPROVED")
    head = git("rev-parse", "HEAD")
    parents = git("show", "-s", "--format=%P", head).split()
    if len(parents) == 2:
        _require(parents[0] == binding["base"], "CI_BASE_PARENT_NOT_APPROVED")
        candidate = parents[1]
        _require(git("rev-parse", f"{head}^{{tree}}") == git("rev-parse", f"{candidate}^{{tree}}"), "CI_MERGE_TREE_CHANGED")
    else:
        candidate = head
    _require(git("show", "-s", "--format=%P", candidate).split() == [implementation], "CANDIDATE_NOT_POLICY_SEAL")
    _require(git("diff", "--name-status", implementation, candidate).splitlines() == [f"A\t{POLICY_PATH}"], "SEAL_CHANGE_SET_NOT_APPROVED")
    tooling = {path: hashlib.sha256(git_bytes("show", f"{implementation}:{path}")).hexdigest()
               for path in manifest["tooling_sha256"]}
    _require(policy["tooling_sha256"] == tooling, "SUCCESSOR_TOOLING_BINDING_CHANGED")
    _require(tooling[".github/workflows/paid-launch.yml"] == manifest["tooling_sha256"][".github/workflows/paid-launch.yml"], "WORKFLOW_CHANGE_NOT_APPROVED")
    conversions = []
    for path, expected in tooling.items():
        raw = git_bytes("show", f"HEAD:{path}")
        checkout = (ROOT / path).read_bytes()
        _require(hashlib.sha256(raw).hexdigest() == expected and checkout.replace(b"\r\n", b"\n") == raw,
                 "UNAUTHORIZED_TOOLING_CHANGE")
        if checkout != raw:
            conversions.append(path)
    return policy, conversions


def _run_integrated(manifest_path: Path, base: str | None, binding: dict) -> tuple[int, dict]:
    # The original run() remains unchanged. Its raw result is retained in full.
    try:
        policy, conversions = _validate_policy(manifest_path, base, binding)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        return 1, {"schema_version": 2, "status": "FAIL", "reason_codes": [str(exc)],
                   "approved_exceptions": [], "approved_integration_changes": [],
                   "protected_changes": [], "existing_test_changes": [], "policy_valid": False}
    _, original = run(manifest_path, base)
    report = copy.deepcopy(original)
    report.update(schema_version=2, policy_version=policy["policy_version"], policy_valid=True,
                  original_guard_report=original, checkout_line_ending_conversions=conversions,
                  approved_integration_changes=policy["integration_changes"],
                  approved_exceptions=[dict(policy["clock_test_exception"], approval_reference=APPROVAL_REFERENCE)])
    report["existing_test_changes"] = [path for path in original["existing_test_changes"] if path != CLOCK_TEST]
    report["self_protected_changes"] = [path for path in original["self_protected_changes"] if path != GUARD_PATH]
    report["tooling_hash_mismatches"] = [path for path in original["tooling_hash_mismatches"]
                                        if path != GUARD_PATH and path not in conversions]
    removable = set()
    if not report["existing_test_changes"]:
        removable.add("EXISTING_TEST_EXPECTATION_CHANGED")
    if not report["self_protected_changes"]:
        removable.add("SCOPE_GUARD_OR_BASELINE_CHANGED")
    if not report["tooling_hash_mismatches"]:
        removable.add("SCOPE_GUARD_TOOLING_HASH_MISMATCH")
    report["reason_codes"] = [reason for reason in original["reason_codes"] if reason not in removable]
    report["status"] = "FAIL" if report["reason_codes"] else "PASS"
    return (1 if report["reason_codes"] else 0), report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    try:
        code, report = _run_integrated(args.manifest, args.base, PRODUCTION_BINDINGS)
    except Exception as exc:
        report = {"schema_version": 1, "status": "ERROR", "reason_codes": ["GUARD_EXECUTION_ERROR"], "error": str(exc)}
        code = 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
