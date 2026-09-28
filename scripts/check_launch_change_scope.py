"""Fail closed when paid-launch work touches protected research dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    try:
        code, report = run(args.manifest, args.base)
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
