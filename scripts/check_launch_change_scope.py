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


V3_POLICY_PATH = "docs/paid-launch/launch-scope-policy-v3.json"
V3_POLICY_VERSION = "paid-launch-provider-health-v3-r1"
V3_APPROVAL_REFERENCE = "Owner-authorized draft and PR #2376 review correction: actual-caller provider health, safe diagnostics and structured ESPN FCS outcomes at main 057abaf8214268d99ba9d1d8ba672997a5aef81f; merge requires separate review"
PROVIDER_PATHS = (
    "core/streamlit_pipeline.py", "core/run_readiness.py", "app/ui/readiness_dashboard.py",
    "app_core/provider_health.py", "app_core/espn_ncaaf_odds.py", GUARD_PATH, "tests/test_provider_caller_health.py",
    "tests/test_provider_health_scope_policy.py", "docs/paid-launch/provider-caller-health-policy.md",
)
PROVIDER_UNCHANGED_PATHS = (
    MANIFEST_PATH, CLOCK_TEST, POLICY_PATH, ".github/workflows/paid-launch.yml",
    "tests/paid_launch/case_isolation_and_scope.py",
)
PROVIDER_BINDINGS = {'base': '057abaf8214268d99ba9d1d8ba672997a5aef81f',
 'base_tree': '9bc0b53a2e1055918c399a36d5a4de65094bd540',
 'manifest_sha256': '2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343',
 'previous_guard_sha256': '8f25398afa9a8620d661227d2621d1065704955dd4d9b0004161d9a9993619a8',
 'previous_policy_blob': '1decb1ca5462b2393f5816d54637dab18d9ed733',
 'clock_blob': 'e610143aff5611235f9cfb44da13e2d54e1c6b48'}
PROVIDER_DIAGNOSTIC_EDITS = {'core/streamlit_pipeline.py': [['    from app_core.odds_api import TheOddsAPIClient, '
                                 'filter_games_today_only\n',
                                 '    from app_core.odds_api import TheOddsAPIClient, '
                                 'filter_games_today_only\n'
                                 '    from app_core.provider_health import failure, health_report, '
                                 'outcome, sanitized_outcome\n'],
                                ['    game_dict = {}\n    mlb_receipt_health = {}\n',
                                 '    game_dict = {}\n'
                                 '    mlb_receipt_health = {}\n'
                                 '    provider_outcomes = {}\n'],
                                ['        games = []\n'
                                 '        try:\n'
                                 '            if client is not None:\n'
                                 '                games = client.get_odds(sk, date=date)\n'
                                 '            if isinstance(games, dict) and "message" in games:\n'
                                 '                logger.error(f"Odds API error for {sk}: '
                                 '{games.get(\'message\')}")\n'
                                 '                games = []\n'
                                 '        except Exception as e:\n'
                                 '            logger.error(f"Network/API failure for {sk}: {e}")\n'
                                 '            games = []\n',
                                 '        games = []\n'
                                 '        provider_outcomes[sk] = outcome("NOT_CONFIGURED")\n'
                                 '        if client is not None:\n'
                                 '            try:\n'
                                 '                games = client.get_odds(sk, date=date)\n'
                                 '                if isinstance(games, list):\n'
                                 '                    provider_outcomes[sk] = outcome("SUCCESS" if '
                                 'games else "SUCCESS_EMPTY", len(games))\n'
                                 '                else:\n'
                                 '                    provider_outcomes[sk] = '
                                 'outcome("INVALID_RESPONSE")\n'
                                 '                    games = []\n'
                                 '            except Exception as exc:\n'
                                 '                detail = failure(exc)\n'
                                 '                provider_outcomes[sk] = outcome(detail["outcome"], '
                                 'http_status=detail["http_status"])\n'
                                 '                games = []\n'
                                 '        current_outcome = provider_outcomes[sk]\n'
                                 '        log_outcome = logger.info if current_outcome["outcome"] in '
                                 '{"SUCCESS", "SUCCESS_EMPTY"} else logger.warning\n'
                                 '        log_outcome("Odds provider sport=%s outcome=%s '
                                 'http_status=%s", sk,\n'
                                 '                    current_outcome["outcome"], '
                                 'current_outcome["http_status"])\n'],
                                ['            games = recover_nfl_novig(games, api_key)\n',
                                 '            try:\n'
                                 '                games = recover_nfl_novig(games, api_key)\n'
                                 '            except Exception as exc:\n'
                                 '                detail = failure(exc)\n'
                                 '                provider_outcomes[sk]["fallback_errors"]["nfl_novig"] '
                                 '= detail\n'
                                 '                logger.warning("Odds fallback sport=%s '
                                 'source=nfl_novig outcome=%s http_status=%s",\n'
                                 '                               sk, detail["outcome"], '
                                 'detail["http_status"])\n'],
                                ['            games = recover_college_novig(games, api_key)\n',
                                 '            try:\n'
                                 '                games = recover_college_novig(games, api_key)\n'
                                 '            except Exception as exc:\n'
                                 '                detail = failure(exc)\n'
                                 '                '
                                 'provider_outcomes[sk]["fallback_errors"]["college_novig"] = detail\n'
                                 '                logger.warning("Odds fallback sport=%s '
                                 'source=college_novig outcome=%s http_status=%s",\n'
                                 '                               sk, detail["outcome"], '
                                 'detail["http_status"])\n'],
                                ['            except Exception as e:\n'
                                 '                logger.error("NCAAF FCS fallback normalization failed '
                                 'closed: %s", e)\n',
                                 '            except Exception as exc:\n'
                                 '                detail = failure(exc)\n'
                                 '                provider_outcomes[sk]["fallback_errors"]["ncaaf_fcs"] '
                                 '= detail\n'
                                 '                logger.warning("Odds fallback sport=%s '
                                 'source=ncaaf_fcs outcome=%s http_status=%s",\n'
                                 '                               sk, detail["outcome"], '
                                 'detail["http_status"])\n'],
                                ['        try:\n'
                                 '            if not games:\n'
                                 '                continue\n'
                                 '\n'
                                 "            # Historical/backfill requests must honor the caller's "
                                 'explicit date.\n',
                                 '        try:\n'
                                 '            if not games:\n'
                                 '                provider_outcomes[sk]["processing"] = "EMPTY"\n'
                                 '                continue\n'
                                 '\n'
                                 "            # Historical/backfill requests must honor the caller's "
                                 'explicit date.\n'],
                                ['            if not sport_games:\n'
                                 '                continue\n'
                                 '\n'
                                 '            if sk == "baseball_mlb" and date is None:\n',
                                 '            if not sport_games:\n'
                                 '                provider_outcomes[sk]["processing"] = '
                                 '"FILTERED_EMPTY"\n'
                                 '                continue\n'
                                 '\n'
                                 '            if sk == "baseball_mlb" and date is None:\n'],
                                ['        except Exception as e:\n'
                                 '            logger.error(f"Odds normalization failure for {sk}: '
                                 '{e}")\n'
                                 '            continue\n'
                                 '\n'
                                 '    if not game_dict:\n'
                                 '        return pd.DataFrame()\n'
                                 '\n'
                                 '    result = pd.DataFrame(list(game_dict.values()))\n'
                                 '    result.attrs["mlb_receipt_health"] = mlb_receipt_health\n'
                                 '    return result\n',
                                 '            provider_outcomes[sk]["processing"] = "SUCCESS"\n'
                                 '        except Exception as exc:\n'
                                 '            detail = failure(exc)\n'
                                 '            provider_outcomes[sk]["processing"] = "FAILED"\n'
                                 '            provider_outcomes[sk]["processing_error"] = detail\n'
                                 '            logger.warning("Odds normalization sport=%s outcome=%s '
                                 'http_status=%s",\n'
                                 '                           sk, detail["outcome"], '
                                 'detail["http_status"])\n'
                                 '            continue\n'
                                 '\n'
                                 '    result = pd.DataFrame(list(game_dict.values())) if game_dict else '
                                 'pd.DataFrame()\n'
                                 '    result.attrs["mlb_receipt_health"] = mlb_receipt_health\n'
                                 '    result.attrs["provider_health"] = '
                                 'health_report(provider_outcomes, len(result))\n'
                                 '    return result\n'],
                                ['    mlb_receipt_health = live_odds_df.attrs.get("mlb_receipt_health", '
                                 '{})\n',
                                 '    mlb_receipt_health = live_odds_df.attrs.get("mlb_receipt_health", '
                                 '{})\n'
                                 '    from app_core.provider_health import sanitized_health\n'
                                 '    provider_health = '
                                 'sanitized_health(live_odds_df.attrs.get("provider_health"))\n'],
                                ['    diagnostics["mlb_receipt_health"] = mlb_receipt_health\n',
                                 '    diagnostics["provider_health"] = provider_health\n'
                                 '    diagnostics["mlb_receipt_health"] = mlb_receipt_health\n'],
                                ['                fallback_games = fetch_espn_ncaaf_fcs_odds(date)\n',
                                 '                fallback_games = fetch_espn_ncaaf_fcs_odds(date)\n'
                                 '                fallback_detail = getattr(fallback_games, '
                                 '"provider_outcome", None)\n'
                                 '                if isinstance(fallback_detail, dict):\n'
                                 '                    detail = sanitized_outcome(fallback_detail)\n'
                                 '                    '
                                 'provider_outcomes[sk]["fallback_outcomes"]["ncaaf_fcs"] = detail\n'
                                 '                    if detail["outcome"] not in {"SUCCESS", '
                                 '"SUCCESS_EMPTY", "NOT_RECORDED"}:\n'
                                 '                        '
                                 'provider_outcomes[sk]["fallback_errors"]["ncaaf_fcs"] = detail\n']],
 'core/run_readiness.py': [['    report["mlb_receipt_health"] = health if isinstance(health, dict) else '
                            '{}\n',
                            '    report["mlb_receipt_health"] = health if isinstance(health, dict) else '
                            '{}\n'
                            '    from app_core.provider_health import sanitized_health\n'
                            '    report["provider_health"] = '
                            'sanitized_health(diagnostics.get("provider_health"))\n']],
 'app/ui/readiness_dashboard.py': [['def render_readiness_dashboard(audit=None, final=None, '
                                    'diagnostics=None):\n',
                                    'def render_provider_health(diagnostics):\n'
                                    '    from app_core.provider_health import sanitized_health\n'
                                    '    health = sanitized_health((diagnostics or '
                                    '{}).get("provider_health"))\n'
                                    '    if not health:\n'
                                    '        return\n'
                                    '    st.caption("Odds provider outcomes: " + '
                                    'health["status"].replace("_", " ").lower())\n'
                                    '    st.caption("These outcomes describe this fetch. Returned games '
                                    'retain their source and still require wager eligibility checks.")\n'
                                    '\n'
                                    '    def fallback_summary(outcomes):\n'
                                    '        return ", ".join(name + ": " + value["outcome"] +\n'
                                    '                         (f" [HTTP {value[\'http_status\']}]" if '
                                    'value["http_status"] is not None else "")\n'
                                    '                         for name, value in outcomes.items())\n'
                                    '\n'
                                    '    rows = []\n'
                                    '    for sport, item in health["sports"].items():\n'
                                    '        rows.append({"Sport": sport, "Provider outcome": '
                                    'item["outcome"],\n'
                                    '                     "HTTP status": item["http_status"], "Games '
                                    'received": item["received_games"],\n'
                                    '                     "Processing": item["processing"],\n'
                                    '                     "Fallback outcomes": '
                                    'fallback_summary(item["fallback_outcomes"]),\n'
                                    '                     "Fallback errors": '
                                    'fallback_summary(item["fallback_errors"])})\n'
                                    '    if rows:\n'
                                    '        st.dataframe(pd.DataFrame(rows), hide_index=True)\n'
                                    '    st.download_button("Download provider outcomes", '
                                    'json.dumps(health, indent=2, allow_nan=False),\n'
                                    '                       file_name="provider-outcomes.json", '
                                    'mime="application/json")\n'
                                    '\n'
                                    '\n'
                                    'def render_readiness_dashboard(audit=None, final=None, '
                                    'diagnostics=None):\n'],
                                   ['        receipt_health = (diagnostics or '
                                    '{}).get("mlb_receipt_health", {})\n',
                                    '        render_provider_health(diagnostics)\n'
                                    '        receipt_health = (diagnostics or '
                                    '{}).get("mlb_receipt_health", {})\n']],
 'app_core/espn_ncaaf_odds.py': [['    """Return FCS games with complete DraftKings markets from '
                                  'ESPN\'s scoreboard."""\n',
                                  '    """Return a list-compatible result with a sanitized '
                                  'provider_outcome receipt."""\n'
                                  '    from app_core.provider_health import ProviderGames, failure\n'],
                                 ['    except Exception as exc:\n'
                                  '        logger.warning("ESPN NCAAF FCS odds fallback failed closed: '
                                  '%s", exc)\n'
                                  '        return []\n'
                                  '    if not isinstance(payload, dict):\n'
                                  '        logger.warning("ESPN NCAAF FCS odds fallback returned a '
                                  'non-object payload")\n'
                                  '        return []\n'
                                  '\n',
                                  '    except Exception as exc:\n'
                                  '        detail = failure(exc)\n'
                                  '        logger.warning("ESPN NCAAF FCS odds fallback '
                                  'sport=americanfootball_ncaaf outcome=%s http_status=%s",\n'
                                  '                       detail["outcome"], detail["http_status"])\n'
                                  '        return ProviderGames([], detail)\n'
                                  '    if not isinstance(payload, dict):\n'
                                  '        logger.warning("ESPN NCAAF FCS odds fallback returned a '
                                  'non-object payload")\n'
                                  '        return ProviderGames([], {"outcome": "INVALID_RESPONSE", '
                                  '"http_status": getattr(response, "status_code", None)})\n'
                                  '\n'],
                                 ['    return games\n',
                                  '    return ProviderGames(games, {"outcome": "SUCCESS" if games else '
                                  '"SUCCESS_EMPTY",\n'
                                  '                                "http_status": getattr(response, '
                                  '"status_code", None)})\n']]}


def _validate_provider_policy(manifest_path: Path, base: str | None, binding: dict) -> tuple[dict, list[str]]:
    _require(manifest_path.resolve() == (ROOT / MANIFEST_PATH).resolve(), "BASELINE_PATH_NOT_APPROVED")
    _require(base in (None, binding["base"], json.loads(git_bytes("show", f"{binding['base']}:{MANIFEST_PATH}"))["base_sha"]),
             "COMPARISON_BASE_NOT_APPROVED")
    _require(git("rev-parse", f"{binding['base']}^{{tree}}") == binding["base_tree"], "STARTING_TREE_NOT_APPROVED")
    original_manifest = git_bytes("show", f"{binding['base']}:{MANIFEST_PATH}")
    _require(hashlib.sha256(original_manifest).hexdigest() == binding["manifest_sha256"], "ORIGINAL_BASELINE_IDENTITY_CHANGED")
    _require(not git("diff", "--name-only") and not git("diff", "--cached", "--name-only"), "LOCAL_TRACKED_CHANGE")
    previous_guard = git_bytes("show", f"{binding['base']}:{GUARD_PATH}")
    _require(hashlib.sha256(previous_guard).hexdigest() == binding["previous_guard_sha256"], "PREVIOUS_TOOLING_IDENTITY_CHANGED")
    _require(blob(binding["base"], POLICY_PATH) == binding["previous_policy_blob"], "PREVIOUS_POLICY_IDENTITY_CHANGED")
    _require(blob(binding["base"], CLOCK_TEST) == binding["clock_blob"], "PREVIOUS_CLOCK_IDENTITY_CHANGED")
    raw = git_bytes("show", f"HEAD:{V3_POLICY_PATH}")
    _require((ROOT / V3_POLICY_PATH).read_bytes().replace(b"\r\n", b"\n") == raw, "POLICY_CHECKOUT_CHANGED")
    policy = json.loads(raw)
    keys = {"schema_version", "policy_version", "approval_reference", "base_sha", "base_tree",
            "original_manifest_sha256", "previous_policy_blob", "clock_test_blob", "implementation_commit",
            "implementation_tree", "implementation_changes", "tooling_sha256", "unchanged_bindings"}
    _require(set(policy) == keys and policy["schema_version"] == 3 and
             policy["policy_version"] == V3_POLICY_VERSION, "SUCCESSOR_POLICY_SCHEMA_INVALID")
    _require(policy["approval_reference"] == V3_APPROVAL_REFERENCE and policy["base_sha"] == binding["base"] and
             policy["base_tree"] == binding["base_tree"] and policy["original_manifest_sha256"] == binding["manifest_sha256"] and
             policy["previous_policy_blob"] == binding["previous_policy_blob"] and policy["clock_test_blob"] == binding["clock_blob"],
             "APPROVAL_BINDING_CHANGED")
    implementation = policy["implementation_commit"]
    _require(git("show", "-s", "--format=%P", implementation).split() == [binding["base"]], "IMPLEMENTATION_PARENT_NOT_APPROVED")
    _require(git("rev-parse", f"{implementation}^{{tree}}") == policy["implementation_tree"], "IMPLEMENTATION_TREE_MISMATCH")
    _require(not exists_at(implementation, V3_POLICY_PATH), "POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION")
    _require(set(git("diff", "--name-only", binding["base"], implementation).splitlines()) == set(PROVIDER_PATHS),
             "IMPLEMENTATION_CHANGE_SET_NOT_APPROVED")
    changes = {path: {"before_blob": blob(binding["base"], path), "after_blob": blob(implementation, path)} for path in PROVIDER_PATHS}
    _require(policy["implementation_changes"] == changes, "IMPLEMENTATION_BLOB_BINDINGS_CHANGED")
    # Reconstruct only the reviewed diagnostic hunks; all other code in these
    # shared scientific/authority files and the fallback must remain identical.
    for path, edits in PROVIDER_DIAGNOSTIC_EDITS.items():
        expected = git_bytes("show", f"{binding['base']}:{path}").decode("utf-8")
        for before, after in edits:
            _require(expected.count(before) == 1, "DIAGNOSTIC_BASE_ANCHOR_CHANGED")
            expected = expected.replace(before, after, 1)
        _require(git_bytes("show", f"{implementation}:{path}").decode("utf-8") == expected,
                 "DIAGNOSTIC_SCOPE_NOT_APPROVED")
    frozen_prefix = previous_guard.split(b"\ndef main() -> int:", 1)[0]
    implementation_guard = git_bytes("show", f"{implementation}:{GUARD_PATH}")
    _require(implementation_guard.split(b"\nV3_POLICY_PATH =", 1)[0] == frozen_prefix,
             "PREVIOUS_GUARD_LOGIC_CHANGED")
    unchanged = {path: blob(binding["base"], path) for path in PROVIDER_UNCHANGED_PATHS}
    _require(policy["unchanged_bindings"] == unchanged, "IMMUTABLE_BINDINGS_CHANGED")
    for path, expected_blob in unchanged.items():
        _require(blob("HEAD", path) == expected_blob and
                 (ROOT / path).read_bytes().replace(b"\r\n", b"\n") == git_bytes("show", f"{binding['base']}:{path}"),
                 "IMMUTABLE_FILE_CHANGED")
    parents = git("show", "-s", "--format=%P", "HEAD").split()
    if len(parents) == 2:
        _require(parents[0] == binding["base"], "CI_BASE_PARENT_NOT_APPROVED")
        candidate = parents[1]
        _require(git("rev-parse", "HEAD^{tree}") == git("rev-parse", f"{candidate}^{{tree}}"), "CI_MERGE_TREE_CHANGED")
    else:
        candidate = git("rev-parse", "HEAD")
    _require(git("show", "-s", "--format=%P", candidate).split() == [implementation], "CANDIDATE_NOT_POLICY_SEAL")
    _require(git("diff", "--name-status", implementation, candidate).splitlines() == [f"A\t{V3_POLICY_PATH}"],
             "SEAL_CHANGE_SET_NOT_APPROVED")
    manifest = json.loads(original_manifest)
    tooling = {path: hashlib.sha256(git_bytes("show", f"{implementation}:{path}")).hexdigest()
               for path in manifest["tooling_sha256"]}
    _require(policy["tooling_sha256"] == tooling, "SUCCESSOR_TOOLING_BINDING_CHANGED")
    _require(tooling[".github/workflows/paid-launch.yml"] == manifest["tooling_sha256"][".github/workflows/paid-launch.yml"],
             "WORKFLOW_CHANGE_NOT_APPROVED")
    conversions = []
    for path, expected_hash in tooling.items():
        committed = git_bytes("show", f"HEAD:{path}"); checkout = (ROOT / path).read_bytes()
        _require(hashlib.sha256(committed).hexdigest() == expected_hash and checkout.replace(b"\r\n", b"\n") == committed,
                 "UNAUTHORIZED_TOOLING_CHANGE")
        if checkout != committed:
            conversions.append(path)
    shadows = []
    for path in ("core/streamlit_pipeline.py", "core/run_readiness.py", "app/ui/readiness_dashboard.py",
                 "app_core/provider_health.py", "app_core/espn_ncaaf_odds.py"):
        suffix = Path(path).parts
        for other in ROOT.rglob(Path(path).name):
            relative = other.relative_to(ROOT)
            if tuple(relative.parts[-len(suffix):]) == tuple(suffix) and relative.as_posix() != path:
                shadows.append(relative.as_posix())
    hooks = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
             if p.is_file() and (p.name in {"sitecustomize.py", "usercustomize.py"} or p.suffix == ".pth")]
    _require(not shadows and not hooks, "PROTECTED_RUNTIME_SHADOWING_RISK")
    return policy, conversions


def _run_provider_integrated(manifest_path: Path, base: str | None, binding: dict) -> tuple[int, dict]:
    try:
        policy, conversions = _validate_provider_policy(manifest_path, base, binding)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        return 1, {"schema_version": 3, "status": "FAIL", "reason_codes": [str(exc)],
                   "approved_exceptions": [], "approved_integration_changes": [],
                   "protected_changes": [], "existing_test_changes": [], "policy_valid": False}
    _, original = run(manifest_path, base)
    report = copy.deepcopy(original)
    report.update(schema_version=3, policy_version=policy["policy_version"], policy_valid=True,
                  original_guard_report=original, checkout_line_ending_conversions=conversions,
                  approved_integration_changes=policy["implementation_changes"],
                  approved_exceptions=[{"path": CLOCK_TEST, "before_blob": PRODUCTION_BINDINGS["before"],
                                        "after_blob": binding["clock_blob"], "retained_unchanged": True,
                                        "approval_reference": APPROVAL_REFERENCE}],
                  approved_tooling_changes={GUARD_PATH: policy["tooling_sha256"][GUARD_PATH]},
                  new_existing_test_exceptions=[])
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


SCHEDULE_POLICY_PATH = "docs/paid-launch/launch-scope-policy-ncaaf-v1.json"
SCHEDULE_POLICY_VERSION = "paid-launch-ncaaf-schedule-v1"
SCHEDULE_APPROVAL_REFERENCE = "Owner-authorized draft: complete NCAAF schedule inventory, exact identity, quote provenance and coverage at main af7b3dcbc3fe9ca7161b886c65074be3d739bae8; merge requires separate review"

SCHEDULE_BINDINGS = {'base': 'af7b3dcbc3fe9ca7161b886c65074be3d739bae8',
 'base_tree': '98a117dbc7f6977bafed53624898d95a4e72f334',
 'manifest_sha256': '2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343',
 'previous_guard_sha256': 'a93f90db1843d6da334adf17979fcecba20fd3549b47864a374e1cf6535baf37',
 'clock_blob': 'e610143aff5611235f9cfb44da13e2d54e1c6b48'}

SCHEDULE_PATHS = ['core/streamlit_pipeline.py',
 'streamlit_app.py',
 'app/ui/sidebar_controls.py',
 'app_core/ncaaf_identity.py',
 'app_core/espn_ncaaf_odds.py',
 'app_core/ncaaf_schedule.py',
 'app/ui/ncaaf_inventory.py',
 'tests/test_ncaaf_schedule_coverage.py',
 'tests/test_ncaaf_schedule_scope_policy.py',
 'docs/paid-launch/ncaaf-schedule-coverage.md',
 'scripts/check_launch_change_scope.py']

SCHEDULE_IMMUTABLE = ['docs/paid-launch/launch-baseline-manifest.json',
 'docs/paid-launch/launch-scope-policy-v2.json',
 'docs/paid-launch/launch-scope-policy-v3.json',
 'tests/test_board_diagnostics.py',
 '.github/workflows/paid-launch.yml',
 'tests/paid_launch/case_isolation_and_scope.py',
 'parlaypicker/app/streamlit_app.py']

SCHEDULE_SHARED_EDITS = {'core/streamlit_pipeline.py': [['    return league + "|" + team_a + "|" + team_b\n'
                                 '\n'
                                 '\n'
                                 'def _matchup_id(df: pd.DataFrame) -> pd.Series:\n'
                                 '    """Canonical matchup id using sorted normalized team names + '
                                 'ET day (direction-independent)."""\n'
                                 '    home = _identity_team_series(df, "home_team")\n',
                                 '    return league + "|" + team_a + "|" + team_b\n'
                                 '\n'
                                 '\n'
                                 'def _ncaaf_schedule_keys(df: pd.DataFrame) -> pd.Series:\n'
                                 '    """Use only the additive inventory identity; keep legacy IDs '
                                 'elsewhere."""\n'
                                 '    ids = _string_series(df, "schedule_event_id")\n'
                                 '    valid = (_string_series(df, '
                                 '"league").str.upper().eq("NCAAF")\n'
                                 '             & _string_series(df, '
                                 '"schedule_match_status").eq("MATCHED")\n'
                                 '             & ids.str.fullmatch(r"espn:college-football:\\d+", '
                                 'na=False))\n'
                                 '    unresolved = _string_series(df, "schedule_inventory_key")\n'
                                 '    unresolved_ok = (_string_series(df, '
                                 '"league").str.upper().eq("NCAAF")\n'
                                 '                     & _string_series(df, '
                                 '"schedule_match_status").isin(["UNRESOLVED", "AMBIGUOUS", '
                                 '"AMBIGUOUS_PROVIDER_ID", "KICKOFF_OR_IDENTITY_CONFLICT"])\n'
                                 '                     & '
                                 'unresolved.str.startswith("ncaaf:unresolved:", na=False))\n'
                                 '    return ids.where(valid, unresolved.where(unresolved_ok, '
                                 '""))\n'
                                 '\n'
                                 '\n'
                                 'def _matchup_id(df: pd.DataFrame) -> pd.Series:\n'
                                 '    """Canonical matchup id using sorted normalized team names + '
                                 'ET day (direction-independent)."""\n'
                                 '    home = _identity_team_series(df, "home_team")\n'],
                                ['    team_a = pd.Series(team_a, index=df.index, dtype="string")\n'
                                 '    team_b = pd.Series(team_b, index=df.index, dtype="string")\n'
                                 '    date_key = _et_day_string(_game_dates(df)).fillna("")\n'
                                 '    return team_a + "|" + team_b + "|" + date_key\n'
                                 '\n'
                                 '\n'
                                 'def _mk_game_key(df: pd.DataFrame) -> pd.Series:\n',
                                 '    team_a = pd.Series(team_a, index=df.index, dtype="string")\n'
                                 '    team_b = pd.Series(team_b, index=df.index, dtype="string")\n'
                                 '    date_key = _et_day_string(_game_dates(df)).fillna("")\n'
                                 '    legacy = team_a + "|" + team_b + "|" + date_key\n'
                                 '    schedule = _ncaaf_schedule_keys(df)\n'
                                 '    return schedule.where(schedule.ne(""), legacy)\n'
                                 '\n'
                                 '\n'
                                 'def _mk_game_key(df: pd.DataFrame) -> pd.Series:\n'],
                                ['        + "|" + '
                                 'pool["home_team"].str.lower().str.replace(r"\\s+", " ", '
                                 'regex=True)\n'
                                 '        + "|" + '
                                 'pool["away_team"].str.lower().str.replace(r"\\s+", " ", '
                                 'regex=True)\n'
                                 '    )\n'
                                 '\n'
                                 '    # Force expected_value to numeric, converting true errors to '
                                 'NaN while preserving negative floats\n'
                                 '    pool["expected_value"] = '
                                 'pd.to_numeric(pool["expected_value"], errors="coerce")\n',
                                 '        + "|" + '
                                 'pool["home_team"].str.lower().str.replace(r"\\s+", " ", '
                                 'regex=True)\n'
                                 '        + "|" + '
                                 'pool["away_team"].str.lower().str.replace(r"\\s+", " ", '
                                 'regex=True)\n'
                                 '    )\n'
                                 '    schedule_keys = _ncaaf_schedule_keys(pool)\n'
                                 '    pool.loc[schedule_keys.ne(""), "matchup_id"] = '
                                 'schedule_keys[schedule_keys.ne("")]\n'
                                 '\n'
                                 '    # Force expected_value to numeric, converting true errors to '
                                 'NaN while preserving negative floats\n'
                                 '    pool["expected_value"] = '
                                 'pd.to_numeric(pool["expected_value"], errors="coerce")\n'],
                                ['    return final_best_df\n'
                                 '\n'
                                 '\n'
                                 'def fetch_live_odds_dataframe(sports: list[str] | None = None, '
                                 'date: str | None = None) -> pd.DataFrame:\n'
                                 '    from app_core.espn_ncaaf_odds import (\n'
                                 '        fetch_espn_ncaaf_fcs_odds,\n'
                                 '        merge_missing_ncaaf_games,\n',
                                 '    return final_best_df\n'
                                 '\n'
                                 '\n'
                                 'def fetch_live_odds_dataframe(sports: list[str] | None = None, '
                                 'date: str | None = None,\n'
                                 '                              *, schedule_start=None, '
                                 'schedule_end=None) -> pd.DataFrame:\n'
                                 '    """Keep independent NCAAF inventory even when no price rows '
                                 'can be built."""\n'
                                 '    inventory = None\n'
                                 '    if schedule_start is not None and (not sports or "NCAAF" in '
                                 '[s.upper() for s in sports]):\n'
                                 '        from app_core.ncaaf_schedule import fetch_schedule\n'
                                 '        inventory = fetch_schedule(schedule_start, schedule_end '
                                 'or schedule_start)\n'
                                 '    result = _fetch_live_odds_dataframe(sports, date, '
                                 '_schedule_inventory=inventory)\n'
                                 '    if inventory is not None:\n'
                                 '        result.attrs["ncaaf_schedule"] = inventory\n'
                                 '        result.attrs["ncaaf_provider_games"] = '
                                 'result.loc[result["league"].eq("NCAAF")].to_dict("records") if '
                                 '"league" in result else []\n'
                                 '    return result\n'
                                 '\n'
                                 '\n'
                                 'def _fetch_live_odds_dataframe(sports: list[str] | None = None, '
                                 'date: str | None = None,\n'
                                 '                               *, _schedule_inventory=None) -> '
                                 'pd.DataFrame:\n'
                                 '    from app_core.espn_ncaaf_odds import (\n'
                                 '        fetch_espn_ncaaf_fcs_odds,\n'
                                 '        merge_missing_ncaaf_games,\n'],
                                ['                    '
                                 'provider_outcomes[sk]["fallback_outcomes"]["ncaaf_fcs"] = '
                                 'detail\n'
                                 '                    if detail["outcome"] not in {"SUCCESS", '
                                 '"SUCCESS_EMPTY", "NOT_RECORDED"}:\n'
                                 '                        '
                                 'provider_outcomes[sk]["fallback_errors"]["ncaaf_fcs"] = detail\n'
                                 '                games = merge_missing_ncaaf_games(games, '
                                 'fallback_games)\n'
                                 '            except Exception as exc:\n'
                                 '                detail = failure(exc)\n'
                                 '                '
                                 'provider_outcomes[sk]["fallback_errors"]["ncaaf_fcs"] = detail\n'
                                 '                logger.warning("Odds fallback sport=%s '
                                 'source=ncaaf_fcs outcome=%s http_status=%s",\n'
                                 '                               sk, detail["outcome"], '
                                 'detail["http_status"])\n'
                                 '\n'
                                 '        try:\n'
                                 '            if not games:\n',
                                 '                    '
                                 'provider_outcomes[sk]["fallback_outcomes"]["ncaaf_fcs"] = '
                                 'detail\n'
                                 '                    if detail["outcome"] not in {"SUCCESS", '
                                 '"SUCCESS_EMPTY", "NOT_RECORDED"}:\n'
                                 '                        '
                                 'provider_outcomes[sk]["fallback_errors"]["ncaaf_fcs"] = detail\n'
                                 '                if _schedule_inventory is not None:\n'
                                 '                    from app_core.ncaaf_schedule import '
                                 'merge_schedule_odds\n'
                                 '                    games = merge_schedule_odds(games, '
                                 'fallback_games, _schedule_inventory)\n'
                                 '                else:\n'
                                 '                    games = merge_missing_ncaaf_games(games, '
                                 'fallback_games)\n'
                                 '            except Exception as exc:\n'
                                 '                detail = failure(exc)\n'
                                 '                '
                                 'provider_outcomes[sk]["fallback_errors"]["ncaaf_fcs"] = detail\n'
                                 '                logger.warning("Odds fallback sport=%s '
                                 'source=ncaaf_fcs outcome=%s http_status=%s",\n'
                                 '                               sk, detail["outcome"], '
                                 'detail["http_status"])\n'
                                 '\n'
                                 '        if sk == "americanfootball_ncaaf" and '
                                 '_schedule_inventory is not None:\n'
                                 '            from app_core.ncaaf_schedule import '
                                 'merge_schedule_odds\n'
                                 '            games = merge_schedule_odds(games, [], '
                                 '_schedule_inventory)\n'
                                 '\n'
                                 '        try:\n'
                                 '            if not games:\n'],
                                ['            # Historical/backfill requests must honor the '
                                 "caller's explicit date.\n"
                                 '            # The today-only guard is appropriate only for the '
                                 'live slate.\n'
                                 '            sport_games = games if date else '
                                 'filter_games_today_only(games)\n'
                                 '            football_sport = {"americanfootball_nfl": "NFL", '
                                 '"americanfootball_ncaaf": "NCAAF"}.get(sk)\n'
                                 '            if football_sport:\n'
                                 '                from app_core.football_identity_capture import '
                                 'collect as collect_football_identity\n'
                                 '                sport_games = '
                                 'collect_football_identity(sport_games, football_sport)\n'
                                 '\n'
                                 '            if not sport_games:\n'
                                 '                provider_outcomes[sk]["processing"] = '
                                 '"FILTERED_EMPTY"\n',
                                 '            # Historical/backfill requests must honor the '
                                 "caller's explicit date.\n"
                                 '            # The today-only guard is appropriate only for the '
                                 'live slate.\n'
                                 '            sport_games = games if date else '
                                 'filter_games_today_only(games)\n'
                                 '            if sk == "americanfootball_ncaaf" and '
                                 '_schedule_inventory is not None:\n'
                                 '                from app_core.ncaaf_schedule import timestamp, '
                                 'window, ET\n'
                                 '                first, last = '
                                 'window(_schedule_inventory["start_date"], '
                                 '_schedule_inventory["end_date"])\n'
                                 '                sport_games = [g for g in games if (kickoff := '
                                 'timestamp(g.get("commence_time"))) and first <= '
                                 'kickoff.astimezone(ET).date() <= last]\n'
                                 '            football_sport = {"americanfootball_nfl": "NFL", '
                                 '"americanfootball_ncaaf": "NCAAF"}.get(sk)\n'
                                 '            if football_sport:\n'
                                 '                from app_core.football_identity_capture import '
                                 'collect as collect_football_identity\n'
                                 '                if football_sport == "NCAAF" and '
                                 '_schedule_inventory is not None:\n'
                                 '                    from app_core.football_identity_capture '
                                 'import attach\n'
                                 '                    sport_games = attach(sport_games, "NCAAF", '
                                 '_schedule_inventory["identity_events"], '
                                 '_schedule_inventory["observed_at"])\n'
                                 '                    for game in sport_games:\n'
                                 '                        if game.get("schedule_match_status") == '
                                 '"AMBIGUOUS_PROVIDER_ID":\n'
                                 '                            game["football_identity_status"] = '
                                 '"CONFLICT"\n'
                                 '                else:\n'
                                 '                    sport_games = '
                                 'collect_football_identity(sport_games, football_sport)\n'
                                 '\n'
                                 '            if not sport_games:\n'
                                 '                provider_outcomes[sk]["processing"] = '
                                 '"FILTERED_EMPTY"\n'],
                                ['                    }\n'
                                 '\n'
                                 '                row = game_dict[matchup_id]\n'
                                 '                import json\n'
                                 '                for field in ("home_team_id", "away_team_id", '
                                 '"team_ids", "provider_ids", "football_identity_status", '
                                 '"football_identity_observed_at", '
                                 '"football_identity_source_hash", "mlb_provider_event_id", '
                                 '"mlb_pregame_receipts"):\n'
                                 '                    if field in game:\n',
                                 '                    }\n'
                                 '\n'
                                 '                row = game_dict[matchup_id]\n'
                                 '                for field in ("schedule_event_id", '
                                 '"schedule_match_status", "historical_matchup_id", '
                                 '"schedule_inventory_key"):\n'
                                 '                    if field in game:\n'
                                 '                        row[field] = game[field]\n'
                                 '                import json\n'
                                 '                for field in ("home_team_id", "away_team_id", '
                                 '"team_ids", "provider_ids", "football_identity_status", '
                                 '"football_identity_observed_at", '
                                 '"football_identity_source_hash", "mlb_provider_event_id", '
                                 '"mlb_pregame_receipts"):\n'
                                 '                    if field in game:\n'],
                                ['    # Required identity columns\n'
                                 '    id_cols = [\n'
                                 '        "league", "home_team", "away_team", "game_date", '
                                 '"matchup_id",\n'
                                 '        "commence_time_raw", "odds_feed_source", '
                                 '"provider_quotes",\n'
                                 '        "home_team_id", "away_team_id", "team_ids", '
                                 '"provider_ids", "football_identity_status", '
                                 '"football_identity_observed_at", '
                                 '"football_identity_source_hash", "mlb_provider_event_id", '
                                 '"mlb_pregame_receipts",\n'
                                 '    ]\n',
                                 '    # Required identity columns\n'
                                 '    id_cols = [\n'
                                 '        "league", "home_team", "away_team", "game_date", '
                                 '"matchup_id",\n'
                                 '        "schedule_event_id", "schedule_match_status", '
                                 '"historical_matchup_id", "schedule_inventory_key",\n'
                                 '        "commence_time_raw", "odds_feed_source", '
                                 '"provider_quotes",\n'
                                 '        "home_team_id", "away_team_id", "team_ids", '
                                 '"provider_ids", "football_identity_status", '
                                 '"football_identity_observed_at", '
                                 '"football_identity_source_hash", "mlb_provider_event_id", '
                                 '"mlb_pregame_receipts",\n'
                                 '    ]\n'],
                                ['    use_ml: bool = True,\n'
                                 '    spreads_df: pd.DataFrame | None = None,\n'
                                 '    totals_df: pd.DataFrame | None = None,\n'
                                 ') -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:\n'
                                 '\n'
                                 '    # 1. Build the enrichment frame (TheOver) BEFORE expanding '
                                 'the Master Slate\n',
                                 '    use_ml: bool = True,\n'
                                 '    spreads_df: pd.DataFrame | None = None,\n'
                                 '    totals_df: pd.DataFrame | None = None,\n'
                                 '    schedule_start=None,\n'
                                 '    schedule_end=None,\n'
                                 ') -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:\n'
                                 '\n'
                                 '    # 1. Build the enrichment frame (TheOver) BEFORE expanding '
                                 'the Master Slate\n'],
                                ['    theover_rows = _dedupe_inverted_matchups(theover_rows)\n'
                                 '\n'
                                 '    # 2. Expand TheOdds API into the Master Slate dynamically '
                                 'using theover_rows\n'
                                 '    live_odds_df = fetch_live_odds_dataframe(sports)\n'
                                 '    mlb_receipt_health = '
                                 'live_odds_df.attrs.get("mlb_receipt_health", {})\n'
                                 '    from app_core.provider_health import sanitized_health\n'
                                 '    provider_health = '
                                 'sanitized_health(live_odds_df.attrs.get("provider_health"))\n',
                                 '    theover_rows = _dedupe_inverted_matchups(theover_rows)\n'
                                 '\n'
                                 '    # 2. Expand TheOdds API into the Master Slate dynamically '
                                 'using theover_rows\n'
                                 '    live_odds_df = (fetch_live_odds_dataframe(sports, '
                                 'schedule_start=schedule_start, schedule_end=schedule_end)\n'
                                 '                    if schedule_start is not None else '
                                 'fetch_live_odds_dataframe(sports))\n'
                                 '    ncaaf_schedule = live_odds_df.attrs.get("ncaaf_schedule")\n'
                                 '    ncaaf_provider_games = '
                                 'live_odds_df.attrs.get("ncaaf_provider_games", [])\n'
                                 '    mlb_receipt_health = '
                                 'live_odds_df.attrs.get("mlb_receipt_health", {})\n'
                                 '    from app_core.provider_health import sanitized_health\n'
                                 '    provider_health = '
                                 'sanitized_health(live_odds_df.attrs.get("provider_health"))\n'],
                                ['\n'
                                 '    diagnostics["provider_health"] = provider_health\n'
                                 '    diagnostics["mlb_receipt_health"] = mlb_receipt_health\n'
                                 '    diagnostics["loaded_model_identity"] = '
                                 'loaded_model_identity\n'
                                 '    return (analysis_df, best_picks_df, diagnostics)\n'
                                 '\n',
                                 '\n'
                                 '    diagnostics["provider_health"] = provider_health\n'
                                 '    diagnostics["mlb_receipt_health"] = mlb_receipt_health\n'
                                 '    if ncaaf_schedule is not None:\n'
                                 '        from app_core.ncaaf_schedule import refresh_coverage\n'
                                 '        diagnostics["ncaaf_schedule"] = ncaaf_schedule\n'
                                 '        diagnostics["ncaaf_provider_games"] = '
                                 'ncaaf_provider_games\n'
                                 '        refresh_coverage(diagnostics, analysis_df)\n'
                                 '    diagnostics["loaded_model_identity"] = '
                                 'loaded_model_identity\n'
                                 '    return (analysis_df, best_picks_df, diagnostics)\n'
                                 '\n']],
 'streamlit_app.py': [['        float(controls.get("bankroll", 0.0)),\n'
                       '        _upload_fingerprint(controls.get("theover_spreads")),\n'
                       '        _upload_fingerprint(controls.get("theover_totals")),\n'
                       '    )\n'
                       '\n'
                       '\n'
                       'def _analysis_inputs_stale(state: dict[str, Any], controls: dict[str, '
                       'Any]) -> bool:\n'
                       '    """Return True when displayed results predate the current analysis '
                       'inputs."""\n'
                       '    analysis = state.get("analysis_df")\n'
                       '    has_results = isinstance(analysis, pd.DataFrame) and not '
                       'analysis.empty\n'
                       '    if not has_results:\n'
                       '        return False\n'
                       '    last_successful = state.get("last_successful_pipeline_signature")\n',
                       '        float(controls.get("bankroll", 0.0)),\n'
                       '        _upload_fingerprint(controls.get("theover_spreads")),\n'
                       '        _upload_fingerprint(controls.get("theover_totals")),\n'
                       '        str(controls.get("schedule_start") or ""),\n'
                       '        str(controls.get("schedule_end") or ""),\n'
                       '    )\n'
                       '\n'
                       '\n'
                       'def _analysis_inputs_stale(state: dict[str, Any], controls: dict[str, '
                       'Any]) -> bool:\n'
                       '    """Return True when displayed results predate the current analysis '
                       'inputs."""\n'
                       '    analysis = state.get("analysis_df")\n'
                       '    has_results = ((isinstance(analysis, pd.DataFrame) and not '
                       'analysis.empty)\n'
                       '                   or isinstance(state.get("diagnostics", '
                       '{}).get("ncaaf_schedule"), dict))\n'
                       '    if not has_results:\n'
                       '        return False\n'
                       '    last_successful = state.get("last_successful_pipeline_signature")\n'],
                      ['        use_ml=bool(controls["use_ml"]),\n'
                       '        spreads_df=spreads_df,\n'
                       '        totals_df=totals_df,\n'
                       '    )\n'
                       '\n'
                       '    timer.start("Market enrichment and candidate selection")\n',
                       '        use_ml=bool(controls["use_ml"]),\n'
                       '        spreads_df=spreads_df,\n'
                       '        totals_df=totals_df,\n'
                       '        schedule_start=controls.get("schedule_start"),\n'
                       '        schedule_end=controls.get("schedule_end"),\n'
                       '    )\n'
                       '\n'
                       '    timer.start("Market enrichment and candidate selection")\n'],
                      ['        diagnostics["prediction_snapshot_error"] = str(exc)\n'
                       '        deferred_warnings.append(f"Prediction evidence was not saved: '
                       '{exc}")\n'
                       '\n'
                       '    timer.finish()\n'
                       '    state_updates = {\n'
                       '        "pipeline_status": "using stored results",\n',
                       '        diagnostics["prediction_snapshot_error"] = str(exc)\n'
                       '        deferred_warnings.append(f"Prediction evidence was not saved: '
                       '{exc}")\n'
                       '\n'
                       '    from app_core.ncaaf_schedule import refresh_coverage\n'
                       '    refresh_coverage(diagnostics, diagnostics.get("candidate_audit_df", '
                       'candidate_pool), best_picks_df)\n'
                       '    timer.finish()\n'
                       '    state_updates = {\n'
                       '        "pipeline_status": "using stored results",\n'],
                      ['\n'
                       '    if _analysis_inputs_stale(st.session_state, controls):\n'
                       '        st.error(\n'
                       '            "Analysis inputs changed after the displayed results were '
                       'generated. "\n'
                       '            "Click **Run Game Analysis** to apply the current TheOver '
                       'files. "\n'
                       '            "Stale picks and exports are hidden until the rerun '
                       'completes."\n'
                       '        )\n'
                       '        return\n',
                       '\n'
                       '    if _analysis_inputs_stale(st.session_state, controls):\n'
                       '        st.error(\n'
                       '            "Analysis inputs or schedule dates changed after the displayed '
                       'results were generated. "\n'
                       '            "Click **Refresh picks** to apply the current inputs. "\n'
                       '            "Stale picks and exports are hidden until the rerun '
                       'completes."\n'
                       '        )\n'
                       '        return\n'],
                      ['\n'
                       '        from app.ui.readiness_dashboard import render_readiness_dashboard\n'
                       '        '
                       'render_readiness_dashboard(diagnostics.get("candidate_authority_df"), '
                       'best_picks_df, diagnostics)\n'
                       '\n'
                       '    if analysis_df is None or analysis_df.empty:\n'
                       '        # History grading and republication must remain available after a '
                       'restart,\n',
                       '\n'
                       '        from app.ui.readiness_dashboard import render_readiness_dashboard\n'
                       '        '
                       'render_readiness_dashboard(diagnostics.get("candidate_authority_df"), '
                       'best_picks_df, diagnostics)\n'
                       '\n'
                       '    with today_tab:\n'
                       '        from app.ui.ncaaf_inventory import render_inventory\n'
                       '        render_inventory(diagnostics)\n'
                       '\n'
                       '    if analysis_df is None or analysis_df.empty:\n'
                       '        # History grading and republication must remain available after a '
                       'restart,\n']],
 'app/ui/sidebar_controls.py': [['    advanced.subheader("Analysis Engines")\n'
                                 '\n'
                                 '    use_ml = advanced.checkbox("Enable ML Predictions", True, '
                                 'key="use_ml")\n'
                                 '    use_gemini = advanced.checkbox(\n'
                                 '        "Require Gemini Review for Bets",\n'
                                 '        value=True,\n',
                                 '    advanced.subheader("Analysis Engines")\n'
                                 '\n'
                                 '    use_ml = advanced.checkbox("Enable ML Predictions", True, '
                                 'key="use_ml")\n'
                                 '    from app.ui.ncaaf_inventory import schedule_controls\n'
                                 '    schedule_start, schedule_end = schedule_controls(advanced, '
                                 'sports)\n'
                                 '    use_gemini = advanced.checkbox(\n'
                                 '        "Require Gemini Review for Bets",\n'
                                 '        value=True,\n'],
                                ['        "show_kalshi_diagnostics": show_kalshi_diagnostics,\n'
                                 '        "theover_spreads": theover_spreads,\n'
                                 '        "theover_totals": theover_totals,\n'
                                 '        "prop_results_log": active_ledger,\n'
                                 '        "run_analysis_counter": run_counter,\n'
                                 '        "run_player_props": run_player_props,\n',
                                 '        "show_kalshi_diagnostics": show_kalshi_diagnostics,\n'
                                 '        "theover_spreads": theover_spreads,\n'
                                 '        "theover_totals": theover_totals,\n'
                                 '        "schedule_start": schedule_start,\n'
                                 '        "schedule_end": schedule_end,\n'
                                 '        "prop_results_log": active_ledger,\n'
                                 '        "run_analysis_counter": run_counter,\n'
                                 '        "run_player_props": run_player_props,\n']],
 'app_core/ncaaf_identity.py': [['\n'
                                 '\n'
                                 'GROUPS = (\n'
                                 '    ("missouri", "mizzou", "missouri tigers"),\n'
                                 '    ("app state", "appalachian state", "appalachian state '
                                 'mountaineers"),\n'
                                 '    ("army", "army black knights"),\n'
                                 '    ("illinois", "illinois fighting illini"),\n'
                                 '    ("vanderbilt", "vanderbilt commodores"),\n'
                                 '    ("gardner webb", "gardner-webb", "gardner-webb runnin '
                                 'bulldogs", "gardner-webb running bulldogs"),\n'
                                 '    ("bowling green", "bgsu", "bowling green falcons"),\n'
                                 '    ("kennesaw state", "kennesaw state owls"),\n'
                                 '    ("southern", "southern university", "southern university '
                                 'jaguars"),\n',
                                 '\n'
                                 '\n'
                                 'GROUPS = (\n'
                                 '    ("mcneese", "mcneese state", "mcneese cowboys", "mcneese '
                                 'state cowboys"),\n'
                                 '    ("lsu", "lsu tigers", "louisiana state", "louisiana state '
                                 'tigers"),\n'
                                 '    ("missouri", "mizzou", "missouri tigers"),\n'
                                 '    ("app state", "appalachian state", "appalachian state '
                                 'mountaineers"),\n'
                                 '    ("army", "army black knights"),\n'
                                 '    ("illinois", "illinois fighting illini"),\n'
                                 '    ("vanderbilt", "vanderbilt commodores"),\n'
                                 '    ("gardner webb", "gardnerwebb", "gardner-webb", '
                                 '"gardner-webb runnin bulldogs", "gardner-webb running '
                                 'bulldogs"),\n'
                                 '    ("bowling green", "bgsu", "bowling green falcons"),\n'
                                 '    ("kennesaw state", "kennesaw state owls"),\n'
                                 '    ("southern", "southern university", "southern university '
                                 'jaguars"),\n']],
 'app_core/espn_ncaaf_odds.py': [['    # The fallback and primary feeds use different school names '
                                  'and mascots.\n'
                                  "    # Share the pipeline's exact aliases before comparing event "
                                  'identities so\n'
                                  '    # recovery cannot append a second game (or displace the '
                                  'primary quote).\n'
                                  '    normalized = normalize_team_name(str(value or "")).lower()\n'
                                  '    compact = re.sub(r"[^a-z0-9]", "", normalized)\n'
                                  '    # Explicit school aliases used by the primary and ESPN '
                                  'college feeds.\n'
                                  '    return {"gramblingstate": "grambling", '
                                  '"gramblingstatetigers": "grambling",\n'
                                  '            "southernuniversity": "southern", '
                                  '"southernjaguars": "southern"}.get(compact, compact)\n'
                                  '\n'
                                  '\n'
                                  'def _game_key(game: dict[str, Any]) -> tuple[str, str]:\n'
                                  '    teams = sorted(\n'
                                  '        [_canonical_team(game.get("home_team")), '
                                  '_canonical_team(game.get("away_team"))]\n'
                                  '    )\n'
                                  '    return teams[0], teams[1]\n'
                                  '\n'
                                  '\n'
                                  'def merge_missing_ncaaf_games(\n',
                                  '    # The fallback and primary feeds use different school names '
                                  'and mascots.\n'
                                  "    # Share the pipeline's exact aliases before comparing event "
                                  'identities so\n'
                                  '    # recovery cannot append a second game (or displace the '
                                  'primary quote).\n'
                                  '    from app_core.ncaaf_identity import normalize_ncaaf_team\n'
                                  '    normalized = normalize_ncaaf_team(value)\n'
                                  '    compact = re.sub(r"[^a-z0-9]", "", normalized)\n'
                                  '    # Explicit school aliases used by the primary and ESPN '
                                  'college feeds.\n'
                                  '    return {"gramblingstate": "grambling", '
                                  '"gramblingstatetigers": "grambling",\n'
                                  '            "southernuniversity": "southern", '
                                  '"southernjaguars": "southern"}.get(compact, compact)\n'
                                  '\n'
                                  '\n'
                                  'def _game_key(game: dict[str, Any]) -> tuple[str, str, str]:\n'
                                  '    teams = sorted(\n'
                                  '        [_canonical_team(game.get("home_team")), '
                                  '_canonical_team(game.get("away_team"))]\n'
                                  '    )\n'
                                  '    from app_core.ncaaf_schedule import timestamp\n'
                                  '    kickoff = timestamp(game.get("commence_time"))\n'
                                  '    return teams[0], teams[1], kickoff.isoformat() if kickoff '
                                  'else ""\n'
                                  '\n'
                                  '\n'
                                  'def merge_missing_ncaaf_games(\n'],
                                 ['        if not isinstance(game, dict):\n'
                                  '            continue\n'
                                  '        key = _game_key(game)\n'
                                  '        if not all(key) or key in seen:\n'
                                  '            continue\n'
                                  '        merged.append(game)\n'
                                  '        seen.add(key)\n',
                                  '        if not isinstance(game, dict):\n'
                                  '            continue\n'
                                  '        key = _game_key(game)\n'
                                  '        if not all(key):\n'
                                  '            # No usable kickoff: retain the source event, never '
                                  'infer equality.\n'
                                  '            merged.append(game)\n'
                                  '            continue\n'
                                  '        if key in seen:\n'
                                  '            # Different explicit ESPN IDs prove distinct source '
                                  'events even\n'
                                  '            # when schools and kickoff coincide. Preserve that '
                                  'distinction.\n'
                                  '            event_id = str(game.get("id") or "")\n'
                                  '            same_key = [g for g in merged if _game_key(g) == '
                                  'key]\n'
                                  '            if event_id.startswith("espn-") and same_key and '
                                  'all(\n'
                                  '                str(g.get("id") or "").startswith("espn-") and '
                                  'g.get("id") != event_id for g in same_key\n'
                                  '            ):\n'
                                  '                merged.append(game)\n'
                                  '            continue\n'
                                  '        merged.append(game)\n'
                                  '        seen.add(key)\n']]}

SCHEDULE_FIXED_SHA256 = {'core/streamlit_pipeline.py': '92eafdffcdcb3dcb2efe801688f0130d6ac0fd35e51eeacde72c1c8277eb0be1',
 'streamlit_app.py': '7e438d76e8210c96a7b1fd9a549d9a4b68bc5548b6267392a86f9ea5e3c01f19',
 'app/ui/sidebar_controls.py': 'c674ff2be375ba5ad6464028ffb5e6b0f586e525f4c6b96e13670fa6bada071f',
 'app_core/ncaaf_identity.py': 'cd83e7b0915d9c98b6b7c3abbf4f1d28f0157c185d0bcd1c05d9ad9d4918f438',
 'app_core/espn_ncaaf_odds.py': '70e19bd69c23b286d2d401f5f8ebcbd4adcf7b82717d10f4153f963b6958c302',
 'app_core/ncaaf_schedule.py': '791ec09c4cc0eb83a027856684244487c9109a08f73a7a7c19a08ccd05cef13e',
 'app/ui/ncaaf_inventory.py': '68cfc102d6d1e86c9cfa0eaf002c80b81b32f05c3d0ae02f68147a4e327289db',
 'tests/test_ncaaf_schedule_coverage.py': '25a718a9e049d4a3f1b22d96080154b0a2d25826969e2e9b781aa657517bb47b',
 'tests/test_ncaaf_schedule_scope_policy.py': 'd5264893c98112f4c2cd2fde46fd100b411310efe929a6aee300b89d7f4a3428',
 'docs/paid-launch/ncaaf-schedule-coverage.md': 'daba80525870036e55ce2f07b67cadea999da4ea26eebd930048f8645be655dc'}

SCHEDULE_GUARD_LOGIC_SHA256 = 'a0fa2ff3ecf94c66219e7632da49808daa8f7e4e762504e031e1cb194456c5e1'

def _validate_schedule_policy(manifest_path, base, binding):
    _require(manifest_path.resolve() == (ROOT / MANIFEST_PATH).resolve(), "BASELINE_PATH_NOT_APPROVED")
    original_manifest = git_bytes("show", f"{binding['base']}:{MANIFEST_PATH}")
    manifest = json.loads(original_manifest)
    _require(base in (None, binding["base"], manifest["base_sha"]), "COMPARISON_BASE_NOT_APPROVED")
    _require(git("rev-parse", f"{binding['base']}^{{tree}}") == binding["base_tree"], "STARTING_TREE_NOT_APPROVED")
    _require(hashlib.sha256(original_manifest).hexdigest() == binding["manifest_sha256"], "ORIGINAL_BASELINE_IDENTITY_CHANGED")
    _require(not git("diff", "--name-only") and not git("diff", "--cached", "--name-only"), "LOCAL_TRACKED_CHANGE")
    previous_guard = git_bytes("show", f"{binding['base']}:{GUARD_PATH}")
    _require(hashlib.sha256(previous_guard).hexdigest() == binding["previous_guard_sha256"], "PREVIOUS_TOOLING_IDENTITY_CHANGED")
    policy_raw = git_bytes("show", f"HEAD:{SCHEDULE_POLICY_PATH}")
    _require((ROOT / SCHEDULE_POLICY_PATH).read_bytes().replace(b"\r\n", b"\n") == policy_raw, "POLICY_CHECKOUT_CHANGED")
    policy = json.loads(policy_raw)
    fields = {"schema_version", "policy_version", "approval_reference", "base_sha", "base_tree", "original_manifest_sha256",
              "implementation_commit", "implementation_tree", "implementation_changes", "tooling_sha256", "unchanged_bindings"}
    _require(set(policy) == fields and policy["schema_version"] == 1 and policy["policy_version"] == SCHEDULE_POLICY_VERSION,
             "SUCCESSOR_POLICY_SCHEMA_INVALID")
    _require(policy["approval_reference"] == SCHEDULE_APPROVAL_REFERENCE and policy["base_sha"] == binding["base"] and
             policy["base_tree"] == binding["base_tree"] and policy["original_manifest_sha256"] == binding["manifest_sha256"],
             "APPROVAL_BINDING_CHANGED")
    implementation = policy["implementation_commit"]
    _require(git("show", "-s", "--format=%P", implementation).split() == [binding["base"]], "IMPLEMENTATION_PARENT_NOT_APPROVED")
    _require(git("rev-parse", f"{implementation}^{{tree}}") == policy["implementation_tree"], "IMPLEMENTATION_TREE_MISMATCH")
    _require(not exists_at(implementation, SCHEDULE_POLICY_PATH), "POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION")
    _require(set(git("diff", "--name-only", binding["base"], implementation).splitlines()) == set(SCHEDULE_PATHS),
             "IMPLEMENTATION_CHANGE_SET_NOT_APPROVED")
    changes = {p: {"before_blob": blob(binding["base"], p), "after_blob": blob(implementation, p)} for p in SCHEDULE_PATHS}
    _require(policy["implementation_changes"] == changes, "IMPLEMENTATION_BLOB_BINDINGS_CHANGED")
    for path, edits in SCHEDULE_SHARED_EDITS.items():
        expected = git_bytes("show", f"{binding['base']}:{path}").decode()
        for before, after in edits:
            _require(expected.count(before) == 1, "SCHEDULE_BASE_ANCHOR_CHANGED")
            expected = expected.replace(before, after, 1)
        _require(git_bytes("show", f"{implementation}:{path}").decode() == expected, "SCHEDULE_SCOPE_NOT_APPROVED")
    for path, expected_hash in SCHEDULE_FIXED_SHA256.items():
        _require(hashlib.sha256(git_bytes("show", f"{implementation}:{path}")).hexdigest() == expected_hash,
                 "IMPLEMENTATION_BYTES_NOT_APPROVED")
    _require(git_bytes("show", f"{implementation}:{GUARD_PATH}").split(b"\nSCHEDULE_POLICY_PATH =", 1)[0] ==
             previous_guard.split(b"\ndef main() -> int:", 1)[0], "PREVIOUS_GUARD_LOGIC_CHANGED")
    logic = git_bytes("show", f"{implementation}:{GUARD_PATH}").split(b"\ndef _validate_schedule_policy", 1)[1]
    _require(hashlib.sha256(b"def _validate_schedule_policy" + logic).hexdigest() == SCHEDULE_GUARD_LOGIC_SHA256,
             "SUCCESSOR_GUARD_LOGIC_CHANGED")
    unchanged = {path: blob(binding["base"], path) for path in SCHEDULE_IMMUTABLE}
    _require(policy["unchanged_bindings"] == unchanged, "IMMUTABLE_BINDINGS_CHANGED")
    for path, expected_blob in unchanged.items():
        _require(blob("HEAD", path) == expected_blob and (ROOT / path).read_bytes().replace(b"\r\n", b"\n") ==
                 git_bytes("show", f"{binding['base']}:{path}"), "IMMUTABLE_FILE_CHANGED")
    _require(unchanged[CLOCK_TEST] == binding["clock_blob"], "PREVIOUS_CLOCK_IDENTITY_CHANGED")
    parents = git("show", "-s", "--format=%P", "HEAD").split()
    if len(parents) == 2:
        _require(parents[0] == binding["base"], "CI_BASE_PARENT_NOT_APPROVED")
        candidate = parents[1]
        _require(git("rev-parse", "HEAD^{tree}") == git("rev-parse", f"{candidate}^{{tree}}"), "CI_MERGE_TREE_CHANGED")
    else:
        candidate = git("rev-parse", "HEAD")
    _require(git("show", "-s", "--format=%P", candidate).split() == [implementation], "CANDIDATE_NOT_POLICY_SEAL")
    _require(git("diff", "--name-status", implementation, candidate).splitlines() == [f"A\t{SCHEDULE_POLICY_PATH}"], "SEAL_CHANGE_SET_NOT_APPROVED")
    tooling = {p: hashlib.sha256(git_bytes("show", f"{implementation}:{p}")).hexdigest() for p in manifest["tooling_sha256"]}
    _require(policy["tooling_sha256"] == tooling, "SUCCESSOR_TOOLING_BINDING_CHANGED")
    _require(tooling[".github/workflows/paid-launch.yml"] == manifest["tooling_sha256"][".github/workflows/paid-launch.yml"], "WORKFLOW_CHANGE_NOT_APPROVED")
    conversions = []
    for path in set(SCHEDULE_PATHS) | set(tooling):
        committed = git_bytes("show", f"HEAD:{path}")
        checkout = (ROOT / path).read_bytes()
        _require(checkout.replace(b"\r\n", b"\n") == committed, "UNAUTHORIZED_CHECKOUT_CHANGE")
        if checkout != committed:
            conversions.append(path)
    shadows = []
    for path in set(SCHEDULE_PATHS) | set(manifest["protected_files"]):
        if not path.endswith(".py"):
            continue
        suffix = Path(path).parts
        for other in ROOT.rglob(Path(path).name):
            relative = other.relative_to(ROOT)
            if tuple(relative.parts[-len(suffix):]) == tuple(suffix) and relative.as_posix() != path:
                # The existing nested Streamlit entry point is already pinned
                # above as immutable evidence. This permits only its exact
                # retained bytes; arbitrary same-name runtime copies fail.
                if relative.as_posix() == "parlaypicker/app/streamlit_app.py" and relative.as_posix() in unchanged:
                    continue
                shadows.append(relative.as_posix())
    hooks = [p for p in ROOT.rglob("*") if p.is_file() and (p.name in {"sitecustomize.py", "usercustomize.py"} or p.suffix == ".pth")]
    _require(not shadows and not hooks, "PROTECTED_RUNTIME_SHADOWING_RISK")
    return policy, conversions


def _run_schedule_integrated(manifest_path, base, binding):
    try:
        policy, conversions = _validate_schedule_policy(manifest_path, base, binding)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        return 1, {"schema_version": 4, "status": "FAIL", "reason_codes": [str(exc)],
                   "approved_exceptions": [], "approved_integration_changes": [], "protected_changes": [],
                   "existing_test_changes": [], "policy_valid": False}
    _, original = run(manifest_path, base)
    report = copy.deepcopy(original)
    report.update(schema_version=4, policy_version=policy["policy_version"], policy_valid=True,
                  original_guard_report=original, checkout_line_ending_conversions=conversions,
                  approved_integration_changes=policy["implementation_changes"],
                  approved_exceptions=[{"path": CLOCK_TEST, "before_blob": blob(original["recorded_base_sha"], CLOCK_TEST),
                                        "after_blob": binding["clock_blob"], "retained_unchanged": True,
                                        "approval_reference": APPROVAL_REFERENCE}],
                  approved_tooling_changes={GUARD_PATH: policy["tooling_sha256"][GUARD_PATH]}, new_existing_test_exceptions=[])
    report["existing_test_changes"] = [p for p in original["existing_test_changes"] if p != CLOCK_TEST]
    report["self_protected_changes"] = [p for p in original["self_protected_changes"] if p != GUARD_PATH]
    report["tooling_hash_mismatches"] = [p for p in original["tooling_hash_mismatches"] if p != GUARD_PATH and p not in conversions]
    removable = set()
    for key, reason in (("existing_test_changes", "EXISTING_TEST_EXPECTATION_CHANGED"),
                        ("self_protected_changes", "SCOPE_GUARD_OR_BASELINE_CHANGED"),
                        ("tooling_hash_mismatches", "SCOPE_GUARD_TOOLING_HASH_MISMATCH")):
        if not report[key]: removable.add(reason)
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
        if exists_at("HEAD", SCHEDULE_POLICY_PATH):
            code, report = _run_schedule_integrated(args.manifest, args.base, SCHEDULE_BINDINGS)
        elif exists_at("HEAD", V3_POLICY_PATH):
            code, report = _run_provider_integrated(args.manifest, args.base, PROVIDER_BINDINGS)
        else:
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
