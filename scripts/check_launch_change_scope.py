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


V4_POLICY_PATH = "docs/paid-launch/launch-scope-policy-v4.json"
V4_POLICY_VERSION = "paid-launch-dfs-projection-identity-v4"
V4_APPROVAL_REFERENCE = "Owner-authorized bounded draft: DFS zero/missing/invalid projections, identity joins and canonical player uniqueness at main af7b3dcbc3fe9ca7161b886c65074be3d739bae8; no merge or deployment authorization"
DFS_PATHS = (
    "app_core/draftkings_classic.py", "app/ui/draftkings.py",
    "tests/test_dfs_projection_identity.py", GUARD_PATH,
    "tests/test_dfs_scope_policy.py", "docs/paid-launch/dfs-projection-identity-policy.md",
)
DFS_UNCHANGED_PATHS = (
    *PROVIDER_UNCHANGED_PATHS, V3_POLICY_PATH, ".github/workflows/ci.yml",
    "tests/test_draftkings_classic.py", "tests/test_draftkings_mlb_classic.py",
    "tests/test_draftkings_panel.py", "core/probability_calibration.py",
    "data/calibration/effective_prob_calibration.json", "data/calibration/bucket_stats.json",
    *(path for path in PROVIDER_PATHS if path != GUARD_PATH),
)
DFS_BINDINGS = {
    "base": "af7b3dcbc3fe9ca7161b886c65074be3d739bae8",
    "base_tree": "98a117dbc7f6977bafed53624898d95a4e72f334",
    "manifest_sha256": "2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343",
    "previous_guard_sha256": "a93f90db1843d6da334adf17979fcecba20fd3549b47864a374e1cf6535baf37",
    "previous_policy_blob": "fd145cd5cc8c8e23b8f891dbf17220e4b2fba4bb",
    "clock_blob": "e610143aff5611235f9cfb44da13e2d54e1c6b48",
    "successor_guard_sha256": "d59447b3c015f31025c8acea72011287d3446ab817417b19ac43fcc62fa08031",
    "reviewed_dfs_blobs": {
        "app_core/draftkings_classic.py": "d4d3fb823bebaa3d3a4c4889947d5b981e126c23",
        "app/ui/draftkings.py": "f17dcc0dc9bf7c880022e4bdf143e99bff0e67d6",
        "tests/test_dfs_projection_identity.py": "251f09c45c0f4977046c47bbd8bc0cfeecfa5261",
    },
}


def _dfs_blobs(revision: str, paths) -> dict[str, str | None]:
    """Read exact Git object identities in one process, including absent paths."""
    ordered = sorted(paths)
    result = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype)"],
                            cwd=ROOT, input="".join(f"{revision}:{path}\n" for path in ordered),
                            text=True, capture_output=True, check=True)
    lines = result.stdout.splitlines()
    _require(len(lines) == len(ordered), "BLOB_BATCH_INCOMPLETE")
    values = {}
    for path, line in zip(ordered, lines):
        if line.endswith(" missing"):
            values[path] = None
        else:
            identity, kind = line.split()
            _require(kind == "blob", "SCOPE_PATH_NOT_BLOB")
            values[path] = identity
    return values


def _validate_dfs_policy(manifest_path: Path, base: str | None, binding: dict) -> tuple[dict, list[str]]:
    _require(manifest_path.resolve() == (ROOT / MANIFEST_PATH).resolve(), "BASELINE_PATH_NOT_APPROVED")
    original_manifest = git_bytes("show", f"{binding['base']}:{MANIFEST_PATH}")
    manifest = json.loads(original_manifest)
    _require(base in (None, binding["base"], manifest["base_sha"]), "COMPARISON_BASE_NOT_APPROVED")
    _require(git("rev-parse", f"{binding['base']}^{{tree}}") == binding["base_tree"], "STARTING_TREE_NOT_APPROVED")
    _require(hashlib.sha256(original_manifest).hexdigest() == binding["manifest_sha256"], "ORIGINAL_BASELINE_IDENTITY_CHANGED")
    previous_guard = git_bytes("show", f"{binding['base']}:{GUARD_PATH}")
    _require(hashlib.sha256(previous_guard).hexdigest() == binding["previous_guard_sha256"], "PREVIOUS_TOOLING_IDENTITY_CHANGED")
    _require(blob(binding["base"], V3_POLICY_PATH) == binding["previous_policy_blob"] and
             blob(binding["base"], CLOCK_TEST) == binding["clock_blob"], "PREVIOUS_POLICY_IDENTITY_CHANGED")
    _require(not git("diff", "--name-only") and not git("diff", "--cached", "--name-only"), "LOCAL_TRACKED_CHANGE")
    raw = git_bytes("show", f"HEAD:{V4_POLICY_PATH}")
    _require((ROOT / V4_POLICY_PATH).read_bytes().replace(b"\r\n", b"\n") == raw, "POLICY_CHECKOUT_CHANGED")
    policy = json.loads(raw)
    keys = {"schema_version", "policy_version", "approval_reference", "base_sha", "base_tree",
            "original_manifest_sha256", "previous_policy_blob", "clock_test_blob", "implementation_commit",
            "implementation_tree", "implementation_changes", "tooling_sha256", "unchanged_bindings"}
    _require(set(policy) == keys and policy["schema_version"] == 4 and
             policy["policy_version"] == V4_POLICY_VERSION, "SUCCESSOR_POLICY_SCHEMA_INVALID")
    expected = {"approval_reference": V4_APPROVAL_REFERENCE, "base_sha": binding["base"],
                "base_tree": binding["base_tree"], "original_manifest_sha256": binding["manifest_sha256"],
                "previous_policy_blob": binding["previous_policy_blob"], "clock_test_blob": binding["clock_blob"]}
    _require(all(policy[key] == value for key, value in expected.items()), "APPROVAL_BINDING_CHANGED")
    implementation = policy["implementation_commit"]
    _require(git("show", "-s", "--format=%P", implementation).split() == [binding["base"]], "IMPLEMENTATION_PARENT_NOT_APPROVED")
    _require(git("rev-parse", f"{implementation}^{{tree}}") == policy["implementation_tree"], "IMPLEMENTATION_TREE_MISMATCH")
    _require(not exists_at(implementation, V4_POLICY_PATH), "POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION")
    _require(set(git("diff", "--name-only", binding["base"], implementation).splitlines()) == set(DFS_PATHS),
             "IMPLEMENTATION_CHANGE_SET_NOT_APPROVED")
    before = _dfs_blobs(binding["base"], DFS_PATHS)
    after = _dfs_blobs(implementation, DFS_PATHS)
    changes = {path: {"before_blob": before[path], "after_blob": after[path]} for path in DFS_PATHS}
    _require(policy["implementation_changes"] == changes, "IMPLEMENTATION_BLOB_BINDINGS_CHANGED")
    for path, reviewed in binding["reviewed_dfs_blobs"].items():
        _require(after[path] == reviewed, "DFS_REVIEWED_BLOB_CHANGED")
    frozen_prefix = previous_guard.split(b"\ndef main() -> int:", 1)[0]
    implementation_guard = git_bytes("show", f"{implementation}:{GUARD_PATH}")
    _require(implementation_guard.split(b"\nV4_POLICY_PATH =", 1)[0] == frozen_prefix, "PREVIOUS_GUARD_LOGIC_CHANGED")
    reviewed_guard = binding["successor_guard_sha256"]
    _require(implementation_guard.count(reviewed_guard.encode("ascii")) == 1 and
             hashlib.sha256(implementation_guard.replace(reviewed_guard.encode("ascii"), b"0" * 64)).hexdigest() == reviewed_guard,
             "SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    # Exact implementation/seal change sets preserve every other starting-main
    # path. Bind the original protected and prior integration evidence explicitly.
    retained_paths = set(manifest["protected_files"]) | set(DFS_UNCHANGED_PATHS)
    unchanged = _dfs_blobs(binding["base"], retained_paths)
    _require(policy["unchanged_bindings"] == unchanged, "IMMUTABLE_BINDINGS_CHANGED")
    _require(_dfs_blobs("HEAD", retained_paths) == unchanged, "IMMUTABLE_FILE_CHANGED")
    parents = git("show", "-s", "--format=%P", "HEAD").split()
    if len(parents) == 2:
        _require(parents[0] == binding["base"], "CI_BASE_PARENT_NOT_APPROVED")
        candidate = parents[1]
        _require(git("rev-parse", "HEAD^{tree}") == git("rev-parse", f"{candidate}^{{tree}}"), "CI_MERGE_TREE_CHANGED")
    else:
        candidate = git("rev-parse", "HEAD")
    _require(git("show", "-s", "--format=%P", candidate).split() == [implementation], "CANDIDATE_NOT_POLICY_SEAL")
    _require(git("diff", "--name-status", implementation, candidate).splitlines() == [f"A\t{V4_POLICY_PATH}"],
             "SEAL_CHANGE_SET_NOT_APPROVED")
    tooling = {path: hashlib.sha256(git_bytes("show", f"{implementation}:{path}")).hexdigest()
               for path in manifest["tooling_sha256"]}
    _require(policy["tooling_sha256"] == tooling, "SUCCESSOR_TOOLING_BINDING_CHANGED")
    _require(tooling[".github/workflows/paid-launch.yml"] == manifest["tooling_sha256"][".github/workflows/paid-launch.yml"],
             "WORKFLOW_CHANGE_NOT_APPROVED")
    conversions = []
    for path in (*DFS_PATHS, *manifest["tooling_sha256"], MANIFEST_PATH, POLICY_PATH, V3_POLICY_PATH, CLOCK_TEST):
        committed = git_bytes("show", f"HEAD:{path}"); checkout = (ROOT / path).read_bytes()
        _require(checkout.replace(b"\r\n", b"\n") == committed, "UNAUTHORIZED_CHECKOUT_CHANGE")
        if path in tooling and checkout != committed:
            conversions.append(path)
    shadow_paths = set(manifest["protected_files"]) | set(PROVIDER_PATHS) | {"app_core/draftkings_classic.py"}
    for path in shadow_paths:
        suffix = Path(path).parts
        for other in ROOT.rglob(Path(path).name):
            relative = other.relative_to(ROOT)
            _require(tuple(relative.parts[-len(suffix):]) != tuple(suffix) or relative.as_posix() == path,
                     "PROTECTED_RUNTIME_SHADOWING_RISK")
    _require(not any(p.is_file() and (p.name in {"sitecustomize.py", "usercustomize.py"} or p.suffix == ".pth")
                     for p in ROOT.rglob("*")), "PROTECTED_RUNTIME_SHADOWING_RISK")
    return policy, conversions


def _run_dfs_integrated(manifest_path: Path, base: str | None, binding: dict) -> tuple[int, dict]:
    try:
        policy, conversions = _validate_dfs_policy(manifest_path, base, binding)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        return 1, {"schema_version": 4, "status": "FAIL", "reason_codes": [str(exc)], "policy_valid": False,
                   "approved_exceptions": [], "approved_integration_changes": [], "new_existing_test_exceptions": []}
    _, original = run(manifest_path, base)
    report = copy.deepcopy(original)
    report.update(schema_version=4, policy_version=policy["policy_version"], policy_valid=True,
                  original_guard_report=original, checkout_line_ending_conversions=conversions,
                  approved_integration_changes=policy["implementation_changes"], new_existing_test_exceptions=[],
                  approved_exceptions=[{"path": CLOCK_TEST, "before_blob": PRODUCTION_BINDINGS["before"],
                                        "after_blob": binding["clock_blob"], "retained_unchanged": True,
                                        "approval_reference": APPROVAL_REFERENCE}],
                  approved_tooling_changes={GUARD_PATH: policy["tooling_sha256"][GUARD_PATH]})
    report["existing_test_changes"] = [p for p in original["existing_test_changes"] if p != CLOCK_TEST]
    report["self_protected_changes"] = [p for p in original["self_protected_changes"] if p != GUARD_PATH]
    report["tooling_hash_mismatches"] = [p for p in original["tooling_hash_mismatches"] if p != GUARD_PATH and p not in conversions]
    removable = {reason for key, reason in (("existing_test_changes", "EXISTING_TEST_EXPECTATION_CHANGED"),
                 ("self_protected_changes", "SCOPE_GUARD_OR_BASELINE_CHANGED"),
                 ("tooling_hash_mismatches", "SCOPE_GUARD_TOOLING_HASH_MISMATCH")) if not report[key]}
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
        if exists_at("HEAD", V4_POLICY_PATH):
            code, report = _run_dfs_integrated(args.manifest, args.base, DFS_BINDINGS)
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
