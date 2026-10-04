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

COVERAGE_POLICY_PATH = "docs/paid-launch/launch-scope-policy-ncaaf-v2.json"
COVERAGE_POLICY_VERSION = "paid-launch-ncaaf-coverage-v2"
COVERAGE_APPROVAL_REFERENCE = "Owner-authorized separate draft for late #2377 revision-fact and unpriced-quote findings at main ab30a6aa74a9fddd23aa207cd0a49baf15e1793f; preserve #2378 unmerged; no deployment or scientific/authority changes"
COVERAGE_PATHS = (
    "app_core/ncaaf_schedule.py", "tests/test_ncaaf_coverage_corrections.py",
    "tests/test_ncaaf_schedule_scope_policy.py", GUARD_PATH,
    "tests/test_ncaaf_coverage_scope_policy.py", "docs/paid-launch/ncaaf-coverage-corrections.md",
)
COVERAGE_UNCHANGED_PATHS = (
    *PROVIDER_UNCHANGED_PATHS, *SCHEDULE_IMMUTABLE, SCHEDULE_POLICY_PATH, V3_POLICY_PATH,
    *(p for p in SCHEDULE_PATHS if p not in COVERAGE_PATHS),
    *(p for p in PROVIDER_PATHS if p != GUARD_PATH),
    ".github/workflows/ci.yml", ".github/workflows/qualification-operations.yml",
    "tests/test_ncaaf_schedule_coverage.py", "app_core/prediction_evidence.py",
    "core/live_wager_contract.py", "core/wager_decisions.py", "core/sport_policy.py",
    "core/market_policy.py", "core/price_value.py", "core/probability_calibration.py",
    "data/calibration/effective_prob_calibration.json", "data/calibration/bucket_stats.json",
    "app_core/draftkings_classic.py", "app/ui/draftkings.py",
)
COVERAGE_MODULE_EDITS = [['import json\nimport re\n', 'import json\nimport math\nimport re\n'],
 ['                if (old["home_team_id"], old["away_team_id"]) != (row["home_team_id"], '
  'row["away_team_id"]) or len(old["kickoff_revisions"]) > 1 or old["schedule_status"] != '
  'row["schedule_status"]:\n'
  '                    old["identity_conflict"] = True\n'
  '                    issues.append("SCHEDULE_REVISION_CONFLICT")',
  '                conflict = ((old["home_team_id"], old["away_team_id"]) != (row["home_team_id"], '
  'row["away_team_id"])\n'
  '                            or len(old["kickoff_revisions"]) > 1 or old["schedule_status"] != '
  'row["schedule_status"]\n'
  '                            or row["identity_conflict"])\n'
  '                repaired = False\n'
  '                if not row["identity_conflict"]:\n'
  '                    for side in ("home", "away"):\n'
  '                        identity = side + "_team_id"\n'
  '                        if old[identity] and row[identity] and old[identity] != row[identity]:\n'
  "                            continue  # Do not attach another school's names to a known ID.\n"
  '                        for field in (side + "_team", identity):\n'
  '                            if not old[field] and row[field]:\n'
  '                                old[field] = row[field]\n'
  '                                repaired = True\n'
  '                        aliases = side + "_aliases"\n'
  '                        if not old[aliases] and row[aliases]:\n'
  '                            repaired = True\n'
  '                        old[aliases] = sorted(set(old[aliases] + row[aliases]))\n'
  '                    if not old["kickoff"] and row["kickoff"]:\n'
  '                        old["kickoff"] = row["kickoff"]\n'
  '                        repaired = True\n'
  '                    if old["schedule_status"] == "UNKNOWN" and row["schedule_status"] != "UNKNOWN":\n'
  '                        old["schedule_status"] = row["schedule_status"]\n'
  '                        repaired = True\n'
  '                # Recovered display facts never silently resolve identity or coverage.\n'
  '                if conflict or repaired:\n'
  '                    old["identity_conflict"] = True\n'
  '                    issues.append("SCHEDULE_REVISION_CONFLICT")'],
 ['\ndef coverage(inventory, games=(), candidates=(), selections=(), provider_health=None, *, now=None):',
  '\n'
  'def _priced_quote(quote):\n'
  '    """Coverage requires a finite American price; receipts remain untouched."""\n'
  '    value = quote.get("price")\n'
  '    if isinstance(value, bool):\n'
  '        return False\n'
  '    try:\n'
  '        price = float(value)\n'
  '    except (TypeError, ValueError, OverflowError):\n'
  '        return False\n'
  '    return math.isfinite(price) and abs(price) >= 100\n'
  '\n'
  '\n'
  'def coverage(inventory, games=(), candidates=(), selections=(), provider_health=None, *, now=None):'],
 ['        quotes = [q for r in matched + pool for q in _quotes(r)]\n        valid_quotes =',
  '        # One receipt copied through games/candidates is still one observation.\n'
  '        receipts = list({json.dumps(q, sort_keys=True, default=str): q\n'
  '                         for r in matched + pool for q in _quotes(r)}.values())\n'
  '        quotes = [q for q in receipts if _priced_quote(q)]\n'
  '        valid_quotes ='],
 ['for q in quotes})),\n                   provider_failure',
  'for q in receipts})),\n                   provider_failure'],
 ['                   quote_count=len({json.dumps(q, sort_keys=True, default=str) for q in quotes}),',
  '                   quote_count=len(quotes), quote_receipt_count=len(receipts),\n'
  '                   invalid_price_quote_count=len(receipts) - len(quotes), '
  'timestamped_quote_count=len(valid_quotes),']]

COVERAGE_V1_TEST_EDITS = [('for p in g.SCHEDULE_PATHS:write(repo,p,(SOURCE/p).read_bytes().replace(b"\\r\\n",b"\\n"))',
  'for p in g.SCHEDULE_PATHS:write(repo,p,original_schedule_source(p))',
  3),
 ('\ndef assess(fx):',
  '\n'
  '\n'
  'def original_schedule_source(path):\n'
  '    """Exercise the original v1 contract, never rebind it to the correction."""\n'
  '    source = (SOURCE / path).read_bytes().replace(b"\\r\\n", b"\\n")\n'
  '    if path == g.GUARD_PATH:\n'
  '        return source.split(b"\\nCOVERAGE_POLICY_PATH =", 1)[0] + g.COVERAGE_PREVIOUS_CLI\n'
  '    if path == "app_core/ncaaf_schedule.py":\n'
  '        text = source.decode()\n'
  '        for before, after in reversed(g.COVERAGE_MODULE_EDITS):\n'
  '            assert text.count(after) == 1\n'
  '            text = text.replace(after, before, 1)\n'
  '        return text.encode()\n'
  '    if path == "tests/test_ncaaf_schedule_scope_policy.py":\n'
  '        text = source.decode()\n'
  '        for before, after, count in reversed(g.COVERAGE_V1_TEST_EDITS):\n'
  '            assert text.count(after) == count\n'
  '            text = text.replace(after, before)\n'
  '        return text.encode()\n'
  '    return source\n'
  '\n'
  'def assess(fx):',
  1)]

# Frozen prior CLI is test data for reconstructing the original v1 seal.
COVERAGE_PREVIOUS_CLI = b'\ndef main() -> int:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)\n    parser.add_argument("--base")\n    parser.add_argument("--json-output", type=Path)\n    args = parser.parse_args()\n    try:\n        if exists_at("HEAD", SCHEDULE_POLICY_PATH):\n            code, report = _run_schedule_integrated(args.manifest, args.base, SCHEDULE_BINDINGS)\n        elif exists_at("HEAD", V3_POLICY_PATH):\n            code, report = _run_provider_integrated(args.manifest, args.base, PROVIDER_BINDINGS)\n        else:\n            code, report = _run_integrated(args.manifest, args.base, PRODUCTION_BINDINGS)\n    except Exception as exc:\n        report = {"schema_version": 1, "status": "ERROR", "reason_codes": ["GUARD_EXECUTION_ERROR"], "error": str(exc)}\n        code = 2\n    rendered = json.dumps(report, indent=2, sort_keys=True)\n    if args.json_output:\n        args.json_output.parent.mkdir(parents=True, exist_ok=True)\n        args.json_output.write_text(rendered + "\\n", encoding="utf-8")\n    print(rendered)\n    return code\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'

COVERAGE_BINDINGS = {'base': 'ab30a6aa74a9fddd23aa207cd0a49baf15e1793f',
 'base_tree': 'c7c41cd049702a9ef53975577d844b0e0999d8d4',
 'clock_blob': 'e610143aff5611235f9cfb44da13e2d54e1c6b48',
 'manifest_sha256': '2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343',
 'previous_guard_sha256': 'ff8eefd620de8aad670a6baeb59a2b2f32d41058b6faa3e0343e3f26de3fb70a',
 'previous_policy_blob': '0732a3d5f0a044fce57c09d0687afef5892e0a1e',
 'reviewed_coverage_blobs': {'app_core/ncaaf_schedule.py': 'c8ad20d279d97da195d4339acf738d07a21db482',
                             'tests/test_ncaaf_coverage_corrections.py': 'a7db1ff5582e8327220d31d92a8235157db1784a',
                             'tests/test_ncaaf_schedule_scope_policy.py': '3ad103128d828872216a109683b22308f38bce32'},
 'successor_guard_sha256': 'd819a4b52875b54be481cad00857f336799091dda8488e7a08825f62da40a4b1'}


def _coverage_blobs(revision: str, paths) -> dict[str, str | None]:
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


def _validate_coverage_policy(manifest_path: Path, base: str | None, binding: dict) -> tuple[dict, list[str]]:
    _require(manifest_path.resolve() == (ROOT / MANIFEST_PATH).resolve(), "BASELINE_PATH_NOT_APPROVED")
    original_manifest = git_bytes("show", f"{binding['base']}:{MANIFEST_PATH}")
    manifest = json.loads(original_manifest)
    _require(base in (None, binding["base"], manifest["base_sha"]), "COMPARISON_BASE_NOT_APPROVED")
    _require(git("rev-parse", f"{binding['base']}^{{tree}}") == binding["base_tree"], "STARTING_TREE_NOT_APPROVED")
    _require(hashlib.sha256(original_manifest).hexdigest() == binding["manifest_sha256"], "ORIGINAL_BASELINE_IDENTITY_CHANGED")
    previous_guard = git_bytes("show", f"{binding['base']}:{GUARD_PATH}")
    _require(hashlib.sha256(previous_guard).hexdigest() == binding["previous_guard_sha256"], "PREVIOUS_TOOLING_IDENTITY_CHANGED")
    _require(blob(binding["base"], SCHEDULE_POLICY_PATH) == binding["previous_policy_blob"] and
             blob(binding["base"], CLOCK_TEST) == binding["clock_blob"], "PREVIOUS_POLICY_IDENTITY_CHANGED")
    _require(not git("diff", "--name-only") and not git("diff", "--cached", "--name-only"), "LOCAL_TRACKED_CHANGE")
    raw = git_bytes("show", f"HEAD:{COVERAGE_POLICY_PATH}")
    _require((ROOT / COVERAGE_POLICY_PATH).read_bytes().replace(b"\r\n", b"\n") == raw, "POLICY_CHECKOUT_CHANGED")
    policy = json.loads(raw)
    keys = {"schema_version", "policy_version", "approval_reference", "base_sha", "base_tree",
            "original_manifest_sha256", "previous_policy_blob", "clock_test_blob", "implementation_commit",
            "implementation_tree", "implementation_changes", "tooling_sha256", "unchanged_bindings"}
    _require(set(policy) == keys and policy["schema_version"] == 5 and
             policy["policy_version"] == COVERAGE_POLICY_VERSION, "SUCCESSOR_POLICY_SCHEMA_INVALID")
    expected = {"approval_reference": COVERAGE_APPROVAL_REFERENCE, "base_sha": binding["base"],
                "base_tree": binding["base_tree"], "original_manifest_sha256": binding["manifest_sha256"],
                "previous_policy_blob": binding["previous_policy_blob"], "clock_test_blob": binding["clock_blob"]}
    _require(all(policy[key] == value for key, value in expected.items()), "APPROVAL_BINDING_CHANGED")
    implementation = policy["implementation_commit"]
    _require(git("show", "-s", "--format=%P", implementation).split() == [binding["base"]], "IMPLEMENTATION_PARENT_NOT_APPROVED")
    _require(git("rev-parse", f"{implementation}^{{tree}}") == policy["implementation_tree"], "IMPLEMENTATION_TREE_MISMATCH")
    _require(not exists_at(implementation, COVERAGE_POLICY_PATH), "POLICY_SEAL_MUST_FOLLOW_IMPLEMENTATION")
    _require(set(git("diff", "--name-only", binding["base"], implementation).splitlines()) == set(COVERAGE_PATHS),
             "IMPLEMENTATION_CHANGE_SET_NOT_APPROVED")
    before = _coverage_blobs(binding["base"], COVERAGE_PATHS)
    after = _coverage_blobs(implementation, COVERAGE_PATHS)
    changes = {path: {"before_blob": before[path], "after_blob": after[path]} for path in COVERAGE_PATHS}
    _require(policy["implementation_changes"] == changes, "IMPLEMENTATION_BLOB_BINDINGS_CHANGED")
    for path, reviewed in binding["reviewed_coverage_blobs"].items():
        _require(after[path] == reviewed, "COVERAGE_REVIEWED_BLOB_CHANGED")
    expected_module = git_bytes("show", f"{binding['base']}:app_core/ncaaf_schedule.py").decode()
    for before_edit, after_edit in COVERAGE_MODULE_EDITS:
        _require(expected_module.count(before_edit) == 1, "COVERAGE_BASE_ANCHOR_CHANGED")
        expected_module = expected_module.replace(before_edit, after_edit, 1)
    _require(git_bytes("show", f"{implementation}:app_core/ncaaf_schedule.py").decode() == expected_module,
             "COVERAGE_DIAGNOSTIC_SCOPE_CHANGED")
    frozen_prefix = previous_guard.split(b"\ndef main() -> int:", 1)[0]
    implementation_guard = git_bytes("show", f"{implementation}:{GUARD_PATH}")
    _require(implementation_guard.split(b"\nCOVERAGE_POLICY_PATH =", 1)[0] == frozen_prefix, "PREVIOUS_GUARD_LOGIC_CHANGED")
    reviewed_guard = binding["successor_guard_sha256"]
    _require(implementation_guard.count(reviewed_guard.encode("ascii")) == 1 and
             hashlib.sha256(implementation_guard.replace(reviewed_guard.encode("ascii"), b"0" * 64)).hexdigest() == reviewed_guard,
             "SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    # Exact implementation/seal change sets preserve every other starting-main
    # path. Bind the original protected and prior integration evidence explicitly.
    retained_paths = set(manifest["protected_files"]) | set(COVERAGE_UNCHANGED_PATHS)
    unchanged = _coverage_blobs(binding["base"], retained_paths)
    _require(policy["unchanged_bindings"] == unchanged, "IMMUTABLE_BINDINGS_CHANGED")
    _require(_coverage_blobs("HEAD", retained_paths) == unchanged, "IMMUTABLE_FILE_CHANGED")
    parents = git("show", "-s", "--format=%P", "HEAD").split()
    if len(parents) == 2:
        _require(parents[0] == binding["base"], "CI_BASE_PARENT_NOT_APPROVED")
        candidate = parents[1]
        _require(git("rev-parse", "HEAD^{tree}") == git("rev-parse", f"{candidate}^{{tree}}"), "CI_MERGE_TREE_CHANGED")
    else:
        candidate = git("rev-parse", "HEAD")
    _require(git("show", "-s", "--format=%P", candidate).split() == [implementation], "CANDIDATE_NOT_POLICY_SEAL")
    _require(git("diff", "--name-status", implementation, candidate).splitlines() == [f"A\t{COVERAGE_POLICY_PATH}"],
             "SEAL_CHANGE_SET_NOT_APPROVED")
    tooling = {path: hashlib.sha256(git_bytes("show", f"{implementation}:{path}")).hexdigest()
               for path in manifest["tooling_sha256"]}
    _require(policy["tooling_sha256"] == tooling, "SUCCESSOR_TOOLING_BINDING_CHANGED")
    _require(tooling[".github/workflows/paid-launch.yml"] == manifest["tooling_sha256"][".github/workflows/paid-launch.yml"],
             "WORKFLOW_CHANGE_NOT_APPROVED")
    conversions = []
    for path in (*COVERAGE_PATHS, *manifest["tooling_sha256"], *retained_paths):
        committed = git_bytes("show", f"HEAD:{path}"); checkout = (ROOT / path).read_bytes()
        _require(checkout.replace(b"\r\n", b"\n") == committed, "UNAUTHORIZED_CHECKOUT_CHANGE")
        if path in tooling and checkout != committed:
            conversions.append(path)
    shadow_paths = retained_paths | set(PROVIDER_PATHS) | set(SCHEDULE_PATHS) | set(COVERAGE_PATHS)
    for path in shadow_paths:
        suffix = Path(path).parts
        for other in ROOT.rglob(Path(path).name):
            relative = other.relative_to(ROOT)
            # The original nested entry point is pinned and checkout-verified above.
            if relative.as_posix() == "parlaypicker/app/streamlit_app.py" and relative.as_posix() in unchanged:
                continue
            _require(tuple(relative.parts[-len(suffix):]) != tuple(suffix) or relative.as_posix() == path,
                     "PROTECTED_RUNTIME_SHADOWING_RISK")
    _require(not any(p.is_file() and (p.name in {"sitecustomize.py", "usercustomize.py"} or p.suffix == ".pth")
                     for p in ROOT.rglob("*")), "PROTECTED_RUNTIME_SHADOWING_RISK")
    return policy, conversions


def _run_coverage_integrated(manifest_path: Path, base: str | None, binding: dict) -> tuple[int, dict]:
    try:
        policy, conversions = _validate_coverage_policy(manifest_path, base, binding)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        return 1, {"schema_version": 5, "status": "FAIL", "reason_codes": [str(exc)], "policy_valid": False,
                   "approved_exceptions": [], "approved_integration_changes": [], "new_existing_test_exceptions": []}
    _, original = run(manifest_path, base)
    report = copy.deepcopy(original)
    report.update(schema_version=5, policy_version=policy["policy_version"], policy_valid=True,
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


V4_POLICY_PATH = "docs/paid-launch/launch-scope-policy-v4.json"
V4_POLICY_VERSION = "paid-launch-dfs-projection-identity-v4-r2"
V4_APPROVAL_REFERENCE = "Owner-authorized bounded DFS draft continuation after verified #2379 merge b1401f42c12c26823b6fc6d223e6de0231572284; preserve reviewed DFS and both NCAAF corrections, original baseline/policies/clock/workflows/subscriber/scientific authority; no merge, deployment or financial authorization"
DFS_PATHS = (
    "app_core/draftkings_classic.py", "app/ui/draftkings.py",
    "tests/test_dfs_projection_identity.py", GUARD_PATH,
    "tests/test_dfs_scope_policy.py", "docs/paid-launch/dfs-projection-identity-policy.md",
    "tests/test_ncaaf_coverage_scope_policy.py",
)
DFS_UNCHANGED_PATHS = (
    *PROVIDER_UNCHANGED_PATHS, *SCHEDULE_IMMUTABLE, SCHEDULE_POLICY_PATH, V3_POLICY_PATH,
    *(path for path in SCHEDULE_PATHS if path != GUARD_PATH), ".github/workflows/ci.yml",
    "tests/test_draftkings_classic.py", "tests/test_draftkings_mlb_classic.py",
    "tests/test_draftkings_panel.py", "core/probability_calibration.py",
    "data/calibration/effective_prob_calibration.json", "data/calibration/bucket_stats.json",
    *(path for path in PROVIDER_PATHS if path != GUARD_PATH),
    *(path for path in COVERAGE_UNCHANGED_PATHS if path not in DFS_PATHS),
    COVERAGE_POLICY_PATH,
    *(path for path in COVERAGE_PATHS if path not in DFS_PATHS),
    '.github/workflows/activation-grading.yml',
    '.github/workflows/ci.yml',
    '.github/workflows/football-stage1.yml',
    '.github/workflows/football-stage2.yml',
    '.github/workflows/mlb-receipt-reconciliation.yml',
    '.github/workflows/paid-launch.yml',
    '.github/workflows/qualification-operations.yml',
    '.github/workflows/read-only-census.yml',
    '.github/workflows/research-scheduler.yml',
    '.github/workflows/subscriber-completion-postgres.yml',
)
DFS_BINDINGS = {
    "base": "b1401f42c12c26823b6fc6d223e6de0231572284",
    "base_tree": "94d263ab4f00fe6c06151c9a07eec7ac78d9e0d3",
    "manifest_sha256": "2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343",
    "previous_guard_sha256": "b125915b489a2bfaf182c90e689c65a8409364fbf3b8ab588c5800ecf92e4ddf",
    "previous_policy_blob": "0b633a8225c725dd56e260de2d5e0b523fb74b04",
    "clock_blob": "e610143aff5611235f9cfb44da13e2d54e1c6b48",
    "successor_guard_sha256": "868f2bd881a2aa7f77084834d4b3bf0e1a034a3cb225b8d0c36fa0e46425c681",
    "reviewed_dfs_blobs": {
        "tests/test_dfs_scope_policy.py": "a2e737b5457cdb6169800a05b209ff8dcc8ab1be",
        "tests/test_ncaaf_coverage_scope_policy.py": "76240a38a0f57e0082dc872c7ef19f6cf0155002",
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
    _require(blob(binding["base"], COVERAGE_POLICY_PATH) == binding["previous_policy_blob"] and
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
    _require(_dfs_guard_matches(implementation_guard, binding["successor_guard_sha256"]),
             "SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    expected_fixture = git_bytes("show", f"{binding['base']}:tests/test_ncaaf_coverage_scope_policy.py")
    for before_edit, after_edit in DFS_COVERAGE_FIXTURE_EDITS:
        _require(expected_fixture.count(before_edit) == 1, "PRIOR_FIXTURE_ANCHOR_CHANGED")
        expected_fixture = expected_fixture.replace(before_edit, after_edit, 1)
    _require(git_bytes("show", f"{implementation}:tests/test_ncaaf_coverage_scope_policy.py") == expected_fixture,
             "PRIOR_FIXTURE_RECONSTRUCTION_CHANGED")
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
    for path in (*DFS_PATHS, *manifest["tooling_sha256"], *retained_paths):
        committed = git_bytes("show", f"HEAD:{path}"); checkout = (ROOT / path).read_bytes()
        _require(checkout.replace(b"\r\n", b"\n") == committed, "UNAUTHORIZED_CHECKOUT_CHANGE")
        if path in tooling and checkout != committed:
            conversions.append(path)
    shadow_paths = retained_paths | set(PROVIDER_PATHS) | set(SCHEDULE_PATHS) | set(COVERAGE_PATHS) | set(DFS_PATHS)
    for path in shadow_paths:
        suffix = Path(path).parts
        for other in ROOT.rglob(Path(path).name):
            relative = other.relative_to(ROOT)
            # This existing entry point is accepted only after exact retained
            # blob and checkout verification above, including reseal attacks.
            if relative.as_posix() == "parlaypicker/app/streamlit_app.py" and relative.as_posix() in unchanged:
                continue
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
    report["approved_fixture_source_reconstructions"] = {
        "tests/test_ncaaf_coverage_scope_policy.py": policy["implementation_changes"]["tests/test_ncaaf_coverage_scope_policy.py"]}
    report["existing_test_changes"] = [p for p in original["existing_test_changes"]
                                       if p not in {CLOCK_TEST, "tests/test_ncaaf_coverage_scope_policy.py"}]
    report["self_protected_changes"] = [p for p in original["self_protected_changes"] if p != GUARD_PATH]
    report["tooling_hash_mismatches"] = [p for p in original["tooling_hash_mismatches"] if p != GUARD_PATH and p not in conversions]
    removable = {reason for key, reason in (("existing_test_changes", "EXISTING_TEST_EXPECTATION_CHANGED"),
                 ("self_protected_changes", "SCOPE_GUARD_OR_BASELINE_CHANGED"),
                 ("tooling_hash_mismatches", "SCOPE_GUARD_TOOLING_HASH_MISMATCH")) if not report[key]}
    report["reason_codes"] = [reason for reason in original["reason_codes"] if reason not in removable]
    report["status"] = "FAIL" if report["reason_codes"] else "PASS"
    return (1 if report["reason_codes"] else 0), report

DFS_PREVIOUS_CLI = b'\ndef main() -> int:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)\n    parser.add_argument("--base")\n    parser.add_argument("--json-output", type=Path)\n    args = parser.parse_args()\n    try:\n        if exists_at("HEAD", COVERAGE_POLICY_PATH):\n            code, report = _run_coverage_integrated(args.manifest, args.base, COVERAGE_BINDINGS)\n        elif exists_at("HEAD", SCHEDULE_POLICY_PATH):\n            code, report = _run_schedule_integrated(args.manifest, args.base, SCHEDULE_BINDINGS)\n        elif exists_at("HEAD", V3_POLICY_PATH):\n            code, report = _run_provider_integrated(args.manifest, args.base, PROVIDER_BINDINGS)\n        else:\n            code, report = _run_integrated(args.manifest, args.base, PRODUCTION_BINDINGS)\n    except Exception as exc:\n        report = {"schema_version": 1, "status": "ERROR", "reason_codes": ["GUARD_EXECUTION_ERROR"], "error": str(exc)}\n        code = 2\n    rendered = json.dumps(report, indent=2, sort_keys=True)\n    if args.json_output:\n        args.json_output.parent.mkdir(parents=True, exist_ok=True)\n        args.json_output.write_text(rendered + "\\n", encoding="utf-8")\n    print(rendered)\n    return code\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'

DFS_COVERAGE_FIXTURE_EDITS = [(b'def policy_for(repo, binding, implementation):',
  b'def original_coverage_source(path):\n    """Reconstruct the reviewed prior fixtures without c'
  b'hanging assertions."""\n    source = (SOURCE / path).read_bytes().replace(b"\\r\\n", b"\\n")'
  b'\n    if path == guard.GUARD_PATH:\n        return guard._dfs_previous_guard_source(source)\n  '
  b'  if path == "tests/test_ncaaf_coverage_scope_policy.py":\n        return guard._dfs_restore_'
  b'coverage_fixture(source)\n    return source\n\n\ndef policy_for(repo, binding, implementatio'
  b'n):'),
 (b'    source_guard = (SOURCE / guard.GUARD_PATH).read_bytes().replace(b"\\r\\n", b"\\n")',
  b'    source_guard = original_coverage_source(guard.GUARD_PATH)'),
 (b'        write(repo, path, (SOURCE / path).read_bytes().replace(b"\\r\\n", b"\\n"))',
  b'        write(repo, path, original_coverage_source(path))')]

DFS_PRIOR_COVERAGE_FIXTURE_SHA256 = '044051ee882b64403fdabe84549ab1b322383965b1eb44dd531789e2bcfc8fbd'


def _dfs_guard_matches(source: bytes, reviewed: str) -> bool:
    digest = reviewed.encode("ascii")
    return (source.count(digest) == 1 and
            hashlib.sha256(source.replace(digest, b"0" * 64)).hexdigest() == reviewed)


def _dfs_previous_guard_source(source: bytes) -> bytes:
    """Reconstruct the entire reviewed main guard after verifying the successor."""
    _require(_dfs_guard_matches(source, DFS_BINDINGS["successor_guard_sha256"]),
             "SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    return source.split(b"\nV4_POLICY_PATH =", 1)[0] + DFS_PREVIOUS_CLI


def _dfs_restore_coverage_fixture(source: bytes) -> bytes:
    for before, after in reversed(DFS_COVERAGE_FIXTURE_EDITS):
        _require(source.count(after) == 1, "PRIOR_FIXTURE_ANCHOR_CHANGED")
        source = source.replace(after, before, 1)
    return source

DRIVE_POLICY_PATH = "docs/paid-launch/launch-scope-policy-drive-history-v1.json"
DRIVE_POLICY_VERSION = "paid-launch-drive-history-v1"
DRIVE_APPROVAL_REFERENCE = "Owner-authorized bounded Drive history/evidence draft; no merge, deployment, live operations, science, qualification, authority or financial changes"
DRIVE_PATHS = (
    "app_core/evidence_drive.py", "app_core/evidence_remote.py", "app_core/public_history.py",
    "app_core/scoped_reads.py", "app_core/market_stage_metrics.py",
    "app/ui/public_results.py", "app/ui/publish_panel.py", "streamlit_app.py",
    "tests/test_evidence_drive.py", "tests/test_dfs_scope_policy.py",
    "tests/test_ncaaf_schedule_scope_policy.py",
    "tests/test_drive_history_loading.py", "tests/test_drive_history_scope_policy.py",
    "scripts/benchmark_drive_history_loading.py", "scripts/drive_history_scope.py",
    "scripts/check_launch_change_scope.py", "docs/paid-launch/drive-history-loading.md",
    "tests/test_ncaaf_coverage_scope_policy.py",
)
DRIVE_BINDINGS = {'base': 'befe6a09cfcec3e6dbe01f53b92602828c3a0db7',
 'base_tree': 'd2aa023d4e5bc1770fbbc6c8f284c21ec4e87e07',
 'manifest_sha256': '2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343',
 'previous_policy_blob': '240f499c737a6978f6bade6578bc40432a942a14',
 'previous_guard_sha256': '661a3d174f768c38f7ef32d2feb843c864ff5e7bc61d9e2b366b57754e22b090',
 'scope_module_sha256': 'ab9bebbfbc27fad6da56d905545b810237628d23551aba77e9d7d8b10def0022',
 'successor_guard_sha256': '006220e9cd9ff6f1798ba5a1e3b39ca533ad61548ccc31df14e18029faf99d4e',
 'reviewed_blobs': {'app_core/evidence_drive.py': '6e83e6616908148c94b8512be5436db225c39fd3',
                    'app_core/evidence_remote.py': '92fe7ac522d885a0bd61703f42a5f9979c8fb038',
                    'app_core/public_history.py': '1841415c5bdfd3cffdfeb90adcc8855c188d5f4c',
                    'app_core/scoped_reads.py': '7418aa87a0c1c23da00dd045ff6d3ddc2c7c3dec',
                    'app_core/market_stage_metrics.py': '1ee69c297e074d38ee2d04a61e4496e9ccf945f7',
                    'app/ui/public_results.py': 'eac5f694cd49310829d1a557b2f1d307502ee6a7',
                    'app/ui/publish_panel.py': '6fc6aef6d3e8e10197f78db347bc0cd0249fed1a',
                    'streamlit_app.py': '050f4193ec4232a91daa38238e8d3c07e2170959',
                    'tests/test_evidence_drive.py': '5c3138c3da434b6c8d1991ab80807dbb9d3011e3',
                    'tests/test_dfs_scope_policy.py': 'a0cdd858b0bb6a0b3d5d92f1f04997d657c1a43a',
                    'tests/test_ncaaf_schedule_scope_policy.py': '5f3fdb8e05dd51459d134606ec32e90c3729f62c',
                    'tests/test_drive_history_loading.py': 'c785f1f07d7fddd3c0b957b0e367b463f2bcdfb0',
                    'tests/test_drive_history_scope_policy.py': '1d0d19d1e3a052305b55349a881893c828b1db4a',
                    'scripts/benchmark_drive_history_loading.py': '78d950dd915a3dc88ead35c58fe6f6ebb9bf3461',
                    'scripts/drive_history_scope.py': '9f5e1864a55d9e103f22885e64d9d645812a7174',
                    'docs/paid-launch/drive-history-loading.md': '8bc16c5d4da49c2054846f477ad9475b3e6dd67a',
                    'tests/test_ncaaf_coverage_scope_policy.py': 'e4bac80c356e8ec2358a1b627b33b902d5740f69'}}
DRIVE_PREVIOUS_CLI = b'\ndef main() -> int:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)\n    parser.add_argument("--base")\n    parser.add_argument("--json-output", type=Path)\n    args = parser.parse_args()\n    try:\n        if exists_at("HEAD", V4_POLICY_PATH):\n            code, report = _run_dfs_integrated(args.manifest, args.base, DFS_BINDINGS)\n        elif exists_at("HEAD", COVERAGE_POLICY_PATH):\n            code, report = _run_coverage_integrated(args.manifest, args.base, COVERAGE_BINDINGS)\n        elif exists_at("HEAD", SCHEDULE_POLICY_PATH):\n            code, report = _run_schedule_integrated(args.manifest, args.base, SCHEDULE_BINDINGS)\n        elif exists_at("HEAD", V3_POLICY_PATH):\n            code, report = _run_provider_integrated(args.manifest, args.base, PROVIDER_BINDINGS)\n        else:\n            code, report = _run_integrated(args.manifest, args.base, PRODUCTION_BINDINGS)\n    except Exception as exc:\n        report = {"schema_version": 1, "status": "ERROR", "reason_codes": ["GUARD_EXECUTION_ERROR"], "error": str(exc)}\n        code = 2\n    rendered = json.dumps(report, indent=2, sort_keys=True)\n    if args.json_output:\n        args.json_output.parent.mkdir(parents=True, exist_ok=True)\n        args.json_output.write_text(rendered + "\\n", encoding="utf-8")\n    print(rendered)\n    return code\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
DRIVE_PRIOR_SOURCE_RECONSTRUCTIONS = {'streamlit_app.py': {'sha256': '7e438d76e8210c96a7b1fd9a549d9a4b68bc5548b6267392a86f9ea5e3c01f19',
                      'edits': [(b'    )\n\n    timer.start("Market enrichment and candidate sele'
                                 b'ction")\n    diagnostics["stage_seconds"] = timer.timings\n   '
                                 b' parlay_columns = ["slate_date", "pipeline_build", "export_run_i'
                                 b'd", "parlay_rank", "parlay_legs", "combined_probability", "combi'
                                 b'ned_decimal_odds", "parlay_ev", "legs", "unique_game_count", "on'
                                 b'e_leg_per_game", "card_unique_games", "card_game_exposure_cap", '
                                 b'"card_unique_game_count", "parlay_source", "risk_tier", "group_i'
                                 b'd", "best_payout_book", "Conviction_Score", "min_leg_prob", "has'
                                 b'_actionable_anchor", "production_safety_mode", "parlay_class", "'
                                 b'premium_eligible", "sellable_as_premium", "commercial_warning", '
                                 b'"kelly_fraction", "recommended_bet"]\n    empty_per_leg = {f"'
                                 b'parlays_{lc}_df": pd.DataFrame(columns=parlay_columns) for lc in'
                                 b' (2, 3)}\n',
                                 b'    )\n\n    timer.start("Market enrichment and candidate sele'
                                 b'ction")\n    from app_core.market_stage_metrics import measur'
                                 b'ed_call\n    def market_call(name, operation, *args, **kwargs'
                                 b'):\n        return measured_call(name, operation, *args, ids='
                                 b'timer.ids, **kwargs)\n    diagnostics["stage_seconds"] = time'
                                 b'r.timings\n    parlay_columns = ["slate_date", "pipeline_buil'
                                 b'd", "export_run_id", "parlay_rank", "parlay_legs", "combined_pro'
                                 b'bability", "combined_decimal_odds", "parlay_ev", "legs", "unique'
                                 b'_game_count", "one_leg_per_game", "card_unique_games", "card_gam'
                                 b'e_exposure_cap", "card_unique_game_count", "parlay_source", "ris'
                                 b'k_tier", "group_id", "best_payout_book", "Conviction_Score", "mi'
                                 b'n_leg_prob", "has_actionable_anchor", "production_safety_mode", '
                                 b'"parlay_class", "premium_eligible", "sellable_as_premium", "comm'
                                 b'ercial_warning", "kelly_fraction", "recommended_bet"]\n    em'
                                 b'pty_per_leg = {f"parlays_{lc}_df": pd.DataFrame(columns=parlay_c'
                                 b'olumns) for lc in (2, 3)}\n'),
                                (b'        if "game_date" not in analysis_df.columns or analysis_df'
                                 b'["game_date"].isna().all():\n            deferred_warnings.ap'
                                 b'pend("game_date missing from analysis_df \xe2\x80\x94 Kalshi '
                                 b'matching skipped.")\n        else:\n            analysis_df, k'
                                 b'alshi_err = _enrich_with_kalshi_safe(analysis_df)\n          '
                                 b'  if kalshi_err:\n                deferred_warnings.append(ka'
                                 b'lshi_err)\n\n    if controls.get("use_ml"):\n        try:\n     '
                                 b'       analysis_df = _sync_ml_probabilities(analysis_df, pipelin'
                                 b'e_best_picks_df)\n        except ValueError as exc:\n         '
                                 b'   deferred_errors.append(f"ML Merge Failed: {exc}")\n       '
                                 b'     timer.finish()\n',
                                 b'        if "game_date" not in analysis_df.columns or analysis_df'
                                 b'["game_date"].isna().all():\n            deferred_warnings.ap'
                                 b'pend("game_date missing from analysis_df \xe2\x80\x94 Kalshi '
                                 b'matching skipped.")\n        else:\n            analysis_df, k'
                                 b'alshi_err = market_call("kalshi_enrichment", _enrich_with_kalshi'
                                 b'_safe, analysis_df)\n            if kalshi_err:\n             '
                                 b'   deferred_warnings.append(kalshi_err)\n\n    if controls.get'
                                 b'("use_ml"):\n        try:\n            analysis_df = market_ca'
                                 b'll("ml_probability_join", _sync_ml_probabilities, analysis_df, p'
                                 b'ipeline_best_picks_df)\n        except ValueError as exc:\n   '
                                 b'         deferred_errors.append(f"ML Merge Failed: {exc}")\n '
                                 b'           timer.finish()\n'),
                                (b'        ml_required = False\n\n    try:\n        analysis_df = '
                                 b'_recompute_consensus_from_kalshi(\n            analysis_df,\n '
                                 b'           require_ml=ml_required,\n        )\n',
                                 b'        ml_required = False\n\n    try:\n        analysis_df = '
                                 b'market_call("consensus", _recompute_consensus_from_kalshi,\n '
                                 b'           analysis_df,\n            require_ml=ml_required,\n'
                                 b'        )\n'),
                                (b'\n    # We pass the diagnostics dictionary so that selection '
                                 b'metrics and preview_df\n    # can be injected without relying'
                                 b' on pandas DataFrame.attrs serialization.\n    best_picks_df '
                                 b'= build_best_picks_df(analysis_df, diagnostics_out=diagnosti'
                                 b'cs)\n    best_picks_df = ensure_best_pick_export_columns(best'
                                 b'_picks_df, diagnostics_out=diagnostics)\n    diagnostics["ide'
                                 b'ntity_columns_ready_before_portfolio"] = bool(\n        all(c'
                                 b' in best_picks_df.columns for c in ["export_run_id", "pick_id", '
                                 b'"canonical_pick_key"])\n        and best_picks_df["canonical_'
                                 b'pick_key"].astype(str).str.strip().ne("").all()\n',
                                 b'\n    # We pass the diagnostics dictionary so that selection '
                                 b'metrics and preview_df\n    # can be injected without relying'
                                 b' on pandas DataFrame.attrs serialization.\n    best_picks_df '
                                 b'= market_call("candidate_selection", build_best_picks_df, analys'
                                 b'is_df, diagnostics_out=diagnostics)\n    best_picks_df = mark'
                                 b'et_call("export_columns", ensure_best_pick_export_columns, best_'
                                 b'picks_df, diagnostics_out=diagnostics)\n    diagnostics["iden'
                                 b'tity_columns_ready_before_portfolio"] = bool(\n        all(c '
                                 b'in best_picks_df.columns for c in ["export_run_id", "pick_id", "'
                                 b'canonical_pick_key"])\n        and best_picks_df["canonical_p'
                                 b'ick_key"].astype(str).str.strip().ne("").all()\n'),
                                (b'    # the probability-first display winner for each game.\n  '
                                 b'  trial_now = pd.Timestamp.now(tz="UTC").to_pydatetime()\n   '
                                 b' from app_core.controlled_trial_pipeline import prepare_review_c'
                                 b'andidates\n    candidate_pool, trial_candidates = prepare_rev'
                                 b'iew_candidates(\n        diagnostics, now=trial_now\n    )\n   '
                                 b' diagnostics["candidate_authority_df"] = candidate_pool\n',
                                 b'    # the probability-first display winner for each game.\n  '
                                 b'  trial_now = pd.Timestamp.now(tz="UTC").to_pydatetime()\n   '
                                 b' from app_core.controlled_trial_pipeline import prepare_review_c'
                                 b'andidates\n    candidate_pool, trial_candidates = market_call'
                                 b'("review_candidate_preparation", prepare_review_candidates,\n'
                                 b'        diagnostics, now=trial_now\n    )\n    diagnostics["ca'
                                 b'ndidate_authority_df"] = candidate_pool\n'),
                                (b'                game_seconds = sum(state_updates.get("diagnostic'
                                 b's", {}).get("stage_seconds", {}).values())\n                g'
                                 b'ame_status.update(label=f"Game analysis finished in {game_second'
                                 b's:.0f}s", state="complete")\n            st.session_state.upd'
                                 b'ate(state_updates)\n            st.session_state["history_ref'
                                 b'resh_requested"] = True\n            st.session_state["last_s'
                                 b'uccessful_pipeline_signature"] = (\n                _analysis'
                                 b'_input_signature(controls)\n            )\n',
                                 b'                game_seconds = sum(state_updates.get("diagnostic'
                                 b's", {}).get("stage_seconds", {}).values())\n                g'
                                 b'ame_status.update(label=f"Game analysis finished in {game_second'
                                 b's:.0f}s", state="complete")\n            st.session_state.upd'
                                 b'ate(state_updates)\n            st.session_state["last_succes'
                                 b'sful_pipeline_signature"] = (\n                _analysis_inpu'
                                 b't_signature(controls)\n            )\n'),
                                (b'                st.caption("Run Game Analysis to collect timings'
                                 b'.")\n        with st.expander("Prediction Evidence Status", e'
                                 b'xpanded=False):\n            from app_core.evidence_health im'
                                 b'port evidence_health\n            from app_core.evidence_remo'
                                 b'te import restore_once, restore, sync\n            try:\n     '
                                 b'           restore_once()\n            except RuntimeError as'
                                 b' exc:\n                st.error(str(exc))\n            if st.b'
                                 b'utton("Restore and sync evidence storage", key="sync_remote_evid'
                                 b'ence"):\n                try:\n                    restore()\n '
                                 b'                   sync()\n                except Exception a'
                                 b's exc:\n                    from app_core.evidence_config imp'
                                 b'ort safe_error\n',
                                 b'                st.caption("Run Game Analysis to collect timings'
                                 b'.")\n        with st.expander("Prediction Evidence Status", e'
                                 b'xpanded=False):\n            from app_core.evidence_health im'
                                 b'port evidence_health\n            from app_core.evidence_remo'
                                 b'te import restore, sync\n            if st.button("Restore an'
                                 b'd sync evidence storage", key="sync_remote_evidence"):\n     '
                                 b'           try:\n                    restore(full_verificatio'
                                 b'n=True)\n                    sync()\n                except Ex'
                                 b'ception as exc:\n                    from app_core.evidence_c'
                                 b'onfig import safe_error\n'),
                                (b'\n    with publish_tab:\n        from app.ui.publish_panel imp'
                                 b'ort render_publish_panel\n        render_publish_panel(public'
                                 b'ation_games, _publication_candidates(diagnostics), publication_p'
                                 b'rops, publication_dfs)\n\n    with tab4:\n        st.subheader('
                                 b'"Best Parlays")\n',
                                 b'\n    with publish_tab:\n        from app.ui.publish_panel imp'
                                 b'ort render_publish_panel\n        render_publish_panel(public'
                                 b'ation_games, _publication_candidates(diagnostics), publication_p'
                                 b'rops, publication_dfs, lazy_history=True)\n\n    with tab4:\n  '
                                 b'      st.subheader("Best Parlays")\n')]},
 'tests/test_evidence_drive.py': {'sha256': '5714de5251327a3f8d4cd482c36d57a424cf7f23cb8e580937a4ffe317f604ad',
                                  'edits': [(b"    assert store.read_cached_objects(Prefix='receipt"
                                             b"/', cache_dir=tmp_path)==[('receipt/key',raw)]\n "
                                             b"   assert store.read_cached_objects(Prefix='receipt/"
                                             b"', cache_dir=tmp_path)==[('receipt/key',raw)]\n  "
                                             b"  assert len(listings)==2 and reads==['one']\n   "
                                             b" (tmp_path/sha).write_bytes(b'corrupt')\n    stor"
                                             b"e.read_cached_objects(Prefix='receipt/', cache_dir=t"
                                             b"mp_path)\n    assert reads==['one','one']\n    fil"
                                             b'es.clear()\n',
                                             b"    assert store.read_cached_objects(Prefix='receipt"
                                             b"/', cache_dir=tmp_path)==[('receipt/key',raw)]\n "
                                             b"   assert store.read_cached_objects(Prefix='receipt/"
                                             b"', cache_dir=tmp_path)==[('receipt/key',raw)]\n  "
                                             b"  assert len(listings)==2 and reads==['one']\n   "
                                             b" (store.verified_cache_root(tmp_path, 'receipt/')/sh"
                                             b"a).write_bytes(b'corrupt')\n    store.read_cached"
                                             b"_objects(Prefix='receipt/', cache_dir=tmp_path)\n"
                                             b"    assert reads==['one','one']\n    files.clear("
                                             b')\n')]},
 'tests/test_dfs_scope_policy.py': {'sha256': '2da033d78d07592a796d3669c90854c84257ec4497abc3e13f95cd94717bc360',
                                    'edits': [(b'    git(repo, "config", "core.autocrlf", "false"'
                                               b')\n    git(repo, "config", "user.name", "Offline '
                                               b'Test")\n    git(repo, "config", "user.email", "of'
                                               b'fline@example.invalid")\n    source_guard = (SOUR'
                                               b'CE / guard.GUARD_PATH).read_bytes().replace(b"\\r'
                                               b'\\n", b"\\n")\n    previous_guard = source_guar'
                                               b'd.split(b"\\nV4_POLICY_PATH =", 1)[0] + guard.DFS'
                                               b'_PREVIOUS_CLI\n    original_guard = source_guard.'
                                               b'split(b"\\nPOLICY_PATH =", 1)[0] + b"\\n"\n    '
                                               b'write(repo, "README.md", b"offline DFS fixture\\n'
                                               b'")\n',
                                               b'    git(repo, "config", "core.autocrlf", "false"'
                                               b')\n    git(repo, "config", "user.name", "Offline '
                                               b'Test")\n    git(repo, "config", "user.email", "of'
                                               b'fline@example.invalid")\n    source_guard = guard'
                                               b'._drive_previous_guard_source((SOURCE / guard.GU'
                                               b'ARD_PATH).read_bytes().replace(b"\\r\\n", b"\\n'
                                               b'"))\n    previous_guard = source_guard.split(b"\\n'
                                               b'V4_POLICY_PATH =", 1)[0] + guard.DFS_PREVIOUS_CL'
                                               b'I\n    original_guard = source_guard.split(b"\\nPO'
                                               b'LICY_PATH =", 1)[0] + b"\\n"\n    write(repo, "REA'
                                               b'DME.md", b"offline DFS fixture\\n")\n'),
                                              (b'               "clock_blob": guard.blob(base, gu'
                                               b'ard.CLOCK_TEST),\n               "successor_guard'
                                               b'_sha256": guard.DFS_BINDINGS["successor_guard_sh'
                                               b'a256"]}\n    for path in guard.DFS_PATHS:\n       '
                                               b' write(repo, path, (SOURCE / path).read_bytes().'
                                               b'replace(b"\\r\\n", b"\\n"))\n    binding["review'
                                               b'ed_dfs_blobs"] = {p: git(repo, "hash-object", "-'
                                               b'-", p)\n                                     for '
                                               b'p in guard.DFS_BINDINGS["reviewed_dfs_blobs"]}\n '
                                               b'   implementation, candidate, policy = seal(repo'
                                               b', binding)\n',
                                               b'               "clock_blob": guard.blob(base, gu'
                                               b'ard.CLOCK_TEST),\n               "successor_guard'
                                               b'_sha256": guard.DFS_BINDINGS["successor_guard_sh'
                                               b'a256"]}\n    for path in guard.DFS_PATHS:\n       '
                                               b' write(repo, path, guard._drive_previous_main_so'
                                               b'urce(path, (SOURCE / path).read_bytes().replace('
                                               b'b"\\r\\n", b"\\n")))\n    binding["reviewed_dfs_'
                                               b'blobs"] = {p: git(repo, "hash-object", "--", p)\n'
                                               b'                                     for p in gu'
                                               b'ard.DFS_BINDINGS["reviewed_dfs_blobs"]}\n    impl'
                                               b'ementation, candidate, policy = seal(repo, bindi'
                                               b'ng)\n'),
                                              (b'\n\ndef test_prior_guard_and_fixture_reconstruct_e'
                                               b'xact_reviewed_bytes():\n    source = (SOURCE / gu'
                                               b'ard.GUARD_PATH).read_bytes().replace(b"\\r\\n", b"'
                                               b'\\n")\n    previous = guard._dfs_previous_guard_so'
                                               b'urce(source)\n    assert hashlib.sha256(previous)'
                                               b'.hexdigest() == guard.DFS_BINDINGS["previous_gua'
                                               b'rd_sha256"]\n    fixture = (SOURCE / "tests/test_'
                                               b'ncaaf_coverage_scope_policy.py").read_bytes().re'
                                               b'place(b"\\r\\n", b"\\n")\n',
                                               b'\n\ndef test_prior_guard_and_fixture_reconstruct_e'
                                               b'xact_reviewed_bytes():\n    source = guard._drive'
                                               b'_previous_guard_source((SOURCE / guard.GUARD_PAT'
                                               b'H).read_bytes().replace(b"\\r\\n", b"\\n"))\n   '
                                               b' previous = guard._dfs_previous_guard_source(sou'
                                               b'rce)\n    assert hashlib.sha256(previous).hexdige'
                                               b'st() == guard.DFS_BINDINGS["previous_guard_sha25'
                                               b'6"]\n    fixture = (SOURCE / "tests/test_ncaaf_co'
                                               b'verage_scope_policy.py").read_bytes().replace(b"'
                                               b'\\r\\n", b"\\n")\n')]},
 'tests/test_ncaaf_schedule_scope_policy.py': {'sha256': '81d20cf7facf63c6f469dc0af52f7f5d740260a86be86128d41d206119ac1407',
                                               'edits': [(b'    previous=(SOURCE/g.GUARD_PATH).r'
                                                          b'ead_bytes().replace(b"\\r\\n",b"\\n").s'
                                                          b'plit(b"\\nSCHEDULE_POLICY_PATH =",1)['
                                                          b'0]+b"\\ndef main() -> int:\\n    pass\\'
                                                          b'n"\n    write(repo,g.GUARD_PATH,previ'
                                                          b'ous)\n    for p,edits in g.SCHEDULE_S'
                                                          b'HARED_EDITS.items():\n        value=('
                                                          b'SOURCE/p).read_bytes().replace(b"\\r\\'
                                                          b'n",b"\\n").decode()\n        for befor'
                                                          b'e,after in reversed(edits):\n        '
                                                          b'    assert value.count(after)==1\n   '
                                                          b'         value=value.replace(after,b'
                                                          b'efore,1)\n',
                                                          b'    previous=(SOURCE/g.GUARD_PATH).r'
                                                          b'ead_bytes().replace(b"\\r\\n",b"\\n").s'
                                                          b'plit(b"\\nSCHEDULE_POLICY_PATH =",1)['
                                                          b'0]+b"\\ndef main() -> int:\\n    pass\\'
                                                          b'n"\n    write(repo,g.GUARD_PATH,previ'
                                                          b'ous)\n    for p,edits in g.SCHEDULE_S'
                                                          b'HARED_EDITS.items():\n        value=g'
                                                          b'._drive_previous_main_source(p,(SOUR'
                                                          b'CE/p).read_bytes().replace(b"\\r\\n",b'
                                                          b'"\\n")).decode()\n        for before,a'
                                                          b'fter in reversed(edits):\n           '
                                                          b' assert value.count(after)==1\n      '
                                                          b'      value=value.replace(after,befo'
                                                          b're,1)\n'),
                                                         (b'\ndef original_schedule_source(path):'
                                                          b'\n    """Exercise the original v1 con'
                                                          b'tract, never rebind it to the correc'
                                                          b'tion."""\n    source = (SOURCE / path'
                                                          b').read_bytes().replace(b"\\r\\n", b"\\n'
                                                          b'")\n    if path == g.GUARD_PATH:\n    '
                                                          b'    return source.split(b"\\nCOVERAGE'
                                                          b'_POLICY_PATH =", 1)[0] + g.COVERAGE_'
                                                          b'PREVIOUS_CLI\n    if path == "app_cor'
                                                          b'e/ncaaf_schedule.py":\n',
                                                          b'\ndef original_schedule_source(path):'
                                                          b'\n    """Exercise the original v1 con'
                                                          b'tract, never rebind it to the correc'
                                                          b'tion."""\n    source = g._drive_previ'
                                                          b'ous_main_source(path, (SOURCE / path'
                                                          b').read_bytes().replace(b"\\r\\n", b"\\n'
                                                          b'"))\n    if path == g.GUARD_PATH:\n   '
                                                          b'     return source.split(b"\\nCOVERAG'
                                                          b'E_POLICY_PATH =", 1)[0] + g.COVERAGE'
                                                          b'_PREVIOUS_CLI\n    if path == "app_co'
                                                          b're/ncaaf_schedule.py":\n')]},
 'tests/test_ncaaf_coverage_scope_policy.py': {'sha256': 'ff5f4b9bb934536724d6e6273c75cd2ef769e82f0f36fc07cc9ea29c3eaf344b',
                                               'edits': [(b'\ndef original_coverage_source(path):'
                                                          b'\n    """Reconstruct the reviewed pri'
                                                          b'or fixtures without changing asserti'
                                                          b'ons."""\n    source = (SOURCE / path)'
                                                          b'.read_bytes().replace(b"\\r\\n", b"\\n"'
                                                          b')\n    if path == guard.GUARD_PATH:\n '
                                                          b'       return guard._dfs_previous_gu'
                                                          b'ard_source(source)\n    if path == "t'
                                                          b'ests/test_ncaaf_coverage_scope_polic'
                                                          b'y.py":\n',
                                                          b'\ndef original_coverage_source(path):'
                                                          b'\n    """Reconstruct the reviewed pri'
                                                          b'or fixtures without changing asserti'
                                                          b'ons."""\n    source = guard._drive_pr'
                                                          b'evious_main_source(path, (SOURCE / p'
                                                          b'ath).read_bytes().replace(b"\\r\\n", b'
                                                          b'"\\n"))\n    if path == guard.GUARD_PA'
                                                          b'TH:\n        return guard._dfs_previo'
                                                          b'us_guard_source(source)\n    if path '
                                                          b'== "tests/test_ncaaf_coverage_scope_'
                                                          b'policy.py":\n')]}}


def _drive_previous_main_source(path, source):
    """Reconstruct reviewed predecessor fixture inputs, never their assertions."""
    if path == GUARD_PATH:
        return _drive_previous_guard_source(source)
    frozen = DRIVE_PRIOR_SOURCE_RECONSTRUCTIONS.get(path)
    if frozen is None or hashlib.sha256(source).hexdigest() == frozen["sha256"]:
        return source
    for before, after in reversed(frozen["edits"]):
        _require(source.count(after) == 1, "PRIOR_FIXTURE_ANCHOR_CHANGED")
        source = source.replace(after, before, 1)
    _require(hashlib.sha256(source).hexdigest() == frozen["sha256"], "PRIOR_ASSERTIONS_CHANGED")
    return source



def _drive_previous_guard_source(source, binding=None):
    binding = DRIVE_BINDINGS if binding is None else binding
    if b"\nDRIVE_POLICY_PATH =" not in source:
        return source
    _require(_dfs_guard_matches(source, binding["successor_guard_sha256"]),
             "SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    return source.split(b"\nDRIVE_POLICY_PATH =", 1)[0] + DRIVE_PREVIOUS_CLI


_drive_prior_dfs_source = _dfs_previous_guard_source


def _dfs_previous_guard_source(source):
    return _drive_prior_dfs_source(_drive_previous_guard_source(source))


_drive_prior_coverage_fixture = _dfs_restore_coverage_fixture


def _dfs_restore_coverage_fixture(source):
    return _drive_prior_coverage_fixture(_drive_previous_main_source(
        "tests/test_ncaaf_coverage_scope_policy.py", source))


def _run_drive_integrated(manifest_path, base, binding):
    import importlib.util
    path = ROOT / "scripts/drive_history_scope.py"
    _require(hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() ==
             binding["scope_module_sha256"], "SUCCESSOR_SCOPE_MODULE_CHANGED")
    spec = importlib.util.spec_from_file_location("parlaypicker_drive_scope_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(sys.modules[__name__], manifest_path, base, binding)

ESTIMATE_POLICY_PATH = "docs/paid-launch/launch-scope-policy-estimate-v1.json"
ESTIMATE_POLICY_VERSION = "paid-launch-estimate-v1"
ESTIMATE_APPROVAL_REFERENCE = "Owner-authorized bounded estimate availability draft and qualification route inspection; no merge, deployment, live acquisition, recovery, fitting, activation, authority or financial changes"
ESTIMATE_PATHS = ('app_core/market_probability_model.py', 'app_core/research_display.py', 'app_core/per_game_boards.py', 'app_core/research_estimate_trace.py', 'core/streamlit_pipeline.py', 'tests/test_estimate_availability.py', 'tests/test_estimate_scope_policy.py', 'tests/test_drive_history_scope_policy.py', 'scripts/estimate_scope.py', 'scripts/check_launch_change_scope.py', 'docs/paid-launch/estimate-availability.md')
ESTIMATE_BINDINGS = {'base': '2e96b5e336e72f8cfdf5faf3e6b0d6ffe1f8ec8d',
 'base_tree': '21f5e5b57a12683e4d770e19a85bf95ff6777f15',
 'manifest_sha256': '2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343',
 'previous_policy_blob': '9082b039a70609620f94102fb8b278eac28cb5f2',
 'previous_guard_sha256': '2116a73036a6243da92120a29c5a4dcf242d21fa735e31beeeceedd0174ef386',
 'scope_module_sha256': 'd93ec615dd45a81739719fda47619a14f9eb8563d6054cd59320ca9bb9a251de',
 'successor_guard_sha256': '53aad6863d8574042a6f4a75652b51be0ec4dc7c4375aff5d410e55920390701',
 'reviewed_blobs': {'app_core/market_probability_model.py': 'a114edc13ca9e7ba3880c365c3559110fc99f2f8',
                    'app_core/research_display.py': 'd45e9085aabd921e9df452f1136dbfcf9baf8b43',
                    'app_core/per_game_boards.py': 'd831301d0d12ef00f0c082232ad2442c35ae6851',
                    'app_core/research_estimate_trace.py': 'f3d02422e8c3c9a8a220a4133f38f4108ad75dd9',
                    'core/streamlit_pipeline.py': 'a2634611a7919788a365e8305cb8c98c48fca48f',
                    'tests/test_estimate_availability.py': '0945d3713ac82ecfbd8160845be8970952b69a28',
                    'tests/test_estimate_scope_policy.py': '4dedb0277e1db860993553c5863e11ed652a2c98',
                    'tests/test_drive_history_scope_policy.py': '8894eff9a96de25a897639211e92a3f43d1d863b',
                    'scripts/estimate_scope.py': '3736461db23d240f12e540cf749d7fef90c47235',
                    'docs/paid-launch/estimate-availability.md': '9b94a1fcf536ddadd93453708d6b9d63aeab4b17'}}
ESTIMATE_PREVIOUS_CLI = b'\ndef main() -> int:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)\n    parser.add_argument("--base")\n    parser.add_argument("--json-output", type=Path)\n    args = parser.parse_args()\n    try:\n        if exists_at("HEAD", DRIVE_POLICY_PATH):\n            code, report = _run_drive_integrated(args.manifest, args.base, DRIVE_BINDINGS)\n        elif exists_at("HEAD", V4_POLICY_PATH):\n            code, report = _run_dfs_integrated(args.manifest, args.base, DFS_BINDINGS)\n        elif exists_at("HEAD", COVERAGE_POLICY_PATH):\n            code, report = _run_coverage_integrated(args.manifest, args.base, COVERAGE_BINDINGS)\n        elif exists_at("HEAD", SCHEDULE_POLICY_PATH):\n            code, report = _run_schedule_integrated(args.manifest, args.base, SCHEDULE_BINDINGS)\n        elif exists_at("HEAD", V3_POLICY_PATH):\n            code, report = _run_provider_integrated(args.manifest, args.base, PROVIDER_BINDINGS)\n        else:\n            code, report = _run_integrated(args.manifest, args.base, PRODUCTION_BINDINGS)\n    except Exception as exc:\n        report = {"schema_version": 1, "status": "ERROR", "reason_codes": ["GUARD_EXECUTION_ERROR"], "error": str(exc)}\n        code = 2\n    rendered = json.dumps(report, indent=2, sort_keys=True)\n    if args.json_output:\n        args.json_output.parent.mkdir(parents=True, exist_ok=True)\n        args.json_output.write_text(rendered + "\\n", encoding="utf-8")\n    print(rendered)\n    return code\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
ESTIMATE_PRIOR_SOURCE_RECONSTRUCTIONS = {'app_core/market_probability_model.py': {'sha256': 'f719ceb95356acb71082381b14f36b8f5dc80d48f9075d5609f694f2473b6780',
                                          'edits': [(b'    result["ml_residual_scale"] = pd.Series(np.nan, '
                                                     b'index=result.index, dtype="float64")\n    result["ml_'
                                                     b'unavailable_reason"] = pd.Series("", index=result.in'
                                                     b'dex, dtype="string")\n    result["ml_feature_quality"'
                                                     b'] = pd.Series("unavailable", index=result.index, dty'
                                                     b'pe="string")\n\n    if frame is None or frame.empt'
                                                     b'y:\n        return result\n',
                                                     b'    result["ml_residual_scale"] = pd.Series(np.nan, '
                                                     b'index=result.index, dtype="float64")\n    result["ml_'
                                                     b'unavailable_reason"] = pd.Series("", index=result.in'
                                                     b'dex, dtype="string")\n    result["ml_feature_quality"'
                                                     b'] = pd.Series("unavailable", index=result.index, dty'
                                                     b'pe="string")\n    result["ml_inference_status"] = pd.'
                                                     b'Series("unavailable", index=result.index, dtype="str'
                                                     b'ing")\n    result["ml_estimate_metadata"] = pd.Series'
                                                     b'("", index=result.index, dtype="string")\n\n    if fra'
                                                     b'me is None or frame.empty:\n        return result\n'),
                                                    (b'\n        probability = 0.5 + params["reliability"] *'
                                                     b' (raw_probability - 0.5)\n        probability = float'
                                                     b'(np.clip(probability, 0.20, 0.80))\n        result.at'
                                                     b'[idx, "ml_probability"] = probability\n        result'
                                                     b'.at[idx, "ml_probability_source"] = f"{MODEL_VERSION'
                                                     b'}:{lg.lower()}"\n        result.at[idx, "ml_target"] '
                                                     b'= target\n',
                                                     b'\n        probability = 0.5 + params["reliability"] *'
                                                     b' (raw_probability - 0.5)\n        probability = float'
                                                     b'(np.clip(probability, 0.20, 0.80))\n        # Explici'
                                                     b't outcome at the originating computation, not numeri'
                                                     b'c inference.\n        result.at[idx, "ml_inference_st'
                                                     b'atus"] = "success"\n        result.at[idx, "ml_probab'
                                                     b'ility"] = probability\n        result.at[idx, "ml_pro'
                                                     b'bability_source"] = f"{MODEL_VERSION}:{lg.lower()}"\n'
                                                     b'        result.at[idx, "ml_target"] = target\n'),
                                                    (b'            else "resolved_team_scoring_stats"\n     '
                                                     b'   )\n\n    return result\n',
                                                     b'            else "resolved_team_scoring_stats"\n     '
                                                     b'   )\n\n    from app_core.research_estimate_trace impo'
                                                     b'rt origin_metadata, generated_time\n    generated = g'
                                                     b'enerated_time()\n    for idx in frame.index:\n        '
                                                     b'line = total_line.loc[idx] if str(market_type.loc[id'
                                                     b'x]).startswith("total") else spread_line.loc[idx]\n  '
                                                     b'      result.at[idx, "ml_estimate_metadata"] = origi'
                                                     b'n_metadata(\n            frame.loc[idx], result.loc[i'
                                                     b'dx], float(line) if np.isfinite(line) else None,\n   '
                                                     b'         generated_at=generated)\n\n    return res'
                                                     b'ult\n')]},
 'app_core/research_display.py': {'sha256': 'e26121e738f7fb06bf5e095a039de2d385f52bc47c3956fa934043d06bceddf2',
                                  'edits': [(b'# Explicit public-research provenance only; never an arbitra'
                                             b'ry source-column copy.\nEXPORT_PROVENANCE_COLUMNS = ["quote_i'
                                             b'd", "prospective_quote_id", "market_period", "period",\n    "'
                                             b'settlement_rules", "inference_status", "model_status", "spre'
                                             b'ad_line", "total_line",\n    "market_line_used", "push_probab'
                                             b'ility", "probability_semantics", "research_source_semantics"'
                                             b']\nSEMANTIC_FIELDS = ("probability_semantics", "push_probabil'
                                             b'ity", "inference_status", "model_status")\nSEMANTICS = frozen'
                                             b'set({"win_conditional_on_decision","win_unconditional_with_p'
                                             b'ush",\n                      "unconditional_win_push_loss","u'
                                             b'nconditional"})\n',
                                             b'# Explicit public-research provenance only; never an arbitra'
                                             b'ry source-column copy.\nEXPORT_PROVENANCE_COLUMNS = ["quote_i'
                                             b'd", "prospective_quote_id", "market_period", "period",\n    "'
                                             b'settlement_rules", "inference_status", "model_status", "spre'
                                             b'ad_line", "total_line",\n    "market_line_used", "push_probab'
                                             b'ility", "probability_semantics", "research_source_semantics"'
                                             b',\n    "ml_inference_status", "ml_estimate_metadata"]\nSEMANTI'
                                             b'C_FIELDS = ("probability_semantics", "push_probability", "in'
                                             b'ference_status", "model_status")\nSEMANTICS = frozenset({"win'
                                             b'_conditional_on_decision","win_unconditional_with_push",\n   '
                                             b'                   "unconditional_win_push_loss","unconditio'
                                             b'nal"})\n'),
                                            (b'    # Nonzero-push reversal cannot reinterpret an unconditio'
                                             b'nal source.\n    return not (_text(original) in unconditional'
                                             b' and current_name=="win_conditional_on_decision"\n           '
                                             b'     and not (_valid_push(push) and math.isclose(_number(pus'
                                             b'h),0.0,rel_tol=0,abs_tol=1e-9)))\n\n\ndef _identity(row):\n',
                                             b'    # Nonzero-push reversal cannot reinterpret an unconditio'
                                             b'nal source.\n    return not (_text(original) in unconditional'
                                             b' and current_name=="win_conditional_on_decision"\n           '
                                             b'     and not (_valid_push(push) and math.isclose(_number(pus'
                                             b'h),0.0,rel_tol=0,abs_tol=1e-9)))\n\n\ndef captured_legacy_h'
                                             b'alf_point(source, line):\n    """Display-only compatibility f'
                                             b"or a captured, originally undeclared row.\n\n    Capture's no-"
                                             b'push canonical label cannot manufacture a missing integer pu'
                                             b'sh\n    model. Original declarations and every current contra'
                                             b'diction still reject.\n    This helper supplies no inference,'
                                             b' qualification or wager authority.\n    """\n    if (line is N'
                                             b'one or abs(line*2-round(line*2))>1e-9\n            or abs(lin'
                                             b'e-round(line))<=1e-9\n            or _absent(source.get("rese'
                                             b'arch_source_semantics"))):\n        return False\n    original'
                                             b'=_source_semantics(source)\n    return bool(original is not N'
                                             b'one\n        and _absent(original.get("probability_semantics"'
                                             b'))\n        and _absent(original.get("push_probability"))\n   '
                                             b'     and _text(source.get("probability_semantics"))=="win_co'
                                             b'nditional_on_decision"\n        and _absent(source.get("push_'
                                             b'probability")))\n\n\ndef _identity(row):\n')]},
 'app_core/per_game_boards.py': {'sha256': 'ca6c6698f6be7a089140acd9e7a7b3277437fc125229a369106f261544dc7ff2',
                                 'edits': [(b'                        # coerce it to the legacy no-push co'
                                            b'mpatibility route.\n                        mass=None\n       '
                                            b"             elif semantics == 'win_conditional_on_decision'"
                                            b':\n                        mass=unconditional_from_conditiona'
                                            b"l(probability,push)\n                    elif semantics in {'"
                                            b"win_unconditional_with_push','unconditional_win_push_loss','"
                                            b"unconditional'} and push is not None:\n                      "
                                            b"  mass=({'p_win':probability,'p_push':push}\n                "
                                            b'              if 0<=push<1 and probability+push<=1 else None'
                                            b')\n',
                                            b'                        # coerce it to the legacy no-push co'
                                            b'mpatibility route.\n                        mass=None\n       '
                                            b"             elif semantics == 'win_conditional_on_decision'"
                                            b':\n                        from app_core.research_display imp'
                                            b'ort captured_legacy_half_point\n                        if ca'
                                            b'ptured_legacy_half_point(source,line):\n                     '
                                            b"       mass={'p_win':probability,'p_push':0.0}\n             "
                                            b'               approved=False\n                        else:\n'
                                            b'                            mass=unconditional_from_conditio'
                                            b'nal(probability,push)\n                    elif semantics in '
                                            b"{'win_unconditional_with_push','unconditional_win_push_loss'"
                                            b",'unconditional'} and push is not None:\n                    "
                                            b"    mass=({'p_win':probability,'p_push':push}\n              "
                                            b'                if 0<=push<1 and probability+push<=1 else No'
                                            b'ne)\n'),
                                           (b"                     'ml_target':text(source,'ml_target') if"
                                            b" source is not None else '',\n                     'market_pe"
                                            b"riod':text(source,'market_period','period') if source is not"
                                            b" None else '',\n                     'settlement_rules':text("
                                            b"source,'settlement_rules') if source is not None else ''}\n  "
                                            b'      from app_core.research_display import from_export\n    '
                                            b"    import json\n        exported['research_display']=json.du"
                                            b'mps(from_export(exported, source=source, source_field=probab'
                                            b"ility_field), allow_nan=False, sort_keys=True, separators=('"
                                            b",',':'))\n        rows.append(exported)\n    return pd.DataFra"
                                            b'me(rows)\n',
                                            b"                     'ml_target':text(source,'ml_target') if"
                                            b" source is not None else '',\n                     'market_pe"
                                            b"riod':text(source,'market_period','period') if source is not"
                                            b" None else '',\n                     'settlement_rules':text("
                                            b"source,'settlement_rules') if source is not None else ''}\n  "
                                            b'      # Preserve supplied producer clocks, including seconds'
                                            b' and UTC offsets.\n        # A display label or capture run I'
                                            b'D cannot replace the original facts.\n        if source is no'
                                            b't None:\n            from datetime import datetime\n          '
                                            b'  from app_core.candidate_evidence_schema import missing\n   '
                                            b"         for field in ('prediction_generated_at','game_start"
                                            b"_utc'):\n                if field in source and not missing(s"
                                            b'ource[field]):\n                    value=source[field]\n     '
                                            b'               # One exported start clock: downstream start '
                                            b'updates and\n                    # existing date/lock checks '
                                            b"must not be hidden by an alias.\n                    target='"
                                            b"start' if field=='game_start_utc' else field\n               "
                                            b'     exported[target]=value.isoformat() if isinstance(value,'
                                            b'datetime) else value\n        from app_core.research_display '
                                            b'import from_export\n        from app_core.research_estimate_t'
                                            b'race import boundary_trace\n        import json\n        displ'
                                            b'ay=from_export(exported, source=source, source_field=probabi'
                                            b"lity_field)\n        exported['research_display']=json.dumps("
                                            b"display, allow_nan=False, sort_keys=True, separators=(',',':"
                                            b"'))\n        # Owner export only: public_board's allowlist ne"
                                            b"ver publishes this trace.\n        exported['research_estimat"
                                            b"e_trace']=boundary_trace(source,exported,display)\n        ro"
                                            b'ws.append(exported)\n    return pd.DataFrame(rows)\n')]},
 'core/streamlit_pipeline.py': {'sha256': '92eafdffcdcb3dcb2efe801688f0130d6ac0fd35e51eeacde72c1c8277eb0be1',
                                'edits': [(b'    "settlement_rules", "inference_status", "prediction_generate'
                                           b'd_at", "game_start_utc",\n    "odds_recorded_at", "quote_time'
                                           b'", "quote_timestamp", "quote_bookmaker", "provider_quotes",\n'
                                           b'    "candidate_id", "export_run_id", "research_source_semantics"'
                                           b',\n    "probability_semantics", "push_probability", "model_st'
                                           b'atus"]\nCANONICAL_BET_COLUMNS = list(dict.fromkeys(CANONICAL_'
                                           b'BET_COLUMNS + _CANONICAL_RESEARCH_COLUMNS))\n\n\n',
                                           b'    "settlement_rules", "inference_status", "prediction_generate'
                                           b'd_at", "game_start_utc",\n    "odds_recorded_at", "quote_time'
                                           b'", "quote_timestamp", "quote_bookmaker", "provider_quotes",\n'
                                           b'    "candidate_id", "export_run_id", "research_source_semantics"'
                                           b',\n    "probability_semantics", "push_probability", "model_st'
                                           b'atus", "ml_inference_status", "ml_estimate_metadata"]\nCANONI'
                                           b'CAL_BET_COLUMNS = list(dict.fromkeys(CANONICAL_BET_COLUMNS + _CA'
                                           b'NONICAL_RESEARCH_COLUMNS))\n\n\n'),
                                          (b'                ]\n        if "ml_unavailable_reason" in mark'
                                           b'et_model_predictions:\n            merged.loc[market_model_pr'
                                           b'edictions.index, "ml_unavailable_reason"] = market_model_predict'
                                           b'ions["ml_unavailable_reason"]\n        merged.loc[market_avai'
                                           b'lable, "model_status"] = "Market Score Model"\n        logger'
                                           b'.info(\n            "MARKET MODEL: generated %s target-specif'
                                           b'ic spread/total probabilities.",\n',
                                           b'                ]\n        if "ml_unavailable_reason" in mark'
                                           b'et_model_predictions:\n            merged.loc[market_model_pr'
                                           b'edictions.index, "ml_unavailable_reason"] = market_model_predict'
                                           b'ions["ml_unavailable_reason"]\n        from app_core.research'
                                           b'_estimate_trace import ORIGIN_COLUMNS\n        for column in '
                                           b'ORIGIN_COLUMNS:\n            if column in market_model_predic'
                                           b'tions:\n                merged.loc[market_model_predictions.i'
                                           b'ndex,column] = market_model_predictions[column]\n        merg'
                                           b'ed.loc[market_available, "model_status"] = "Market Score Mod'
                                           b'el"\n        logger.info(\n            "MARKET MODEL: generate'
                                           b'd %s target-specific spread/total probabilities.",\n')]},
 'tests/test_drive_history_scope_policy.py': {'sha256': '79c1498ddcf23d3c6e417cea0fc93ebc1003affe8ccf5ae017f2d43e0cc90689',
                                              'edits': [(b'        git(repo,"config","user.name","Offline T'
                                                         b'est")\n        git(repo,"config","user.email","of'
                                                         b'fline@example.invalid")\n        git(repo,"config'
                                                         b'","core.autocrlf","false")\n        current_guard'
                                                         b'=(SOURCE/guard.GUARD_PATH).read_bytes().replace('
                                                         b'b"\\r\\n",b"\\n")\n        previous=guard._drive'
                                                         b'_previous_guard_source(current_guard)\n        or'
                                                         b'iginal=current_guard.split(b"\\nPOLICY_PATH =",1)'
                                                         b'[0]+b"\\n"\n        write(repo,"README.md",b"offli'
                                                         b'ne fixture\\n")\n',
                                                         b'        git(repo,"config","user.name","Offline T'
                                                         b'est")\n        git(repo,"config","user.email","of'
                                                         b'fline@example.invalid")\n        git(repo,"config'
                                                         b'","core.autocrlf","false")\n        current_guard'
                                                         b'=guard._estimate_previous_main_source(guard.GUAR'
                                                         b'D_PATH,(SOURCE/guard.GUARD_PATH).read_bytes().re'
                                                         b'place(b"\\r\\n",b"\\n"))\n        previous=guard'
                                                         b'._drive_previous_guard_source(current_guard)\n   '
                                                         b'     original=current_guard.split(b"\\nPOLICY_PAT'
                                                         b'H =",1)[0]+b"\\n"\n        write(repo,"README.md",'
                                                         b'b"offline fixture\\n")\n'),
                                                        (b'            source_path=SOURCE/path\n            '
                                                         b'if not source_path.exists():\n                con'
                                                         b'tinue\n            value=source_path.read_bytes()'
                                                         b'.replace(b"\\r\\n",b"\\n")\n            if path '
                                                         b'in guard.DRIVE_PATHS:\n                if path in'
                                                         b' guard.DRIVE_PRIOR_SOURCE_RECONSTRUCTIONS:\n     '
                                                         b'               value=guard._drive_previous_main_'
                                                         b'source(path,value)\n',
                                                         b'            source_path=SOURCE/path\n            '
                                                         b'if not source_path.exists():\n                con'
                                                         b'tinue\n            value=guard._estimate_previous'
                                                         b'_main_source(path,source_path.read_bytes().repla'
                                                         b'ce(b"\\r\\n",b"\\n"))\n            if path in gu'
                                                         b'ard.DRIVE_PATHS:\n                if path in guar'
                                                         b'd.DRIVE_PRIOR_SOURCE_RECONSTRUCTIONS:\n          '
                                                         b'          value=guard._drive_previous_main_sourc'
                                                         b'e(path,value)\n'),
                                                        (b'            previous_guard_sha256=hashlib.sha256'
                                                         b'(previous).hexdigest(),\n            previous_pol'
                                                         b'icy_blob=guard.blob(base,guard.V4_POLICY_PATH))\n'
                                                         b'        for path in guard.DRIVE_PATHS:\n         '
                                                         b'   write(repo,path,(SOURCE/path).read_bytes().re'
                                                         b'place(b"\\r\\n",b"\\n"))\n        binding["revie'
                                                         b'wed_blobs"]={p:git(repo,"hash-object","--",p) fo'
                                                         b'r p in guard.DRIVE_PATHS if p!=guard.GUARD_PATH}'
                                                         b'\n        implementation,candidate,policy=seal(re'
                                                         b'po,binding)\n        guard.ROOT=original_root\n',
                                                         b'            previous_guard_sha256=hashlib.sha256'
                                                         b'(previous).hexdigest(),\n            previous_pol'
                                                         b'icy_blob=guard.blob(base,guard.V4_POLICY_PATH))\n'
                                                         b'        for path in guard.DRIVE_PATHS:\n         '
                                                         b'   write(repo,path,guard._estimate_previous_main'
                                                         b'_source(path,(SOURCE/path).read_bytes().replace('
                                                         b'b"\\r\\n",b"\\n")))\n        binding["reviewed_b'
                                                         b'lobs"]={p:git(repo,"hash-object","--",p) for p i'
                                                         b'n guard.DRIVE_PATHS if p!=guard.GUARD_PATH}\n    '
                                                         b'    implementation,candidate,policy=seal(repo,bi'
                                                         b'nding)\n        guard.ROOT=original_root\n')]}}


def _estimate_previous_guard_source(source, binding=None):
    binding = ESTIMATE_BINDINGS if binding is None else binding
    if b"\nESTIMATE_POLICY_PATH =" not in source:
        return source
    _require(_dfs_guard_matches(source,binding["successor_guard_sha256"]),
             "SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    return source.split(b"\nESTIMATE_POLICY_PATH =",1)[0]+ESTIMATE_PREVIOUS_CLI


def _estimate_previous_main_source(path,source):
    if path==GUARD_PATH:
        return _estimate_previous_guard_source(source)
    frozen=ESTIMATE_PRIOR_SOURCE_RECONSTRUCTIONS.get(path)
    if frozen is None or hashlib.sha256(source).hexdigest()==frozen["sha256"]:
        return source
    for before,after in reversed(frozen["edits"]):
        _require(source.count(after)==1,"PRIOR_FIXTURE_ANCHOR_CHANGED")
        source=source.replace(after,before,1)
    _require(hashlib.sha256(source).hexdigest()==frozen["sha256"],"PRIOR_ASSERTIONS_CHANGED")
    return source


_estimate_prior_drive_guard = _drive_previous_guard_source
_estimate_prior_drive_main = _drive_previous_main_source


def _drive_previous_guard_source(source,binding=None):
    return _estimate_prior_drive_guard(_estimate_previous_guard_source(source),binding)


def _drive_previous_main_source(path,source):
    return _estimate_prior_drive_main(path,_estimate_previous_main_source(path,source))


def _run_estimate_integrated(manifest_path,base,binding):
    import importlib.util
    path=ROOT/"scripts/estimate_scope.py"
    _require(hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==
             binding["scope_module_sha256"],"SUCCESSOR_SCOPE_MODULE_CHANGED")
    spec=importlib.util.spec_from_file_location("parlaypicker_estimate_scope_policy",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(sys.modules[__name__],manifest_path,base,binding)

NFL_POLICY_PATH = "docs/paid-launch/launch-scope-policy-nfl-provenance-v1.json"
NFL_POLICY_VERSION = "paid-launch-nfl-provenance-v1"
NFL_APPROVAL_REFERENCE = "Owner-authorized separate October 4 software provenance/display correction and minimal private local replay retention; no acquisition, fitting, scientific qualification, authority change, activation, merge or deployment; owner-authorized verified replay bundle download"
NFL_PATHS = ('app/ui/publish_panel.py', 'app_core/prediction_evidence.py', 'app_core/research_display.py', 'app_core/research_estimate_trace.py', 'app_core/research_replay.py', 'tests/test_nfl_research_replay.py', 'tests/test_nfl_provenance_scope.py', 'tests/test_estimate_scope_policy.py', 'scripts/nfl_provenance_scope.py', 'scripts/check_launch_change_scope.py', 'docs/paid-launch/nfl-oct4-provenance.md')
NFL_UNCHANGED_PATHS = ('.github/workflows/activation-grading.yml', '.github/workflows/ci.yml', '.github/workflows/football-stage1.yml', '.github/workflows/football-stage2.yml', '.github/workflows/mlb-receipt-reconciliation.yml', '.github/workflows/paid-launch.yml', '.github/workflows/qualification-operations.yml', '.github/workflows/read-only-census.yml', '.github/workflows/research-scheduler.yml', '.github/workflows/subscriber-completion-postgres.yml', 'app/ui/draftkings.py', 'app/ui/lock_picks.py', 'app/ui/ncaaf_inventory.py', 'app/ui/public_results.py', 'app/ui/readiness_dashboard.py', 'app/ui/sidebar_controls.py', 'app_core/draftkings_classic.py', 'app_core/espn_ncaaf_odds.py', 'app_core/evidence_drive.py', 'app_core/evidence_remote.py', 'app_core/market_probability_model.py', 'app_core/market_stage_metrics.py', 'app_core/ncaaf_identity.py', 'app_core/ncaaf_schedule.py', 'app_core/per_game_boards.py', 'app_core/performance_spans.py', 'app_core/provider_health.py', 'app_core/public_history.py', 'app_core/scoped_reads.py', 'app_core/stage_timing.py', 'core/live_wager_contract.py', 'core/market_policy.py', 'core/price_value.py', 'core/probability_calibration.py', 'core/run_readiness.py', 'core/sport_policy.py', 'core/streamlit_pipeline.py', 'core/wager_decisions.py', 'data/calibration/bucket_stats.json', 'data/calibration/effective_prob_calibration.json', 'docs/paid-launch/dfs-projection-identity-policy.md', 'docs/paid-launch/drive-history-loading.md', 'docs/paid-launch/estimate-availability.md', 'docs/paid-launch/launch-baseline-manifest.json', 'docs/paid-launch/launch-scope-policy-drive-history-v1.json', 'docs/paid-launch/launch-scope-policy-estimate-v1.json', 'docs/paid-launch/launch-scope-policy-ncaaf-v1.json', 'docs/paid-launch/launch-scope-policy-ncaaf-v2.json', 'docs/paid-launch/launch-scope-policy-v2.json', 'docs/paid-launch/launch-scope-policy-v3.json', 'docs/paid-launch/launch-scope-policy-v4.json', 'docs/paid-launch/ncaaf-coverage-corrections.md', 'docs/paid-launch/ncaaf-schedule-coverage.md', 'docs/paid-launch/provider-caller-health-policy.md', 'parlaypicker/app/streamlit_app.py', 'scripts/benchmark_drive_history_loading.py', 'scripts/benchmark_refresh_lock_storage.py', 'scripts/drive_history_scope.py', 'scripts/estimate_scope.py', 'streamlit_app.py', 'tests/paid_launch/case_isolation_and_scope.py', 'tests/test_board_diagnostics.py', 'tests/test_dfs_projection_identity.py', 'tests/test_dfs_scope_policy.py', 'tests/test_draftkings_classic.py', 'tests/test_draftkings_mlb_classic.py', 'tests/test_draftkings_panel.py', 'tests/test_drive_history_loading.py', 'tests/test_drive_history_scope_policy.py', 'tests/test_estimate_availability.py', 'tests/test_evidence_drive.py', 'tests/test_lock_storage_performance.py', 'tests/test_ncaaf_coverage_corrections.py', 'tests/test_ncaaf_coverage_scope_policy.py', 'tests/test_ncaaf_schedule_coverage.py', 'tests/test_ncaaf_schedule_scope_policy.py', 'tests/test_prediction_evidence.py', 'tests/test_provider_caller_health.py', 'tests/test_provider_health_scope_policy.py', 'tests/test_refresh_lock_performance.py')
NFL_BINDINGS = {'base': '3e7f1d5d91de7f3b7741172ae2ad630d6f0e0825',
 'base_tree': 'f7cd888b55206b00dcd104424f2f64f9d3c5d787',
 'manifest_sha256': '2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343',
 'previous_guard_sha256': '5aec2314281757774d59f4976644b8a70e781261ef4b8607590fedbf51f21317',
 'previous_policy_blob': '2ae20bc6927735f6b1b99c1b3164a39ac8f46dfd',
 'reviewed_blobs': {'app/ui/publish_panel.py': '75524da0c3bd5da6a9c5756b5ca66bbde50c68a7',
                    'app_core/prediction_evidence.py': 'd2f87ea6bfbeb91d181842635be40e39d7622ed5',
                    'app_core/research_display.py': '783c127b9d85f2f45ace0f01daa5a310bd9e34c3',
                    'app_core/research_estimate_trace.py': 'e12869179927d00a4a227bdb32c13f1834abd4bb',
                    'app_core/research_replay.py': '82c19ce2db85b41350020b01a81d5ccff0f35209',
                    'docs/paid-launch/nfl-oct4-provenance.md': '0be36582c2da903a7b450558f8993a0eb912d375',
                    'scripts/nfl_provenance_scope.py': 'd1aa16b6a7ba12ba176f5322b9241f1c201bd769',
                    'tests/test_estimate_scope_policy.py': '21ce84b32ed1616081ac254c8bcf68959cdfd4d0',
                    'tests/test_nfl_provenance_scope.py': 'a579bddb98a07a27a4df802271e24515ff51de57',
                    'tests/test_nfl_research_replay.py': '3df921632778254b1ef6a8636c35c3ca7c3b203d'},
 'scope_module_sha256': 'a5c6a79d4361a0a297510d8215d3062ff30f3583887c3cd41e662c39896a574e',
 'successor_guard_sha256': 'ee909ddba6d0996f4a4d14b46fa9991c8171ffcea1e17651cb4f936793c09e87'}
NFL_PREVIOUS_CLI = b'\ndef main() -> int:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)\n    parser.add_argument("--base")\n    parser.add_argument("--json-output", type=Path)\n    args = parser.parse_args()\n    try:\n        if exists_at("HEAD", ESTIMATE_POLICY_PATH):\n            code, report = _run_estimate_integrated(args.manifest, args.base, ESTIMATE_BINDINGS)\n        elif exists_at("HEAD", DRIVE_POLICY_PATH):\n            code, report = _run_drive_integrated(args.manifest, args.base, DRIVE_BINDINGS)\n        elif exists_at("HEAD", V4_POLICY_PATH):\n            code, report = _run_dfs_integrated(args.manifest, args.base, DFS_BINDINGS)\n        elif exists_at("HEAD", COVERAGE_POLICY_PATH):\n            code, report = _run_coverage_integrated(args.manifest, args.base, COVERAGE_BINDINGS)\n        elif exists_at("HEAD", SCHEDULE_POLICY_PATH):\n            code, report = _run_schedule_integrated(args.manifest, args.base, SCHEDULE_BINDINGS)\n        elif exists_at("HEAD", V3_POLICY_PATH):\n            code, report = _run_provider_integrated(args.manifest, args.base, PROVIDER_BINDINGS)\n        else:\n            code, report = _run_integrated(args.manifest, args.base, PRODUCTION_BINDINGS)\n    except Exception as exc:\n        report = {"schema_version": 1, "status": "ERROR", "reason_codes": ["GUARD_EXECUTION_ERROR"], "error": str(exc)}\n        code = 2\n    rendered = json.dumps(report, indent=2, sort_keys=True)\n    if args.json_output:\n        args.json_output.parent.mkdir(parents=True, exist_ok=True)\n        args.json_output.write_text(rendered + "\\n", encoding="utf-8")\n    print(rendered)\n    return code\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
NFL_PRIOR_SOURCE_RECONSTRUCTIONS = {'app/ui/publish_panel.py': {'edits': [(b"            )\n            saved = {'fingerprint':fingerprint, 'package':package, 'ht"
            b"ml':html,\n                     'private_candidate_trace':private_trace}\n        "
            b"    st.session_state['publication_preview'] = saved\n        except (ValueError, Type"
            b"Error, KeyError) as exc:\n            st.session_state.pop('publication_preview',None"
            b')\n',
            b"            )\n            saved = {'fingerprint':fingerprint, 'package':package, 'ht"
            b"ml':html,\n                     'private_candidate_trace':private_trace}\n        "
            b'    # Local owner evidence is distinct from publishing and remote sync.\n            '
            b'import sqlite3\n            from app_core.research_replay import retain_export\n  '
            b"          try:\n                saved['research_replay_receipt'] = retain_export(boar"
            b'ds, package, games, candidates)\n            except (OSError, sqlite3.Error, ValueErr'
            b"or, TypeError) as exc:\n                saved['research_replay_error'] = str(exc)"
            b"\n                st.warning('Private research replay evidence was not saved: '+str(e"
            b"xc))\n            st.session_state['publication_preview'] = saved\n        except "
            b"(ValueError, TypeError, KeyError) as exc:\n            st.session_state.pop('publicat"
            b"ion_preview',None)\n"),
           (b"                'current-wagers-candidate-trace.json', 'application/json',\n         "
            b"       key='download_current_wagers_candidate_trace',\n            )\n    from app"
            b"_core.public_parlays import parlay_funnel\n    with st.expander('Parlay eligibility f"
            b"unnel', expanded=False):\n        if package.get('parlay_policy')=='canonical-v3':\n",
            b"                'current-wagers-candidate-trace.json', 'application/json',\n         "
            b"       key='download_current_wagers_candidate_trace',\n            )\n    replay_r"
            b"eceipt = saved.get('research_replay_receipt')\n    if replay_receipt:\n        wit"
            b"h st.expander('Private research replay evidence', expanded=False):\n            st.ca"
            b"ption('Download the retained source and per-game traces for this preview. Missing or"
            b"iginal sources remain UNKNOWN.')\n            import sqlite3\n            from app"
            b'_core.research_replay import download_bundle, digest, encode\n            try:\n  '
            b'              bundle, verified = download_bundle(replay_receipt, expected_package_ha'
            b'sh=digest(encode(package)))\n            except (OSError, sqlite3.Error, ValueError, '
            b"TypeError, KeyError) as exc:\n                st.warning('Private research replay dow"
            b"nload is unavailable: '+str(exc))\n            else:\n                st.write({ke"
            b"y:verified[key] for key in ('export_id','package_hash','source_boundary','source_lin"
            b"ks')})\n                if verified['source_boundary'] == 'UNKNOWN':\n            "
            b"        st.warning('Original source evidence is unavailable for one or more snapshot"
            b" links. The bundle preserves UNKNOWN.')\n                st.download_button(\n    "
            b"                'Download private research replay bundle', bundle,\n                 "
            b"   'private-research-replay-'+verified['export_id']+'.zip', 'application/zip',\n     "
            b"               key='download_private_research_replay', on_click='ignore',\n          "
            b'      )\n    from app_core.public_parlays import parlay_funnel\n    with st.expand'
            b"er('Parlay eligibility funnel', expanded=False):\n        if package.get('parlay_poli"
            b"cy')=='canonical-v3':\n")],
 'sha256': 'fd783d6d0146c57935e45137d3f4dd3dee9fadf442017d5ce61029348747cd2f'},
 'app_core/prediction_evidence.py': {'edits': [(b'        for action in ("UPDATE", "DELETE"):\n            '
                                                b'db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{tab'
                                                b'le}_{action} BEFORE {action} ON {table} "\n              '
                                                b'         "BEGIN SELECT RAISE(ABORT, \'prediction evidence'
                                                b' is append-only\'); END")\n    return db\n\n\n',
                                                b'        for action in ("UPDATE", "DELETE"):\n            '
                                                b'db.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{tab'
                                                b'le}_{action} BEFORE {action} ON {table} "\n              '
                                                b'         "BEGIN SELECT RAISE(ABORT, \'prediction evidence'
                                                b' is append-only\'); END")\n    from app_core.research_repl'
                                                b'ay import setup\n    setup(db)\n    return db\n\n\n'),
                                               (b'        raise ValueError("Model/configuration artifacts '
                                                b'changed during analysis; run analysis again")\n    if aud'
                                                b'it is None or audit.empty or final is None or final.empt'
                                                b'y:\n        raise ValueError("Cannot capture an empty can'
                                                b'didate audit or final card")\n    audit, final = audit.co'
                                                b'py(), final.copy()\n    # Source facts first; metadata an'
                                                b'd canonical derivation follow. There is no\n    # post-de'
                                                b'rivation restoration that could erase facts with project'
                                                b'ed nulls.\n',
                                                b'        raise ValueError("Model/configuration artifacts '
                                                b'changed during analysis; run analysis again")\n    if aud'
                                                b'it is None or audit.empty or final is None or final.empt'
                                                b'y:\n        raise ValueError("Cannot capture an empty can'
                                                b'didate audit or final card")\n    from app_core.research_'
                                                b'replay import original_frames, retain_source\n    origina'
                                                b'l = original_frames(audit, final, inputs)\n    audit, fin'
                                                b'al = audit.copy(), final.copy()\n    # Source facts first'
                                                b'; metadata and canonical derivation follow. There is no\n'
                                                b'    # post-derivation restoration that could erase facts'
                                                b' with projected nulls.\n'),
                                               (b'        db.execute("INSERT INTO snapshots VALUES (?, ?, '
                                                b'?, ?, ?, ?, ?)",\n                   (context["snapshot_i'
                                                b'd"], context["model_version"], generated, *payload, dige'
                                                b'st))\n        db.execute("INSERT INTO snapshot_runtime VA'
                                                b'LUES (?, ?)", (context["snapshot_id"], PROCESS_INSTANCE)'
                                                b')\n    if path is None:\n        from app_core.evidence_re'
                                                b'mote import sync\n        sync(incremental=True)\n',
                                                b'        db.execute("INSERT INTO snapshots VALUES (?, ?, '
                                                b'?, ?, ?, ?, ?)",\n                   (context["snapshot_i'
                                                b'd"], context["model_version"], generated, *payload, dige'
                                                b'st))\n        db.execute("INSERT INTO snapshot_runtime VA'
                                                b'LUES (?, ?)", (context["snapshot_id"], PROCESS_INSTANCE)'
                                                b')\n        retain_source(db, context["snapshot_id"], run_'
                                                b'id, digest, original, audit, final)\n    if path is None:'
                                                b'\n        from app_core.evidence_remote import sync\n     '
                                                b'   sync(incremental=True)\n')],
                                     'sha256': '14b2663af599bb46de30eaf188ae429b6eaee93e9307c314f2162009d165873b'},
 'app_core/research_display.py': {'edits': [(b'def _time(value):\n    from app_core.public_board import timestamp\n    try:\n     '
            b'   return timestamp(_text(value))\n    except (ValueError, TypeError):\n        re'
            b'turn None\n\n',
            b'def _time(value):\n    from app_core.public_board import timestamp\n    try:\n     '
            b'   from datetime import datetime\n        return timestamp(value.isoformat() if isins'
            b'tance(value,datetime) else _text(value))\n    except (ValueError, TypeError):\n   '
            b'     return None\n\n'),
           (b'    return mass,priced\n\n\ndef from_export(row, *, source=None, source_field="win_'
            b'probability"):\n    """Capture the actual export estimate before contract authorizati'
            b'on replaces it."""\n    identity=_identity(row)\n',
            b'    return mass,priced\n\n\ndef _source_identity_matches(source, identity):\n    """'
            b'Every supplied source alias must match the exact exported research ticket."""\n    la'
            b'bels={"matchup_id":"event_id","candidate_id":"candidate_id","export_run_id":"export_'
            b'run_id",\n        "league":"sport","market_type":"market","best_pick":"selection","di'
            b'splay_pick":"selection",\n        "market_period":"period","period":"period","settlem'
            b'ent_rules":"rules"}\n    for field,key in labels.items():\n        value=source.ge'
            b't(field)\n        if not _absent(value) and not (isinstance(value,float) and math.isn'
            b'an(value)):\n            if not _text(value) or _text(value)!=identity[key]: return F'
            b'alse\n    for field in ("quote_bookmaker","quote_source","sportsbook","book"):\n  '
            b'      value=source.get(field)\n        if not _absent(value) and not (isinstance(valu'
            b'e,float) and math.isnan(value)):\n            if not _text(value) or _text(value).cas'
            b'efold()!=identity["sportsbook"].casefold(): return False\n    for field in ("quote_ti'
            b'me","odds_recorded_at","quote_timestamp","selected_quote_recorded_at"):\n        valu'
            b'e=source.get(field)\n        if not _absent(value) and not (isinstance(value,float) a'
            b'nd math.isnan(value)):\n            if _time(value) is None or _time(value)!=identity'
            b'["quote_time"]: return False\n    for field,key in (("prediction_generated_at","analy'
            b'sis_time"),("game_start_utc","start")):\n        value=source.get(field)\n        '
            b'if not _absent(value) and not (isinstance(value,float) and math.isnan(value)):\n     '
            b'       if _time(value) is None or _time(value)!=identity[key]: return False\n    for '
            b'field,key in (("odds_american","odds"),("quote_id","quote_id"),("prospective_quote_i'
            b'd","quote_id")):\n        value=source.get(field)\n        if not _absent(value) a'
            b'nd not (isinstance(value,float) and math.isnan(value)):\n            if (_number(valu'
            b'e) if key=="odds" else _text(value))!=identity[key]: return False\n    return Tru'
            b'e\n\n\ndef missing_identity_fields(identity):\n    required=("event_id","candidate_i'
            b'd","export_run_id","sport","market","selection","line",\n              "model_target"'
            b',"sportsbook","quote_id","quote_time","analysis_time","start","period","rules")\n    '
            b'return [key for key in required if identity.get(key) is None or identity.get(key)=="'
            b'"]\n\n\ndef from_export(row, *, source=None, source_field="win_probability"):\n    "'
            b'""Capture the actual export estimate before contract authorization replaces it."""\n '
            b'   identity=_identity(row)\n'),
           (b'    if target not in allowed:\n        result["availability_reason"]="TARGET_MISMATCH'
            b'"\n        return result\n    probability=_number(row.get("win_probability"))\n    '
            b'if probability is None:\n        result["availability_reason"]="UNSUPPORTED_PROBABILI'
            b'TY_SEMANTICS"\n',
            b'    if target not in allowed:\n        result["availability_reason"]="TARGET_MISMATCH'
            b'"\n        return result\n    if not direct and not _source_identity_matches(sourc'
            b'e,identity):\n        return _empty(identity,source_field,basis,reason="ESTIMATE_IDEN'
            b'TITY_MISMATCH")\n    from app_core.research_estimate_trace import origin_rejectio'
            b'n\n    rejection=origin_rejection(source)\n    if rejection:\n        return _empty'
            b'(identity,source_field,basis,reason=rejection)\n    probability=_number(row.get("win_'
            b'probability"))\n    if probability is None:\n        result["availability_reason"]'
            b'="UNSUPPORTED_PROBABILITY_SEMANTICS"\n')],
 'sha256': '64734dda2a50f399e0158664be81e9af2837a408e7ec11ce1db40d6e35c8b333'},
 'app_core/research_estimate_trace.py': {'edits': [(b'EXPORT_FIELDS = """candidate_id matchup_id export_ru'
                                                    b'n_id league market_type pick line\nodds quote_id quot'
                                                    b'e_source quote_time market_period settlement_rules\np'
                                                    b'rediction_generated_at start win_probability probabi'
                                                    b'lity_basis ev\nprobability_semantics push_probability'
                                                    b' status Play_Stake""".split()\n\n\ndef fact(value):\n',
                                                    b'EXPORT_FIELDS = """candidate_id matchup_id export_ru'
                                                    b'n_id league market_type pick line\nodds quote_id quot'
                                                    b'e_source quote_time market_period settlement_rules\np'
                                                    b'rediction_generated_at start win_probability probabi'
                                                    b'lity_basis ev\nprobability_semantics push_probability'
                                                    b' status Play_Stake ml_target""".split()\n\n\ndef fact(v'
                                                    b'alue):\n'),
                                                   (b'    Raw model and blended research estimates retain '
                                                    b'separate fields. Provider\n    payloads, features, se'
                                                    b'crets, review prose and private configuration are om'
                                                    b'itted.\n    """\n    return encode(dict(version=1,\n   '
                                                    b'     source={k:fact(source.get(k)) for k in SOURCE_F'
                                                    b'IELDS} if source is not None else None,\n        expo'
                                                    b'rt={k:fact(export.get(k)) for k in EXPORT_FIELDS},\n '
                                                    b'       display={k:display[k] for k in ("source_field'
                                                    b'","basis","identity","inference_status",\n           '
                                                    b' "availability_reason","value_reason","probability",'
                                                    b'"push_probability","ev")}))\n',
                                                    b'    Raw model and blended research estimates retain '
                                                    b'separate fields. Provider\n    payloads, features, se'
                                                    b'crets, review prose and private configuration are om'
                                                    b'itted.\n    """\n    from app_core.research_display im'
                                                    b'port missing_identity_fields\n    return encode(dict('
                                                    b'version=2,\n        missing_identity_fields=missing_i'
                                                    b'dentity_fields(display["identity"]),\n        source='
                                                    b'{k:fact(source.get(k)) for k in SOURCE_FIELDS} if so'
                                                    b'urce is not None else None,\n        export={k:fact(e'
                                                    b'xport.get(k)) for k in EXPORT_FIELDS},\n        displ'
                                                    b'ay={k:display[k] for k in ("source_field","basis","i'
                                                    b'dentity","inference_status",\n            "availabili'
                                                    b'ty_reason","value_reason","probability","push_probab'
                                                    b'ility","ev")}))\n\n\nORIGIN_IDENTITY_FIELDS = frozenset'
                                                    b'("""candidate_id matchup_id export_run_id provider_e'
                                                    b'vent_id\nprovider_namespace prediction_generated_at g'
                                                    b'ame_start_utc market_period period settlement_rules"'
                                                    b'"".split())\n\n\ndef origin_rejection(source):\n    '
                                                    b'"""Validate supplied producer diagnostics without pr'
                                                    b'omoting a blend\'s inference."""\n    from app_core.re'
                                                    b'search_display import _absent, _number, _text, _time'
                                                    b'\n    raw=source.get("ml_estimate_metadata")\n    stat'
                                                    b'us=source.get("ml_inference_status")\n    if _absent('
                                                    b'raw) or (isinstance(raw,float) and math.isnan(raw)):'
                                                    b'\n        return None if _absent(status) or (isinstan'
                                                    b'ce(status,float) and math.isnan(status)) else "ESTIM'
                                                    b'ATE_PROVENANCE_NOT_RECORDED"\n    try:\n        item=j'
                                                    b'son.loads(raw)\n        keys={"version","generated_at'
                                                    b'","identity","inference_status","line","market_type"'
                                                    b',"predictor_id",\n              "probability","probab'
                                                    b'ility_field","probability_semantics","push_probabili'
                                                    b'ty","reason","target"}\n        if not isinstance(ite'
                                                    b'm,dict) or set(item)!=keys or type(item["version"]) '
                                                    b'is not int or item["version"]!=1:\n            raise '
                                                    b'ValueError("Invalid origin schema")\n        if item['
                                                    b'"inference_status"] not in {"success","failed","unav'
                                                    b'ailable"} or item["inference_status"]!=_text(status)'
                                                    b':\n            raise ValueError("Contradictory origin'
                                                    b' outcome")\n        if item["probability_field"]!="ml'
                                                    b'_probability" or _time(item["generated_at"]) is None'
                                                    b' or not isinstance(item["reason"],str):\n            '
                                                    b'raise ValueError("Invalid origin provenance")\n      '
                                                    b'  if not isinstance(item["identity"],dict) or set(it'
                                                    b'em["identity"])!=ORIGIN_IDENTITY_FIELDS:\n           '
                                                    b' raise ValueError("Invalid origin identity")\n       '
                                                    b' for field in ("line","market_type","predictor_id","'
                                                    b'probability","target"):\n            value=item[field'
                                                    b']\n            if value not in ({"state":"MISSING"},{'
                                                    b'"state":"INVALID"}):\n                if not isinstan'
                                                    b'ce(value,dict) or set(value)!={"state","value"} or v'
                                                    b'alue["state"]!="VALUE" or fact(value["value"])!=valu'
                                                    b'e:\n                    raise ValueError("Invalid ori'
                                                    b'gin fact")\n        for value in item["identity"].val'
                                                    b'ues():\n            if value not in ({"state":"MISSIN'
                                                    b'G"},{"state":"INVALID"}):\n                if not isi'
                                                    b'nstance(value,dict) or set(value)!={"state","value"}'
                                                    b' or value["state"]!="VALUE" or not isinstance(value['
                                                    b'"value"],str):\n                    raise ValueError('
                                                    b'"Invalid origin identity fact")\n        # Capture as'
                                                    b'signs a new run ID; it does not change these recorde'
                                                    b'd\n        # event/target facts. Compare clocks by in'
                                                    b'stant without overwriting them.\n        def event(va'
                                                    b'lue):\n            import re\n            from core.te'
                                                    b'am_mapper import normalize_team_name\n            par'
                                                    b'ts=value.split("|")\n            if len(parts)!=3: re'
                                                    b'turn value\n            if re.fullmatch(r"\\d{4}-\\d{2}'
                                                    b'-\\d{2}",parts[0]): day,home,away=parts\n            e'
                                                    b'lif re.fullmatch(r"\\d{4}-\\d{2}-\\d{2}",parts[2]): hom'
                                                    b'e,away,day=parts\n            else: return value\n    '
                                                    b'        return (day,normalize_team_name(home),normal'
                                                    b'ize_team_name(away))\n        for field in ORIGIN_IDE'
                                                    b'NTITY_FIELDS-{"export_run_id"}:\n            original'
                                                    b'=item["identity"][field]\n            if original.get'
                                                    b'("state")!="VALUE": continue\n            current=sou'
                                                    b'rce.get(field)\n            if _absent(current) or (i'
                                                    b'sinstance(current,float) and math.isnan(current)):\n '
                                                    b'               return "ESTIMATE_PROVENANCE_NOT_RECOR'
                                                    b'DED"\n            if field in {"prediction_generated_'
                                                    b'at","game_start_utc"}:\n                if _time(orig'
                                                    b'inal["value"]) is None or _time(original["value"])!='
                                                    b'_time(current):\n                    return "ESTIMATE'
                                                    b'_IDENTITY_MISMATCH"\n            elif field=="matchup'
                                                    b'_id":\n                if event(original["value"])!=e'
                                                    b'vent(_text(current)): return "ESTIMATE_IDENTITY_MISM'
                                                    b'ATCH"\n            elif original["value"]!=_text(curr'
                                                    b'ent): return "ESTIMATE_IDENTITY_MISMATCH"\n        if'
                                                    b' item["inference_status"]=="failed": return "INFEREN'
                                                    b'CE_FAILED"\n        if item["inference_status"]=="una'
                                                    b'vailable":\n            return "INFERENCE_UNAVAILABLE'
                                                    b'" if _number(source.get("ml_probability")) is not No'
                                                    b'ne else None\n        line=_number(source.get("total_'
                                                    b'line" if _text(source.get("market_type")).startswith'
                                                    b'("total") else "spread_line"))\n        expected={"pr'
                                                    b'obability":fact(source.get("ml_probability")),"predi'
                                                    b'ctor_id":fact(source.get("ml_probability_source")),\n'
                                                    b'                  "target":fact(source.get("ml_targe'
                                                    b't")),"market_type":fact(source.get("market_type")),"'
                                                    b'line":fact(line)}\n        if any(item[k]!=v for k,v '
                                                    b'in expected.items()): return "ESTIMATE_IDENTITY_MISM'
                                                    b'ATCH"\n        probability=_number(source.get("ml_pro'
                                                    b'bability"))\n        if probability is None or not 0<'
                                                    b'=probability<=1: return "INVALID_PROBABILITY"\n      '
                                                    b'  half=line is not None and abs(line*2-round(line*2)'
                                                    b')<=1e-9 and abs(line-round(line))>1e-9\n        if (i'
                                                    b'tem["probability_semantics"]!=("win_unconditional_wi'
                                                    b'th_push" if half else "UNDECLARED_PUSH_MODEL")\n     '
                                                    b'           or item["push_probability"]!=(0.0 if half'
                                                    b' else None) or isinstance(item["push_probability"],b'
                                                    b'ool)):\n            return "UNSUPPORTED_PROBABILITY_S'
                                                    b'EMANTICS"\n        return None\n    except (ValueError'
                                                    b',TypeError,KeyError,AttributeError):\n        return '
                                                    b'"ESTIMATE_PROVENANCE_NOT_RECORDED"\n')],
                                         'sha256': '7a859e0cc630819a6f82a0039738431a73e9f000dc89a4a43362ab520f68f08e'},
 'tests/test_estimate_scope_policy.py': {'edits': [(b'        git(repo,"config","user.name","Offline Test"'
                                                    b')\n        git(repo,"config","user.email","offline@ex'
                                                    b'ample.invalid")\n        git(repo,"config","core.auto'
                                                    b'crlf","false")\n        current_guard=(SOURCE/guard.G'
                                                    b'UARD_PATH).read_bytes().replace(b"\\r\\n",b"\\n")\n '
                                                    b'       previous=guard._estimate_previous_guard_sourc'
                                                    b'e(current_guard)\n        original=current_guard.spli'
                                                    b't(b"\\nPOLICY_PATH =",1)[0]+b"\\n"\n        write(repo,'
                                                    b'"README.md",b"offline fixture\\n")\n',
                                                    b'        git(repo,"config","user.name","Offline Test"'
                                                    b')\n        git(repo,"config","user.email","offline@ex'
                                                    b'ample.invalid")\n        git(repo,"config","core.auto'
                                                    b'crlf","false")\n        current_guard=guard._nfl_prev'
                                                    b'ious_main_source(guard.GUARD_PATH,(SOURCE/guard.GUAR'
                                                    b'D_PATH).read_bytes().replace(b"\\r\\n",b"\\n"))\n   '
                                                    b'     previous=guard._estimate_previous_guard_source('
                                                    b'current_guard)\n        original=current_guard.split('
                                                    b'b"\\nPOLICY_PATH =",1)[0]+b"\\n"\n        write(repo,"R'
                                                    b'EADME.md",b"offline fixture\\n")\n'),
                                                   (b'            source_path=SOURCE/path\n            if n'
                                                    b'ot source_path.exists():\n                continue\n  '
                                                    b'          value=source_path.read_bytes().replace(b"\\'
                                                    b'r\\n",b"\\n")\n            if path in guard.ESTIMATE_PA'
                                                    b'THS:\n                if path in guard.ESTIMATE_PRIOR'
                                                    b'_SOURCE_RECONSTRUCTIONS:\n                    value=g'
                                                    b'uard._estimate_previous_main_source(path,value)\n',
                                                    b'            source_path=SOURCE/path\n            if n'
                                                    b'ot source_path.exists():\n                continue\n  '
                                                    b'          value=guard._nfl_previous_main_source(path'
                                                    b',source_path.read_bytes().replace(b"\\r\\n",b"\\n")'
                                                    b')\n            if path in guard.ESTIMATE_PATHS:\n     '
                                                    b'           if path in guard.ESTIMATE_PRIOR_SOURCE_RE'
                                                    b'CONSTRUCTIONS:\n                    value=guard._esti'
                                                    b'mate_previous_main_source(path,value)\n'),
                                                   (b'            previous_guard_sha256=hashlib.sha256(pre'
                                                    b'vious).hexdigest(),\n            previous_policy_blob'
                                                    b'=guard.blob(base,guard.DRIVE_POLICY_PATH))\n        f'
                                                    b'or path in guard.ESTIMATE_PATHS:\n            write(r'
                                                    b'epo,path,(SOURCE/path).read_bytes().replace(b"\\r\\n",'
                                                    b'b"\\n"))\n        binding["reviewed_blobs"]={p:git(rep'
                                                    b'o,"hash-object","--",p) for p in guard.ESTIMATE_PATH'
                                                    b'S if p!=guard.GUARD_PATH}\n        implementation,can'
                                                    b'didate,policy=seal(repo,binding)\n        guard.ROOT='
                                                    b'original_root\n',
                                                    b'            previous_guard_sha256=hashlib.sha256(pre'
                                                    b'vious).hexdigest(),\n            previous_policy_blob'
                                                    b'=guard.blob(base,guard.DRIVE_POLICY_PATH))\n        f'
                                                    b'or path in guard.ESTIMATE_PATHS:\n            write(r'
                                                    b'epo,path,guard._nfl_previous_main_source(path,(SOURC'
                                                    b'E/path).read_bytes().replace(b"\\r\\n",b"\\n")))\n  '
                                                    b'      binding["reviewed_blobs"]={p:git(repo,"hash-ob'
                                                    b'ject","--",p) for p in guard.ESTIMATE_PATHS if p!=gu'
                                                    b'ard.GUARD_PATH}\n        implementation,candidate,pol'
                                                    b'icy=seal(repo,binding)\n        guard.ROOT=original_r'
                                                    b'oot\n')],
                                         'sha256': 'feb3c8cb5e3b0cf78c0cd3bbefaa2beb6abe8ef8df81f4a8baa5bdc271e4b53f'}}


def _nfl_previous_guard_source(source, binding=None):
    binding = NFL_BINDINGS if binding is None else binding
    if b"\nNFL_POLICY_PATH =" not in source: return source
    _require(_dfs_guard_matches(source,binding["successor_guard_sha256"]),"SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    return source.split(b"\nNFL_POLICY_PATH =",1)[0]+NFL_PREVIOUS_CLI


def _nfl_previous_main_source(path,source):
    if path==GUARD_PATH: return _nfl_previous_guard_source(source)
    frozen=NFL_PRIOR_SOURCE_RECONSTRUCTIONS.get(path)
    if frozen is None or hashlib.sha256(source).hexdigest()==frozen["sha256"]: return source
    for before,after in reversed(frozen["edits"]):
        _require(source.count(after)==1,"PRIOR_FIXTURE_ANCHOR_CHANGED")
        source=source.replace(after,before,1)
    _require(hashlib.sha256(source).hexdigest()==frozen["sha256"],"PRIOR_ASSERTIONS_CHANGED")
    return source


_nfl_prior_estimate_guard = _estimate_previous_guard_source
_nfl_prior_estimate_main = _estimate_previous_main_source


def _estimate_previous_guard_source(source,binding=None):
    return _nfl_prior_estimate_guard(_nfl_previous_guard_source(source),binding)


def _estimate_previous_main_source(path,source):
    return _nfl_prior_estimate_main(path,_nfl_previous_main_source(path,source))


def _run_nfl_integrated(manifest_path,base,binding):
    import importlib.util
    path=ROOT/"scripts/nfl_provenance_scope.py"
    _require(hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==binding["scope_module_sha256"],"SUCCESSOR_SCOPE_MODULE_CHANGED")
    spec=importlib.util.spec_from_file_location("parlaypicker_nfl_provenance_scope_policy",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(sys.modules[__name__],manifest_path,base,binding)

HOME_POLICY_PATH = 'docs/paid-launch/launch-scope-policy-home-runline-v1.json'
HOME_POLICY_VERSION = 'paid-launch-home-runline-contract-v1'
HOME_APPROVAL_REFERENCE = 'Owner-authorized bounded exact-home Run Line versioned feature contract and offline harness; no fitting, acquisition, registration, activation, wagering, merge or deployment'
HOME_PATHS = ('app_core/mlb_home_runline_contract.py', 'tests/test_mlb_home_runline_contract.py', 'scripts/home_runline_scope.py', 'tests/test_home_runline_scope_policy.py', 'scripts/check_launch_change_scope.py', 'tests/test_nfl_provenance_scope.py', 'docs/paid-launch/mlb-home-runline-contract.md')
HOME_BINDINGS = {'base': 'e24e814b6cb08a8d983de615a499035594893830',
 'base_tree': 'a7a8c8d1a831f24d51754fefc35405c58cca6e92',
 'manifest_sha256': '2faf43204d045c81a1fdf589fff2d8ff76515c3a7c1c9b7d628d6b7f47cd1343',
 'previous_policy_blob': 'e202937efde63fb81f7a9e332aafa16aebb689f6',
 'previous_guard_sha256': '8995dfacb57fd63c2017e7d35b013fdac4ba4d3b45975802777e73063fa1690a',
 'scope_module_sha256': '8fcf20cf79af722d7752f17c6834e0e455db97b0789fd90b0995c74158230f80',
 'successor_guard_sha256': '046a9dc9d58e40cc47497d1e281ea2045e84f098211995d93b54a7ec0c8ccf5c',
 'reviewed_blobs': {'app_core/mlb_home_runline_contract.py': 'a86ba1a783a9f0b302053d40b81e543590aca806',
                    'tests/test_mlb_home_runline_contract.py': '6cd50521f8086d8d9b662ec15223b834721bb218',
                    'scripts/home_runline_scope.py': '73f5df38e41505c05c062e419670862b5fcf5659',
                    'tests/test_home_runline_scope_policy.py': 'f3b52a32677cdca5841a40b9a710e531b6a062d9',
                    'tests/test_nfl_provenance_scope.py': 'e8e8b40fc6e2e0eda2249a9bea758c97130e30fd',
                    'docs/paid-launch/mlb-home-runline-contract.md': 'bdc28800d9c8b826fa9beba00c4826efab6b7544'}}
HOME_PREVIOUS_CLI = b'\ndef main() -> int:\n    parser = argparse.ArgumentParser()\n    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)\n    parser.add_argument("--base")\n    parser.add_argument("--json-output", type=Path)\n    args = parser.parse_args()\n    try:\n        if exists_at("HEAD", NFL_POLICY_PATH):\n            code, report = _run_nfl_integrated(args.manifest, args.base, NFL_BINDINGS)\n        elif exists_at("HEAD", ESTIMATE_POLICY_PATH):\n            code, report = _run_estimate_integrated(args.manifest, args.base, ESTIMATE_BINDINGS)\n        elif exists_at("HEAD", DRIVE_POLICY_PATH):\n            code, report = _run_drive_integrated(args.manifest, args.base, DRIVE_BINDINGS)\n        elif exists_at("HEAD", V4_POLICY_PATH):\n            code, report = _run_dfs_integrated(args.manifest, args.base, DFS_BINDINGS)\n        elif exists_at("HEAD", COVERAGE_POLICY_PATH):\n            code, report = _run_coverage_integrated(args.manifest, args.base, COVERAGE_BINDINGS)\n        elif exists_at("HEAD", SCHEDULE_POLICY_PATH):\n            code, report = _run_schedule_integrated(args.manifest, args.base, SCHEDULE_BINDINGS)\n        elif exists_at("HEAD", V3_POLICY_PATH):\n            code, report = _run_provider_integrated(args.manifest, args.base, PROVIDER_BINDINGS)\n        else:\n            code, report = _run_integrated(args.manifest, args.base, PRODUCTION_BINDINGS)\n    except Exception as exc:\n        report = {"schema_version": 1, "status": "ERROR", "reason_codes": ["GUARD_EXECUTION_ERROR"], "error": str(exc)}\n        code = 2\n    rendered = json.dumps(report, indent=2, sort_keys=True)\n    if args.json_output:\n        args.json_output.parent.mkdir(parents=True, exist_ok=True)\n        args.json_output.write_text(rendered + "\\n", encoding="utf-8")\n    print(rendered)\n    return code\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
HOME_PRIOR_SOURCE_RECONSTRUCTIONS = {'tests/test_nfl_provenance_scope.py': {'sha256': '7bf2ab487f20c8a66e48a853923122f5123877ccb98d088166a7b085de14d28c',
                                        'edits': [(b'        current_guard=(S'
                                                   b'OURCE/guard.GUARD_PATH).'
                                                   b'read_bytes().replace(b"\\'
                                                   b'r\\n",b"\\n")\n',
                                                   b'        current_guard=gu'
                                                   b'ard._home_previous_main_'
                                                   b'source(guard.GUARD_PATH,'
                                                   b'(SOURCE/guard.GUARD_PATH'
                                                   b').read_bytes().replace(b'
                                                   b'"\\r\\n",b"\\n"))\n'),
                                                  (b'            value=guard.'
                                                   b'_nfl_previous_main_sourc'
                                                   b'e(path,source_path.read_'
                                                   b'bytes().replace(b"\\r\\n",'
                                                   b'b"\\n"))\n',
                                                   b'            value=guard.'
                                                   b'_nfl_previous_main_sourc'
                                                   b'e(path,guard._home_previ'
                                                   b'ous_main_source(path,sou'
                                                   b'rce_path.read_bytes().re'
                                                   b'place(b"\\r\\n",b"\\n")'
                                                   b'))\n'),
                                                  (b'            write(repo,p'
                                                   b'ath,(SOURCE/path).read_b'
                                                   b'ytes().replace(b"\\r\\n",b'
                                                   b'"\\n"))\n',
                                                   b'            write(repo,p'
                                                   b'ath,guard._home_previous'
                                                   b'_main_source(path,(SOURC'
                                                   b'E/path).read_bytes().rep'
                                                   b'lace(b"\\r\\n",b"\\n"))'
                                                   b')\n')]}}


def _home_previous_guard_source(source, binding=None):
    binding = HOME_BINDINGS if binding is None else binding
    if b"\nHOME_POLICY_PATH =" not in source:
        return source
    _require(_dfs_guard_matches(source,binding["successor_guard_sha256"]),
             "SUCCESSOR_GUARD_REVIEWED_BYTES_CHANGED")
    return source.split(b"\nHOME_POLICY_PATH =",1)[0]+HOME_PREVIOUS_CLI


def _home_previous_main_source(path,source):
    if path==GUARD_PATH:
        return _home_previous_guard_source(source)
    frozen=HOME_PRIOR_SOURCE_RECONSTRUCTIONS.get(path)
    if frozen is None or hashlib.sha256(source).hexdigest()==frozen["sha256"]:
        return source
    for before,after in reversed(frozen["edits"]):
        _require(source.count(after)==1,"PRIOR_FIXTURE_ANCHOR_CHANGED")
        source=source.replace(after,before,1)
    _require(hashlib.sha256(source).hexdigest()==frozen["sha256"],"PRIOR_ASSERTIONS_CHANGED")
    return source


_home_prior_nfl_guard = _nfl_previous_guard_source
_home_prior_nfl_main = _nfl_previous_main_source


def _nfl_previous_guard_source(source,binding=None):
    return _home_prior_nfl_guard(_home_previous_guard_source(source),binding)


def _nfl_previous_main_source(path,source):
    return _home_prior_nfl_main(path,_home_previous_main_source(path,source))


def _run_home_integrated(manifest_path,base,binding):
    import importlib.util
    path=ROOT/"scripts/home_runline_scope.py"
    _require(hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()==
             binding["scope_module_sha256"],"SUCCESSOR_SCOPE_MODULE_CHANGED")
    spec=importlib.util.spec_from_file_location("parlaypicker_home_runline_scope_policy",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run(sys.modules[__name__],manifest_path,base,binding)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--base")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    try:
        if exists_at("HEAD", HOME_POLICY_PATH):
            code, report = _run_home_integrated(args.manifest, args.base, HOME_BINDINGS)
        elif exists_at("HEAD", NFL_POLICY_PATH):
            code, report = _run_nfl_integrated(args.manifest, args.base, NFL_BINDINGS)
        elif exists_at("HEAD", ESTIMATE_POLICY_PATH):
            code, report = _run_estimate_integrated(args.manifest, args.base, ESTIMATE_BINDINGS)
        elif exists_at("HEAD", DRIVE_POLICY_PATH):
            code, report = _run_drive_integrated(args.manifest, args.base, DRIVE_BINDINGS)
        elif exists_at("HEAD", V4_POLICY_PATH):
            code, report = _run_dfs_integrated(args.manifest, args.base, DFS_BINDINGS)
        elif exists_at("HEAD", COVERAGE_POLICY_PATH):
            code, report = _run_coverage_integrated(args.manifest, args.base, COVERAGE_BINDINGS)
        elif exists_at("HEAD", SCHEDULE_POLICY_PATH):
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
