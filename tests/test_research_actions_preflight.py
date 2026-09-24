"""The Actions preflight must expose setting names without exposing values."""

import os
from pathlib import Path
import subprocess
import sys
import yaml

from scripts.research_actions_preflight import configuration_status


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "research_actions_preflight.py"


def test_presence_report_never_contains_values(tmp_path):
    canaries = {
        "PARLAYPICKER_DRIVE_FOLDER_ID": "CANARY_FOLDER_ID",
        "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT": "CANARY_SERVICE_ACCOUNT_JSON",
        "ODDS_API_KEY": "CANARY_ODDS_KEY",
        "CFBD_API_KEY": "CANARY_CFBD_KEY",
        "PARLAYPICKER_NETLIFY_SITE_ID": "CANARY_NETLIFY_ID",
    }
    flags = {
        "PREFLIGHT_RESEARCH_ENABLED": "true",
        "PREFLIGHT_DRIVE_FOLDER_PRESENT": "true",
        "PREFLIGHT_SERVICE_ACCOUNT_PRESENT": "true",
        "PREFLIGHT_ODDS_KEY_PRESENT": "true",
        "PREFLIGHT_CFBD_KEY_PRESENT": "true",
        "PREFLIGHT_NETLIFY_SITE_PRESENT": "true",
    }
    summary = tmp_path / "summary.md"
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True,
                            text=True, env={**os.environ, **canaries, **flags,
                                            "GITHUB_STEP_SUMMARY": str(summary)}, check=False)
    assert result.returncode == 0
    report = result.stdout + result.stderr + summary.read_text(encoding="utf-8")
    assert "configured" in report
    for value in canaries.values():
        assert value not in report


def test_missing_required_settings_fail_with_names_only():
    missing, report = configuration_status({
        "PREFLIGHT_RESEARCH_ENABLED": "false",
        "PREFLIGHT_DRIVE_FOLDER_PRESENT": "true",
        "PARLAYPICKER_DRIVE_FOLDER_ID": "PRIVATE_FOLDER_VALUE",
    })
    assert missing == ["RESEARCH_SCHEDULER_ENABLED", "PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT",
                       "ODDS_API_KEY", "CFBD_API_KEY"]
    assert "PRIVATE_FOLDER_VALUE" not in report
    assert "PARLAYPICKER_NETLIFY_SITE_ID | optional" in report


def test_mlb_only_does_not_require_other_provider_keys():
    missing, report = configuration_status({
        "PREFLIGHT_RESEARCH_ENABLED": "true",
        "RESEARCH_SPORTS": "MLB",
        "PREFLIGHT_DRIVE_FOLDER_PRESENT": "true",
        "PREFLIGHT_SERVICE_ACCOUNT_PRESENT": "true",
    })
    assert missing == []
    assert "ODDS_API_KEY | not required for selected sports" in report
    assert "CFBD_API_KEY | not required for selected sports" in report


def test_workflow_preflight_receives_booleans_instead_of_secret_values():
    workflow = yaml.safe_load((SCRIPT.parents[1] / ".github/workflows/research-scheduler.yml").read_text())
    env = workflow["jobs"]["configuration-preflight"]["env"]
    assert set(env) == {
        "PREFLIGHT_RESEARCH_ENABLED", "PREFLIGHT_DRIVE_FOLDER_PRESENT",
        "PREFLIGHT_SERVICE_ACCOUNT_PRESENT", "PREFLIGHT_ODDS_KEY_PRESENT",
        "PREFLIGHT_CFBD_KEY_PRESENT", "PREFLIGHT_NETLIFY_SITE_PRESENT",
        "RESEARCH_SPORTS",
    }
    for name, expression in env.items():
        if name != "RESEARCH_SPORTS":
            assert expression.endswith(" != '' }}") or expression.endswith(" == 'true' }}")
    assert not any("runner.temp" in value for value in workflow["jobs"]["research"]["env"].values())
