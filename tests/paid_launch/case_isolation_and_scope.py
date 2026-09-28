from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_subscriber_runtime_never_imports_research_or_lock_modules():
    forbidden = {"app", "app_core", "streamlit_app", "core.streamlit_pipeline"}
    violations = []
    for path in (ROOT / "services" / "subscriber").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                if any(name == item or name.startswith(item + ".") for item in forbidden):
                    violations.append(f"{path.relative_to(ROOT)}:{name}")
    assert violations == []


def test_release_submit_command_contains_no_lock_drive_or_analysis_imports():
    path = ROOT / "scripts" / "submit_customer_release.py"
    source = path.read_text(encoding="utf-8")
    assert "app_core" not in source
    assert "lock_picks" not in source
    assert "evidence_drive" not in source


def test_protected_scope_guard_passes_for_additive_work():
    result = subprocess.run(
        [sys.executable, "scripts/check_launch_change_scope.py"], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == 0, payload
    assert payload["protected_changes"] == []
    assert payload["existing_test_changes"] == []
