from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
HTML = ROOT / "web" / "subscriber" / "index.html"
SCRIPT = ROOT / "web" / "subscriber" / "subscriber.js"
BROWSER = ROOT / "tests" / "subscriber_journey_browser.cjs"


def test_subscriber_shell_exposes_complete_existing_service_controls():
    html = HTML.read_text(encoding="utf-8")
    for control in (
        'id="sign-in"',
        'id="checkout"',
        'id="billing-portal"',
        'id="cancel-subscription"',
        'id="logout"',
        'id="alerts-form"',
        'id="result-filter"',
        'id="refresh-current"',
    ):
        assert control in html
    assert "Support address pending owner configuration" in html
    assert "support@example.invalid" not in html


def test_subscriber_script_has_no_browser_authority_or_client_settlement_math():
    script = SCRIPT.read_text(encoding="utf-8")
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "provider_price_id" not in script
    assert "paper_return=" not in script
    assert "'/offer'" in script
    assert "'/billing/checkout'" in script
    assert "'/billing/portal'" in script
    assert "'/billing/cancel'" in script
    assert "'/me/alerts'" in script
    assert "visibilitychange" in script
    assert "offline" in script and "online" in script
    assert "requestNumber!==state.latestCurrentRequest" in script


def test_subscriber_javascript_and_browser_harness_parse():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    for path in (SCRIPT, BROWSER):
        subprocess.run([node, "--check", str(path)], check=True, capture_output=True, text=True)


def test_browser_harness_covers_all_result_states_and_journey_edges():
    harness = BROWSER.read_text(encoding="utf-8")
    for state in ("WIN", "LOSS", "PUSH", "VOID", "PENDING", "NEEDS_REVIEW", "CORRECTED"):
        assert f"'{state}'" in harness
    for evidence in (
        "checkoutPosts",
        "portalPosts",
        "cancelPosts",
        "logoutPosts",
        "alertPosts",
        "lateResponseSuppressed",
        "expiryCleared",
        "mobileKeyboard",
        "premiumLeakage",
    ):
        assert evidence in harness
