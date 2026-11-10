"""The readiness report published to Drive for the app (owner 2026-11-10). SYNTHETIC client; no Drive."""
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json

import pytest
from streamlit.testing.v1 import AppTest

from app_core import readiness_report_remote as remote
from app_core.evidence_config import EvidenceStorageError
from app_core.evidence_drive import AlreadyExists

NOW = datetime(2026, 11, 10, 11, 42, tzinfo=timezone.utc)


class Drive:
    def __init__(self):
        self.objects, self.asked = {}, []

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch, **kwargs):
        assert IfNoneMatch == "*"
        if Key in self.objects:
            raise AlreadyExists()
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        self.asked.append(Key)
        if Key not in self.objects:
            raise EvidenceStorageError(remote.MISSING)
        return {"Body": BytesIO(self.objects[Key])}


def test_publish_writes_one_object_outside_the_evidence_and_the_app_reads_the_newest():
    drive = Drive()
    out = remote.publish(drive, "f", {"markets": []}, "| old |", clock=lambda: NOW - timedelta(hours=5))
    assert out["status"] == "PUBLISHED" and out["key"] == "parlaypicker/market-readiness-v1/report/20261110T06.json"
    assert not out["key"].startswith("parlaypicker/evidence-v1")
    assert remote.publish(drive, "f", {"markets": [1]}, "| new |", clock=lambda: NOW)["status"] == "PUBLISHED"
    again = remote.publish(drive, "f", {"markets": [2]}, "| later |", clock=lambda: NOW + timedelta(minutes=5))
    assert again["status"] == "ALREADY_PUBLISHED_THIS_HOUR" and len(drive.objects) == 2      # create-only: the first stands
    drive.asked.clear()
    found = remote.latest(drive, "f", now=NOW + timedelta(hours=2))
    assert found["markdown"] == "| new |" and found["report"] == {"markets": [1]} and found["generated_at"] == NOW.isoformat()
    assert len(drive.asked) == 3                                                           # exact names, newest hour first
    assert remote.latest(drive, "f", now=NOW + timedelta(hours=60)) is None                 # nothing in the last 48 hours
    assert remote.latest(Drive(), "f", now=NOW) is None


def test_a_bad_object_or_a_drive_error_is_raised_not_skipped():
    drive = Drive()
    drive.objects[remote.key_for(NOW)] = json.dumps({"markdown": 3, "report": {}}).encode()
    with pytest.raises(ValueError, match="readiness_report_shape"):
        remote.latest(drive, "f", now=NOW)

    class Broken(Drive):
        def get_object(self, **kwargs):
            raise EvidenceStorageError("Drive listing was incomplete; restore cannot be verified")
    with pytest.raises(EvidenceStorageError):
        remote.latest(Broken(), "f", now=NOW)

    class Altered(Drive):
        def get_object(self, **kwargs):
            return {"Body": BytesIO(b"{}")}
    with pytest.raises(ValueError, match="readiness_report_readback"):
        remote.publish(Altered(), "f", {}, "x", clock=lambda: NOW)


def test_the_script_publishes_only_when_asked_and_a_failed_upload_never_fails_the_report(tmp_path, monkeypatch):
    from scripts import market_readiness_report as mr
    drive = Drive()
    assert mr.publish({"markets": []}, "| t |", client=drive, folder="f")["status"] == "PUBLISHED"

    class Down:
        def put_object(self, **kwargs):
            raise ConnectionError("no network")
    assert mr.publish({}, "t", client=Down(), folder="f") == {"status": "PUBLISH_FAILED", "error": "ConnectionError"}
    monkeypatch.setattr(mr, "report", lambda *a, **k: {"markets": []})
    called = []
    monkeypatch.setattr(mr, "publish", lambda result, table: called.append(table) or {"status": "PUBLISHED"})
    base = ["--directory", str(tmp_path / "r"), "--output", str(tmp_path / "o" / "x.json")]
    assert mr.main(base) == 0 and called == []
    assert mr.main(base + ["--publish"]) == 0 and len(called) == 1
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / ".github/workflows/market-readiness.yml").read_text(encoding="utf-8")
    assert "--publish" in text and "contents: read" in text


def test_the_panel_shows_the_full_report_loaded_from_drive():
    payload = {"generated_at": NOW.isoformat(), "markdown": "| market | bets |\n|---|---|\n| NBA SPREAD | 5 |",
               "report": {"research_bindings": []}}
    script = ("import json\nimport streamlit as st\n"
              "import app.ui.qualification_progress as panel\n"
              f"panel._from_drive = lambda: json.loads({json.dumps(json.dumps(payload))})\n"
              "panel.render_qualification_progress()\n")
    at = AppTest.from_string(script)
    at.run(timeout=30)
    assert not at.exception and not any("Full readiness report" in m.value for m in at.markdown)
    next(b for b in at.button if b.key == "qualification_progress_drive").click().run(timeout=30)
    assert not at.exception and not at.error
    assert any("Full readiness report" in m.value for m in at.markdown)
    assert any("| NBA SPREAD | 5 |" in m.value for m in at.markdown)
