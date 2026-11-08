"""Drive uploads rejected with a definite 429/5xx are retried after a readback check (owner 2026-11-08)."""
import pytest
import requests

from app_core import evidence_drive
from app_core.evidence_drive import DriveStore
from tests.test_evidence_drive import DriveSession, Response


class Rejected(Response):
    def __init__(self, status, reason=None):
        super().__init__({"error": {"code": status, "errors": [{"reason": reason}] if reason else []}})
        self.status_code = status

    def raise_for_status(self):
        raise requests.HTTPError(f"{self.status_code} rejected", response=self)


class FlakySession(DriveSession):
    """Rejects the first ``rejections`` uploads with ``status``; optionally creates the file anyway."""

    def __init__(self, status, rejections, *, created_anyway=False, reason=None):
        super().__init__()
        self.status, self.rejections, self.created_anyway, self.posts = status, rejections, created_anyway, 0
        self.reason = reason

    def post(self, url, params, headers, data, timeout):
        self.posts += 1
        if self.posts <= self.rejections:
            if self.created_anyway:
                super().post(url, params, headers, data, timeout)
            return Rejected(self.status, self.reason)
        return super().post(url, params, headers, data, timeout)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(evidence_drive.time, "sleep", lambda seconds: None)


def test_a_rate_limited_upload_is_retried_and_lands_once():
    session = FlakySession(429, rejections=1)
    store = DriveStore("folder", session=session)
    store.put_object(Key="prefix/a.json", Body=b'{"x":1}', IfNoneMatch="*")
    assert session.posts == 2 and [f["name"] for f in session.files] == ["prefix/a.json"]
    assert store.get_object(Key="prefix/a.json")["Body"].read() == b'{"x":1}'


def test_a_rejected_upload_that_landed_anyway_is_not_sent_twice():
    session = FlakySession(503, rejections=1, created_anyway=True)
    store = DriveStore("folder", session=session)
    store.put_object(Key="prefix/a.json", Body=b'{"x":1}', IfNoneMatch="*")
    assert session.posts == 1 and len(session.files) == 1                 # confirmed by readback


def test_retries_are_bounded_and_other_errors_are_not_retried():
    session = FlakySession(503, rejections=10)
    with pytest.raises(requests.HTTPError):
        DriveStore("folder", session=session).put_object(Key="prefix/a.json", Body=b"{}", IfNoneMatch="*")
    assert session.posts == evidence_drive.UPLOAD_ATTEMPTS and session.files == []
    forbidden = FlakySession(403, rejections=10)
    with pytest.raises(requests.HTTPError):
        DriveStore("folder", session=forbidden).put_object(Key="prefix/a.json", Body=b"{}", IfNoneMatch="*")
    assert forbidden.posts == 1                                          # a 403 is final


def test_a_drive_rate_limit_403_is_retried():
    session = FlakySession(403, rejections=1, reason="userRateLimitExceeded")
    DriveStore("folder", session=session).put_object(Key="prefix/a.json", Body=b"{}", IfNoneMatch="*")
    assert session.posts == 2 and len(session.files) == 1
