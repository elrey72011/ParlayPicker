"""Canonical research evidence survives an authenticated remote restore."""
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pytest

from app_core import prospective_evidence as evidence
from app_core import prospective_remote
from app_core import nfl_market_store as nfl
from app_core.prospective_reconciliation import reconcile_sport


class Cloud:
    def __init__(self):
        self.objects = {}

    def read_objects(self, *, Prefix):
        return [(key, value) for key, value in self.objects.items() if key.startswith(Prefix)]

    def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
        assert IfNoneMatch == "*"
        assert Key not in self.objects
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        return {"Body": BytesIO(self.objects[Key])}


def event(path, *, home="Home"):
    now = datetime.now(timezone.utc)
    return evidence.insert_event(path, {
        "event_id": "event:1", "sport": "NBA", "game_id": "1",
        "provider_namespace": "THE_ODDS_API", "provider_event_id": "1",
        "home_team": home, "away_team": "Away", "home_team_id": "h",
        "away_team_id": "a", "scheduled_start": (now + timedelta(days=1)).isoformat(),
        "observed_at": now.isoformat(), "source_id": "source:1",
        "raw_source": {"provider_event_id": "1", "home": home},
    })


def test_remote_restore_is_idempotent_and_readback_verified(tmp_path):
    cloud = Cloud()
    first, second = tmp_path / "first.sqlite3", tmp_path / "second.sqlite3"
    event(first)
    result = prospective_remote.sync(first, cloud, "folder")
    assert result["new_records_verified"] == 1
    assert result["records_verified"] == 1
    assert prospective_remote.sync(second, cloud, "folder")["remote_records_read"] == 1
    repeated = prospective_remote.sync(second, cloud, "folder")
    assert repeated["remote_records_read"] == 1
    assert repeated["records_restored"] == 0
    assert len(evidence.read_records(second, "prospective_event")) == 1
    assert len(cloud.objects) == 1


def test_remote_detects_immutable_identity_conflict(tmp_path):
    cloud = Cloud()
    first, second = tmp_path / "first.sqlite3", tmp_path / "second.sqlite3"
    event(first)
    prospective_remote.sync(first, cloud, "folder")
    event(second, home="Different Home")
    with pytest.raises(ValueError, match="canonical_remote_local_conflict"):
        prospective_remote.sync(second, cloud, "folder")


def test_remote_rejects_tampered_payload(tmp_path):
    cloud = Cloud()
    first = tmp_path / "first.sqlite3"
    event(first)
    prospective_remote.sync(first, cloud, "folder")
    key = next(iter(cloud.objects))
    cloud.objects[key] = cloud.objects[key].replace(b"Home", b"H0me")
    with pytest.raises(ValueError, match="canonical_remote_integrity_conflict|canonical_remote_evidence_hash_conflict"):
        prospective_remote.sync(tmp_path / "second.sqlite3", cloud, "folder")


def test_interrupted_fact_upload_requires_independent_source_recovery(tmp_path):
    cloud = Cloud()
    source = tmp_path / "nfl-market.sqlite3"
    first, second = tmp_path / "first.sqlite3", tmp_path / "second.sqlite3"
    now = datetime.now(timezone.utc)
    nfl.insert({"schema": 1, "kind": "capture", "created_at": now.isoformat(),
                "data": {"sport": "NFL", "protocol": "nfl-market-v1", "events": [{
                    "event_id": "orphan-1", "home": "Home", "away": "Away",
                    "start": (now + timedelta(hours=2)).isoformat(),
                    "quotes": [{"market": "totals", "selection": "Over", "point": 42.5,
                                "odds_american": -110, "book": "book-a",
                                "recorded_at": now.isoformat()}]}]}}, source)
    reconcile_sport("NFL", first, source)
    prospective_remote.sync(first, cloud, "folder")
    parent = next(key for key in cloud.objects if "prospective_reconciled_source/" in key)
    del cloud.objects[parent]  # Simulate a prior run that published the fact first.

    with pytest.raises(prospective_remote.CanonicalMissingDependencies) as error:
        prospective_remote.sync(second, cloud, "folder")
    assert error.value.sports == {"NFL"}
    assert not cloud.objects.get(parent)

    reconcile_sport("NFL", second, source)  # Recovered from independent source backup.
    restored = prospective_remote.sync(second, cloud, "folder")
    assert restored["new_records_verified"] == 1
    assert parent in cloud.objects
    assert prospective_remote.sync(tmp_path / "third.sqlite3", cloud, "folder")["records_restored"] == 2
