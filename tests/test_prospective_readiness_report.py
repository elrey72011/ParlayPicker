"""Readiness distinguishes local absence from authenticated remote emptiness."""
from io import BytesIO

from app_core.prospective_readiness_report import load_readiness
from app_core.prospective_remote import sync
from app_core.prospective_validation_plans import freeze_current_validation_plans


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


def test_empty_local_inventory_is_not_a_verified_remote_zero(tmp_path):
    report = load_readiness(tmp_path)
    assert len(report["markets"]) == 12
    assert report["remote"]["status"] == "REMOTE_NOT_RESTORED"
    assert all(row["remote_count_interpretation"] == "LOCAL_COUNTS_ONLY_REMOTE_UNKNOWN"
               and row["canonical_predictions"] == 0
               and row["production_eligible"] is False
               for row in report["markets"])


def test_authenticated_restore_exposes_frozen_plan_ids_without_activation(tmp_path):
    cloud = Cloud()
    source = tmp_path / "source"
    source.mkdir()
    canonical = source / "prospective-evidence.sqlite3"
    freeze_current_validation_plans(canonical, source_commit="a" * 40)
    assert sync(canonical, cloud, "folder")["records_verified"] == 12
    restored = load_readiness(tmp_path / "restored", authenticate=True,
                              client=cloud, folder="folder")
    assert restored["remote"]["status"] == "RESTORED_AND_READBACK_VERIFIED"
    assert len(restored["markets"]) == 12
    assert all(row["validation_plan_id"] and row["remote_evidence_verified"]
               and row["production_eligible"] is False
               and row["recommended_stake"] == 0
               for row in restored["markets"])
