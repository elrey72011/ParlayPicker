"""V3 feasibility audit cannot promote history or create a validation plan."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import types
import unittest
from unittest import mock

from app_core import football_v3_feasibility as v3
from app_core import prospective_evidence as evidence


class EmptyReadOnlyCloud:
    def __init__(self):
        self.writes = 0
        self.reads = []

    def read_objects(self, *, Prefix):
        self.reads.append(Prefix)
        return iter(())

    def put_object(self, **kwargs):
        self.writes += 1
        raise AssertionError("remote write attempted")


class FeasibilityTests(unittest.TestCase):
    def test_empty_authenticated_restore_is_fail_closed_and_read_only(self):
        with TemporaryDirectory() as name:
            cloud = EmptyReadOnlyCloud()
            # The bundled local Python omits requests; CI uses the real
            # dependency. No HTTP call occurs in this synthetic restore.
            with mock.patch.dict(sys.modules, {"requests": sys.modules.get("requests") or
                                  types.ModuleType("requests")}):
                report = v3.restore_and_audit(Path(name), cloud, "folder")
            assert report["authenticated_remote_verified"] is True
            assert report["result"] == v3.CONCLUSION
            assert report["v3_plans_created"] == 0
            assert cloud.writes == 0 and len(cloud.reads) == 3
            assert len(report["markets"]) == 4
            assert all(row["replay_eligible_games"] == 0 and
                       row["primary_blocker"] == "MISSING_EXACT_SCOPE_MODEL_CALIBRATION" and
                       row["production_eligible"] is False and row["recommended_stake"] == 0
                       for row in report["markets"])
            assert evidence.read_records(Path(name)/"prospective-evidence.sqlite3",
                                         "prospective_validation_plan") == []

    def test_local_zero_never_claims_authenticated_remote_zero(self):
        with TemporaryDirectory() as name:
            report = v3.build_report(name)
            assert report["authenticated_remote_verified"] is False
            assert all(row["evidence_status"] == "LOCAL_ONLY_REMOTE_UNKNOWN" and
                       row["primary_blocker"] == "AUTHENTICATED_RESTORE_REQUIRED"
                       for row in report["markets"])

    def test_cluster_study_deduplicates_events_and_is_reproducible(self):
        start = datetime(2024, 9, 1, tzinfo=timezone.utc)
        rows = []
        for i in range(40):
            stamp = start + timedelta(weeks=i//4, days=i%4)
            row = {"provider_event_id": f"game-{i}",
                   "scheduled_start": stamp.isoformat(),
                   "prediction_timestamp": (stamp-timedelta(hours=2)).isoformat(),
                   "mean_probability": .45 + .01*(i%10),
                   "result_outcome": "WIN" if i%2 else "LOSS",
                   "home_team_id": f"team-{i%8}", "away_team_id": f"team-{(i+1)%8}"}
            rows.append(row)
            rows.append(dict(row, prediction_timestamp=(stamp-timedelta(hours=1)).isoformat(),
                             mean_probability=.99))
        first = v3.cluster_study(rows)
        second = v3.cluster_study(rows)
        assert first == second
        assert first["decided_events"] == 40
        assert first["season_week_clusters"] >= 8
        assert first["teams_repeated"] > 0
        assert first["status"] == "EXPLORATORY_CLUSTER_ESTIMATES_ONLY"
        assert first["brier"]["event_variance"] > 0

    def test_too_few_clusters_cannot_justify_sample_size(self):
        assert v3.cluster_study([])["status"] == "INSUFFICIENT_INDEPENDENT_REPLAY_CLUSTERS"
        assert v3._methods(0)[0]["initial_minimum"] == 9451
        assert all(x["initial_minimum"] is None for x in v3._methods(0)[1:])

    def test_readonly_wrapper_rejects_upload(self):
        cloud = EmptyReadOnlyCloud()
        with self.assertRaisesRegex(RuntimeError, "remote_write_prohibited"):
            v3.ReadOnlyEvidenceStore(cloud).put_object(Key="anything")
        assert cloud.writes == 0

    def test_model_calibration_binding_requires_exact_artifacts_and_chronology(self):
        model = {"model_id": "nfl-spread", "artifact_hash": "model-hash",
                 "feature_version": "v1", "training_cutoff": "2025-06-01T00:00:00Z",
                 "available_at": "2025-06-02T00:00:00Z"}
        calibration = {"calibration_id": "cal", "model_id": "nfl-spread",
                       "artifact_hash": "cal-hash", "fit_cutoff": "2025-06-03T00:00:00Z",
                       "available_at": "2025-06-04T00:00:00Z"}
        assert v3._bound_models([model], [calibration]) == [
            {"model_id": "nfl-spread", "calibration_id": "cal"}]
        assert not v3._bound_models([model], [dict(calibration, model_id="ncaaf-spread")])
        assert not v3._bound_models([model], [dict(calibration, fit_cutoff="2025-06-01T00:00:00Z")])
        assert not v3._bound_models([model], [dict(calibration, artifact_hash="")])

    def test_v3_replay_requires_proven_probability_settlement_and_source(self):
        row = {"mean_probability": .6, "result_outcome": "WIN",
               "evidence_hash": "sha", "source_record_id": "record"}
        assert v3._legal_replay_blockers(row) == []
        assert "REPLAY_PROBABILITY_UNAVAILABLE" in v3._legal_replay_blockers(
            dict(row, mean_probability=None))
        assert "REPLAY_SETTLEMENT_UNAVAILABLE" in v3._legal_replay_blockers(
            dict(row, result_outcome="NEEDS_REVIEW"))
        assert "REPLAY_SOURCE_HASH_UNAVAILABLE" in v3._legal_replay_blockers(
            dict(row, evidence_hash=None))

    def test_replay_manifest_counts_first_event_once_across_books_and_reprices(self):
        early = {"provider_event_id": "game", "prediction_timestamp": "2025-09-01T12:00:00Z",
                 "sportsbook": "book-a", "mean_probability": .55, "result_outcome": "WIN"}
        late = dict(early, prediction_timestamp="2025-09-01T13:00:00Z",
                    sportsbook="book-b", mean_probability=.70)
        manifest = v3._replay_manifest([late, early])
        assert len(manifest) == 1
        assert manifest[0]["sportsbook"] == "book-a"
        assert manifest[0]["role"] == "UNASSIGNED_HISTORICAL_REPLAY"


if __name__ == "__main__":
    unittest.main()
