"""Fixed-methodology football V2 invariants; fixtures never represent live evidence."""
import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app_core import prospective_evidence as evidence
from app_core import prospective_validation_plans as v1
from app_core import football_validation_v2 as v2


class FootballValidationV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "canonical.sqlite3"
        self.old_clock = evidence._clock
        evidence._clock = lambda: datetime(2026, 9, 24, 12, tzinfo=timezone.utc)
        self.addCleanup(lambda: setattr(evidence, "_clock", self.old_clock))

    def freeze(self):
        original = v1.freeze_current_validation_plans(self.path, source_commit="a" * 40)
        new = v2.freeze_plans(self.path, source_commit="a" * 40)
        return original, new

    def test_v1_unchanged_and_four_immutable_exact_scope_plans(self):
        original, new = self.freeze()
        assert len(new) == 4
        assert {(x["sport"], x["market_family"]) for x in new} == set(v2.SCOPES)
        again = v2.freeze_plans(self.path, source_commit="b" * 40)
        assert again == new
        for old in original:
            actual = evidence.load_record(self.path, "prospective_validation_plan", old["validation_plan_id"])
            assert actual["artifact_hash"] == old["artifact_hash"]
        assert len(evidence.read_records(self.path, "prospective_validation_plan")) == 16
        assert all(x["production_eligible"] is False and x["recommended_stake"] == 0 for x in new)
        assert all(datetime.fromisoformat(x["validation_start"]) >
                   datetime.fromisoformat(x["frozen_at"]) for x in new)

    def test_deterministic_minimum_and_all_scopes_separate(self):
        n = v2.calculated_minimum()["effective_decided_events"]
        assert n == v2.calculated_minimum()["effective_decided_events"]
        assert all(v2.uncertainty_radius(n, metric=k) <= value for k, value in v2.PRECISION.items())
        assert any(v2.uncertainty_radius(n - 1, metric=k) > value for k, value in v2.PRECISION.items())
        self.freeze()
        rows = v2.timeline(self.path)
        assert len(rows) == 4 and all(r["required_count"] == n for r in rows)
        assert all(r["effective_count"] == 0 and r["remaining_count"] == n for r in rows)
        assert all(r["status"] == "UNVALIDATED" and r["production_eligible"] is False for r in rows)

    def test_replay_rejects_unverified_price_future_features_and_result_leakage(self):
        start = datetime(2025, 9, 1, tzinfo=timezone.utc)
        row = {"sport": "NFL", "market_family": "SPREAD", "provider_event_id": "event-1",
               "home_team_id": "h", "away_team_id": "a", "game_start_utc": start.isoformat(),
               "prediction_generated_at": (start - timedelta(hours=2)).isoformat(),
               "odds_recorded_at": (start - timedelta(hours=3)).isoformat(), "line": -3.5,
               "sportsbook": "book", "decimal_odds": 1.91, "quote_verified": True,
               "features_generated_at": (start - timedelta(hours=4)).isoformat(),
               "feature_snapshot_id": "features", "feature_replay_verified": True,
               "model_id": "model", "model_replay_verified": True,
               "model_artifact_hash": "abc", "runtime_hash": "runtime",
               "model_available_at": (start - timedelta(days=2)).isoformat(),
               "model_trained_through": (start - timedelta(days=3)).isoformat(),
               "training_inputs_latest_available_at": (start - timedelta(days=4)).isoformat(),
               "result_available_at": (start + timedelta(hours=4)).isoformat(),
               "result_verified": True}
        good = v2.historical_replay_eligible(row)
        assert good["eligible"] and good["prospective_count"] == 0
        assert good["classification"] == "REPLAYABLE_HISTORICAL"
        invalid = copy.deepcopy(row)
        invalid.update(quote_verified=False,
                       features_generated_at=(start + timedelta(hours=1)).isoformat(),
                       result_available_at=(start - timedelta(hours=3)).isoformat(),
                       home_team_id=None)
        blockers = set(v2.historical_replay_eligible(invalid)["blockers"])
        assert {"HISTORICAL_PRICE_UNVERIFIED", "FEATURE_ASOF_UNAVAILABLE",
                "RESULT_LEAKAGE_RISK", "EVENT_IDENTITY_AMBIGUOUS"} <= blockers
        invalid = copy.deepcopy(row)
        invalid["odds_recorded_at"] = (start + timedelta(minutes=1)).isoformat()
        assert {"QUOTE_AFTER_START", "QUOTE_AFTER_PREDICTION"} <= set(
            v2.historical_replay_eligible(invalid)["blockers"])

    def test_conservative_probability_never_above_mean(self):
        interval = {"method_version": v2.POLICY_VERSION, "calibration_id": "calibration",
                    "lower": 0.57, "upper": 0.67}
        first = v2.conservative_probability(0.62, interval)
        assert first == v2.conservative_probability(0.62, interval)
        assert first["conservative_probability"] <= first["mean_probability"]
        assert first["uncertainty_interval"][0] == first["conservative_probability"]
        with self.assertRaises(ValueError):
            v2.conservative_probability(0.62, {})

    def test_positive_paper_roi_cannot_substitute_for_proper_scores_or_close(self):
        _, new = self.freeze()
        plan = evidence.load_record(self.path, "prospective_validation_plan", new[0]["validation_plan_id"])
        method = __import__("json").loads(plan["payload"])
        n = plan["minimum_independent_sample"]
        cohort = {"effective_observations": n, "unique_events": n,
                  "missing_provenance_count": 0, "missing_stable_identity_count": 0,
                  "unsupported_probability_semantics_count": 0,
                  "wrong_model_or_calibration_count": 0,
                  "outcomes": {"PENDING": 0, "NEEDS_REVIEW": 0},
                  "verified_entry_coverage": 1.0, "comparable_close_coverage": 0.0,
                  "brier": 0.25, "log_loss": 0.69, "calibration_error": 0.09,
                  "coverage": 1.0, "paper_roi": 1.0}
        decision = v2.evaluate(plan, method, cohort, cohort,
                               datetime(2027, 4, 2, tzinfo=timezone.utc))
        assert decision["status"] == "UNVALIDATED"
        assert "MISSING_MODEL" in decision["blockers"]
        assert "MISSING_CALIBRATION" in decision["blockers"]
        assert "CONFIRMATION_CERTIFIED_CLOSE_INCOMPLETE" in decision["blockers"]
        assert "CONFIRMATION_BRIER_UPPER_BOUND_NOT_MET" in decision["blockers"]
        assert decision["standard_status"] == "BLOCKED"

    def test_timeline_and_inventory_do_not_write_or_activate(self):
        self.freeze()
        before = len(evidence.read_records(self.path, "prospective_validation_plan"))
        timeline = v2.timeline(self.path)
        inventory = v2.evidence_inventory(self.path.parent)
        assert len(timeline) == len(inventory) == 4
        assert all(r["coverage_audit"]["full_slate_verified"] is False for r in inventory)
        assert len(evidence.read_records(self.path, "prospective_validation_plan")) == before

    def test_missing_exact_price_does_not_qualify_even_with_good_scores(self):
        _, new = self.freeze()
        plan = evidence.load_record(self.path, "prospective_validation_plan", new[0]["validation_plan_id"])
        method = __import__("json").loads(plan["payload"])
        n = plan["minimum_independent_sample"]
        cohort = {"effective_observations": n, "unique_events": n,
                  "missing_provenance_count": 0, "missing_stable_identity_count": 0,
                  "unsupported_probability_semantics_count": 0,
                  "wrong_model_or_calibration_count": 0,
                  "outcomes": {"PENDING": 0, "NEEDS_REVIEW": 0},
                  "verified_entry_coverage": 0.99, "comparable_close_coverage": 1.0,
                  "brier": 0.0, "log_loss": 0.0, "calibration_error": 0.0,
                  "coverage": 1.0, "paper_roi": 1.0}
        decision = v2.evaluate(plan, method, cohort, cohort,
                               datetime(2028, 4, 2, tzinfo=timezone.utc))
        assert "CONFIRMATION_EXACT_PRICE_INCOMPLETE" in decision["blockers"]
        assert "HOLDOUT_EXACT_PRICE_INCOMPLETE" in decision["standard_blockers"]
        assert "FULL_SLATE_COVERAGE_UNVERIFIED" in decision["blockers"]

    def test_no_historical_or_opposing_side_count_in_prospective_timeline(self):
        self.freeze()
        before = v2.timeline(self.path)
        # A read-only historical replay verdict has no path to the canonical
        # V2 cohort and cannot change the timeline's prospective count.
        v2.historical_replay_eligible({"sport": "NFL", "market_family": "SPREAD",
                                        "provider_event_id": "old-game"})
        after = v2.timeline(self.path)
        assert before == after
        assert all(r["prospective_confirmation_count"] == 0 for r in after)


if __name__ == "__main__":
    unittest.main()
