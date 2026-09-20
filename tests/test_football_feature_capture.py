import json
from datetime import datetime, timezone
import pandas as pd
from app_core.football_feature_capture import capture
from core.exposure_ledger import digest


def test_capture_records_actual_values_without_authority():
    frame = pd.DataFrame([dict(league="NFL",commence_time="2026-09-21T12:00:00Z",
        feature_rating=1.2,feature_missing=float("nan"),stats_source="fallback")])
    result = capture(frame, datetime(2026,9,20,tzinfo=timezone.utc))
    receipt = json.loads(result.iloc[0].football_feature_receipt)
    assert receipt["sha256"] == digest(receipt["payload"])
    assert receipt["payload"]["features"] == {"feature_rating":1.2}
    assert receipt["payload"]["stats_source"] == "fallback"
    assert receipt["payload"]["historical_availability_verified"] is False
    assert receipt["payload"]["training_authorized"] is False
    assert "features_generated_at" not in result
    assert "football_feature_receipt" not in frame


def test_late_missing_and_other_sports_cannot_create_receipt():
    frame = pd.DataFrame([dict(league=s,commence_time=t,feature_a=1) for s,t in
        [("NFL","2026-09-19T12:00:00Z"),("NCAAF",None),("MLB","2026-09-21T12:00:00Z")]])
    result = capture(frame, datetime(2026,9,20,tzinfo=timezone.utc))
    assert result.football_feature_receipt.isna().all()


def test_expanded_candidate_raw_kickoff_is_captured_before_start():
    frame = pd.DataFrame([dict(league="NFL", commence_time_raw="2026-09-20T20:25:00Z",
                              feature_rating=1.2, home_team_id="espn:nfl:6", away_team_id="espn:nfl:28")])
    result = capture(frame, datetime(2026,9,20,19,tzinfo=timezone.utc))
    receipt = json.loads(result.iloc[0].football_feature_receipt)
    assert result.iloc[0].football_feature_capture_status == "OBSERVED_RESEARCH_INPUTS"
    assert receipt["payload"]["game_start_utc"] == "2026-09-20T20:25:00+00:00"
    assert receipt["sha256"] == digest(receipt["payload"])
    assert receipt["payload"]["home_team_id"] == "espn:nfl:6"


def test_raw_kickoff_at_start_or_invalid_still_rejected():
    frame = pd.DataFrame([dict(league="NCAAF", commence_time_raw=value, feature_a=1)
                          for value in ["2026-09-20T19:00:00Z", "bad", "2026-09-20T20:00:00"]])
    result = capture(frame, datetime(2026,9,20,19,tzinfo=timezone.utc))
    assert result.football_feature_receipt.isna().all()
    assert result.football_feature_capture_status.eq("NOT_VERIFIED_PREGAME").all()


def test_raw_field_does_not_override_existing_started_kickoff():
    frame = pd.DataFrame([dict(league="NFL",game_start_utc="2026-09-20T17:00:00Z",
                              commence_time_raw="2026-09-20T20:25:00Z",feature_a=1)])
    result = capture(frame, datetime(2026,9,20,19,tzinfo=timezone.utc))
    assert result.football_feature_receipt.isna().all()
