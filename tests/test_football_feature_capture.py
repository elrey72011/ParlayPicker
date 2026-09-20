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
