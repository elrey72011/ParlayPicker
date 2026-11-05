"""ET conversion of provider start times must work for current dates (zoneinfo, not dateutil)."""
from datetime import timedelta

import pandas as pd

from app_core import odds_api


def test_iso8601_to_est_has_a_real_offset_across_dst():
    evening = odds_api.iso8601_to_est("2026-11-06T00:00:00Z")      # 7 PM EST
    assert evening.utcoffset() == timedelta(hours=-5) and evening.hour == 19
    summer = odds_api.iso8601_to_est("2026-07-04T23:00:00Z")       # 7 PM EDT
    assert summer.utcoffset() == timedelta(hours=-4) and summer.hour == 19
    assert str(pd.Timestamp(evening).tz_convert("UTC")) == "2026-11-06 00:00:00+00:00"


def test_eastern_prefers_zoneinfo():
    from zoneinfo import ZoneInfo
    assert isinstance(odds_api._eastern(), ZoneInfo)
