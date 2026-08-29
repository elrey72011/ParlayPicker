import pandas as pd

from app_core.kalshi_integrator import (
    _event_date_from_ticker,
    _event_match_score,
    _is_within_72h,
)


SLATE_DATE = pd.Timestamp("2026-08-29T00:00:00Z")


def test_event_ticker_date_overrides_metadata_update_time() -> None:
    event = {
        "event_ticker": "KXNCAAFSPREAD-26AUG29UNCTCU",
        "title": "North Carolina vs TCU: Spread",
        "last_updated_ts": "2026-09-10T18:00:00Z",
    }

    assert _event_date_from_ticker(event) == SLATE_DATE
    assert _is_within_72h(event, SLATE_DATE)


def test_future_ticker_is_rejected_even_when_recently_updated() -> None:
    event = {
        "event_ticker": "KXNCAAFSPREAD-26SEP04FRESUSC",
        "title": "Fresno St. vs USC: Spread",
        "last_updated_ts": "2026-08-29T12:00:00Z",
    }

    assert not _is_within_72h(event, SLATE_DATE)


def test_scheduled_time_is_fallback_when_ticker_has_no_date() -> None:
    event = {
        "event_ticker": "CUSTOM-EVENT",
        "close_time": "2026-08-29T23:30:00Z",
        "last_updated_ts": "2026-09-20T12:00:00Z",
    }

    assert _event_date_from_ticker(event) is None
    assert _is_within_72h(event, SLATE_DATE)


def test_date_filter_keeps_exact_ncaaf_match_and_removes_future_same_team() -> None:
    events = [
        {
            "event_ticker": "KXNCAAFSPREAD-26AUG29SJSUUSC",
            "title": "San Jose St. vs USC: Spread",
            "sub_title": "SJSU vs USC (Aug 29)",
            "last_updated_ts": "2026-09-10T12:00:00Z",
        },
        {
            "event_ticker": "KXNCAAFSPREAD-26SEP04FRESUSC",
            "title": "Fresno St. vs USC: Spread",
            "sub_title": "FRES vs USC (Sep 4)",
            "last_updated_ts": "2026-08-29T12:00:00Z",
        },
    ]

    dated = [event for event in events if _is_within_72h(event, SLATE_DATE)]

    assert [event["event_ticker"] for event in dated] == ["KXNCAAFSPREAD-26AUG29SJSUUSC"]
    assert _event_match_score(dated[0], "Usc", "San Jose State", "NCAAF", "26AUG29") >= 25
