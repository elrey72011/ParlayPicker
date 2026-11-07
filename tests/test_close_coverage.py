"""Comparable certified-close coverage by entry timing (2026-11-07)."""
from datetime import datetime, timezone

from app_core.close_coverage import coverage

NOW = datetime(2026, 11, 7, 12, tzinfo=timezone.utc)
EVENT = {"event_id": "e1", "scheduled_start": "2026-11-07T01:00:00Z"}


def _quote(minutes_before, line, book="draftkings", verified=1):
    from datetime import timedelta
    taken = datetime(2026, 11, 7, 1, tzinfo=timezone.utc) - timedelta(minutes=minutes_before)
    return {"event_id": "e1", "market_family": "SPREAD", "selection": "Home", "sportsbook": book,
            "line": line, "quote_timestamp": taken.isoformat(), "quote_verified": verified}


def test_late_quotes_match_the_close_line_early_ones_do_not():
    closes = [{"event_id": "e1", "market_family": "SPREAD", "selection": "Home", "sportsbook": "draftkings",
               "line": -8.0, "close_verified": 1}]
    quotes = [_quote(10, -8.0), _quote(20, -8.0), _quote(400, -7.5), _quote(200, -7.5),
              _quote(10, -8.0, book="pinnacle"), _quote(10, -8.0, book="fanduel")]
    out = coverage([EVENT], quotes, closes, now=NOW)
    assert out["under 30 min"] == {"quotes": 3, "with_close_fetched": 2, "comparable": 2, "share": 1.0}
    assert out["over 6 h"]["share"] == 0.0 and out["3-6 h"]["share"] == 0.0
    assert out["30-60 min"]["share"] is None


def test_unstarted_games_and_unverified_quotes_are_ignored():
    future = {"event_id": "e1", "scheduled_start": "2026-11-08T01:00:00Z"}
    assert coverage([future], [_quote(10, -8.0)], [], now=NOW)["under 30 min"]["quotes"] == 0
    assert coverage([EVENT], [_quote(10, -8.0, verified=0)], [], now=NOW)["under 30 min"]["quotes"] == 0
