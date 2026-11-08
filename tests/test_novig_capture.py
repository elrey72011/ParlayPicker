"""Novig in NBA/NCAAB Pinnacle captures, behind PARLAYPICKER_NOVIG_SPORTS (owner 2026-11-08)."""
from datetime import datetime, timezone

import pytest

from app_core import research_api_budget as rab
from app_core.public_quote_policy import canonical_book_label
from app_core.pinnacle_model import PLAN_BOOKS


def test_novig_is_added_only_where_switched_on_and_never_for_ncaaf():
    off = {}.get
    assert rab.capture_bookmakers("NCAAB", off) == rab.PINNACLE_BOOKMAKERS               # default: unchanged
    on = {rab.NOVIG_ENV: "nba, ncaab, ncaaf"}.get
    assert rab.novig_sports(on) == {"NBA", "NCAAB"}                                       # NCAAF is never added
    assert rab.capture_bookmakers("NBA", on) == rab.PINNACLE_BOOKMAKERS + ",novig"
    assert rab.capture_bookmakers("NCAAF", on) == rab.PINNACLE_BOOKMAKERS
    books = rab.PINNACLE_NOVIG_BOOKMAKERS.split(",")
    assert len(books) == 13 and len(set(books)) == 13                                     # still 2 region equivalents
    assert canonical_book_label("novig") in PLAN_BOOKS                                    # a plan entry book


def _budget():
    state = {}
    return rab.Budget(state, lambda: None, clock=lambda: datetime(2026, 11, 8, 23, tzinfo=timezone.utc))


@pytest.mark.parametrize("books", [rab.PINNACLE_BOOKMAKERS, rab.PINNACLE_NOVIG_BOOKMAKERS])
def test_budget_accepts_the_novig_request_at_the_same_cost(monkeypatch, books):
    monkeypatch.setenv(rab.NOVIG_ENV, "NCAAB")
    budget = _budget()
    params = {"bookmakers": books, "markets": "spreads,totals", "oddsFormat": "american", "eventIds": "g1"}
    calls = []

    class Response:
        status_code = 200
        headers = {"x-requests-last": "4"}

        def json(self):
            return []
    budget.get = lambda url, **kwargs: calls.append(kwargs["params"]) or Response()
    budget.request("https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds", params=dict(params, apiKey="k"))
    assert calls and calls[0]["bookmakers"] == books
