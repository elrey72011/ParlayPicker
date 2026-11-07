"""NCAAB gets its own daily Odds API allowance (owner 2026-11-07); NBA and NHL keep 40."""
from datetime import datetime, timezone

import pytest

from app_core.research_api_budget import Budget, BudgetLimit

PARAMS = {"regions": "us", "markets": "spreads,totals", "oddsFormat": "american"}


class _Response:
    status_code = 200

    def __init__(self, credits):
        self.headers = {"x-requests-last": str(credits)}

    def json(self):
        return []


def _budget(monkeypatch, calls):
    monkeypatch.setattr("app_core.research_api_budget.is_open", lambda at: True)
    now = datetime.now(timezone.utc)
    return Budget({}, lambda: None, clock=lambda: now,
                  get=lambda url, **kwargs: calls.append(url) or _Response(2))


def _odds_url(sport_key):
    return f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"


def test_ncaab_daily_credit_cap_is_120(monkeypatch):
    calls = []
    budget = _budget(monkeypatch, calls)
    for _ in range(60):                                   # 60 requests x 2 credits
        budget.request(_odds_url("basketball_ncaab"), params=PARAMS)
    with pytest.raises(BudgetLimit, match="api_budget:NCAAB:credits:daily"):
        budget.request(_odds_url("basketball_ncaab"), params=PARAMS)
    assert len(calls) == 60
    report = budget.report()
    assert report["by_sport"]["NCAAB"]["provider_units"]["daily"] == 120
    assert report["sport_credit_limits"]["NCAAB"]["daily"] == 120
    assert report["sport_call_limits"]["NCAAB"]["daily"] == 135


@pytest.mark.parametrize("sport,key", [("NBA", "basketball_nba"), ("NHL", "icehockey_nhl")])
def test_nba_and_nhl_keep_the_default_40(monkeypatch, sport, key):
    calls = []
    budget = _budget(monkeypatch, calls)
    for _ in range(20):
        budget.request(_odds_url(key), params=PARAMS)
    with pytest.raises(BudgetLimit, match=f"api_budget:{sport}:credits:daily"):
        budget.request(_odds_url(key), params=PARAMS)
    assert budget.report()["sport_credit_limits"][sport]["daily"] == 40


def test_ncaab_still_counts_against_the_plan_wide_odds_cap(monkeypatch):
    calls = []
    budget = _budget(monkeypatch, calls)
    budget.ledger["ODDS"] = {datetime.now(timezone.utc).date().isoformat(): 199}
    with pytest.raises(BudgetLimit, match="api_budget:ODDS:daily"):
        budget.request(_odds_url("basketball_ncaab"), params=PARAMS)
    assert calls == []
