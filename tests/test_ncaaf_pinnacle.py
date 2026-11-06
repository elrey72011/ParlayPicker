"""D4 option B: Pinnacle as the NCAAF fair-price reference. SYNTHETIC data; no network."""
from datetime import datetime, timezone
import json

import pytest

from app_core import ncaaf_prospective as ncaaf
from app_core import research_api_budget as rab
from app_core import research_model_recovery as recovery

NOW = datetime(2026, 11, 6, 20, tzinfo=timezone.utc)
URL = "https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"


def test_the_reviewed_runtime_is_the_code_on_disk():
    assert ncaaf.runtime_hash() == recovery.PINNACLE_RUNTIME


def _model(runtime):
    return {"kind": "model", "id": "frozen", "data": {"artifact": {"protocol": {"production_eligible": False}},
            "artifact_hash": recovery.ARTIFACT, "runtime_hash": runtime}}


@pytest.mark.parametrize("source", ["ALIAS_RUNTIME", "TARGET_RUNTIME", "SOURCE_RUNTIME"])
def test_frozen_cohort_carries_over_to_the_pinnacle_runtime(monkeypatch, tmp_path, source):
    monkeypatch.setattr(recovery.ncaaf, "runtime_hash", lambda: recovery.PINNACLE_RUNTIME)
    monkeypatch.setattr(recovery.ncaaf, "digest", lambda _: recovery.ARTIFACT)
    new = recovery.recover_ncaaf(_model(getattr(recovery, source)), tmp_path / "cohorts.sqlite3")
    assert new["data"]["runtime_hash"] == recovery.PINNACLE_RUNTIME
    assert "pinnacle" in new["data"]["recovery"]["reason"] and new["data"]["recovery"]["production_eligible"] is False


def test_pinnacle_runtime_never_goes_back(monkeypatch, tmp_path):
    monkeypatch.setattr(recovery.ncaaf, "runtime_hash", lambda: recovery.ALIAS_RUNTIME)
    monkeypatch.setattr(recovery.ncaaf, "digest", lambda _: recovery.ARTIFACT)
    with pytest.raises(ValueError, match="stale_frozen_model"):
        recovery.recover_ncaaf(_model(recovery.PINNACLE_RUNTIME), tmp_path / "absent.sqlite3")


class Response:
    def __init__(self, cost):
        self.headers = {"x-requests-last": str(cost)}


def test_ncaaf_pinnacle_request_charges_three_extra_credits_to_the_pinnacle_allowance():
    budget = rab.Budget({}, lambda: None, clock=lambda: NOW, get=lambda *a, **kw: Response(6))
    budget.request(URL, params={"bookmakers": rab.PINNACLE_BOOKMAKERS, "markets": "h2h,spreads,totals",
                                "oddsFormat": "american"})
    assert budget.usage("ODDS")["daily"] == 6
    assert budget.usage("NCAAF", budget.sport_ledger)["daily"] == 3
    assert budget.usage("PINNACLE", budget.pinnacle_ledger)["daily"] == 3
    with pytest.raises(ValueError):          # NFL is not in scope
        budget.request("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
                       params={"bookmakers": rab.PINNACLE_BOOKMAKERS, "markets": "h2h,spreads,totals"})


def test_fetch_uses_pinnacle_only_when_switched_on_and_falls_back_when_spent(monkeypatch):
    calls = []

    def fake_fetch(url, token, params, *, cfbd=True, get=None):
        calls.append(params)
        if "bookmakers" in params and spent[0]:
            raise rab.BudgetLimit("api_budget:PINNACLE:daily")
        return []

    spent = [False]
    monkeypatch.setattr(ncaaf, "fetch", fake_fetch)
    monkeypatch.delenv(rab.PINNACLE_ENV, raising=False)
    ncaaf._fetch_odds("key", None)
    assert calls[-1]["regions"] == "us"
    monkeypatch.setenv(rab.PINNACLE_ENV, "NCAAF")
    ncaaf._fetch_odds("key", None)
    assert calls[-1]["bookmakers"] == rab.PINNACLE_BOOKMAKERS and "regions" not in calls[-1]
    spent[0] = True
    calls.clear()
    ncaaf._fetch_odds("key", None)
    assert "bookmakers" in calls[0] and calls[1]["regions"] == "us"


def test_quote_candidates_never_select_pinnacle(monkeypatch):
    from app_core import prediction_evidence
    stamp = NOW.isoformat()
    quotes = [{"book": b, "market_type": t, "point": p, "price": price, "recorded_at": stamp}
              for b, price in (("pinnacle", 150), ("draftkings", -110))
              for t, p in (("spread_home", -3.5), ("spread_away", 3.5))]
    monkeypatch.setattr(prediction_evidence, "provider_quotes", lambda event: json.dumps(quotes))
    monkeypatch.setattr(ncaaf, "centers", lambda model, rows, target: [7.0])
    monkeypatch.setattr(ncaaf, "probabilities", lambda mu, sigma, threshold, total=False: {"over": 0.6, "under": 0.4, "push": 0.0})
    model = {"margin": {"sigma": 14.0}, "total": {"sigma": 14.0}}
    result = ncaaf.quote_candidates({}, model, {}, NOW)
    assert result and {q["book"] for q in result} == {"draftkings"}
