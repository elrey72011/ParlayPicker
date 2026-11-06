"""D4 option A: Pinnacle as the fair-price reference for NBA and NCAAB spreads. SYNTHETIC data; no network."""
from datetime import datetime, timedelta, timezone

import pytest

from app_core import odds_research_adapter as ora
from app_core import price_edge_screen as pes
from app_core import research_api_budget as rab

NOW = datetime(2026, 11, 6, 20, tzinfo=timezone.utc)   # inside the operating window
NCAAB_ODDS = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"
PIN = {"bookmakers": rab.PINNACLE_BOOKMAKERS, "markets": "spreads,totals", "oddsFormat": "american"}


class Response:
    def __init__(self, cost):
        self.headers = {"x-requests-last": str(cost)}


def test_switch_is_off_by_default_and_limited_to_nba_and_ncaab():
    assert rab.pinnacle_sports(lambda *_: "") == set()
    assert rab.pinnacle_sports(lambda *_: "nba, NCAAB,NHL") == {"NBA", "NCAAB"}
    assert len(rab.PINNACLE_BOOKMAKERS.split(",")) == 12          # 11-20 books = 2 region equivalents


def test_pinnacle_request_charges_the_extra_credits_to_their_own_allowance():
    state = {}
    budget = rab.Budget(state, lambda: None, clock=lambda: NOW, get=lambda *a, **kw: Response(4))
    budget.request(NCAAB_ODDS, params=PIN)
    assert budget.usage("ODDS")["daily"] == 4                                  # whole request, plan-wide
    assert budget.usage("NCAAB", budget.sport_ledger)["daily"] == 2            # capture allowance unchanged
    assert budget.usage("PINNACLE", budget.pinnacle_ledger)["daily"] == 2
    assert not state.get("api_budget_cost_mismatch")


def test_spent_allowance_refuses_before_any_reservation():
    state = {"api_budget_pinnacle_v1": {"PINNACLE": {NOW.date().isoformat(): rab.PINNACLE_LIMITS["daily"]}}}
    calls = []
    budget = rab.Budget(state, lambda: None, clock=lambda: NOW, get=lambda *a, **kw: calls.append(1))
    with pytest.raises(rab.BudgetLimit, match="PINNACLE"):
        budget.request(NCAAB_ODDS, params=PIN)
    assert not calls and budget.usage("ODDS")["daily"] == 0


def test_only_the_exact_book_list_and_sports_are_budgeted():
    budget = rab.Budget({}, lambda: None, clock=lambda: NOW, get=lambda *a, **kw: Response(4))
    with pytest.raises(ValueError):
        budget.request(NCAAB_ODDS, params={**PIN, "bookmakers": "pinnacle"})
    with pytest.raises(ValueError):
        budget.request(NCAAB_ODDS, params={**PIN, "regions": "us"})
    with pytest.raises(ValueError):
        budget.request("https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds", params=PIN)


def _upcoming():
    now = ora.utcnow()
    return {"g1": {"id": "g1", "commence_time": (now + timedelta(hours=1)).isoformat()}}


def _capture(monkeypatch, tmp_path, fail_pinnacle=False):
    calls = []

    def fake_fetch(sport, endpoint, key, request_get, *, params=None):
        calls.append((endpoint, params))
        if endpoint == "odds" and "bookmakers" in (params or {}) and fail_pinnacle:
            raise rab.BudgetLimit("api_budget:PINNACLE:daily")
        return {} if endpoint == "participants" else []

    monkeypatch.setattr(ora, "fetch", fake_fetch)
    monkeypatch.setattr(ora, "participant_ids", lambda response: {})
    result = ora.OddsResearchAdapter("NCAAB").capture_pregame(tmp_path / "SYNTHETIC.sqlite3", {}, "key", None,
                                                              lambda: None, upcoming=_upcoming())
    return result, [p for e, p in calls if e == "odds"]


def test_capture_requests_us_books_plus_pinnacle_only_when_switched_on(monkeypatch, tmp_path):
    monkeypatch.delenv(rab.PINNACLE_ENV, raising=False)
    result, odds = _capture(monkeypatch, tmp_path)
    assert result["pinnacle"] == "off" and odds[0]["regions"] == "us" and "bookmakers" not in odds[0]
    monkeypatch.setenv(rab.PINNACLE_ENV, "NCAAB")
    result, odds = _capture(monkeypatch, tmp_path)
    assert result["pinnacle"] == "on" and odds == [{**PIN, "eventIds": "g1"}]


def test_spent_pinnacle_allowance_falls_back_to_the_normal_capture(monkeypatch, tmp_path):
    monkeypatch.setenv(rab.PINNACLE_ENV, "NCAAB")
    result, odds = _capture(monkeypatch, tmp_path, fail_pinnacle=True)
    assert result["pinnacle"] == "allowance_spent"
    assert odds[-1]["regions"] == "us" and len(odds) == 2


def _quote(book, selection, line, american):
    dec = 1 + (american / 100 if american > 0 else 100 / abs(american))
    return {"event_id": "e", "market_family": "SPREAD", "selection": selection, "line": line,
            "american_odds": american, "decimal_odds": dec, "sportsbook": book, "quote_verified": True}


def test_screen_keeps_pinnacle_out_of_consensus_and_best_price_but_reports_ev_against_it():
    event = {"event_id": "e", "sport": "NCAAB", "home_team": "H", "away_team": "A"}
    quotes = []
    for book, home, away in (("a", -110, -110), ("b", -110, -110), ("c", -105, -115), ("pinnacle", -130, 115)):
        quotes += [_quote(book, "H", -3.5, home), _quote(book, "A", 3.5, away)]
    rows = {r["selection"]: r for r in pes.screen([event], quotes, [], [])}
    assert rows["H"]["sportsbook"] == "c" and rows["H"]["books"] == 3      # pinnacle neither best nor counted
    assert rows["H"]["fair_probability"] == pytest.approx(0.5)
    assert rows["H"]["pinnacle_fair"] > 0.54 and rows["H"]["ev_vs_pinnacle"] > rows["H"]["ev"]
    summary = pes.summarize(list(rows.values()), ev_key="ev_vs_pinnacle")
    assert summary["all"]["picks"] == 2


def test_pinnacle_is_never_a_plan_reference_quote_or_a_research_offer(monkeypatch):
    from app_core import prospective_research_models as models
    from app_core import research_picks as rp
    monkeypatch.setattr(models.evidence, "verify_provider_offer", lambda event, quote: True)
    event = {"event_id": "e", "home_team": "H", "away_team": "A"}
    early, late = "2026-11-06T18:00:00+00:00", "2026-11-06T19:00:00+00:00"
    pin = {**_quote("pinnacle", "H", -3.5, -130), "quote_id": "q-pin", "quote_timestamp": early}
    book = {**_quote("a", "H", -3.5, -110), "quote_id": "q-a", "quote_timestamp": late}
    chosen = models._reference_quotes({"e": event}, [pin, book], datetime(2026, 11, 6, 20, tzinfo=timezone.utc))
    assert chosen["e"]["sportsbook"] == "a"          # the earlier Pinnacle quote is skipped
    offers = rp._offers(event, [pin, book], "SPREAD", datetime(2026, 11, 6, 20, tzinfo=timezone.utc))
    assert set(offers[-3.5]) == {"a"}


def test_certified_closes_never_name_pinnacle(monkeypatch):
    from app_core import prospective_closes as pc
    event = {"event_id": "e", "sport": "NBA", "provider_namespace": "THE_ODDS_API",
             "scheduled_start": "2026-11-06T19:00:00+00:00"}
    quotes = [{**_quote(b, "H", -3.5, -110), "quote_id": b} for b in ("a", "pinnacle")]
    records = {"prospective_event": [event], "prospective_close": [], "prospective_quote": quotes}
    monkeypatch.setattr(pc.evidence, "read_records", lambda path, kind, sport=None: records[kind])
    monkeypatch.setattr(pc, "enabled_families", lambda sport, setting: {"SPREAD"})
    due = pc.due("unused", datetime(2026, 11, 6, 20, tzinfo=timezone.utc), sports=("NBA",))
    assert [q["sportsbook"] for q in due[0][1]] == ["a"]
