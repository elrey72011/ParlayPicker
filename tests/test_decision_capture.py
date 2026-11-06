"""C6 step 1: fresh decision quotes and both-side bound predictions. SYNTHETIC store and provider.

Nothing here is live evidence or a model-quality claim.
"""
from datetime import timedelta
import math

import pytest

from app_core import decision_capture as dc
from app_core import prospective_calibration as pc
from app_core import prospective_evidence as pe
from app_core import prospective_research_models as models
from app_core import qualified_bridge as qb
from tests.test_prospective_research_models import _at, _event

UPCOMING = 80


@pytest.fixture
def clock(monkeypatch):
    state = [_at(0)]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    monkeypatch.setattr(models, "_utcnow", lambda: state[0])
    monkeypatch.setattr(models, "source_commit", lambda: "a" * 40)
    return state


def _response(start, stamp, *, home_price, away_price, line=-2.5, books=("book",)):
    """A provider odds response for the upcoming game, observed later than its first capture."""
    market = {"key": "spreads", "last_update": stamp.isoformat(), "outcomes": [
        {"name": "Ana", "point": line, "price": home_price},
        {"name": "Bos", "point": -line, "price": away_price}]}
    return {"id": f"provider-NBA-{UPCOMING}", "sport_key": "basketball_nba", "home_team": "Ana",
            "away_team": "Bos", "commence_time": start.isoformat(),
            "bookmakers": [{"key": b, "markets": [dict(market)]} for b in books]}


class _Reply:
    status_code = 200

    def __init__(self, payload, spent="1"):
        self.payload, self.headers = payload, {"x-requests-last": spent}

    def json(self):
        return self.payload


class _Ledger:
    """In-memory stand-in for the loop's remote state."""

    def __init__(self, clock):
        self.state, self.saves = {}, 0
        self.budget = dc.budget_for(self.state, self.save, clock=lambda: clock[0])

    def save(self):
        self.saves += 1


def _store(tmp_path, clock):
    path = tmp_path / "SYNTHETIC-evidence.sqlite3"
    for index in range(70):
        _event(path, clock, "NBA", index)
    clock[0] = _at(71)
    fit = models.train_scope(path, "NBA", "SPREAD")
    assert fit["status"] == "FITTED_RESEARCH_ONLY"
    # The upcoming game: first captured 3 hours out, like the research cycle.
    start = _at(UPCOMING, 2)
    first = start - timedelta(hours=3)
    clock[0] = first
    raw = _response(start, first, home_price=-110, away_price=-110)
    pe.insert_event(path, {"event_id": "upcoming", "sport": "NBA", "game_id": raw["id"],
                           "provider_namespace": "THE_ODDS_API", "provider_event_id": raw["id"],
                           "home_team": "Ana", "away_team": "Bos", "home_team_id": "team-a",
                           "away_team_id": "team-b", "scheduled_start": start.isoformat(),
                           "observed_at": first.isoformat(), "source_id": "capture-upcoming", "raw_source": raw})
    pe.insert_quote(path, {"quote_id": "first-quote", "event_id": "upcoming", "sport": "NBA", "market_family": "SPREAD",
                           "selection": "Ana", "line": -2.5, "american_odds": -110, "decimal_odds": 1 + 100 / 110,
                           "sportsbook": "book", "quote_timestamp": first.isoformat(), "quote_source": "THE_ODDS_API",
                           "quote_verified": True, "source_id": f"{raw['id']}:book:spreads:Ana:{first.isoformat()}",
                           "raw_source": raw["bookmakers"][0]["markets"][0]})
    return path, fit["model_id"], start


def _teams_have_history(monkeypatch):
    # The synthetic training teams are not "Ana"/"Bos"; give the upcoming game fixed features.
    monkeypatch.setattr(models, "_features", lambda event, now, events, scores, market: ({"diff": 0.0}, []))
    monkeypatch.setattr(models, "_expected", lambda fit, features: 4.0)  # home by 4 on average


def test_markets_switch_is_off_by_default_and_parses_owner_names():
    assert dc.markets(lambda *_: "") == frozenset()
    assert dc.markets(lambda *_: " ncaab:spread , NBA:TOTAL, NFL:SPREAD, NBA:MONEYLINE") == {
        ("NCAAB", "SPREAD"), ("NBA", "TOTAL")}


def test_a_later_capture_replays_only_with_its_own_response(tmp_path, clock):
    path, _, start = _store(tmp_path, clock)
    later = start - timedelta(minutes=40)
    clock[0] = later + timedelta(minutes=1)
    response = _response(start, later, home_price=-105, away_price=-115)
    market = response["bookmakers"][0]["markets"][0]
    quote = {"quote_id": "q", "event_id": "upcoming", "sport": "NBA", "market_family": "SPREAD",
             "selection": "Ana", "line": -2.5, "american_odds": -105, "decimal_odds": 1 + 100 / 105,
             "sportsbook": "book", "quote_timestamp": later.isoformat(), "quote_source": "THE_ODDS_API",
             "quote_verified": True, "source_id": f"{response['id']}:book:spreads:Ana:{later.isoformat()}"}
    with pytest.raises(ValueError, match="not replayable"):           # the fragment alone: first capture only
        pe.insert_quote(path, dict(quote, raw_source=market))
    tampered = dict(response, home_team="Someone Else")
    with pytest.raises(ValueError, match="not replayable"):           # its response must be the same game
        pe.insert_quote(path, dict(quote, raw_source=pe.quote_source(tampered, market, clock[0].isoformat())))
    with pytest.raises(ValueError, match="future"):                  # received after "now"
        pe.insert_quote(path, dict(quote, raw_source=pe.quote_source(
            response, market, (clock[0] + timedelta(minutes=5)).isoformat())))
    pe.insert_quote(path, dict(quote, raw_source=pe.quote_source(response, market, clock[0].isoformat())))
    stored = pe.load_record(path, "prospective_quote", "q")
    event = pe.load_record(path, "prospective_event", "upcoming")
    assert stored["quote_verified"] == 1 and pe.verify_provider_offer(event, stored)


def test_capture_stores_both_sides_skips_pinnacle_and_dedupes(tmp_path, clock):
    path, _, start = _store(tmp_path, clock)
    later = start - timedelta(minutes=40)
    clock[0] = later + timedelta(minutes=1)
    ledger = _Ledger(clock)
    calls = []

    def get(url, **kwargs):
        calls.append(kwargs["params"])
        return _Reply([_response(start, later, home_price=-105, away_price=-115, books=("book", "pinnacle"))])

    first = dc.capture(path, "NBA", "SPREAD", key="k", request_get=get, budget=ledger.budget, clock=lambda: clock[0])
    assert first["stored_quotes"] == 2 and first["replay_blocked"] == 0 and first["due_events"] == 1
    assert calls[0]["markets"] == "spreads" and calls[0]["regions"] == "us" and calls[0]["eventIds"] == f"provider-NBA-{UPCOMING}"
    books = {q["sportsbook"] for q in pe.read_records(path, "prospective_quote", sport="NBA", market_family="SPREAD")
             if q["event_id"] == "upcoming"}
    assert books == {"book"}                                           # Pinnacle is never an entry price
    again = dc.capture(path, "NBA", "SPREAD", key="k", request_get=get, budget=ledger.budget, clock=lambda: clock[0])
    assert again["stored_quotes"] == 0                                 # same provider update, same quote id
    assert ledger.budget.usage("ODDS")["daily"] == 2 and ledger.saves >= 2   # charged and saved before each call


def test_no_due_games_spends_nothing_and_limits_hold(tmp_path, clock):
    path, _, start = _store(tmp_path, clock)
    ledger = _Ledger(clock)
    clock[0] = start - timedelta(hours=2)                              # outside the 60-minute horizon

    def never(*_, **__):
        raise AssertionError("no request expected")

    assert dc.capture(path, "NBA", "SPREAD", key="k", request_get=never, budget=ledger.budget,
                      clock=lambda: clock[0])["requested"] is False
    clock[0] = start - timedelta(minutes=30)
    ledger.state["api_budget_v1"]["ODDS"] = {clock[0].date().isoformat(): dc.DECISION_LIMITS["daily"]}
    spent = dc.capture(path, "NBA", "SPREAD", key="k", request_get=never, budget=ledger.budget, clock=lambda: clock[0])
    assert spent["requested"] is False and "ODDS:daily" in spent["blocker"]


def test_an_unexpected_provider_charge_stops_the_loop_until_reviewed(tmp_path, clock):
    path, _, start = _store(tmp_path, clock)
    later = start - timedelta(minutes=40)
    clock[0] = later + timedelta(minutes=1)
    ledger = _Ledger(clock)

    def get(url, **kwargs):
        return _Reply([_response(start, later, home_price=-105, away_price=-115)], spent="3")

    first = dc.capture(path, "NBA", "SPREAD", key="k", request_get=get, budget=ledger.budget, clock=lambda: clock[0])
    assert first["blocker"] == dc.COST_FLAG and ledger.state[dc.COST_FLAG] == "3"
    second = dc.capture(path, "NBA", "SPREAD", key="k", request_get=get, budget=ledger.budget, clock=lambda: clock[0])
    assert second["requested"] is False and second["blocker"] == dc.COST_FLAG


def test_both_sides_are_predicted_and_the_bridge_offers_the_better_ev(tmp_path, clock, monkeypatch):
    path, model_id, start = _store(tmp_path, clock)
    _teams_have_history(monkeypatch)
    later = start - timedelta(minutes=40)
    clock[0] = later + timedelta(minutes=1)
    ledger = _Ledger(clock)
    # Home -2.5 at -105, Bos +2.5 at -115: with the model at home by 4, home is the better side.
    dc.capture(path, "NBA", "SPREAD", key="k", budget=ledger.budget, clock=lambda: clock[0],
               request_get=lambda url, **kw: _Reply([_response(start, later, home_price=-105, away_price=-115)]))
    model = pe.load_record(path, "prospective_model", model_id)
    now = clock[0] + timedelta(minutes=1)
    clock[0] = now
    unpredicted = pc._predict(path, "NBA", "SPREAD", model, None, now=now, policy_version="synthetic", tag="DECISION",
                              select=lambda events, quotes, at: dc.decision_quotes(events, quotes, at, predicted=set()))
    assert unpredicted["predictions"] == 0          # never a game's first prediction (validation scores that one)
    made = pc._predict(path, "NBA", "SPREAD", model, None, now=now, policy_version="synthetic", tag="DECISION",
                       select=lambda events, quotes, at: dc.decision_quotes(events, quotes, at, predicted={"upcoming"}))
    assert made["predictions"] == 2 and made["blocker"] is None
    rows = {p["selection"]: p for p in pe.read_records(path, "prospective_prediction", sport="NBA", market_family="SPREAD")
            if p["event_id"] == "upcoming" and p["observation_id"].startswith("decision-")}
    assert set(rows) == {"Ana", "Bos"}
    # The two sides are one distribution: home's win is away's loss at the same line, and vice versa.
    assert math.isclose(rows["Ana"]["mean_probability"], rows["Bos"]["loss_probability"])
    assert math.isclose(rows["Ana"]["loss_probability"], rows["Bos"]["mean_probability"])
    assert rows["Ana"]["mean_probability"] > 0.5
    picked = qb.rows(path, {("NBA", "SPREAD")}, now=now)
    assert len(picked) == 1 and picked[0]["selection"] == "Ana" and picked[0]["market_type"] == "spread_home"
    # The research cycle's own selector still uses the first capture's quote.
    events, quotes, _ = models._read_scope(path, "NBA", "SPREAD")
    assert models._reference_quotes(events, quotes, now)["upcoming"]["quote_id"] == "first-quote"


def test_decision_quotes_window_and_freshness(tmp_path, clock):
    path, _, start = _store(tmp_path, clock)
    later = start - timedelta(minutes=40)
    clock[0] = later + timedelta(minutes=1)
    ledger = _Ledger(clock)
    dc.capture(path, "NBA", "SPREAD", key="k", budget=ledger.budget, clock=lambda: clock[0],
               request_get=lambda url, **kw: _Reply([_response(start, later, home_price=-105, away_price=-115)]))
    events, quotes, _ = models._read_scope(path, "NBA", "SPREAD")
    assert len(dc.decision_quotes(events, quotes, later + timedelta(minutes=5))) == 2
    assert dc.decision_quotes(events, quotes, later + timedelta(minutes=31)) == {}   # older than 30 minutes
    assert dc.decision_quotes(events, quotes, later - timedelta(minutes=1)) == {}    # not yet quoted
    assert dc.decision_quotes(events, quotes, start) == {}                            # started


def test_run_once_is_off_without_a_switched_market_and_skips_unqualified(tmp_path):
    def never(*_, **__):
        raise AssertionError("no request expected")

    off = dc.run_once(tmp_path / "x.sqlite3", key="k", request_get=never, budget=None, setting=lambda *_: "")
    assert off["status"] == "OFF"
    ran = dc.run_once(tmp_path / "x.sqlite3", key="k", request_get=never, budget=None,
                      setting=lambda *_: "NCAAB:SPREAD", assess=lambda path: set())
    assert ran["markets"] == {"NCAAB SPREAD": {"blocker": "NOT_QUALIFIED"}}


def test_loop_restores_first_and_backs_up_only_after_changes(tmp_path):
    from scripts.decision_capture_loop import run
    events = []

    class Remote:
        def get_paginator(self, _):
            class P:
                def paginate(self, **_):
                    return [{"Contents": []}]
            return P()

    reports = iter([{"markets": {"NCAAB SPREAD": {"capture": {"stored_quotes": 2}, "predictions": {"predictions": 2}}}},
                    {"markets": {"NCAAB SPREAD": {"capture": {"stored_quotes": 0}, "predictions": {"predictions": 0}}}}])
    ticks = iter([0, 0, 1000])
    totals = run(30, 15, root=tmp_path, client=Remote(), folder="f", key="k", request_get=None,
                 setting=lambda *_: "NCAAB:SPREAD", cycle=lambda *a, **k: next(reports),
                 sync=lambda *a: events.append("sync"), clock=lambda: next(ticks), sleep=lambda s: None)
    assert events == ["sync", "sync"]                                  # restore, then one backup
    assert totals["cycles"] == 2 and totals["stored_quotes"] == 2 and totals["backups"] == 1
    assert run(30, 15, root=tmp_path, client=None, folder="f", key="k", request_get=None,
               setting=lambda *_: "")["status"] == "OFF"


def test_pinnacle_plan_capture_requests_pinnacle_and_keeps_it_reference_only(tmp_path, clock):
    from app_core.research_api_budget import PINNACLE_BOOKMAKERS
    path, _, start = _store(tmp_path, clock)
    later = start - timedelta(minutes=40)
    clock[0] = later + timedelta(minutes=1)
    ledger, calls = _Ledger(clock), []

    def get(url, **kwargs):
        calls.append(kwargs["params"])
        return _Reply([_response(start, later, home_price=-105, away_price=-115, books=("book", "pinnacle"))], spent="2")

    out = dc.capture(path, "NBA", "SPREAD", key="k", request_get=get, budget=ledger.budget,
                     clock=lambda: clock[0], pinnacle=True)
    assert calls[0]["bookmakers"] == PINNACLE_BOOKMAKERS and "regions" not in calls[0]
    assert out["stored_quotes"] == 4 and out["blocker"] is None           # 2 credits is the expected cost
    assert ledger.budget.usage("ODDS")["daily"] == 2
    events, quotes, _ = models._read_scope(path, "NBA", "SPREAD")
    assert {q["sportsbook"] for q in dc.decision_quotes(events, quotes, clock[0]).values()} == {"book"}
