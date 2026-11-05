"""SYNTHETIC certified closes (roadmap C1): the provider's last pregame snapshot, fetched after start."""
from datetime import datetime, timedelta, timezone
import copy

import pytest

from app_core import prospective_closes as pc, prospective_evidence as pe

BASE = datetime(2030, 1, 1, tzinfo=timezone.utc)


def at(hours=0, minutes=0):
    return (BASE + timedelta(hours=hours, minutes=minutes)).isoformat()


def market(stamp_minutes, line=-2.5, price=-110):
    return {"key": "spreads", "last_update": at(0, stamp_minutes), "outcomes": [
        {"name": "Home", "point": line, "price": price}, {"name": "Away", "point": -line, "price": price}]}


def provider_event(stamp_minutes=0, **kw):
    return {"id": "SYNTHETIC-E1", "sport_key": "basketball_nba", "home_team": "Home", "away_team": "Away",
            "commence_time": at(1), "bookmakers": [{"key": "TestBook", "markets": [market(stamp_minutes, **kw)]}]}


@pytest.fixture
def clock(monkeypatch):
    state = [BASE]
    monkeypatch.setattr(pe, "_clock", lambda: state[0])
    return lambda h=0, m=0: state.__setitem__(0, BASE + timedelta(hours=h, minutes=m))


@pytest.fixture
def store(tmp_path, clock):
    path = tmp_path / 'SYNTHETIC-evidence.sqlite3'
    pe.insert_event(path, dict(event_id="SYNTHETIC-E1", sport="NBA", game_id="SYNTHETIC-E1",
        provider_namespace="THE_ODDS_API", provider_event_id="SYNTHETIC-E1", home_team="Home", away_team="Away",
        home_team_id="provider:1", away_team_id="provider:2", scheduled_start=at(1), observed_at=at(0),
        source_id="event-SYNTHETIC-E1", raw_source=provider_event()))
    pe.insert_quote(path, dict(quote_id="SYNTHETIC-Q1", event_id="SYNTHETIC-E1", sport="NBA", market_family="SPREAD",
        selection="Home", line=-2.5, american_odds=-110, decimal_odds=1.909090909, sportsbook="TestBook",
        quote_timestamp=at(0), quote_source="THE_ODDS_API", quote_verified=True,
        source_id=f"SYNTHETIC-E1:TestBook:spreads:Home:{at(0)}", raw_source=market(0)))
    return path


def snapshot(timestamp=at(0, 55), next_timestamp=at(1, 0), price=-105, line=-2.5):
    return {"timestamp": timestamp, "previous_timestamp": at(0, 50), "next_timestamp": next_timestamp,
            "data": provider_event(50, price=price, line=line)}


class Provider:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def __call__(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        payload = self.payload
        return type('R', (), {'raise_for_status': lambda self: None, 'json': lambda self: copy.deepcopy(payload)})()


def test_due_lists_only_started_events_missing_a_certified_close(store, clock):
    assert pc.due(store, BASE + timedelta(minutes=30), sports=('NBA',)) == []  # not started yet
    [(event, quotes)] = pc.due(store, BASE + timedelta(hours=1, minutes=5), sports=('NBA',))
    assert event['event_id'] == 'SYNTHETIC-E1' and [q['quote_id'] for q in quotes] == ['SYNTHETIC-Q1']
    url, params, credits = pc.request_plan(event, quotes)
    assert url.endswith('/historical/sports/basketball_nba/events/SYNTHETIC-E1/odds')
    assert params == dict(bookmakers='TestBook', markets='spreads', oddsFormat='american', date='2030-01-01T00:59:59Z')
    assert credits == 10


def test_record_stores_a_replayable_certified_close_once(store, clock):
    clock(1, 5)
    now = BASE + timedelta(hours=1, minutes=5)
    provider = Provider(snapshot())
    report = pc.record(store, now=now, key='SYNTHETIC-KEY', request_get=provider, sports=('NBA',))
    assert (report['requests'], report['stored'], report['credits'], report['errors']) == (1, 1, 10, [])
    [close] = pe.read_records(store, 'prospective_close', sport='NBA')
    assert close['close_verified'] == 1 and close['american_odds'] == -105 and close['close_timestamp'] == at(0, 55)
    assert provider.calls[0][1]['apiKey'] == 'SYNTHETIC-KEY'
    again = pc.record(store, now=now, key='SYNTHETIC-KEY', request_get=provider, sports=('NBA',))
    assert again['events'] == 0 and len(provider.calls) == 1  # never re-buys a close


@pytest.mark.parametrize('payload, retrieved_minutes', [
    (snapshot(timestamp=at(1, 0)), 65),                     # snapshot not before the start
    (snapshot(next_timestamp=at(0, 58)), 65),               # a later pregame snapshot exists
    (snapshot(), 58),                                        # retrieved before the game started
])
def test_non_closing_snapshots_are_rejected(store, clock, payload, retrieved_minutes):
    clock(0, retrieved_minutes)
    now = BASE + timedelta(minutes=retrieved_minutes)
    event = pe.read_records(store, 'prospective_event', sport='NBA')[0]
    quotes = pe.read_records(store, 'prospective_quote', sport='NBA')
    [row] = pc.closes_from_snapshot(event, quotes, payload, now)
    with pytest.raises(ValueError):
        pe.insert_certified_close(store, row)
    assert pe.read_records(store, 'prospective_close', sport='NBA') == []


def test_tampered_price_or_live_source_is_never_certified(store, clock):
    clock(1, 5)
    now = BASE + timedelta(hours=1, minutes=5)
    event = pe.read_records(store, 'prospective_event', sport='NBA')[0]
    quotes = pe.read_records(store, 'prospective_quote', sport='NBA')
    [row] = pc.closes_from_snapshot(event, quotes, snapshot(), now)
    with pytest.raises(ValueError, match='not replayable'):
        pe.insert_certified_close(store, dict(row, american_odds=-200, decimal_odds=1.5))
    with pytest.raises(ValueError, match='verified historical'):
        pe.insert_certified_close(store, dict(row, close_source='THE_ODDS_API'))


def test_disabled_by_default_and_budget_limits_stop_before_any_request(store, clock):
    provider = Provider(snapshot())
    now = BASE + timedelta(hours=1, minutes=5)
    off = pc.record(store, now=now, key='SYNTHETIC-KEY', request_get=provider, setting=lambda k, d='': d)
    assert off['skipped'] == ['CERTIFIED_CLOSES_DISABLED'] and provider.calls == []

    from app_core.research_api_budget import BudgetLimit

    class Exhausted:
        def reserve_certified_close(self, sport, units, already_spent=False):
            raise BudgetLimit('api_budget:NBA:certified_close:daily')
    capped = pc.record(store, now=now, key='SYNTHETIC-KEY', request_get=provider, budget=Exhausted(), sports=('NBA',))
    assert provider.calls == [] and capped['stored'] == 0 and 'BudgetLimit' in capped['errors'][0]
    assert pc.enabled_sports(lambda k, d='': 'nba, nhl, XFL') == ('NBA', 'NHL')


def test_a_certified_close_is_comparable_for_plan_evaluation(store, clock):
    import sqlite3
    from contextlib import closing
    clock(1, 5)
    now = BASE + timedelta(hours=1, minutes=5)
    pc.record(store, now=now, key='SYNTHETIC-KEY', request_get=Provider(snapshot()), sports=('NBA',))
    [quote] = pe.read_records(store, 'prospective_quote', sport='NBA')
    with closing(sqlite3.connect(store)) as db:
        db.row_factory = sqlite3.Row
        found = pe._comparable_close(db, quote, now)
    assert found is not None and found['close_verified'] == 1  # same event, book, selection and line



def test_spread_only_by_default_and_totals_only_when_asked(store, clock):
    pe.insert_quote(store, dict(quote_id="SYNTHETIC-Q2", event_id="SYNTHETIC-E1", sport="NBA", market_family="TOTAL",
        selection="Over", line=221.5, american_odds=-110, decimal_odds=1.909090909, sportsbook="TestBook",
        quote_timestamp=at(0), quote_source="THE_ODDS_API", quote_verified=False,
        source_id=f"SYNTHETIC-E1:TestBook:totals:Over:{at(0)}", raw_source={"synthetic": True}))
    assert pc.enabled_families('NBA', lambda k, d='': d) == {'SPREAD'}
    assert pc.enabled_families('NBA', lambda k, d='': 'spread,total') == {'SPREAD', 'TOTAL'}
    assert pc.enabled_families('MLB', lambda k, d='': 'SPREAD') == {'RUN_LINE'}
    now = BASE + timedelta(hours=1, minutes=5)
    [(_, quotes)] = pc.due(store, now, sports=('NBA',), setting=lambda k, d='': d)
    assert {q['market_family'] for q in quotes} == {'SPREAD'}


def test_certified_close_allowance_is_separate_and_capped():
    from app_core.research_api_budget import Budget, BudgetLimit, CERTIFIED_CLOSE_LIMITS
    assert CERTIFIED_CLOSE_LIMITS == {'NBA': {'daily': 100, 'rolling_31_days': 2500}}
    saves = []
    budget = Budget({}, lambda: saves.append(1), clock=lambda: BASE)
    for _ in range(10):
        budget.reserve_certified_close('NBA', 10)
    with pytest.raises(BudgetLimit, match='NBA:certified_close:daily'):
        budget.reserve_certified_close('NBA', 10)
    with pytest.raises(BudgetLimit, match='NHL:certified_close:not_budgeted'):
        budget.reserve_certified_close('NHL', 10)
    budget.reserve_certified_close('NBA', 5, already_spent=True)  # an overcharge is always recorded
    assert budget.usage('NBA', budget.close_ledger)['daily'] == 105 and len(saves) == 11
    assert budget.usage('ODDS') == {'daily': 0, 'rolling_31_days': 0}  # capture allowance untouched


def test_a_provider_overcharge_is_recorded_and_stops_the_batch(store, clock):
    from app_core.research_api_budget import Budget
    clock(1, 5)
    now = BASE + timedelta(hours=1, minutes=5)

    class Overcharging(Provider):
        def __call__(self, url, params=None, timeout=None):
            response = super().__call__(url, params, timeout)
            response.headers = {'x-requests-last': '30'}
            return response
    budget = Budget({}, lambda: None, clock=lambda: now)
    report = pc.record(store, now=now, key='SYNTHETIC-KEY', request_get=Overcharging(snapshot()), budget=budget,
                       sports=('NBA',))
    assert report['stored'] == 1 and report['credits'] == 30 and report['stopped'] == 'COST_MISMATCH'
    assert budget.usage('NBA', budget.close_ledger)['daily'] == 30


def test_a_game_started_most_of_a_day_ago_is_still_due(tmp_path, monkeypatch):
    # D16: runs start only a few times a day, so the close lookback spans a full day.
    assert pc.LOOKBACK == timedelta(hours=24)
