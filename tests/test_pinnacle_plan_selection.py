"""Pinnacle plan entry quotes, window rate and line-aware scoring (owner 2026-11-07)."""
from datetime import datetime, timedelta, timezone

import pytest

from app_core import pinnacle_model as pm
from app_core import prospective_evidence as pe

START = "2030-01-01T20:00:00+00:00"
EVENT = {"event_id": "e", "home_team": "H", "away_team": "A", "observed_at": "2030-01-01T10:00:00+00:00",
         "scheduled_start": START}
FIRST = "2030-01-01T10:00:00+00:00"    # first capture (event observed_at)
LATER = "2030-01-01T16:00:00+00:00"    # a later capture's response time


def _decimal(american):
    return 1 + american / 100 if american > 0 else 1 + 100 / abs(american)


def q(book, selection, line, price, received=None, stamp=None, event="e"):
    raw = ({"event_response": {}, "market": {}, "response_received_at": received} if received else {"key": "spreads"})
    return {"event_id": event, "sportsbook": book, "market_family": "SPREAD", "selection": selection, "line": line,
            "decimal_odds": _decimal(price), "quote_verified": 1, "raw_source": raw,
            "quote_timestamp": stamp or ("2030-01-01T09:58:00+00:00" if not received else "2030-01-01T15:58:00+00:00")}


def both(book, home_line, home_price, away_price, received=None, stamp=None, event="e"):
    return [q(book, "H", home_line, home_price, received, stamp, event),
            q(book, "A", -home_line, away_price, received, stamp, event)]


NOW = datetime(2030, 1, 1, 19, 0, tzinfo=timezone.utc)


def test_earliest_capture_where_a_us_book_posts_pinnacles_line_best_price():
    quotes = (
        # First capture: Pinnacle at -4.5, US books only at -3.5 -> no shared line, no entry.
        both("pinnacle", -4.5, -105, -115) + both("bookA", -3.5, -110, -110)
        # Later capture: Pinnacle and two US books at -3.5.
        + both("pinnacle", -3.5, -102, -118, received=LATER) + both("bookA", -3.5, -110, -110, received=LATER)
        + both("bookB", -3.5, +105, -125, received=LATER))
    chosen = pm.plan_quotes({"e": EVENT}, quotes, NOW)["e"]
    # Pinnacle no-vig home ~0.483: bookB's +105 home side has the best EV of any side (-1.0%).
    assert (chosen["sportsbook"], chosen["selection"], chosen["line"]) == ("bookB", "H", -3.5)
    assert pm.capture_time(EVENT, chosen) == pe._time(LATER, "t")
    # The plan never enters at the first canonical quote when Pinnacle has another line there,
    # never at Pinnacle itself, and never after the start.
    assert all(c["sportsbook"] != "pinnacle" for c in pm.plan_quotes({"e": EVENT}, quotes, NOW).values())


def test_away_side_value_still_enters_at_the_canonical_quote_and_one_sided_books_are_skipped():
    quotes = (both("pinnacle", -2.5, -130, +110) + both("bookA", -2.5, -110, -110)
              + [q("bookC", "H", -2.5, +150)])                                    # bookC: home side only
    chosen = pm.plan_quotes({"e": EVENT}, quotes, NOW)["e"]
    assert chosen["sportsbook"] == "bookA" and chosen["selection"] == "H"      # canonical side stored
    assert pm.plan_quotes({"e": EVENT}, both("pinnacle", -2.5, -110, -110) + [q("bookC", "H", -2.5, 150)], NOW) == {}


def test_no_entry_after_start_or_before_now():
    late = both("pinnacle", -2.5, -110, -110, received="2030-01-01T20:30:00+00:00",
                stamp="2030-01-01T20:29:00+00:00") + both("bookA", -2.5, -105, -115,
                                                            received="2030-01-01T20:30:00+00:00",
                                                            stamp="2030-01-01T20:29:00+00:00")
    assert pm.plan_quotes({"e": EVENT}, late, datetime(2030, 1, 2, tzinfo=timezone.utc)) == {}
    early = both("pinnacle", -2.5, -110, -110) + both("bookA", -2.5, -105, -115)
    assert pm.plan_quotes({"e": EVENT}, early, datetime(2030, 1, 1, 9, 0, tzinfo=timezone.utc)) == {}


def test_window_rate_counts_plan_games_over_the_observed_days(monkeypatch):
    now = datetime(2030, 1, 10, tzinfo=timezone.utc)
    events, quotes = {}, []
    for i in range(12):                                    # 12 games over the last 3 days
        start = now - timedelta(days=3) + timedelta(hours=6 * i)
        observed = start - timedelta(hours=8)
        eid = f"g{i}"
        events[eid] = {"event_id": eid, "home_team": "H", "away_team": "A", "observed_at": observed.isoformat(),
                       "scheduled_start": start.isoformat()}
        stamp = (observed - timedelta(minutes=2)).isoformat()
        quotes += both("bookA", -2.5, -110, -110, stamp=stamp, event=eid)
        if i % 2 == 0:                                     # Pinnacle at the same line on half of them
            quotes += both("pinnacle", -2.5, -108, -112, stamp=stamp, event=eid)
    monkeypatch.setattr("app_core.prospective_research_models._read_scope", lambda *a: (events, quotes, []))
    rate = pm.plan_games_per_day("db", "NCAAB", "SPREAD", now=now)
    assert rate == pytest.approx(6 / 3)                    # 6 plan games over 3 observed days, not 14


def test_prediction_at_another_line_is_scored_from_the_same_final_score(monkeypatch):
    monkeypatch.setattr(pe, "_event", lambda db, eid: {"event_id": eid, "sport": "NCAAB", "home_team": "H",
                                                       "away_team": "A"})
    result = {"event_id": "e", "home_score": 80, "away_score": 76, "outcome": "WIN",
              "raw_source": {"rule": pe.SCORE_SETTLEMENT_SOURCE, "selection": "H", "line": -3.5}}
    at = lambda line: {"market_family": "SPREAD", "selection": "H", "line": line}  # noqa: E731
    assert pe._outcome_at_quote_line(None, result, at(-3.5), "WIN") == "WIN"      # same line: unchanged
    assert pe._outcome_at_quote_line(None, result, at(-4.5), "WIN") == "LOSS"     # home won by 4
    assert pe._outcome_at_quote_line(None, result, at(-4.0), "WIN") == "PUSH"
    other = dict(result, raw_source={"rule": "MARKET_GRADE", "line": -3.5})
    assert pe._outcome_at_quote_line(None, other, at(-4.5), "WIN") == "WIN"       # no stored score settlement


def test_replay_estimate_counts_plan_games_and_bets_by_edge_without_results(monkeypatch):
    now = datetime(2030, 1, 10, tzinfo=timezone.utc)
    events, quotes = {}, []
    for i in range(8):                                     # 8 games over the last 4 days
        start = now - timedelta(days=4) + timedelta(hours=12 * i)
        observed = start - timedelta(hours=8)
        eid = f"g{i}"
        events[eid] = {"event_id": eid, "home_team": "H", "away_team": "A", "observed_at": observed.isoformat(),
                       "scheduled_start": start.isoformat()}
        stamp = (observed - timedelta(minutes=2)).isoformat()
        quotes += both("pinnacle", -2.5, -110, -110, stamp=stamp, event=eid)     # fair 50/50
        home_price = {0: +106, 1: +103, 2: -110, 3: -110}.get(i % 4)           # +3%, +1.5%, -4.5%, -4.5% EV
        quotes += both("bookA", -2.5, home_price, -110, stamp=stamp, event=eid)
    monkeypatch.setattr("app_core.prospective_research_models._read_scope",
                        lambda *a: (events, quotes, [["never read"]]))
    out = pm.replay_estimate("db", "NCAAB", "SPREAD", now=now)
    assert out["plan_games"] == 8 and out["plan_games_per_day"] == 2.0 and out["days_for_min_sample"] == 100.0
    assert out["by_edge"]["1%"]["bets"] == 4 and out["by_edge"]["2%"]["bets"] == 2
    assert out["by_edge"]["2%"]["bets_per_day"] == 0.5 and out["by_edge"]["2%"]["days_for_min_bets"] == 100.0
    assert out["by_edge"]["1%"]["days_per_window"] == 100.0                # the 200-game minimum binds
    assert pm.replay_estimate("db", "NCAAB", "SPREAD", now=now - timedelta(days=30))["status"] == "NO_GAMES"


def test_bound_plan_draft_shows_the_replay_estimate():
    import importlib
    bound_plan = importlib.import_module("scripts.bound_plan")
    text = bound_plan.markdown({"mode": "DRAFT_READ_ONLY", "markets": [], "replay_estimate": {
        "sport": "NCAAB", "market_family": "SPREAD", "status": "ESTIMATED", "days_observed": 4.0,
        "games_started": 120, "plan_games": 60, "plan_share": 0.5, "plan_games_per_day": 15.0,
        "days_for_min_sample": 13.3, "min_sample": 200, "min_bets": 50,
        "by_edge": {"1%": {"bets": 20, "bets_per_day": 5.0, "days_for_min_bets": 10.0, "days_per_window": 13.3},
                    "2%": {"bets": 8, "bets_per_day": 2.0, "days_for_min_bets": 25.0, "days_per_window": 25.0}}}})
    assert "### Replay estimate NCAAB SPREAD: ESTIMATED" in text
    assert "| 2% | 8 | 2.0 | 25.0 | 25.0 |" in text and "(50%), 15.0 a day" in text


def test_plans_enter_only_at_books_the_stake_can_use():
    quotes = (both("pinnacle", -3.5, -102, -118, received=LATER) + both("caesars", -3.5, +110, -130, received=LATER)
              + both("fanduel", -3.5, -105, -105, received=LATER) + both("draftkings", -3.5, -110, -110, received=LATER))
    # Every US book: Caesars has the best price. The plan's books: FanDuel's -105 away side is the best.
    assert pm.plan_quotes({"e": EVENT}, quotes, NOW)["e"]["sportsbook"] == "caesars"
    chosen = pm.plan_quotes({"e": EVENT}, quotes, NOW, books=pm.PLAN_BOOKS)["e"]
    assert chosen["sportsbook"] == "fanduel"
    assert pm.PLAN_BOOKS == {"Novig", "DraftKings", "FanDuel", "BetMGM"}
    only_caesars = both("pinnacle", -3.5, -102, -118) + both("caesars", -3.5, +110, -130)
    assert pm.plan_quotes({"e": EVENT}, only_caesars, NOW, books=pm.PLAN_BOOKS) == {}


def test_bound_plan_freezes_its_entry_books_and_predictions_follow_them():
    from app_core import market_skill_validation as ms
    from app_core import prospective_calibration as pc
    rules = ms.methodology(0.02, pm.PLAN_BOOKS)
    assert rules["entry_books"] == ["BetMGM", "DraftKings", "FanDuel", "Novig"]
    assert "entry_books" not in ms.methodology(0.02)                      # older plans: no restriction
    spec = {"sport": "NCAAB", "market_family": "SPREAD", "model_id": "m", "calibration_id": "c",
            "validation_start": "2030-01-02T00:00:00+00:00", "validation_end": "2030-01-12T00:00:00+00:00",
            "holdout_start": "2030-01-12T00:00:00+00:00", "holdout_end": "2030-01-22T00:00:00+00:00",
            "market_skill_methodology": rules}
    assert pc.freeze_inputs(spec)["entry_books"] == rules["entry_books"]   # the freeze redrafts the same plan
    import json
    assert pc._plan_rules({"payload": json.dumps({"market_skill_methodology": rules})})["entry_books"] == \
        rules["entry_books"]
    assert pc._plan_rules({"payload": "{}"}) == {} and pc._plan_rules({}) == {}
