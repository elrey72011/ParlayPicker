"""Strikeout-prop orchestration: schedule parse + name<->id resolution + offline end-to-end.

The runner is the network-facing wrapper around the (unit-tested, pure) scoring. These tests
pin the parts the wrapper owns: parsing the StatsAPI schedule into team/pitcher ids, matching
a propped pitcher to his id (form) and to the OPPONENT lineup (K rate), memoizing fetches, and
running the whole card offline against injected feeds so a propped pitcher with a real edge
surfaces as actionable while a no-form pitcher degrades to no_data.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.mlb_pitcher_stats import parse_schedule_probables
from app_core.prop_pipeline import PROP_MAX_PLAUSIBLE_EDGE
from app_core.prop_runner import (
    _dedupe_prop_card_rows,
    apply_prop_precision_shortlist,
    _resolve_prop_results_history,
    build_prop_card,
    build_resolvers,
    build_strikeout_card,
)


def test_duplicate_provider_event_ids_are_fetched_once():
    calls = []
    diagnostics = {}

    def props_fetch(client, sport_key, event_id):
        calls.append(event_id)
        return []

    card = build_strikeout_card(
        odds_client=object(), date="2026-08-29", season=2026,
        diagnostics=diagnostics,
        list_events=lambda c, sk, d: [{"id": "evt1"}, {"id": "evt1"}],
        props_fetch=props_fetch,
        schedule_fetch=lambda d: [],
    )

    assert card == []
    assert calls == ["evt1"]
    assert diagnostics["strikeout_prop_event_count"] == 1
    assert diagnostics["strikeout_prop_duplicate_event_count"] == 1


def test_prop_card_quote_dedupe_keeps_better_price_and_distinct_events():
    rows = pd.DataFrame([
        {
            "_event_id": "evt1", "_commence_time": "2026-08-29T23:00:00Z",
            "game_date": "2026-08-29", "matchup": "Boston @ New York",
            "player": "George Lombard Jr.", "market_type": "batter_total_bases_over",
            "line": 0.5, "book": "betmgm", "odds_american": -160,
        },
        {
            "_event_id": "evt1", "_commence_time": "2026-08-29T23:00:00Z",
            "game_date": "2026-08-29", "matchup": "Boston @ New York",
            "player": "George Lombard Jr.", "market_type": "batter_total_bases_over",
            "line": 0.5, "book": "betmgm", "odds_american": -155,
        },
        {
            # A separate event id must survive (for example, a doubleheader).
            "_event_id": "evt2", "_commence_time": "2026-08-29T18:00:00Z",
            "game_date": "2026-08-29", "matchup": "Boston @ New York",
            "player": "George Lombard Jr.", "market_type": "batter_total_bases_over",
            "line": 0.5, "book": "betmgm", "odds_american": -150,
        },
    ])

    out, dropped = _dedupe_prop_card_rows(rows)

    assert dropped == 1
    assert sorted(out["odds_american"].tolist()) == [-155, -150]
    assert "_event_id" not in out.columns
    assert "_commence_time" not in out.columns


def test_prop_precision_shortlist_uses_calibrated_rank_without_changing_stakes():
    card = pd.DataFrame({
        "player": [f"P{i}" for i in range(10)],
        "CalibratedProbability": [0.60 + i / 100 for i in range(10)],
        "ConservativeWinProbability": [0.80 - i / 100 for i in range(10)],
        "WinProbability": [0.80 - i / 100 for i in range(10)],
        "expected_value": [0.05] * 10,
        "edge": [0.04] * 10,
        "Kelly_Bet_Size": [float(i) for i in range(10)],
    })

    out = apply_prop_precision_shortlist(card, max_rows=8)

    shortlisted = out[out["Prop_Precision_Shortlist"]].sort_values(
        "Prop_Precision_Rank"
    )
    assert shortlisted["player"].tolist() == [
        "P9", "P8", "P7", "P6", "P5", "P4", "P3", "P2"
    ]
    assert shortlisted["Prop_Precision_Rank"].tolist() == list(range(1, 9))
    assert out["Prop_Ranking_Source"].eq("CalibratedProbability").all()
    assert out["Kelly_Bet_Size"].tolist() == card["Kelly_Bet_Size"].tolist()


_SCHEDULE = {
    "dates": [{
        "games": [{
            "teams": {
                "home": {"team": {"id": 147, "name": "New York Yankees"},
                         "probablePitcher": {"id": 1, "fullName": "Gerrit Cole"}},
                "away": {"team": {"id": 145, "name": "Chicago White Sox"},
                         "probablePitcher": {"id": 2, "fullName": "Garrett Crochet"}},
            }
        }]
    }]
}


def test_empty_explicit_prop_ledger_restores_repo_baseline():
    history = _resolve_prop_results_history(pd.DataFrame())
    assert history is not None
    assert len(history) >= 470
    assert history["result"].isin(["WIN", "LOSS"]).sum() >= 441


def test_parse_schedule_probables_extracts_ids_and_names():
    rows = parse_schedule_probables(_SCHEDULE)
    assert len(rows) == 1
    r = rows[0]
    assert r["home_team"] == "New York Yankees" and r["home_team_id"] == 147
    assert r["home_pitcher"] == "Gerrit Cole" and r["home_pitcher_id"] == 1
    assert r["away_team_id"] == 145 and r["away_pitcher_id"] == 2


def test_parse_schedule_handles_missing_probable():
    sched = {"dates": [{"games": [{"teams": {
        "home": {"team": {"id": 1, "name": "A"}},
        "away": {"team": {"id": 2, "name": "B"}, "probablePitcher": {"id": 9, "fullName": "P"}},
    }}]}]}
    rows = parse_schedule_probables(sched)
    assert rows[0]["home_pitcher_id"] is None
    assert rows[0]["away_pitcher_id"] == 9


def test_resolvers_map_pitcher_to_form_and_opponent_k_rate():
    rows = parse_schedule_probables(_SCHEDULE)
    form_calls, team_calls = [], []

    def form_fetch(pid, season):
        form_calls.append(pid)
        return {"k_per_9": 11.0, "avg_innings": 6.0, "n_games": 5}

    def team_k_fetch(tid, season):
        team_calls.append(tid)
        return 0.26

    form_lookup, opp_k_lookup = build_resolvers(
        rows, 2026, form_fetch=form_fetch, team_k_fetch=team_k_fetch
    )

    # Cole (home, id 1) faces the away lineup -> White Sox (team id 145).
    assert form_lookup("Gerrit Cole")["k_per_9"] == 11.0
    assert opp_k_lookup({"pitcher": "Gerrit Cole"}) == 0.26
    assert team_calls == [145]
    # Memoized: a second lookup of the same pitcher/team does not refetch.
    form_lookup("Gerrit Cole")
    opp_k_lookup({"pitcher": "Gerrit Cole"})
    assert form_calls == [1] and team_calls == [145]
    # Unknown pitcher -> no id, no fetch, None.
    assert form_lookup("Nobody") is None


def test_build_card_end_to_end_offline_surfaces_actionable():
    events = [{"id": "evt1", "home_team": "New York Yankees", "away_team": "Chicago White Sox"}]
    prop = {
        "pitcher": "Gerrit Cole", "line": 6.5, "over_odds": -110, "under_odds": -110,
        "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox",
    }

    card = build_strikeout_card(
        odds_client=object(),
        date="2026-06-23",
        season=2026,
        max_plausible_edge=1.0,  # isolate orchestration from the edge cap (tested in prop_pipeline)
        list_events=lambda c, sk, d: events,
        props_fetch=lambda c, sk, eid: [prop],
        schedule_fetch=lambda d: parse_schedule_probables(_SCHEDULE),
        # High K/9 vs a high-K lineup -> projection well over 6.5 -> over has real edge.
        form_fetch=lambda pid, season: {"k_per_9": 12.0, "avg_innings": 6.5, "n_games": 5},
        team_k_fetch=lambda tid, season: 0.28,
    )
    assert len(card) == 1
    row = card[0]
    assert row["best_side"] == "over"
    assert row["recommendation"] == "over"
    assert row["expected_ks"] > 6.5


def test_event_list_retries_once_after_transient_failure():
    calls = []
    diagnostics = {}

    def flaky_events(client, sport_key, date):
        calls.append(date)
        if len(calls) == 1:
            raise TimeoutError("temporary event-list timeout")
        return [{"id": "evt1"}]

    prop = {
        "pitcher": "Gerrit Cole", "line": 6.5, "over_odds": -110, "under_odds": -110,
        "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox",
    }
    card = build_strikeout_card(
        odds_client=object(), date="2026-06-23", season=2026,
        diagnostics=diagnostics,
        max_plausible_edge=1.0,
        list_events=flaky_events,
        props_fetch=lambda c, sk, eid: [prop],
        schedule_fetch=lambda d: parse_schedule_probables(_SCHEDULE),
        form_fetch=lambda pid, season: {"k_per_9": 12.0, "avg_innings": 6.5, "n_games": 5},
        team_k_fetch=lambda tid, season: 0.28,
    )
    assert len(card) == 1
    assert len(calls) == 2
    assert diagnostics["strikeout_prop_event_list_attempts"] == 2
    assert diagnostics["strikeout_prop_feed_status"] == "ready"


def test_event_list_failure_is_diagnosed_without_raising():
    diagnostics = {}

    def failed_events(client, sport_key, date):
        raise ConnectionError("feed unavailable")

    card = build_strikeout_card(
        odds_client=object(), date="2026-06-23", season=2026,
        diagnostics=diagnostics,
        list_events=failed_events,
    )
    assert card == []
    assert diagnostics["strikeout_prop_feed_status"] == "event_list_failed"
    assert diagnostics["strikeout_prop_event_list_attempts"] == 2
    assert diagnostics["strikeout_prop_feed_error_type"] == "ConnectionError"


def test_empty_event_response_is_retried():
    calls = []
    diagnostics = {}

    def empty_then_ready(client, sport_key, date):
        calls.append(1)
        return [] if len(calls) == 1 else [{"id": "evt1"}]

    card = build_strikeout_card(
        odds_client=object(), date="2026-06-23", season=2026,
        diagnostics=diagnostics,
        list_events=empty_then_ready,
        props_fetch=lambda c, sk, eid: [],
        schedule_fetch=lambda d: [],
    )
    assert card == []
    assert len(calls) == 2
    assert diagnostics["strikeout_prop_event_count"] == 1
    assert diagnostics["strikeout_prop_feed_status"] == "no_prop_markets"


def test_build_card_no_form_is_no_data_not_crash():
    events = [{"id": "e", "home_team": "New York Yankees", "away_team": "Chicago White Sox"}]
    prop = {
        "pitcher": "Gerrit Cole", "line": 6.5, "over_odds": -110, "under_odds": -110,
        "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox",
    }
    card = build_strikeout_card(
        odds_client=object(), date="2026-06-23", season=2026,
        list_events=lambda c, sk, d: events,
        props_fetch=lambda c, sk, eid: [prop],
        schedule_fetch=lambda d: parse_schedule_probables(_SCHEDULE),
        form_fetch=lambda pid, season: None,
        team_k_fetch=lambda tid, season: None,
    )
    assert card[0]["recommendation"] == "no_data"


def test_build_card_empty_when_no_props():
    card = build_strikeout_card(
        odds_client=object(), date="2026-06-23", season=2026,
        list_events=lambda c, sk, d: [{"id": "e"}],
        props_fetch=lambda c, sk, eid: [],
        schedule_fetch=lambda d: [],
    )
    assert card == []


def _prop_card(prop, **overrides):
    events = [{"id": "e1", "home_team": "New York Yankees", "away_team": "Chicago White Sox"}]
    kwargs = dict(
        kelly_per_pick_pct=0.01, kelly_total_pct=0.03, kelly_fraction=0.25,
        max_plausible_edge=1.0,  # isolate staking/cap tests from the edge-plausibility guard
        list_events=lambda c, sk, d: events,
        props_fetch=lambda c, sk, eid: [prop],
        schedule_fetch=lambda d: parse_schedule_probables(_SCHEDULE),
        form_fetch=lambda pid, season: {"k_per_9": 12.0, "avg_innings": 6.5, "n_games": 5},
        team_k_fetch=lambda tid, season: 0.28,
    )
    kwargs.update(overrides)
    return build_prop_card(object(), "2026-06-23", 2026, 1000.0, **kwargs)


def _proven_strikeout_over_log():
    return pd.DataFrame({
        "market_type": ["pitcher_strikeouts_over"] * 200,
        "pick": ["Historical Pitcher Over 5.5 Ks"] * 200,
        "result": ["WIN"] * 150 + ["LOSS"] * 50,
        "raw_probability": [0.70] * 200,
        "odds_american": [-110] * 200,
        "game_date": ["2026-06-01"] * 200,
    })


def test_prop_card_keeps_pitcher_market_research_only_during_hits_pilot():
    prop = {
        "pitcher": "Gerrit Cole", "line": 6.5, "over_odds": -115, "under_odds": -105,
        "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox",
    }
    card = _prop_card(prop, prop_results_log=_proven_strikeout_over_log())
    assert len(card) == 1
    row = card.iloc[0]
    assert row["Pick_Status"] == "Research / No Stake"
    assert row["market_type"] == "pitcher_strikeouts_over"
    assert "Gerrit Cole Over 6.5 Ks" == row["best_pick"]
    assert row["CalibrationSource"] == "directional"
    assert row["WinProbability"] <= row["CalibratedProbability"]
    assert not bool(row["production_eligible"])
    assert row["Stake_Status"] == "Research / No Stake"
    assert row["Kelly_Bet_Size"] == 0.0
    assert "batter-hit props only" in row["Status_Reason"]


def test_prop_card_without_graded_ledger_is_research_only():
    prop = {
        "pitcher": "Gerrit Cole", "line": 6.5,
        "over_odds": -115, "under_odds": -105,
        "book": "novig", "home_team": "New York Yankees",
        "away_team": "Chicago White Sox",
    }
    card = _prop_card(prop, prop_results_log=None)
    row = card.iloc[0]
    assert row["CalibrationSource"] == "market_blend_fallback"
    assert not bool(row["production_eligible"])
    assert row["Kelly_Bet_Size"] == 0.0
    assert "calibration" in row["Status_Reason"].lower()


def test_prop_card_total_cap_scales_down_many_picks():
    # Three strong picks; total must not exceed 3% of bankroll ($30 on 1000).
    props = [
        {"pitcher": f"P{i}", "line": 5.5, "over_odds": -110, "under_odds": -110,
         "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox"}
        for i in range(3)
    ]
    sched = {"dates": [{"games": [{"teams": {
        "home": {"team": {"id": 147, "name": "New York Yankees"},
                 "probablePitcher": {"id": 1, "fullName": "P0"}},
        "away": {"team": {"id": 145, "name": "Chicago White Sox"},
                 "probablePitcher": {"id": 2, "fullName": "P1"}},
    }}]}]}
    card = build_prop_card(
        object(), "2026-06-23", 2026, 1000.0,
        kelly_per_pick_pct=0.01, kelly_total_pct=0.03, kelly_fraction=0.25,
        max_plausible_edge=1.0,
        list_events=lambda c, sk, d: [{"id": "e1", "home_team": "New York Yankees", "away_team": "Chicago White Sox"}],
        props_fetch=lambda c, sk, eid: props,
        schedule_fetch=lambda d: parse_schedule_probables(sched),
        form_fetch=lambda pid, season: {"k_per_9": 13.0, "avg_innings": 6.5, "n_games": 5},
        team_k_fetch=lambda tid, season: 0.30,
    )
    assert float(card["Kelly_Bet_Size"].sum()) <= 30.0 + 1e-6


def test_prop_card_empty_when_nothing_clears_edge():
    # No form -> no_data -> not actionable -> empty card (no crash).
    prop = {
        "pitcher": "Gerrit Cole", "line": 6.5, "over_odds": -110, "under_odds": -110,
        "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox",
    }
    card = _prop_card(prop, form_fetch=lambda pid, season: None, team_k_fetch=lambda tid, season: None)
    assert card.empty


def test_prop_card_excludes_opener_and_implausible_edge_by_default():
    # The live-slate failure shape: an opener (low avg innings) projects ~1.5 Ks, making
    # Under 4.5 look like a 40% edge. With guards at production defaults the card is empty.
    prop = {
        "pitcher": "Gerrit Cole", "line": 4.5, "over_odds": -144, "under_odds": 120,
        "book": "draftkings", "home_team": "New York Yankees", "away_team": "Chicago White Sox",
    }
    card = _prop_card(
        prop,
        form_fetch=lambda pid, season: {"k_per_9": 9.0, "avg_innings": 1.2, "n_games": 2},
        max_plausible_edge=PROP_MAX_PLAUSIBLE_EDGE,  # production default -> guards active
    )
    assert card.empty


def test_prop_card_drops_coinflip_plus_money_picks():
    # Owner preference (3 Jul): the card wants the highest CHANCE OF WINNING, not the
    # highest ROI. A near-coin-flip pick whose value is in the plus-money price (the
    # Rangel Over 50.3% / +125 shape) must not make the card even with a passing edge.
    # min_win_probability=0.60 with a pick modeled ~0.55-0.58 -> excluded.
    prop = {
        "pitcher": "Gerrit Cole", "line": 6.5, "over_odds": 125, "under_odds": -145,
        "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox",
    }
    card = _prop_card(prop, min_win_probability=0.99)
    assert card.empty


def test_prop_card_keeps_favorite_and_orders_by_win_probability():
    # Two qualifying picks: the higher-win-probability one must rank FIRST even when
    # the other carries the larger edge/EV (plus-money price play).
    props = [
        # Strong favorite shape: high k_per_9 vs line 5.5 -> high p(over) at minus odds.
        {"pitcher": "P0", "line": 5.5, "over_odds": -150, "under_odds": 130,
         "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox"},
        # Price-play shape: same model read but at a higher line -> lower p(over), plus odds.
        {"pitcher": "P1", "line": 7.5, "over_odds": 140, "under_odds": -160,
         "book": "novig", "home_team": "New York Yankees", "away_team": "Chicago White Sox"},
    ]
    sched = {"dates": [{"games": [{"teams": {
        "home": {"team": {"id": 147, "name": "New York Yankees"},
                 "probablePitcher": {"id": 1, "fullName": "P0"}},
        "away": {"team": {"id": 145, "name": "Chicago White Sox"},
                 "probablePitcher": {"id": 2, "fullName": "P1"}},
    }}]}]}
    card = build_prop_card(
        object(), "2026-06-23", 2026, 1000.0,
        kelly_per_pick_pct=0.01, kelly_total_pct=0.03, kelly_fraction=0.25,
        max_plausible_edge=1.0, min_win_probability=0.0,
        list_events=lambda c, sk, d: [{"id": "e1", "home_team": "New York Yankees", "away_team": "Chicago White Sox"}],
        props_fetch=lambda c, sk, eid: props,
        schedule_fetch=lambda d: parse_schedule_probables(sched),
        form_fetch=lambda pid, season: {"k_per_9": 12.5, "avg_innings": 6.5, "n_games": 5},
        team_k_fetch=lambda tid, season: 0.29,
    )
    assert len(card) >= 2
    probs = card["WinProbability"].tolist()
    assert probs == sorted(probs, reverse=True), "card must be ordered by win probability"

