import pandas as pd

from app_core.nfl_prop_pipeline import (
    build_nfl_prop_card,
    fetch_nfl_actuals,
    load_nfl_player_forms,
    score_nfl_prop,
)
from app_core.prop_grading import grade_prop_export, merge_prop_ledgers
from app_core.prop_odds_ingest import parse_pitcher_props
from app_core.prop_runner import _prop_history_for_league
from scripts.grade_props import _stat_for_market


def _event_payload():
    return {
        "id": "event-1",
        "commence_time": "2026-09-13T17:00:00Z",
        "home_team": "New York Jets",
        "away_team": "Buffalo Bills",
        "bookmakers": [{
            "key": "novig",
            "markets": [{
                "key": "player_pass_yds",
                "outcomes": [
                    {"description": "Josh Allen", "name": "Over", "point": 244.5, "price": -108},
                    {"description": "Josh Allen", "name": "Under", "point": 244.5, "price": -104},
                ],
            }],
        }],
    }


def _forms():
    return {
        "josh allen": {
            "passing_yards": {"expected": 270.0, "sigma": 40.0, "games": 8.0},
        }
    }


def test_nfl_prop_parser_preserves_event_and_player_identity():
    rows = parse_pitcher_props(_event_payload(), "player_pass_yds")

    assert len(rows) == 1
    assert rows[0]["participant_type"] == "nfl_player"
    assert rows[0]["player"] == "Josh Allen"
    assert rows[0]["event_id"] == "event-1"
    assert rows[0]["market_key"] == "player_pass_yds"


def test_nfl_form_loader_excludes_future_current_season_weeks():
    weekly = pd.DataFrame({
        "season": [2025, 2025, 2025, 2026, 2026],
        "week": [15, 16, 17, 1, 2],
        "player_display_name": ["Josh Allen"] * 5,
        "passing_yards": [250, 260, 270, 280, 999],
        "rushing_yards": [20, 25, 30, 35, 999],
        "receiving_yards": [None] * 5,
        "receptions": [None] * 5,
    })
    schedules = pd.DataFrame({
        "week": [1, 2],
        "gameday": ["2026-09-10", "2026-09-20"],
    })

    forms = load_nfl_player_forms(
        2026,
        "2026-09-15",
        weekly_fetch=lambda seasons: weekly,
        schedule_fetch=lambda seasons: schedules,
    )

    passing = forms["josh allen"]["passing_yards"]
    assert passing["games"] == 4.0
    assert passing["expected"] < 280.0
    assert passing["expected"] > 250.0


def test_nfl_prop_scoring_is_explicitly_research_only():
    prop = parse_pitcher_props(_event_payload(), "player_pass_yds")[0]
    row = score_nfl_prop(prop, _forms())

    assert row is not None
    assert row["league"] == "NFL"
    assert row["market_type"] == "player_pass_yds_over"
    assert row["Stake_Status"] == "Research / No Stake"
    assert row["Kelly_Bet_Size"] == 0.0
    assert row["production_eligible"] is False
    assert row["CalibrationSource"] == "nfl_research_only_uncalibrated"


def test_nfl_prop_without_form_still_exports_market_baseline_for_grading():
    prop = parse_pitcher_props(_event_payload(), "player_pass_yds")[0]
    row = score_nfl_prop(prop, {})

    assert row is not None
    assert row["CalibrationSource"] == "nfl_market_only_no_form"
    assert row["FormSampleSize"] == 0
    assert row["Kelly_Bet_Size"] == 0.0
    assert row["production_eligible"] is False


def test_nfl_card_queries_preseason_and_regular_season_keys():
    calls = []
    events = [{
        "id": "event-1",
        "commence_time": "2026-09-13T17:00:00Z",
        "home_team": "New York Jets",
        "away_team": "Buffalo Bills",
    }]
    prop = parse_pitcher_props(_event_payload(), "player_pass_yds")[0]

    def list_events(client, sport_key, date):
        calls.append(sport_key)
        return events

    card = build_nfl_prop_card(
        object(),
        "2026-09-13",
        2026,
        list_events=list_events,
        props_fetch=lambda client, sport_key, event_id: [prop],
        form_loader=lambda season, date: _forms(),
    )

    assert calls == ["americanfootball_nfl_preseason", "americanfootball_nfl"]
    assert len(card) == 1  # duplicate event across keys is collapsed
    assert card.iloc[0]["NFL_Research_Rank"] == 1


def test_nfl_market_stat_mapping_and_generic_grading():
    assert _stat_for_market("player_pass_yds_over", "Josh Allen Over 244.5 Pass Yards") == "passing_yards"
    assert _stat_for_market("player_rush_yds_under", "Player Under 45.5 Rush Yards") == "rushing_yards"
    assert _stat_for_market("player_reception_yds_over", "Player Over 70.5 Receiving Yards") == "receiving_yards"
    assert _stat_for_market("player_receptions_under", "Player Under 5.5 Receptions") == "receptions"

    card = pd.DataFrame([{
        "league": "NFL",
        "player": "Josh Allen",
        "participant_type": "nfl_player",
        "market_type": "player_pass_yds_over",
        "best_pick": "Josh Allen Over 244.5 Pass Yards",
        "line": 244.5,
        "odds_american": -108,
        "Kelly_Bet_Size": 0.0,
        "WinProbability": 0.56,
    }])
    graded = grade_prop_export(
        card,
        "2026-09-13",
        name_resolver=lambda frame, date: {"josh allen": "nfl:josh allen"},
        actual_fetcher=lambda player_id, participant_type: {"passing_yards": 275},
    )

    assert graded.iloc[0]["league"] == "NFL"
    assert graded.iloc[0]["stat"] == "passing_yards"
    assert graded.iloc[0]["actual_value"] == 275
    assert graded.iloc[0]["result"] == "WIN"


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_espn_nfl_actuals_parse_passing_rushing_and_receiving():
    summary = {
        "boxscore": {"players": [{"statistics": [
            {
                "name": "passing",
                "labels": ["C/ATT", "YDS", "TD", "INT"],
                "athletes": [{"athlete": {"displayName": "Josh Allen"}, "stats": ["22/31", "275", "2", "0"]}],
            },
            {
                "name": "rushing",
                "labels": ["CAR", "YDS", "AVG", "TD"],
                "athletes": [{"athlete": {"displayName": "Josh Allen"}, "stats": ["8", "42", "5.3", "1"]}],
            },
            {
                "name": "receiving",
                "labels": ["REC", "YDS", "AVG", "TD"],
                "athletes": [{"athlete": {"displayName": "Keon Coleman"}, "stats": ["6", "81", "13.5", "1"]}],
            },
        ]}]},
    }

    def http_get(url, params, timeout):
        if "scoreboard" in url:
            return _Response({"events": [{"id": "401"}]})
        assert params == {"event": "401"}
        return _Response(summary)

    actuals = fetch_nfl_actuals("2026-09-13", http_get=http_get)

    assert actuals["josh allen"]["passing_yards"] == 275
    assert actuals["josh allen"]["completions"] == 22
    assert actuals["josh allen"]["pass_attempts"] == 31
    assert actuals["josh allen"]["rushing_yards"] == 42
    assert actuals["keon coleman"]["receptions"] == 6
    assert actuals["keon coleman"]["receiving_yards"] == 81


def test_mlb_calibration_history_excludes_nfl_rows_and_ledger_keys_include_league():
    history = pd.DataFrame({
        "league": ["MLB", "NFL"],
        "game_date": ["2026-09-13", "2026-09-13"],
        "player": ["Same Name", "Same Name"],
        "pick": ["Same Pick", "Same Pick"],
        "result": ["WIN", "LOSS"],
    })

    mlb = _prop_history_for_league(history, "MLB")
    combined = merge_prop_ledgers(history.iloc[[0]], history.iloc[[1]])

    assert mlb["league"].tolist() == ["MLB"]
    assert len(combined) == 2
