import pandas as pd

from app_core.nfl_prop_pipeline import (
    attach_nfl_prop_coverage,
    build_nfl_prop_card,
    fetch_nfl_actuals,
    load_nfl_player_forms,
    nfl_prop_feed_message,
    score_nfl_prop,
)
from app_core.prop_grading import grade_prop_export, merge_prop_ledgers
from app_core.prop_odds_ingest import (
    PropOddsFetchError,
    fetch_event_player_prop_markets,
    fetch_nfl_player_props,
    parse_pitcher_props,
)
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
        "attempts": [30, 31, 32, 33, 99],
        "completions": [20, 21, 22, 23, 99],
        "rushing_yards": [20, 25, 30, 35, 999],
        "carries": [5, 6, 7, 8, 99],
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
    assert forms["josh allen"]["pass_attempts"]["expected"] < 40.0
    assert forms["josh allen"]["completions"]["expected"] < 30.0
    assert forms["josh allen"]["rush_attempts"]["expected"] < 10.0


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


def test_nfl_fetch_recovers_supported_market_across_all_us_books(monkeypatch):
    calls = []

    def fake_fetch(client, sport_key, event_id, market_keys, **kwargs):
        calls.append((tuple(market_keys), kwargs))
        if "bookmakers" in kwargs and kwargs["bookmakers"] is None:
            return [{"market_key": "player_pass_yds", "player": "Josh Allen"}]
        return []

    monkeypatch.setattr(
        "app_core.prop_odds_ingest.fetch_pitcher_props", fake_fetch
    )
    monkeypatch.setattr(
        "app_core.prop_odds_ingest.fetch_event_player_prop_markets",
        lambda client, sport_key, event_id, regions: {"player_pass_yds"},
    )
    diagnostics = {}
    rows = fetch_nfl_player_props(
        object(),
        "americanfootball_nfl",
        "event-1",
        diagnostics=diagnostics,
    )

    assert rows == [{"market_key": "player_pass_yds", "player": "Josh Allen"}]
    assert calls[0][0] == (
        "player_pass_yds",
        "player_pass_attempts",
        "player_pass_completions",
        "player_rush_yds",
        "player_rush_attempts",
        "player_reception_yds",
        "player_receptions",
    )
    assert calls[1][0] == ("player_pass_yds",)
    assert calls[1][1]["bookmakers"] is None
    assert set(calls[1][1]["regions"].split(",")) == {"us", "us2"}
    assert diagnostics["nfl_prop_market_discovery_success_count"] == 1
    assert diagnostics["nfl_prop_events_with_supported_markets"] == 1
    assert diagnostics["nfl_prop_broad_book_fallback_count"] == 1


def test_nfl_market_discovery_queries_all_us_books_without_a_book_filter(monkeypatch):
    request = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "bookmakers": [
                    {
                        "key": "caesars",
                        "markets": [
                            {"key": "player_pass_attempts"},
                            {"key": "h2h"},
                        ],
                    },
                    {
                        "key": "espnbet",
                        "markets": [{"key": "player_receptions"}],
                    },
                ]
            }

    def http_get(url, params, timeout):
        request.update({"url": url, "params": params, "timeout": timeout})
        return Response()

    monkeypatch.setattr("requests.get", http_get)

    class Client:
        BASE_URL = "https://example.test/v4"
        api_key = "test"
        regions = "us2,eu"

    keys = fetch_event_player_prop_markets(
        Client(), "americanfootball_nfl_preseason", "event-1"
    )

    assert keys == {"player_pass_attempts", "player_receptions"}
    assert request["url"].endswith(
        "/americanfootball_nfl_preseason/events/event-1/markets"
    )
    assert set(request["params"]["regions"].split(",")) == {"us", "us2"}
    assert "bookmakers" not in request["params"]


def test_nfl_fetch_stops_when_market_inventory_has_no_supported_props(monkeypatch):
    calls = []

    def fake_fetch(client, sport_key, event_id, market_keys, **kwargs):
        calls.append(tuple(market_keys))
        return []

    monkeypatch.setattr(
        "app_core.prop_odds_ingest.fetch_pitcher_props", fake_fetch
    )
    monkeypatch.setattr(
        "app_core.prop_odds_ingest.fetch_event_player_prop_markets",
        lambda client, sport_key, event_id, regions: {"player_anytime_td"},
    )
    diagnostics = {}

    rows = fetch_nfl_player_props(
        object(),
        "americanfootball_nfl_preseason",
        "event-1",
        diagnostics=diagnostics,
    )

    assert rows == []
    assert len(calls) == 1
    assert diagnostics["nfl_prop_market_discovery_success_count"] == 1
    assert diagnostics["nfl_prop_events_without_supported_markets"] == 1
    assert diagnostics["nfl_prop_available_market_keys"] == "player_anytime_td"


def test_nfl_card_exposes_prop_transport_failures():
    diagnostics = {}

    def failed_fetch(client, sport_key, event_id):
        raise PropOddsFetchError("player_prop_http_401")

    card = build_nfl_prop_card(
        object(),
        "2026-08-22",
        2026,
        sport_keys=("americanfootball_nfl_preseason",),
        diagnostics=diagnostics,
        list_events=lambda client, sport_key, date: [{"id": "event-1"}],
        props_fetch=failed_fetch,
        form_loader=lambda season, date: {},
    )

    assert card.empty
    assert diagnostics["nfl_prop_feed_status"] == "event_fetch_failed"
    assert diagnostics["nfl_prop_event_fetch_errors"] == 1


def test_empty_nfl_feed_is_visible_on_combined_mlb_export():
    mlb = pd.DataFrame([{
        "league": "MLB",
        "player": "Juan Soto",
        "best_pick": "Juan Soto Over 0.5 Hits",
    }])
    diagnostics = {
        "nfl_prop_feed_status": "no_prop_markets",
        "nfl_prop_event_count": 3,
        "nfl_prop_events_without_rows": 3,
        "nfl_prop_raw_count": 0,
    }

    combined = attach_nfl_prop_coverage(
        mlb, {"MLB", "NFL"}, diagnostics, nfl_game_count=3
    )

    assert bool(combined.loc[0, "nfl_prop_requested"])
    assert combined.loc[0, "nfl_selected_game_count"] == 3
    assert combined.loc[0, "nfl_prop_feed_status"] == "no_prop_markets"
    assert combined.loc[0, "nfl_prop_event_count"] == 3
    assert combined.loc[0, "nfl_prop_events_without_rows"] == 3
    assert "Rerun closer to kickoff" in nfl_prop_feed_message(diagnostics)


def test_nfl_card_diagnostics_count_events_without_rows():
    diagnostics = {}
    card = build_nfl_prop_card(
        object(),
        "2026-08-21",
        2026,
        diagnostics=diagnostics,
        list_events=lambda client, sport_key, date: [{"id": sport_key}],
        props_fetch=lambda client, sport_key, event_id: [],
        form_loader=lambda season, date: {},
    )

    assert card.empty
    assert diagnostics["nfl_prop_feed_status"] == "no_prop_markets"
    assert diagnostics["nfl_prop_event_count"] == 2
    assert diagnostics["nfl_prop_events_without_rows"] == 2


def test_nfl_card_reports_discovered_but_unsupported_markets():
    diagnostics = {
        "nfl_prop_market_discovery_success_count": 2,
        "nfl_prop_events_without_supported_markets": 2,
    }
    card = build_nfl_prop_card(
        object(),
        "2026-08-22",
        2026,
        diagnostics=diagnostics,
        list_events=lambda client, sport_key, date: [{"id": sport_key}],
        props_fetch=lambda client, sport_key, event_id: [],
        form_loader=lambda season, date: {},
    )

    assert card.empty
    assert diagnostics["nfl_prop_feed_status"] == "no_supported_prop_markets"
    assert "event-market inventory" in nfl_prop_feed_message(diagnostics)


def test_nfl_market_stat_mapping_and_generic_grading():
    assert _stat_for_market(
        "player_pass_attempts_over", "Josh Allen Over 31.5 Pass Attempts"
    ) == "pass_attempts"
    assert _stat_for_market(
        "player_pass_completions_under", "Josh Allen Under 21.5 Pass Completions"
    ) == "completions"
    assert _stat_for_market(
        "player_rush_attempts_over", "Player Over 9.5 Rush Attempts"
    ) == "rush_attempts"
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
    assert actuals["josh allen"]["rush_attempts"] == 8
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
