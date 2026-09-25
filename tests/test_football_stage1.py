"""Synthetic evidence tests; provider facts are never backfilled by these tests."""
from datetime import datetime, timedelta, timezone
import csv
import copy
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from app_core import football_stage1 as stage1
from app_core import prospective_evidence as evidence
from app_core import prospective_remote
from app_core import football_stage1_cycle as cycle


NOW = datetime(2026, 9, 24, 17, 0, tzinfo=timezone.utc)
START = NOW + timedelta(minutes=90)


def nfl_event(*, start=START, completed=False, event_id="401"):
    return {"id": event_id, "date": start.isoformat(), "season": {"year": 2026, "type": 2},
            "week": {"number": 3}, "status": {"type": {"completed": completed}},
            "competitions": [{"date": start.isoformat(), "neutralSite": False,
                              "competitors": [
                                  {"homeAway": "home", "team": {"id": "8", "displayName": "Atlanta Falcons"},
                                   "score": "24"},
                                  {"homeAway": "away", "team": {"id": "9", "displayName": "Green Bay Packers"},
                                   "score": "20"}]}]}


def odds_event(*, start=START, updated=None, price=-110, line=-3.5, sport_key="americanfootball_nfl"):
    updated = updated or NOW - timedelta(minutes=1)
    return {"id": "odds-1", "sport_key": sport_key, "commence_time": start.isoformat(),
            "home_team": "Atlanta Falcons", "away_team": "Green Bay Packers",
            "bookmakers": [{"key": "book_a", "markets": [
                {"key": "spreads", "last_update": updated.isoformat(), "outcomes": [
                    {"name": "Atlanta Falcons", "point": line, "price": price},
                    {"name": "Green Bay Packers", "point": -line, "price": -110}]},
                {"key": "totals", "last_update": updated.isoformat(), "outcomes": [
                    {"name": "Over", "point": 44.0, "price": -110},
                    {"name": "Under", "point": 44.0, "price": -110}]}]}]}


class FootballStage1Test(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "canonical.sqlite3"

    def rows(self, table):
        with evidence.connect(self.path) as db:
            return [dict(x) for x in db.execute("SELECT * FROM " + table)]

    def test_schedule_denominator_independent_of_odds(self):
        report = stage1.coverage(self.path, [nfl_event()], [], sport="NFL", observed=NOW, run_id="r1")
        self.assertEqual(report["scheduled_target_games"], 1)
        self.assertEqual(report["games"][0]["status"], "NO_ODDS_EVENT")
        self.assertFalse(report["games"][0]["spread_price_available"])
        self.assertEqual(len(self.rows("prospective_football_event")), 1)
        cycle_row = self.rows("prospective_football_cycle_coverage")[0]
        self.assertEqual(json.loads(cycle_row["raw_source"])["games"][0]["status"], "NO_ODDS_EVENT")

    def test_duplicate_schedule_and_unparseable_rows_remain_visible(self):
        event = nfl_event()
        report = stage1.coverage(self.path, [event, event, {"id": "bad"}], [],
                                 sport="NFL", observed=NOW, run_id="r1")
        self.assertEqual(report["scheduled_target_games"], 1)
        self.assertEqual(report["duplicate_schedule_events"], 1)
        self.assertEqual(len(report["games"]), 2)
        self.assertEqual(len(self.rows("prospective_football_event")), 1)
        self.assertEqual(len(self.rows("prospective_football_cycle_coverage")), 1)
        self.assertEqual(len(json.loads(self.rows("prospective_football_cycle_coverage")[0]["raw_source"])["games"]), 2)

    def test_schedule_versions_follow_schedule_facts_not_live_status_metadata(self):
        original = nfl_event()
        first, created = stage1.append_schedule(self.path, "NFL", original, NOW)
        self.assertTrue(created)
        changed_status = copy.deepcopy(original)
        changed_status["status"]["type"]["completed"] = True
        changed_status["presentation_updated"] = "later"
        second, created = stage1.append_schedule(self.path, "NFL", changed_status,
                                                  NOW + timedelta(minutes=30))
        self.assertFalse(created)
        self.assertEqual(first["version_id"], second["version_id"])
        self.assertEqual(len(self.rows("prospective_football_event")), 1)
        rescheduled = copy.deepcopy(changed_status)
        rescheduled["competitions"][0]["date"] = (START + timedelta(hours=1)).isoformat()
        third, created = stage1.append_schedule(self.path, "NFL", rescheduled,
                                                 NOW + timedelta(minutes=31))
        self.assertTrue(created)
        self.assertNotEqual(first["version_id"], third["version_id"])

    def test_capture_horizon_boundaries_cover_mid_pregame_gap(self):
        self.assertEqual(stage1.horizon(START + timedelta(days=8), NOW), "SNAPSHOT_WINDOW_MISSED")
        self.assertEqual(stage1.horizon(NOW + timedelta(days=7), NOW), "EARLY_RESEARCH")
        self.assertEqual(stage1.horizon(NOW + timedelta(hours=24), NOW), "EARLY_RESEARCH")
        self.assertEqual(stage1.horizon(NOW + timedelta(hours=23, minutes=59), NOW), "MID_PREGAME")
        self.assertEqual(stage1.horizon(NOW + timedelta(minutes=121), NOW), "MID_PREGAME")
        self.assertEqual(stage1.horizon(NOW + timedelta(minutes=120), NOW), "FINAL_LEGAL_PREGAME")
        self.assertEqual(stage1.horizon(NOW + timedelta(minutes=5), NOW), "FINAL_LEGAL_PREGAME")
        self.assertEqual(stage1.horizon(NOW + timedelta(minutes=4), NOW), "SNAPSHOT_WINDOW_MISSED")

    def test_mid_pregame_quote_is_preserved_as_distinct_research_horizon(self):
        start = NOW + timedelta(hours=6)
        report = stage1.coverage(self.path, [nfl_event(start=start)], [odds_event(start=start)],
                                 sport="NFL", observed=NOW, run_id="mid")
        self.assertEqual(report["games"][0]["status"], "QUOTE_CAPTURED")
        rows = self.rows("prospective_football_quote")
        self.assertEqual(len(rows), 4)
        self.assertEqual({x["capture_horizon"] for x in rows}, {"MID_PREGAME"})

    def test_unchanged_offer_crossing_horizon_boundary_keeps_both_snapshots(self):
        start = NOW + timedelta(hours=24)
        event = nfl_event(start=start)
        offer = odds_event(start=start)
        first = stage1.coverage(self.path, [event], [offer], sport="NFL", observed=NOW, run_id="early")
        self.assertEqual(first["games"][0]["status"], "QUOTE_CAPTURED")
        early = {row["quote_id"]: row for row in self.rows("prospective_football_quote")}
        observed = NOW + timedelta(minutes=1)
        second = stage1.coverage(self.path, [event], [offer], sport="NFL", observed=observed, run_id="mid")
        self.assertEqual(second["games"][0]["status"], "QUOTE_CAPTURED")
        rows = self.rows("prospective_football_quote")
        self.assertEqual(len(rows), 8)
        self.assertEqual({row["capture_horizon"] for row in rows}, {"EARLY_RESEARCH", "MID_PREGAME"})
        self.assertTrue(all(row in rows for row in early.values()))
        repeated = stage1.coverage(self.path, [event], [offer], sport="NFL", observed=observed,
                                   run_id="mid-repeat")
        self.assertEqual(repeated["games"][0]["status"], "HORIZON_ALREADY_CAPTURED")
        self.assertEqual(len(self.rows("prospective_football_quote")), 8)

    def test_exact_quote_and_idempotence(self):
        event = nfl_event()
        quote = odds_event()
        first = stage1.coverage(self.path, [event], [quote], sport="NFL", observed=NOW, run_id="r1")
        second = stage1.coverage(self.path, [event], [quote], sport="NFL", observed=NOW, run_id="r2")
        self.assertEqual(first["scheduled_target_games"], second["scheduled_target_games"])
        self.assertEqual(second["games"][0]["status"], "HORIZON_ALREADY_CAPTURED")
        self.assertEqual(len(self.rows("prospective_football_quote")), 4)
        self.assertEqual({x["capture_horizon"] for x in self.rows("prospective_football_quote")},
                         {"FINAL_LEGAL_PREGAME"})
        self.assertTrue(first["games"][0]["spread_price_available"])
        self.assertTrue(first["games"][0]["total_price_available"])
        with self.assertRaises(sqlite3.DatabaseError):
            with evidence.connect(self.path) as db:
                db.execute("UPDATE prospective_football_quote SET american_odds=-120")

    def test_all_supported_books_preserved_without_extra_games(self):
        quote = odds_event()
        second = copy.deepcopy(quote["bookmakers"][0])
        second["key"] = "book_b"
        quote["bookmakers"].append(second)
        report = stage1.coverage(self.path, [nfl_event()], [quote], sport="NFL", observed=NOW, run_id="r")
        self.assertEqual(report["scheduled_target_games"], 1)
        self.assertEqual(len(self.rows("prospective_football_quote")), 8)
        self.assertEqual({x["sportsbook"] for x in self.rows("prospective_football_quote")}, {"book_a", "book_b"})

    def test_late_future_and_invalid_price_rejected(self):
        schedule, _ = stage1.append_schedule(self.path, "NFL", nfl_event(), NOW)
        self.assertEqual(stage1.append_offers(self.path, schedule, odds_event(updated=NOW+timedelta(minutes=1)), NOW, "r")[0], 0)
        self.assertEqual(stage1.append_offers(self.path, schedule, odds_event(price=-99), NOW, "r")[0], 2)
        self.assertEqual(stage1.append_offers(self.path, schedule, odds_event(), START, "r")[1], "SNAPSHOT_WINDOW_MISSED")

    def test_wrong_sport_and_ambiguous_identity(self):
        schedule, _ = stage1.append_schedule(self.path, "NFL", nfl_event(), NOW)
        quote = odds_event(sport_key="americanfootball_ncaaf")
        self.assertEqual(stage1.append_offers(self.path, schedule, quote, NOW, "r")[1], "EVENT_IDENTITY_AMBIGUOUS")
        diagnostic = stage1.provider_diagnostic([schedule], [quote], sport="NFL", observed=NOW)
        self.assertEqual(diagnostic[0]["status"], "WRONG_SPORT_KEY")
        future = odds_event(start=NOW + timedelta(days=10))
        diagnostic = stage1.provider_diagnostic([schedule], [future], sport="NFL", observed=NOW,
                                                schedule_window_start=NOW-timedelta(days=8),
                                                schedule_window_end=NOW+timedelta(days=8))
        self.assertEqual(diagnostic[0]["status"], "OUTSIDE_SCHEDULE_WINDOW")

    def test_provider_reconciliation_classifies_each_event_without_forcing_kickoff(self):
        schedule, _ = stage1.append_schedule(self.path, "NFL", nfl_event(), NOW)
        matched = odds_event()
        revised = odds_event(start=START + timedelta(minutes=10))
        revised["id"] = "revised"
        outside = odds_event(start=START + timedelta(days=10))
        outside["id"] = "outside"
        invalid = odds_event(sport_key="americanfootball_ncaaf")
        invalid["id"] = "wrong-sport"
        events = [matched, matched, revised, outside, invalid]
        rows = stage1.provider_diagnostic([schedule], events, sport="NFL", observed=NOW,
                                          schedule_window_start=NOW-timedelta(days=7),
                                          schedule_window_end=NOW+timedelta(days=8))
        self.assertEqual([x["classification"] for x in rows],
                         ["MATCHED_TARGET", "DUPLICATE_PROVIDER_EVENT", "KICKOFF_TIME_REVISION",
                          "OUTSIDE_TARGET_WINDOW", "PROVIDER_DATA_INVALID"])
        self.assertEqual(rows[2]["kickoff_delta_seconds"], 600)
        self.assertEqual(rows[2]["candidate_game_ids"], [schedule["game_id"]])
        self.assertTrue(all(x["classification_reason"] for x in rows))
        coverage = stage1.coverage(self.path, [nfl_event()], [revised], sport="NFL", observed=NOW,
                                   run_id="revision")
        self.assertEqual(coverage["games"][0]["status"], "NO_ODDS_EVENT")

    def test_ambiguous_kickoff_candidates_never_match(self):
        first, _ = stage1.append_schedule(self.path, "NFL", nfl_event(), NOW)
        second, _ = stage1.append_schedule(self.path, "NFL", nfl_event(start=START + timedelta(seconds=30),
                                                                       event_id="402"), NOW)
        rows = stage1.provider_diagnostic([first, second], [odds_event()], sport="NFL", observed=NOW)
        self.assertEqual(rows[0]["classification"], "AMBIGUOUS_MATCH")
        self.assertEqual(len(rows[0]["candidate_game_ids"]), 2)
        revised = odds_event(start=START + timedelta(minutes=10))
        rows = stage1.provider_diagnostic([first, second], [revised], sport="NFL", observed=NOW)
        self.assertEqual(rows[0]["classification"], "AMBIGUOUS_MATCH")
        self.assertIsNone(rows[0]["canonical_match"])

    def test_training_manifest_uses_one_deterministic_observation_per_game_market(self):
        event = nfl_event()
        offer = odds_event()
        second_book = copy.deepcopy(offer["bookmakers"][0])
        second_book["key"] = "book_b"
        offer["bookmakers"].insert(0, second_book)
        denominator = stage1.coverage(self.path, [event], [offer], sport="NFL", observed=NOW,
                                      run_id="manifest")
        schedule = self.rows("prospective_football_event")[0]
        completed = nfl_event(completed=True)
        raw = {"provider_event_id": "401", "home_team_id": "8", "away_team_id": "9",
               "home_score": 24, "away_score": 20, "status": "FINAL", "provider_response": completed}
        result, _ = stage1.append_result(self.path, schedule, raw, START + timedelta(hours=3), source="ESPN")
        self.assertEqual(stage1.settle_game(self.path, schedule, result, START + timedelta(hours=3)), 8)
        with evidence.connect(self.path) as db:
            first = [dict(x) for x in db.execute("SELECT * FROM prospective_football_training_manifest ORDER BY market_family")]
            self.assertEqual(db.execute("SELECT count(*) FROM prospective_football_active_training_row").fetchone()[0], 8)
        self.assertEqual(len(first), 2)
        self.assertEqual({x["sportsbook"] for x in first}, {"book_a"})
        self.assertEqual({x["market_family"] for x in first}, {"SPREAD", "TOTAL"})
        self.assertTrue(all(x["event_source_hash"] and x["quote_source_hash"] and
                            x["result_source_hash"] for x in first))
        self.assertEqual(stage1.settle_game(self.path, schedule, result, START + timedelta(hours=3)), 0)
        with evidence.connect(self.path) as db:
            second = [dict(x) for x in db.execute("SELECT * FROM prospective_football_training_manifest ORDER BY market_family")]
        self.assertEqual(first, second)
        readiness = cycle._readiness(self.path, denominator, START + timedelta(hours=3),
                                     {"401": completed})
        self.assertEqual(readiness["market_summary"]["SPREAD"]["independent_manifest_events"], 1)
        self.assertEqual(readiness["market_summary"]["TOTAL"]["independent_manifest_events"], 1)
        self.assertEqual(cycle._lifecycle(self.path, denominator, readiness)["status"], "PROVED")

    def test_market_summary_counts_only_current_denominator_games(self):
        event = nfl_event(event_id="402")
        offer = odds_event()
        stage1.coverage(self.path, [event], [offer], sport="NFL", observed=NOW, run_id="priced")
        schedule = next(x for x in self.rows("prospective_football_event") if x["provider_event_id"] == "402")
        completed = nfl_event(completed=True, event_id="402")
        raw = {"provider_event_id": "402", "home_team_id": "8", "away_team_id": "9",
               "home_score": 24, "away_score": 20, "status": "FINAL", "provider_response": completed}
        result, _ = stage1.append_result(self.path, schedule, raw, START + timedelta(hours=3), source="ESPN")
        stage1.settle_game(self.path, schedule, result, START + timedelta(hours=3))
        current = stage1.coverage(self.path, [nfl_event(event_id="401")], [], sport="NFL",
                                  observed=NOW, run_id="current")
        readiness = cycle._readiness(self.path, current, NOW)
        self.assertEqual(readiness["target_games"], 1)
        for market in ("SPREAD", "TOTAL"):
            self.assertEqual(readiness["market_summary"][market]["independent_manifest_events"], 0)
            self.assertEqual(readiness["market_summary"][market]["settled_rows"], 0)
            self.assertEqual(readiness["market_summary"][market]["training_blocked_games"], 1)

    def test_result_settlement_and_training_readiness(self):
        event = nfl_event()
        schedule, _ = stage1.append_schedule(self.path, "NFL", event, NOW)
        stage1.append_offers(self.path, schedule, odds_event(), NOW, "r")
        result_raw = {"provider_event_id": "401", "home_team_id": "8", "away_team_id": "9",
                      "home_score": 24, "away_score": 20, "status": "FINAL",
                      "provider_response": nfl_event(completed=True)}
        result, created = stage1.append_result(self.path, schedule, result_raw,
                                               START + timedelta(hours=3), source="ESPN")
        self.assertTrue(created)
        self.assertEqual(stage1.settle_game(self.path, schedule, result, START + timedelta(hours=3)), 4)
        self.assertEqual(stage1.settle_game(self.path, schedule, result, START + timedelta(hours=3)), 0)
        self.assertEqual(len(self.rows("prospective_football_training_row")), 4)
        self.assertEqual({x["training_row_status"] for x in self.rows("prospective_football_training_row")},
                         {"TRAINING_READY"})
        self.assertEqual({x["outcome"] for x in self.rows("prospective_football_settlement")},
                         {"HOME_COVER", "PUSH"})
        with self.assertRaises(ValueError):
            stage1.append_result(self.path, schedule, result_raw, NOW, source="ESPN")

    def test_result_revisions_do_not_expose_duplicate_or_conflicting_labels(self):
        event = nfl_event()
        denominator = stage1.coverage(self.path, [event], [odds_event()], sport="NFL",
                                      observed=NOW, run_id="r")
        schedule = self.rows("prospective_football_event")[0]
        def result_raw(source):
            competitors = {x["homeAway"]: x for x in source["competitions"][0]["competitors"]}
            return {"provider_event_id": "401", "home_team_id": "8", "away_team_id": "9",
                    "home_score": int(competitors["home"]["score"]),
                    "away_score": int(competitors["away"]["score"]),
                    "status": "FINAL", "provider_response": source}
        observed = START + timedelta(hours=3)
        first_source = nfl_event(completed=True)
        first, _ = stage1.append_result(self.path, schedule, result_raw(first_source), observed, source="ESPN")
        self.assertEqual(stage1.settle_game(self.path, schedule, first, observed), 4)
        revision = copy.deepcopy(first_source)
        revision["summary"] = "provider metadata revision"
        same_score, created = stage1.append_result(self.path, schedule, result_raw(revision),
                                                   observed + timedelta(minutes=1), source="ESPN")
        self.assertFalse(created)
        self.assertEqual(same_score["result_id"], first["result_id"])
        self.assertEqual(len(self.rows("prospective_football_result")), 1)
        self.assertEqual(stage1.settle_game(self.path, schedule, same_score,
                                            observed + timedelta(minutes=1)), 0)
        with evidence.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM prospective_football_active_training_row").fetchone()[0], 4)
        corrected = copy.deepcopy(revision)
        corrected["competitions"][0]["competitors"][0]["score"] = "21"
        corrected_result, created = stage1.append_result(
            self.path, schedule, result_raw(corrected), observed + timedelta(minutes=2), source="ESPN")
        self.assertTrue(created)
        self.assertEqual(stage1.settle_game(self.path, schedule, corrected_result,
                                            observed + timedelta(minutes=2)), 4)
        with evidence.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM prospective_football_active_training_row").fetchone()[0], 0)
        readiness = cycle._readiness(self.path, denominator, observed + timedelta(minutes=2))
        self.assertEqual(readiness["training_ready_rows"], 0)
        self.assertEqual({m["blockers"][0] for m in readiness["games"][0]["markets"]},
                         {"RESULT_CORRECTION_CONFLICT"})

    def test_pushes_and_ncaaf_exact_alias(self):
        self.assertEqual(stage1.outcome({"sport": "NFL", "market_family": "TOTAL", "selection": "Over",
                                         "line": 44.0}, {"result_status": "FINAL", "home_score": 24,
                                                        "away_score": 20}), "PUSH")
        self.assertEqual(stage1._name("NCAAF", "Mizzou"), stage1._name("NCAAF", "Missouri"))
        self.assertNotEqual(stage1._name("NCAAF", "Random State"), stage1._name("NCAAF", "Random University"))

    def test_ncaaf_authoritative_team_catalog_aliases(self):
        source = {"id": 77, "season": 2026, "week": 4, "seasonType": "regular",
                  "homeTeam": "Ohio State", "awayTeam": "Missouri", "homeId": 10, "awayId": 20,
                  "startDate": START.isoformat(), "neutralSite": True, "venue": "Test Stadium"}
        catalog = {"10": {"id": 10, "school": "Ohio State", "mascot": "Buckeyes"},
                   "20": {"id": 20, "school": "Missouri", "mascot": "Tigers"}}
        aliases = {"10": {stage1._name("NCAAF", "Ohio State"), stage1._name("NCAAF", "Ohio State Buckeyes")},
                   "20": {stage1._name("NCAAF", "Missouri"), stage1._name("NCAAF", "Missouri Tigers")}}
        odds = odds_event(sport_key="americanfootball_ncaaf")
        odds["home_team"], odds["away_team"] = "Ohio State Buckeyes", "Missouri Tigers"
        for market in odds["bookmakers"][0]["markets"]:
            if market["key"] == "spreads":
                market["outcomes"][0]["name"] = odds["home_team"]
                market["outcomes"][1]["name"] = odds["away_team"]
        report = stage1.coverage(self.path, [source], [odds], sport="NCAAF", observed=NOW,
                                 run_id="r", team_catalog=catalog, aliases=aliases)
        self.assertEqual(report["games"][0]["status"], "QUOTE_CAPTURED")
        self.assertEqual(len(self.rows("prospective_football_team_identity")), 2)
        self.assertEqual(len(self.rows("prospective_football_quote")), 4)
        schedule = self.rows("prospective_football_event")[0]
        completed = dict(source, completed=True, homePoints=24, awayPoints=20)
        raw_result = {"provider_event_id": "77", "home_team_id": "10", "away_team_id": "20",
                      "home_score": 24, "away_score": 20, "status": "FINAL",
                      "provider_response": completed}
        result, created = stage1.append_result(self.path, schedule, raw_result,
                                               START + timedelta(hours=3), source="CFBD")
        self.assertTrue(created)
        self.assertEqual(stage1.settle_game(self.path, schedule, result, START + timedelta(hours=3)), 4)
        labels = {q["selection"]: t["label"] for q in self.rows("prospective_football_quote")
                  for t in self.rows("prospective_football_training_row") if t["quote_id"] == q["quote_id"]}
        self.assertEqual(labels["Ohio State Buckeyes"], "WIN")
        with evidence.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM prospective_football_training_manifest").fetchone()[0], 2)
        odds["away_team"] = "Ambiguous Unknown"
        self.assertFalse(stage1._same_identity(self.rows("prospective_football_event")[0], odds, aliases))

    def test_theover_blank_probability_and_distinct_model_rate(self):
        schedule, _ = stage1.append_schedule(self.path, "NFL", nfl_event(), NOW)
        csv_path = Path(self.temp.name) / "sides.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["League", "HomeTeam", "AwayTeam", "Market",
                                                         "Pick", "Line", "WinProbability", "ModelHitRate"])
            writer.writeheader()
            writer.writerow({"League": "NFL", "HomeTeam": "Atlanta", "AwayTeam": "Green Bay",
                             "Market": "Spread", "Pick": "ATL", "Line": "3.5", "WinProbability": "",
                             "ModelHitRate": "0.615"})
        report = stage1.ingest_theover(self.path, csv_path, [schedule], NOW)
        self.assertEqual(report["rows"][0]["matched_game_id"], schedule["game_id"])
        row = self.rows("prospective_football_theover")[0]
        self.assertIsNone(row["win_probability"])
        self.assertEqual(row["model_hit_rate"], 0.615)

    def test_theover_line_disagreement_is_research_only(self):
        schedule, _ = stage1.append_schedule(self.path, "NFL", nfl_event(), NOW)
        stage1.append_offers(self.path, schedule, odds_event(), NOW, "r")
        csv_path = Path(self.temp.name) / "totals.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["League", "HomeTeam", "AwayTeam", "Market",
                                                         "Pick", "Line", "WinProbability", "ModelHitRate"])
            writer.writeheader()
            writer.writerow({"League": "NFL", "HomeTeam": "Atlanta", "AwayTeam": "Green Bay",
                             "Market": "Total", "Pick": "OVER", "Line": "45.5",
                             "WinProbability": "", "ModelHitRate": "0.615"})
        report = stage1.ingest_theover(self.path, csv_path, [schedule], NOW)
        self.assertEqual(report["rows"][0]["status"], "LINE_DISAGREEMENT")
        self.assertEqual(len(self.rows("prospective_football_quote")), 4)

    def test_authenticated_cycle_order_and_one_sport_failure(self):
        calls = []
        class Response:
            def __init__(self, status, payload):
                self.status_code, self.payload = status, payload
            def json(self):
                return self.payload
        def get(url, *, params, headers=None, **kwargs):
            calls.append(url)
            if url == cycle.ESPN:
                return Response(200, {"events": [nfl_event()] if params["dates"] == "20260924" else []})
            if url.endswith("americanfootball_nfl/odds"):
                return Response(200, [odds_event()])
            if url.endswith("/games"):
                return Response(429, {})
            raise AssertionError(url)
        sync_calls = []
        def sync(path, client, folder, session=None):
            sync_calls.append(len(calls))
            return {"records_restored": 0, "remote_records_read": 0, "records_verified": 0}
        with patch.object(cycle.prospective_remote, "sync", side_effect=sync):
            report = cycle.run_cycle(self.path, "folder", object(), "odds-secret", "cfbd-secret", now=NOW, get=get)
        self.assertEqual(sync_calls[0], 0)
        self.assertGreater(sync_calls[1], 0)
        self.assertFalse(report["requested_slate_success"])
        self.assertTrue(report["sports"]["NFL"]["requested_slate_success"])
        self.assertEqual(report["sports"]["NCAAF"]["errors"][0]["reason"], "CFBD_RATE_LIMIT")
        self.assertEqual(len(self.rows("prospective_football_quote")), 4)
        self.assertNotIn("odds-secret", str(report))
        self.assertNotIn("cfbd-secret", str(report))
        for table in ("prospective_model", "prospective_calibration", "prospective_prediction",
                      "prospective_validation_plan", "prospective_deployment_review"):
            self.assertEqual(len(self.rows(table)), 0)

    def test_due_games_without_verified_quotes_fail_requested_slate(self):
        # A previous valid snapshot must not hide a broken current provider response.
        stage1.coverage(self.path, [nfl_event()], [odds_event()], sport="NFL", observed=NOW,
                        run_id="earlier")
        class Response:
            status_code = 200
            def __init__(self, payload): self.payload = payload
            def json(self): return self.payload
        def get(url, *, params, **kwargs):
            if url == cycle.ESPN:
                return Response({"events": [nfl_event()] if params["dates"] == "20260924" else []})
            if url.endswith("americanfootball_nfl/odds"):
                return Response([])
            if url.endswith("/games"):
                return Response([{"id": 77, "startDate": (NOW + timedelta(days=30)).isoformat()}])
            if url.endswith("/teams/fbs"):
                return Response([{"id": 10, "school": "Ohio State"}])
            if url.endswith("americanfootball_ncaaf/odds"):
                return Response([])
            raise AssertionError(url)
        with patch.object(cycle.prospective_remote, "sync", return_value={"records_verified": 0}):
            report = cycle.run_cycle(self.path, "folder", object(), "o", "c", now=NOW, get=get)
        self.assertFalse(report["requested_slate_success"])
        self.assertTrue(report["sports"]["NFL"]["denominator"]["games"][0]["spread_price_available"])
        self.assertEqual(report["sports"]["NFL"]["errors"][-1]["reason"],
                         "NFL_NO_VERIFIED_PREGAME_PRICE")
        self.assertTrue(report["sports"]["NCAAF"]["requested_slate_success"])

    def test_one_priced_game_cannot_hide_other_due_game_without_price(self):
        class Response:
            status_code = 200
            def __init__(self, payload): self.payload = payload
            def json(self): return self.payload
        other = nfl_event(start=START - timedelta(minutes=30), event_id="402")
        def get(url, *, params, **kwargs):
            if url == cycle.ESPN:
                return Response({"events": [nfl_event(), other] if params["dates"] == "20260924" else []})
            if url.endswith("americanfootball_nfl/odds"):
                return Response([odds_event()])
            if url.endswith("/games"):
                return Response([{"id": 77, "startDate": (NOW + timedelta(days=30)).isoformat()}])
            if url.endswith("/teams/fbs"):
                return Response([{"id": 10, "school": "Ohio State"}])
            if url.endswith("americanfootball_ncaaf/odds"):
                return Response([])
            raise AssertionError(url)
        with patch.object(cycle.prospective_remote, "sync", return_value={"records_verified": 0}):
            report = cycle.run_cycle(self.path, "folder", object(), "o", "c", now=NOW, get=get)
        self.assertFalse(report["requested_slate_success"])
        self.assertEqual(report["sports"]["NFL"]["errors"][-1]["missing_game_ids"],
                         ["nfl:espn:402"])

    def test_ncaaf_fbs_population_and_legitimate_empty_window(self):
        class Response:
            status_code = 200
            def __init__(self, payload): self.payload = payload
            def json(self): return self.payload
        future = NOW + timedelta(days=30)
        game = {"id": 77, "season": 2026, "week": 8, "seasonType": "regular",
                "homeTeam": "Ohio State", "awayTeam": "FCS College", "homeId": 10,
                "awayId": 999, "startDate": future.isoformat()}
        def get(url, **kwargs):
            return Response([game] if url.endswith("/games") else
                            [{"id": 10, "school": "Ohio State", "mascot": "Buckeyes"}])
        from collections import Counter
        ledger = {"CFBD": Counter()}
        games, policy, catalog, aliases, diagnostic = cycle._ncaaf_schedule(NOW, "token", get=get, ledger=ledger)
        self.assertEqual(games, [])
        self.assertEqual(diagnostic["window_games"], 0)
        game["startDate"] = START.isoformat()
        games, policy, catalog, aliases, diagnostic = cycle._ncaaf_schedule(NOW, "token", get=get, ledger=ledger)
        schedule, _ = stage1.append_schedule(self.path, "NCAAF", games[0], NOW, catalog)
        self.assertEqual(policy(schedule), "FBS_VS_NON_FBS_OR_UNKNOWN_EXCLUDED")

    def test_failed_backup_verification_fails_closed(self):
        def sync(path, client, folder, session=None):
            if not session.get("restored"):
                session["restored"] = True
                return {"records_restored": 0}
            raise ValueError("readback conflict")
        def get(url, *, params, **kwargs):
            class Response:
                status_code = 200
                def json(self):
                    if url == cycle.ESPN:
                        return {"events": []}
                    if url.endswith("/games"):
                        return [{"id": 1}]
                    if url.endswith("/teams/fbs"):
                        return [{"id": 1}]
                    return []
            return Response()
        with patch.object(cycle.prospective_remote, "sync", side_effect=sync):
            report = cycle.run_cycle(self.path, "folder", object(), "o", "c", now=NOW, get=get)
        self.assertEqual(report["execution_state"], "BACKUP_VERIFICATION_FAILED")
        self.assertFalse(report["requested_slate_success"])

    def test_canonical_remote_roundtrip_verifies_new_tables(self):
        stage1.coverage(self.path, [nfl_event()], [odds_event()], sport="NFL", observed=NOW, run_id="r")
        class Store:
            def __init__(self):
                self.objects = {}
            def read_objects(self, Prefix):
                return ((k, v) for k, v in self.objects.items() if k.startswith(Prefix))
            def put_object(self, *, Bucket, Key, Body, **kwargs):
                self.objects[Key] = Body
            def get_object(self, *, Bucket, Key):
                return {"Body": BytesIO(self.objects[Key])}
        store = Store()
        saved = prospective_remote.sync(self.path, store, "folder")
        restored_path = Path(self.temp.name) / "restored.sqlite3"
        restored = prospective_remote.sync(restored_path, store, "folder")
        self.assertEqual(saved["new_records_verified"], restored["records_restored"])
        with evidence.connect(restored_path) as db:
            self.assertEqual(db.execute("SELECT count(*) FROM prospective_football_quote").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT count(*) FROM prospective_football_cycle_coverage").fetchone()[0], 1)

    def test_result_coverage_counts_provider_completed_games_only(self):
        started = NOW - timedelta(hours=1)
        event = nfl_event(start=started, completed=False)
        denominator = stage1.coverage(self.path, [event], [], sport="NFL", observed=NOW, run_id="r")
        readiness = cycle._readiness(self.path, denominator, NOW)
        self.assertEqual(readiness["completed_games_in_window"], 0)
        self.assertIsNone(readiness["rates"]["result_coverage_for_completed_games"])
        completed = nfl_event(start=started, completed=True)
        raw_result = {"provider_event_id": "401", "home_team_id": "8", "away_team_id": "9",
                      "home_score": 24, "away_score": 20, "status": "FINAL",
                      "provider_response": completed}
        schedule = self.rows("prospective_football_event")[0]
        stage1.append_result(self.path, schedule, raw_result, NOW, source="ESPN")
        refreshed = cycle._readiness(self.path, denominator, NOW, {"401": completed})
        self.assertEqual(refreshed["completed_games_in_window"], 1)
        self.assertEqual(refreshed["completed_games_with_result"], 1)


if __name__ == "__main__":
    unittest.main()
