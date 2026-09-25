"""Stage 2 gates use synthetic fixtures; no invented row enters production evidence."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from app_core import football_stage1 as stage1
from app_core import football_stage2 as stage2
from app_core import prospective_evidence as evidence
from tests.test_football_stage1 import NOW, START, nfl_event, odds_event


def verified_refresh():
    return {"run_id": "synthetic-stage1-test", "execution_state": "COMPLETE",
            "remote": {"pre_mutation_verified": True, "backup_readback_verified": True}}


def one_game(path, *, two_books=False):
    offer = odds_event()
    if two_books:
        second = copy.deepcopy(offer["bookmakers"][0])
        second["key"] = "second_book"
        offer["bookmakers"].append(second)
    stage1.coverage(path, [nfl_event()], [offer], sport="NFL", observed=NOW, run_id="stage2")
    with evidence.connect(path) as db:
        event = dict(db.execute("SELECT * FROM prospective_football_event").fetchone())
    raw_result = {"provider_event_id": "401", "home_team_id": "8", "away_team_id": "9",
                  "home_score": 24, "away_score": 20, "status": "FINAL",
                  "provider_response": nfl_event(completed=True)}
    result, _ = stage1.append_result(path, event, raw_result,
                                     START + timedelta(hours=3), source="ESPN")
    stage1.settle_game(path, event, result, START + timedelta(hours=3))


def test_authenticated_readback_required_and_empty_scopes_fail_closed(tmp_path):
    path = tmp_path / "evidence.sqlite3"
    with pytest.raises(ValueError, match="STAGE1_AUTHENTICATED_REFRESH_NOT_VERIFIED"):
        stage2.build_reports(path, Path(__file__).resolve().parents[1],
                             stage1_report={"run_id": "x", "execution_state": "COMPLETE"}, source_commit="abc")
    refresh = verified_refresh()
    refresh["sports"] = {"NCAAF": {"readiness": {"market_summary": {
        "SPREAD": {"blocker_counts": {"NO_VERIFIED_PREGAME_PRICE": 68}}}}}}
    reports = stage2.build_reports(path, Path(__file__).resolve().parents[1],
                                   stage1_report=refresh, source_commit="abc")
    assert len(reports) == 8
    assert reports["training_audit"]["scopes"]["NCAAF/SPREAD"]["stage1_current_slate"][
        "blocker_counts"]["NO_VERIFIED_PREGAME_PRICE"] == 68
    for key in ("NFL/SPREAD", "NFL/TOTAL", "NCAAF/SPREAD", "NCAAF/TOTAL"):
        assert reports["training_audit"]["scopes"][key]["legal_independent_n"] == 0
        assert reports["model_inventory"]["scopes"][key]["model_status"] == "INSUFFICIENT_TRAINING_EVIDENCE"
        assert reports["model_inventory"]["scopes"][key]["model_id"] is None
        assert reports["calibration_readiness"]["scopes"][key]["calibration_id"] is None
        research = reports["research_prediction_readiness"]["scopes"][key]
        assert research["production_eligible"] is False and research["stake"] == 0
        assert research["production_parlay_eligible"] is False and research["wager_eligible"] is False


def test_books_and_reprices_do_not_inflate_independent_n_or_write_authority(tmp_path):
    path = tmp_path / "evidence.sqlite3"
    one_game(path, two_books=True)
    with evidence.connect(path) as db:
        before = {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in
                  ("prospective_model", "prospective_calibration", "prospective_prediction",
                   "prospective_deployment_review")}
    reports = stage2.build_reports(path, Path(__file__).resolve().parents[1],
                                   stage1_report=verified_refresh(), source_commit="abc")
    for key in ("NFL/SPREAD", "NFL/TOTAL"):
        item = reports["training_audit"]["scopes"][key]
        assert item["raw_quote_rows"] == 4
        assert item["raw_training_ready_rows"] == 4
        assert item["manifest_selected_rows"] == item["legal_independent_n"] == 1
        assert item["legal_by_season"] == {2026: 1}
        assert item["blocked_or_nonselected_raw_reasons"] == {"NONSELECTED_BOOK_OR_REPRICE": 3}
        assert reports["split_plan"]["scopes"][key]["status"] == "INSUFFICIENT_TRAINING_EVIDENCE"
        assert reports["baselines"]["scopes"][key]["status"] == "NOT_EVALUATED"
    assert reports["training_audit"]["scopes"]["NCAAF/SPREAD"]["legal_independent_n"] == 0
    sources = reports["training_audit"]["historical_candidate_sources"]
    assert any(item["source"] == "data/master_all_sports.csv" and item["scope"] == "NFL/UNASSIGNED"
               and item["source_rows"] == 16 and item["training_ready"] == 0 for item in sources)
    assert any(item["source"] == "data/master_all_sports.csv" and item["scope"] == "NCAAF/UNASSIGNED"
               and item["source_rows"] == 65 and item["training_ready"] == 0 for item in sources)
    with evidence.connect(path) as db:
        after = {t: db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in before}
    assert after == before  # No model, prediction, activation, stake, or wager route.


@pytest.mark.parametrize("market,selection,line,home,away,expected", [
    ("SPREAD", "Home", -3.5, 24, 20, "COVER"),
    ("SPREAD", "Home", -4.0, 24, 20, "PUSH"),
    ("SPREAD", "Home", -4.5, 24, 20, "NO_COVER"),
    ("SPREAD", "Away", 3.5, 24, 20, "NO_COVER"),
    ("TOTAL", "Over", 44.0, 24, 20, "PUSH"),
    ("TOTAL", "Over", 43.5, 24, 20, "OVER"),
    ("TOTAL", "Under", 44.5, 24, 20, "UNDER"),
])
def test_exact_spread_and_total_targets_retain_push(market, selection, line, home, away, expected):
    assert stage2.exact_label(market, selection, line, home, away, "Home", "Away") == expected


def test_home_win_and_moneyline_cannot_satisfy_cover_target():
    with pytest.raises(ValueError):
        stage2.exact_label("MONEYLINE", "Home", 0, 24, 20, "Home", "Away")
    with pytest.raises(ValueError):
        stage2.validate_probability_vector({"HOME_WIN": 0.6, "AWAY_WIN": 0.4}, "SPREAD")
    with pytest.raises(ValueError):
        stage2.validate_probability_vector({"COVER": 0.6, "PUSH": 0.1, "NO_COVER": 0.4}, "SPREAD")


def test_ncaaf_provider_mascot_selection_replays_exact_cover(tmp_path):
    source = {"id": 77, "season": 2026, "week": 4, "seasonType": "regular",
              "homeTeam": "Ohio State", "awayTeam": "Missouri", "homeId": 10, "awayId": 20,
              "startDate": START.isoformat(), "neutralSite": True, "venue": "Test Stadium"}
    catalog = {"10": {"id": 10, "school": "Ohio State", "mascot": "Buckeyes"},
               "20": {"id": 20, "school": "Missouri", "mascot": "Tigers"}}
    aliases = {"10": {stage1._name("NCAAF", "Ohio State"), stage1._name("NCAAF", "Ohio State Buckeyes")},
               "20": {stage1._name("NCAAF", "Missouri"), stage1._name("NCAAF", "Missouri Tigers")}}
    offer = odds_event(sport_key="americanfootball_ncaaf")
    offer["home_team"], offer["away_team"] = "Ohio State Buckeyes", "Missouri Tigers"
    for market in offer["bookmakers"][0]["markets"]:
        if market["key"] == "spreads":
            market["outcomes"][0]["name"] = offer["home_team"]
            market["outcomes"][1]["name"] = offer["away_team"]
    path = tmp_path / "evidence.sqlite3"
    stage1.coverage(path, [source], [offer], sport="NCAAF", observed=NOW,
                    run_id="stage2-ncaaf", team_catalog=catalog, aliases=aliases)
    with evidence.connect(path) as db:
        event = dict(db.execute("SELECT * FROM prospective_football_event").fetchone())
    raw = {"provider_event_id": "77", "home_team_id": "10", "away_team_id": "20",
           "home_score": 24, "away_score": 20, "status": "FINAL",
           "provider_response": dict(source, completed=True, homePoints=24, awayPoints=20)}
    result, _ = stage1.append_result(path, event, raw, START+timedelta(hours=3), source="CFBD")
    stage1.settle_game(path, event, result, START+timedelta(hours=3))
    assert stage2.exact_label("SPREAD", "Ohio State Buckeyes", -3.5, 24, 20,
                              "Ohio State", "Missouri",
                              provider_home_team="Ohio State Buckeyes",
                              provider_away_team="Missouri Tigers") == "COVER"
    reports = stage2.build_reports(path, Path(__file__).resolve().parents[1],
                                   stage1_report=verified_refresh(), source_commit="abc")
    assert reports["training_audit"]["scopes"]["NCAAF/SPREAD"]["legal_independent_n"] == 1
    assert reports["training_audit"]["scopes"]["NCAAF/TOTAL"]["legal_independent_n"] == 1


def _snapshot(quote_time="2026-09-25T12:00:00+00:00"):
    return {"line": {"value": -3.5, "source": "verified_quote", "available_at": quote_time},
            "price_implied_probability": {"value": 110/210, "source": "verified_quote", "available_at": quote_time},
            "neutral_site": {"value": 0, "source": "pregame_schedule", "available_at": "2026-09-25T11:00:00+00:00"},
            "selection_is_home_or_over": {"value": 1, "source": "verified_quote", "available_at": quote_time}}


def test_feature_asof_and_leakage_fails_closed():
    args = {"quote_at": "2026-09-25T12:00:00Z", "kickoff": "2026-09-26T12:00:00Z",
            "event_discovered_at": "2026-09-25T11:00:00Z",
            "result_available_at": "2026-09-26T16:00:00Z"}
    stage2.validate_feature_snapshot(_snapshot(), **args)
    future = _snapshot(); future["line"]["available_at"] = "2026-09-25T13:00:00Z"
    with pytest.raises(ValueError, match="FEATURE_ASOF_UNAVAILABLE"):
        stage2.validate_feature_snapshot(future, **args)
    for field in ("same_event_result", "season_end", "current_roster", "future_rating", "closing_price"):
        leaked = _snapshot(); leaked["line"][field] = True
        with pytest.raises(ValueError, match="RESULT_LEAKAGE_RISK"):
            stage2.validate_feature_snapshot(leaked, **args)
    with pytest.raises(ValueError, match="RESULT_LEAKAGE_RISK"):
        stage2.validate_feature_snapshot(_snapshot(), **dict(args, result_available_at="2026-09-25T10:00:00Z"))
    assert "TheOver ModelHitRate as game probability" in stage2.feature_contract()["forbidden"]


def synthetic_rows():
    rows = []
    origin = datetime(2025, 1, 1, tzinfo=timezone.utc)
    for i in range(400):
        label = ("COVER", "PUSH", "NO_COVER")[i % 3]
        line = {"COVER": 7.0, "PUSH": 0.0, "NO_COVER": -7.0}[label]
        kickoff = origin + timedelta(days=i)
        rows.append({"sport": "NFL", "market_family": "SPREAD", "season": kickoff.year,
                     "game_id": f"nfl:test:{i}", "manifest_id": f"m{i}",
                     "source_manifest_hash": f"h{i}", "kickoff": kickoff.isoformat(),
                     "available_for_training_at": (kickoff+timedelta(hours=4)).isoformat(),
                     "label": label, "selection": "Home",
                     "features": {"line": {"value": line},
                                  "price_implied_probability": {"value": 0.5},
                                  "neutral_site": {"value": 0},
                                  "selection_is_home_or_over": {"value": 1}}})
    return rows


def test_chronological_whole_game_split_and_no_random_holdout():
    rows = synthetic_rows()
    other_market = dict(rows[0], market_family="TOTAL")
    assignment = stage2.chronological_partitions(rows + [other_market])
    flat = {(s,g): p for s,mapping in assignment.items() for g,p in mapping.items()}
    assert flat[("NFL", rows[0]["game_id"])] == flat[("NFL", other_market["game_id"])]
    stage2.validate_partitions(rows+[other_market], flat)
    crossed = [dict(rows[0], assigned_partition="development"),
               dict(other_market, assigned_partition="research_holdout")]
    with pytest.raises(ValueError, match="SAME_GAME_CROSSES_PARTITIONS"):
        stage2.validate_partitions(crossed, {("NFL", rows[0]["game_id"]): "development"})
    bad = {(r["sport"], r["game_id"]): ("research_holdout" if i == 0 else "development")
           for i,r in enumerate(rows)}
    with pytest.raises(ValueError, match="RANDOM_FINAL_HOLDOUT_PROHIBITED"):
        stage2.validate_partitions(rows, bad)


def test_calibration_requires_disjoint_chronological_cohort():
    rows = synthetic_rows()
    parts = [rows[:220], rows[220:280], rows[280:340], rows[340:]]
    stage2.validate_calibration_cohort(*parts)
    with pytest.raises(ValueError, match="CALIBRATION_COHORT_OVERLAP"):
        stage2.validate_calibration_cohort(parts[0], parts[1], parts[1], parts[3])
    with pytest.raises(ValueError, match="CALIBRATION_CHRONOLOGY_INVALID"):
        stage2.validate_calibration_cohort(parts[0], parts[2], parts[1], parts[3])


def test_baseline_distinct_and_training_model_has_immutable_provenance():
    rows = synthetic_rows()
    parts = {"development": rows[:220], "selection_validation": rows[220:280],
             "calibration_candidate": rows[280:340], "research_holdout": rows[340:]}
    base = stage2._baselines(parts["development"], parts["selection_validation"], "SPREAD")
    assert base["base_rate"]["status"] == base["market_implied"]["status"] == "BASELINE_ONLY"
    assert base["base_rate"]["probabilities"] != {"COVER": 0.5, "PUSH": 0.0, "NO_COVER": 0.5}
    split, _, comparison, inventory, calibration, artifact = stage2._artifacts_for_scope(
        rows, parts, Counter(), "a"*40, "2026-09-24T12:00:00+00:00")
    assert split["status"] == "FEASIBLE"
    assert comparison["candidates"][0]["metrics"]["n"] == 60
    assert comparison["candidates"][0]["metrics"]["nonpush_auc"] is not None
    assert inventory["model_status"] == "RESEARCH_MODEL_SELECTED"
    assert artifact["training_cutoff"] and artifact["training_manifest_hash"]
    assert artifact["source_commit"] == "a"*40 and artifact["training_config_hash"]
    assert artifact["metrics_artifact_hash"] == stage2.digest(comparison)
    assert inventory["artifact_hash"] == stage2.digest({k:v for k,v in artifact.items() if k != "model_id"})
    assert calibration["calibration_status"] in ("RESEARCH_CALIBRATION_FIT", "INSUFFICIENT_EVIDENCE")


def _model_for_prediction():
    artifact = {"schema": stage2.VERSION, "sport": "NFL", "market_family": "SPREAD",
                "target_classes": stage2.CLASSES["SPREAD"], "feature_version": stage2.FEATURE_VERSION,
                "training_start": "2025-09-01T00:00:00Z", "training_cutoff": "2026-08-01T08:00:00Z",
                "validation_window": {"first_kickoff": "2026-08-15T00:00:00Z",
                                      "last_kickoff": "2026-09-01T00:00:00Z"},
                "independent_training_n": 200, "validation_n": 60,
                "training_manifest_hash": "a"*64, "runtime_environment_hash": "b"*64,
                "training_config_hash": "c"*64, "metrics_artifact_hash": "d"*64,
                "source_commit": "e"*40, "algorithm": "regularized_multinomial_logistic",
                "created_at": "2026-09-24T11:00:00Z", "available_at": "2026-09-24T12:00:00Z",
                "classes": ["COVER", "PUSH", "NO_COVER"], "coefficients": [[0,0,0,0]]*3,
                "intercept": [1.0,-1.0,0.0], "deployment_state": "UNVALIDATED",
                "production_eligible": False, "stake": 0}
    h = stage2.digest(artifact)
    artifact["model_id"] = "football-stage2-" + h
    return {"model_status": "RESEARCH_MODEL_SELECTED", "model_id": artifact["model_id"],
            "artifact_hash": h, "model_artifact": artifact}


def _prediction_inputs():
    event = {"sport": "NFL", "game_id": "nfl:espn:401", "version_id": "event-v1",
             "home_team": "Home", "away_team": "Away", "neutral_site": 0,
             "discovered_at": "2026-09-25T11:00:00Z", "scheduled_start": "2026-09-26T12:00:00Z"}
    quote = {"quote_id": "q1", "sport": "NFL", "game_id": event["game_id"],
             "event_version_id": event["version_id"], "market_family": "SPREAD", "selection": "Home",
             "line": -3.5, "american_odds": -110, "quote_verified": 1,
             "observed_at": "2026-09-25T12:00:00Z", "provider_last_update": "2026-09-25T11:59:00Z"}
    return event, quote, _snapshot(), "2026-09-25T12:01:00Z"


def test_prediction_binds_exact_scope_quote_time_and_stays_research_only():
    model = _model_for_prediction()
    event, quote, snapshot, when = _prediction_inputs()
    record = stage2.research_prediction(model, None, event, quote, snapshot, when)
    assert record["production_eligible"] is False and record["stake"] == 0
    assert record["deployment_state"] == "UNVALIDATED" and record["calibration_id"] is None
    assert record["probability_semantics"] == "EXACT_SPREAD_COVER"
    assert sum(record["probabilities"].values()) == pytest.approx(1.0)
    for mutation in (
        lambda m,e,q,s,t: m["model_artifact"].update(target_classes=("HOME_WIN", "AWAY_WIN")),
        lambda m,e,q,s,t: e.update(sport="NCAAF"),
        lambda m,e,q,s,t: q.update(market_family="TOTAL"),
        lambda m,e,q,s,t: q.update(line=-4.5),
        lambda m,e,q,s,t: s["price_implied_probability"].update(value=0.7),
        lambda m,e,q,s,t: q.update(quote_id="", quote_verified=0),
        lambda m,e,q,s,t: m["model_artifact"].update(available_at="2026-09-25T13:00:00Z"),
        lambda m,e,q,s,t: m["model_artifact"].pop("training_manifest_hash"),
        lambda m,e,q,s,t: e.update(scheduled_start="2026-09-25T11:00:00Z"),
        lambda m,e,q,s,t: m["model_artifact"].update(production_eligible=True),
    ):
        m,e,q,s = copy.deepcopy((model,event,quote,snapshot))
        mutation(m,e,q,s,when)
        with pytest.raises(ValueError):
            stage2.research_prediction(m, None, e, q, s, when)
    with pytest.raises(ValueError, match="NO_VALID_EXACT_SCOPE_MODEL"):
        stage2.research_prediction(None, None, event, quote, snapshot, when)


def test_reports_write_only_sanitized_named_artifacts(tmp_path):
    path = tmp_path / "db.sqlite3"
    reports = stage2.build_reports(path, Path(__file__).resolve().parents[1],
                                   stage1_report=verified_refresh(), source_commit="abc")
    stage2.write_reports(reports, tmp_path / "out")
    assert {p.name for p in (tmp_path / "out").iterdir()} == set(stage2.ARTIFACT_NAMES.values())
    text = "\n".join(p.read_text() for p in (tmp_path / "out").iterdir())
    assert "private_key" not in text and "ODDS_API_KEY" not in text
