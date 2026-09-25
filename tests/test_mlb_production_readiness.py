from copy import deepcopy
from datetime import timedelta
import json

import pytest

from app_core import mlb_pregame_receipts as receipts
from app_core import mlb_production_readiness as audit
from app_core import mlb_spread_total_model as old_model
from test_mlb_pregame_receipts import fixture, collect, START


VERIFIED = {key: True for key in (
    "receipt_restore_verified", "receipt_readback_verified", "canonical_restore_verified",
    "native_restore_verified", "snapshot_restore_verified")}


def settled(fixture, monkeypatch):
    collect(fixture)
    monkeypatch.setattr(receipts, "now", lambda: START + timedelta(hours=4))
    assert receipts.reconcile(fixture[0], fetch=fixture[2])["outcomes_created"] == 1
    return fixture[0]


def test_exact_targets_and_pushes_never_use_moneyline():
    assert audit.exact_label("spread_home", -2, 5, 3) == "PUSH"
    assert audit.exact_label("spread_away", 2, 5, 3) == "PUSH"
    assert audit.exact_label("spread_home", -1.5, 5, 3) == "COVER"
    assert audit.exact_label("spread_away", 1.5, 5, 3) == "NO_COVER"
    assert audit.exact_label("total_over", 8, 5, 3) == "PUSH"
    assert audit.exact_label("total_under", 8.5, 5, 3) == "UNDER"
    assert audit.exact_label("total_over", 8.5, 5, 3) == "UNDER"
    assert audit.exact_label("spread_home", -1.5, None, None, "VOID") == "VOID"
    with pytest.raises(ValueError, match="WRONG_TARGET"):
        audit.exact_label("moneyline_home", 0, 5, 3)


def test_vectors_are_exact_scope_and_sum_to_one():
    assert audit.validate_vector({"COVER": .5, "PUSH": .1, "NO_COVER": .4}, "MLB/RUN_LINE")
    assert audit.validate_vector({"OVER": .5, "PUSH": .1, "UNDER": .4}, "MLB/TOTAL")
    with pytest.raises(ValueError):
        audit.validate_vector({"WIN": .5, "PUSH": .1, "LOSS": .4}, "MLB/RUN_LINE")
    with pytest.raises(ValueError):
        audit.validate_vector({"OVER": .5, "PUSH": .1, "UNDER": .5}, "MLB/TOTAL")


def test_verified_receipts_one_independent_game_per_market(fixture, monkeypatch):
    path = settled(fixture, monkeypatch)
    rows, manifest, summary, reconciliation = audit.receipt_reports(path)
    assert len(rows) == 4 and len(manifest) == 2
    assert {r["scope"] for r in manifest} == set(audit.SCOPES)
    assert {r["selection"] for r in manifest} == {"spread_home", "total_over"}
    assert all(r["training_status"] == "TRAINING_READY" for r in rows)
    assert all(r["price_status"] == "VERIFIED_PREGAME_PRICE" for r in rows)
    assert all(summary[s]["legal_independent_n"] == 1 for s in audit.SCOPES)
    assert reconciliation["statuses"]["matched"] == 4
    assert len({r["independent_game_market_id"] for r in manifest}) == 2
    snapshot = next(iter(receipts.read("receipts", path).values()))
    payload, values = audit.exact_feature_values(snapshot)
    assert len(values) == 8
    assert values[-2] == float(payload["quote"]["line"])
    assert values[-1] == 1 / float(payload["quote"]["decimal_odds"])


def test_missing_price_and_timestamp_never_default_to_minus_110(fixture):
    collect(fixture)
    snapshot = next(iter(receipts.read("receipts", fixture[0]).values()))
    observations = receipts.read("observations", fixture[0])
    changed = deepcopy(snapshot)
    changed["payload"]["quote"]["decimal_odds"] = None
    changed["sha256"] = old_model.digest(changed["payload"])
    row = audit.classify_receipt(changed, None, observations)
    assert row["price_status"] == "LINE_PRESENT_PRICE_MISSING"
    assert "NO_VERIFIED_PREGAME_PRICE" in row["blockers"]
    changed = deepcopy(snapshot)
    changed["payload"]["quote"].pop("observed_at")
    changed["sha256"] = old_model.digest(changed["payload"])
    row = audit.classify_receipt(changed, None, observations)
    assert row["price_status"] == "PRICE_PRESENT_TIMESTAMP_UNVERIFIED"
    assert "QUOTE_TIMESTAMP_UNVERIFIED" in row["blockers"]


def test_legacy_theover_line_and_price_remain_unverified(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "theover_spreads.csv").write_text(
        "Game Date,Home Team,Away Team,League,Spread Line,Odds American\n"
        "2026-09-24,Home,Away,MLB,-1.5,-110\n", encoding="utf-8")
    source = audit.local_csv_inventory(tmp_path)[0]
    assert source["line_present_rows"] == 1
    assert source["price_present_rows"] == 1
    assert source["verified_pregame_price_rows"] == 0
    assert source["price_status_counts"] == {"PRICE_PRESENT_TIMESTAMP_UNVERIFIED": 1}


def test_source_tampering_and_post_start_quote_fail_closed(fixture):
    collect(fixture)
    snapshot = next(iter(receipts.read("receipts", fixture[0]).values()))
    observations = receipts.read("observations", fixture[0])
    bad = deepcopy(snapshot)
    bad["payload"]["source_observations"]["quotes"] = "0" * 64
    bad["sha256"] = old_model.digest(bad["payload"])
    assert "SOURCE_LINEAGE_UNVERIFIED" in audit.classify_receipt(bad, None, observations)["blockers"]
    bad = deepcopy(snapshot)
    bad["payload"]["quote"]["observed_at"] = bad["payload"]["game_start_utc"]
    bad["sha256"] = old_model.digest(bad["payload"])
    assert audit.classify_receipt(bad, None, observations)["training_status"] == "TRAINING_BLOCKED"


def test_feature_blocker_does_not_erase_verified_price_or_settlement(fixture, monkeypatch):
    path = settled(fixture, monkeypatch)
    snapshot = next(iter(receipts.read("receipts", path).values()))
    outcome = next(iter(receipts.read("outcomes", path).values()))
    observations = receipts.read("observations", path)
    bad = deepcopy(snapshot)
    bad["payload"]["prior_games"][0]["home_score"] += 1
    bad["sha256"] = old_model.digest(bad["payload"])
    row = audit.classify_receipt(bad, outcome, observations)
    assert row["price_status"] == "VERIFIED_PREGAME_PRICE"
    assert row["result_verified"] is True
    assert row["label"] in audit.CLASSES[row["scope"].split("/")[1]]
    assert "FEATURE_ASOF_UNAVAILABLE" in row["blockers"]
    assert "feature_hash" not in row
    assert row["training_status"] == "TRAINING_BLOCKED"


def test_doubleheaders_require_distinct_stable_game_ids(fixture):
    collect(fixture)
    snapshot = next(iter(receipts.read("receipts", fixture[0]).values()))
    observations = receipts.read("observations", fixture[0])
    modified = deepcopy(observations)
    key = snapshot["payload"]["source_observations"]["schedule"]
    schedule = deepcopy(modified[key])
    game = deepcopy(schedule["payload"]["dates"][0]["games"][-1])
    game["gamePk"] = 101
    game["gameDate"] = (START + timedelta(hours=3)).isoformat()
    schedule["payload"]["dates"][0]["games"].append(game)
    new_key = old_model.digest(schedule)
    modified[new_key] = schedule
    bad = deepcopy(snapshot)
    bad["payload"]["source_observations"]["schedule"] = new_key
    bad["sha256"] = old_model.digest(bad["payload"])
    assert "DOUBLEHEADER_IDENTITY_AMBIGUOUS" in audit.classify_receipt(bad, None, modified)["blockers"]


def test_remote_proof_required_and_reports_idempotent(fixture, monkeypatch, tmp_path):
    path = settled(fixture, monkeypatch)
    kwargs = dict(source_commit="a" * 40, canonical={}, native=[], snapshots={})
    with pytest.raises(ValueError, match="AUTHENTICATED_REMOTE_READBACK_REQUIRED"):
        audit.build_reports(path, tmp_path, remote_verification={}, **kwargs)
    first = audit.build_reports(path, tmp_path, remote_verification=VERIFIED, **kwargs)
    second = audit.build_reports(path, tmp_path, remote_verification=VERIFIED, **kwargs)
    assert first == second
    assert set(first) == set(audit.REPORTS)
    assert all(first["models"]["scopes"][s]["model_id"] is None for s in audit.SCOPES)
    assert all(first["calibration"]["scopes"][s]["calibration_id"] is None for s in audit.SCOPES)
    assert all(first["validation"]["scopes"][s]["stake"] == 0 for s in audit.SCOPES)
    assert all(first["validation"]["scopes"][s]["production_eligible"] is False for s in audit.SCOPES)
    files = audit.write_reports(first, tmp_path / "reports")
    assert len(files) == 12
    assert json.loads((tmp_path / "reports" / audit.REPORTS["training"]).read_text())["scopes"]
    sources = {item["source"]: item for item in first["inventory"]["sources"]}
    assert sources["receipt_observations/odds_api"]["price_present_rows"] > 0
    assert sources["receipt_outcomes/mlb_statsapi"]["final_result_rows"] == 1
    assert all(item["lineage_hash_verified"] for item in sources.values()
               if item["source"].startswith("receipt_observations/"))


def test_chronological_plan_whole_game_and_not_random(fixture, monkeypatch):
    path = settled(fixture, monkeypatch)
    manifest = audit.receipt_reports(path)[1]
    plan = audit.chronological_plan(manifest)
    assert all(plan["scopes"][s]["partition_n"]["development"] in (0, 1) for s in audit.SCOPES)
    assert all(plan["scopes"][s]["status"] == "INSUFFICIENT_TRAINING_EVIDENCE" for s in audit.SCOPES)
    assert all("NO_RANDOM_HOLDOUT" in plan["scopes"][s]["strategy"] for s in audit.SCOPES)
    assert plan["scopes"]["MLB/RUN_LINE"]["assignment_hash"] == plan["scopes"]["MLB/TOTAL"]["assignment_hash"]


def test_unvalidated_challenger_never_becomes_exact_model_or_wager(fixture, monkeypatch):
    path = settled(fixture, monkeypatch)
    snapshot = next(iter(receipts.read("receipts", path).values()))
    assert audit.model_audit({})[0]["classification"] == "WRONG_TARGET"
    with pytest.raises(ValueError, match="NO_VALID_EXACT_SCOPE_MODEL"):
        audit.validate_research_prediction({"model_status": "INSUFFICIENT_TRAINING_EVIDENCE"},
                                           snapshot, {}, START - timedelta(hours=1))
