"""Admission applicability, with real pipeline/export/replay and no network."""
import base64
from copy import deepcopy
import hashlib
import json
import pandas as pd
import pytest
from app_core import nfl_inference_evidence as packet, research_replay
from test_nfl_inference_evidence import actual, mutate
from test_source_contract_pipeline import exported, INFERENCE
from scripts.benchmark_drive_history_loading import blocked_network


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def observation(row, change):
    def update(p):
        receipt = json.loads(p["observation_receipt"])
        change(receipt["payload"])
        receipt["sha256"] = packet.digest(receipt["payload"])
        p["observation_receipt"] = json.dumps(receipt)
    result = mutate(row, update)
    result["football_feature_receipt"] = json.loads(result["ml_estimate_metadata"])["nfl_inputs"]["payload"]["observation_receipt"]
    return result


def dependency(row, change):
    def update(p):
        receipt = p["source_dependencies"][packet.FEATURES[0]]
        change(receipt["payload"])
        payload = receipt["payload"]
        original = json.loads(base64.b64decode(payload["source_artifact"]["bytes_base64"]))
        original["scope"] = payload.get("scope")
        original["event_id"] = payload["provider_event_id"]
        raw = json.dumps(original).encode()
        payload["source_artifact"] = dict(bytes_base64=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest())
        receipt["sha256"] = packet.digest(payload)
    return mutate(row, update)


@pytest.mark.parametrize("field,value", [
    ("ml_feature_eligible", False), ("ml_feature_eligible", 1),
    ("stats_resolution_status", "unresolved"), ("stats_resolution_status", "cached"),
])
def test_changed_original_eligibility_rejects_actual_capture_export_replay(monkeypatch, tmp_path, field, value):
    analysis = actual(monkeypatch)
    row = analysis.iloc[0].to_dict()
    row[field] = value
    assert "eligibility.source_conflict:" + field in packet.diagnose(row)["errors"]
    with pytest.raises(ValueError):
        packet.replay(row)
    analysis.at[analysis.index[0], field] = value
    result = exported(monkeypatch, tmp_path, analysis)
    traces = [json.loads(r["research_estimate_trace"]) for f in result["frames"] for r in f.to_dict("records") if r.get("research_estimate_trace")]
    assert any(t["first_rejection_stage"] == "producer.nfl_inputs" and "eligibility.source_conflict:" + field in t["nfl_evidence"]["errors"] for t in traces)
    assert all(r["production_bet_amount"] == 0 for r in result["card"].wager_contract)
    receipt = research_replay.retain_export(result["frames"], result["package"], result["card"], result["captured"], path=result["db"])
    _, sources = research_replay.read_export(receipt["export_id"], path=result["db"])
    producer = research_replay.frame_from_payload(next(iter(sources.values()))["original"]["producer"])
    assert packet.diagnose(producer.iloc[0].to_dict())["status"] == "REJECTED"


@pytest.mark.parametrize("field,value", [("ml_feature_eligible", False), ("stats_resolution_status", "unresolved")])
def test_rehashed_matching_source_cannot_claim_success_after_ineligible_scoring(monkeypatch, field, value):
    row = actual(monkeypatch).iloc[0].to_dict()
    bad = mutate(row, lambda p: p["eligibility"].update({field: research_replay.cell(value)}))
    bad[field] = value
    assert "eligibility.success_conflict:" + field in packet.diagnose(bad)["errors"]
    with pytest.raises(ValueError):
        packet.replay(bad)


@pytest.mark.parametrize("change,diagnostic", [
    (lambda p: p.update(sport="NCAAF"), "features.observation_event:sport"),
    (lambda p: p.update(home_team="Unrelated Home"), "features.observation_event:home"),
    (lambda p: p.update(home_team=p["away_team"], away_team=p["home_team"]), "features.observation_event:home"),
    (lambda p: p.update(game_start_utc="2026-10-08T20:00:00Z"), "features.observation_event:start"),
    (lambda p: p.update(matchup_id="unrelated"), "features.observation_matchup"),
    (lambda p: p.update(stats_resolution_status="unresolved"), "features.observation_resolution"),
])
def test_consistently_rehashed_observation_event_conflicts(monkeypatch, change, diagnostic):
    bad = observation(actual(monkeypatch).iloc[0].to_dict(), change)
    assert diagnostic in packet.diagnose(bad)["errors"]
    with pytest.raises(ValueError):
        packet.replay(bad)


@pytest.mark.parametrize("change,diagnostic", [
    (lambda d: d["scope"]["event"].update(provider_event_id="other-event"), "features.dependency_scope_event:feature_home_ppg:provider_event_id"),
    (lambda d: d["scope"]["event"].update(provider_namespace="other-provider"), "features.dependency_scope_event:feature_home_ppg:provider_namespace"),
    (lambda d: d["scope"]["event"].update(sport="NCAAF"), "features.dependency_scope_event:feature_home_ppg:sport"),
    (lambda d: d["scope"]["event"].update(home="Unrelated Home"), "features.dependency_scope_event:feature_home_ppg:home"),
    (lambda d: d["scope"]["event"].update(home=d["scope"]["event"]["away"], away=d["scope"]["event"]["home"]), "features.dependency_scope_event:feature_home_ppg:home"),
    (lambda d: d["scope"]["event"].update(start="2026-10-08T20:00:00Z"), "features.dependency_scope_event:feature_home_ppg:start"),
    (lambda d: d["scope"].update(feature="feature_away_ppg"), "features.dependency_scope_feature:feature_home_ppg"),
    (lambda d: d["scope"].update(contract="unrelated-contract"), "features.dependency_scope_contract:feature_home_ppg"),
    (lambda d: d["scope"].update(available_at="2026-10-06T18:01:00Z"), "features.dependency_scope_available_at:feature_home_ppg"),
    (lambda d: (d.update(observed_at=INFERENCE), d["scope"].update(observed_at=INFERENCE)), "features.dependency_observation_window:feature_home_ppg"),
])
def test_consistently_rehashed_dependency_applicability(monkeypatch, change, diagnostic):
    bad = dependency(actual(monkeypatch).iloc[0].to_dict(), change)
    assert diagnostic in packet.diagnose(bad)["errors"]
    with pytest.raises(ValueError):
        packet.replay(bad)


def test_dependency_conflict_survives_capture_export_download(monkeypatch, tmp_path):
    analysis = actual(monkeypatch)
    bad = dependency(analysis.iloc[0].to_dict(), lambda d: d["scope"]["event"].update(provider_event_id="other-event"))
    analysis.at[analysis.index[0], "ml_estimate_metadata"] = bad["ml_estimate_metadata"]
    result = exported(monkeypatch, tmp_path, analysis)
    receipt = research_replay.retain_export(result["frames"], result["package"], result["card"], result["captured"], path=result["db"])
    data, verified = research_replay.download_bundle(receipt, expected_package_hash=receipt["package_hash"], path=result["db"])
    assert data and verified["source_boundary"] == "RETAINED"
    _, sources = research_replay.read_export(receipt["export_id"], path=result["db"])
    row = research_replay.frame_from_payload(next(iter(sources.values()))["original"]["producer"]).iloc[0].to_dict()
    assert packet.diagnose(row)["status"] == "REJECTED"
    with pytest.raises(ValueError):
        packet.replay(row)
    displays = [json.loads(r["research_display"]) for f in result["frames"] for r in f.to_dict("records") if r.get("research_display")]
    assert any(d["probability"] is None and d["ev"] is None for d in displays)
    assert all(c["production_bet_amount"] == 0 for c in result["card"].wager_contract)


@pytest.mark.parametrize("missing", ["eligibility_current", "resolution_current", "receipt_current", "dependencies", "dependency", "scope", "empty_features", "one_feature"])
def test_missing_facts_stay_unknown_and_never_replay_complete(monkeypatch, missing):
    row = actual(monkeypatch).iloc[0].to_dict()
    if missing == "eligibility_current": row.pop("ml_feature_eligible")
    elif missing == "resolution_current": row["stats_resolution_status"] = None
    elif missing == "receipt_current": row.pop("football_feature_receipt")
    elif missing == "dependencies": row = mutate(row, lambda p: p.update(source_dependencies=None))
    elif missing == "dependency": row = mutate(row, lambda p: p["source_dependencies"].pop(packet.FEATURES[0]))
    elif missing == "scope": row = dependency(row, lambda d: d.pop("scope"))
    elif missing == "empty_features": row = observation(row, lambda p: p.update(features={}))
    else: row = observation(row, lambda p: p["features"].pop(packet.FEATURES[0]))
    before = deepcopy(row)
    result = packet.diagnose(row)
    assert result["status"] == "INCOMPLETE" and result["unknown"] and not result["errors"]
    with pytest.raises(ValueError): packet.replay(row)
    assert row == before  # Read-only historical diagnostics; no upgrade/backfill.



@pytest.mark.parametrize("missing", ["dependencies", "empty_features"])
def test_missing_coverage_stays_incomplete_through_actual_export_and_replay(monkeypatch, tmp_path, missing):
    analysis = actual(monkeypatch)
    row = analysis.iloc[0].to_dict()
    if missing == "dependencies":
        row = mutate(row, lambda p: p.update(source_dependencies=None))
    else:
        row = observation(row, lambda p: p.update(features={}))
    for name in ("ml_estimate_metadata", "football_feature_receipt"):
        analysis.at[analysis.index[0], name] = row[name]
    result = exported(monkeypatch, tmp_path, analysis)
    receipt = research_replay.retain_export(result["frames"], result["package"], result["card"], result["captured"], path=result["db"])
    _, sources = research_replay.read_export(receipt["export_id"], path=result["db"])
    for boundary in (next(iter(sources.values()))["original"]["producer"], next(iter(sources.values()))["captured_candidates"]):
        retained = research_replay.frame_from_payload(boundary).iloc[0].to_dict()
        assessment = packet.diagnose(retained)
        assert assessment["status"] == "INCOMPLETE" and assessment["unknown"]
        with pytest.raises(ValueError):
            packet.replay(retained)
    assert all(c["production_bet_amount"] == 0 for c in result["card"].wager_contract)
    assert analysis.ml_probability.tolist() == [0.5840939462974203, 0.4159060537025797]


def test_legitimate_bound_inputs_preserve_raw_blend_and_private_comparison_facts(monkeypatch, tmp_path):
    analysis = actual(monkeypatch)
    assert analysis.ml_probability.tolist() == [0.5840939462974203, 0.4159060537025797]
    raw = analysis.ml_probability.tolist(); blend = analysis.calibrated_probability.tolist()
    result = exported(monkeypatch, tmp_path, analysis)
    receipt = research_replay.retain_export(result["frames"], result["package"], result["card"], result["captured"], path=result["db"])
    _, sources = research_replay.read_export(receipt["export_id"], path=result["db"])
    source = next(iter(sources.values()))
    for boundary in [source["original"]["producer"], source["captured_candidates"]]:
        frame = research_replay.frame_from_payload(boundary)
        assert all(k in frame for k in (*packet.ELIGIBILITY, "football_feature_receipt"))
        assert all(packet.diagnose(r)["status"] == "COMPLETE" for r in frame.to_dict("records"))
    producer = research_replay.frame_from_payload(source["original"]["producer"])
    assert producer.ml_probability.tolist() == raw and producer.calibrated_probability.tolist() == blend
    for row in producer.to_dict("records"):
        replay = packet.replay(row)
        assert replay["raw_probability"] == row["ml_probability"]
        assert replay["blended_probability"] == row["calibrated_probability"]
        assert replay["scientific_acceptance"] is False and replay["wagering_authority"] is False
    assert all(c["production_bet_amount"] == 0 for c in result["card"].wager_contract)
    assert not any(str(k) in json.dumps(result["package"]) for k in ("source_dependencies", "nfl_inputs", "bytes_base64"))

@pytest.mark.parametrize("field,value", [("ml_feature_eligible", False), ("stats_resolution_status", "unresolved")])
def test_rehashed_failure_status_cannot_mask_ineligible_numeric_output(monkeypatch, tmp_path, field, value):
    analysis = actual(monkeypatch)
    raw = analysis.ml_probability.tolist(); blend = analysis.calibrated_probability.tolist()
    row = analysis.iloc[0].to_dict()
    def change(p):
        p["eligibility"][field] = research_replay.cell(value)
        p["inference_status"] = "unavailable"
    bad = mutate(row, change)
    item = json.loads(bad["ml_estimate_metadata"])
    item["inference_status"] = "unavailable"
    bad["ml_estimate_metadata"] = json.dumps(item)
    bad[field] = value; bad["ml_inference_status"] = "unavailable"
    assert "origin.non_success_numeric_output" in packet.diagnose(bad)["errors"]
    with pytest.raises(ValueError): packet.replay(bad)
    for name in (field, "ml_inference_status", "ml_estimate_metadata"):
        analysis.at[analysis.index[0], name] = bad[name]
    result = exported(monkeypatch, tmp_path, analysis)
    receipt = research_replay.retain_export(result["frames"], result["package"], result["card"], result["captured"], path=result["db"])
    _, sources = research_replay.read_export(receipt["export_id"], path=result["db"])
    for boundary in (next(iter(sources.values()))["original"]["producer"], next(iter(sources.values()))["captured_candidates"]):
        retained = research_replay.frame_from_payload(boundary).iloc[0].to_dict()
        assert packet.diagnose(retained)["status"] == "REJECTED"
        with pytest.raises(ValueError): packet.replay(retained)
    assert analysis.ml_probability.tolist() == raw and analysis.calibrated_probability.tolist() == blend
    assert all(c["production_bet_amount"] == 0 for c in result["card"].wager_contract)


@pytest.mark.parametrize("status", [None, "unknown"])
def test_missing_inference_status_is_unknown_never_inferred_from_numeric(monkeypatch, status):
    row = actual(monkeypatch).iloc[0].to_dict()
    bad = mutate(row, lambda p: p.update(inference_status=status))
    item = json.loads(bad["ml_estimate_metadata"]); item["inference_status"] = status
    bad["ml_estimate_metadata"] = json.dumps(item); bad["ml_inference_status"] = status
    assessment = packet.diagnose(bad)
    assert assessment["status"] == "INCOMPLETE" and "origin.inference_status" in assessment["unknown"]
    assert not assessment["errors"]
    assert bad["ml_probability"] == row["ml_probability"]
    with pytest.raises(ValueError): packet.replay(bad)
