"""Explicitly synthetic actual prospective capture/export/replay; no fitting."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
import pandas as pd
import pytest

from app_core import ncaaf_prospective as p, ncaaf_prospective_store as store
from app_core import ncaaf_research as research, ncaaf_research_contract as contract
from app_core.market_probability_model import predict_market_probabilities
from scripts.benchmark_drive_history_loading import blocked_network
from test_ncaaf_prospective import inputs, event, response, NOW


class FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz or timezone.utc)


@pytest.fixture(autouse=True)
def offline():
    with blocked_network():
        yield


def captured(tmp_path, monkeypatch, *, half=False, book="draftkings"):
    monkeypatch.setattr(p, "utcnow", lambda: NOW)
    monkeypatch.setattr(store, "datetime", FrozenDateTime)
    artifact = dict(protocol=deepcopy(research.PROTOCOL),
        source_hash=hashlib.sha256(Path(research.__file__).read_bytes()).hexdigest(),
        train_hash=research.digest("SYNTHETIC_TRAINING_IDENTITY_NO_FIT"),
        calibration_hash=research.digest("SYNTHETIC_CALIBRATION_IDENTITY_NO_FIT"),
        models={"constant": {"margin": dict(kind="constant", intercept=10., bias=0., sigma=12.),
                             "total": dict(kind="constant", intercept=50., bias=0., sigma=12.)}})
    data = dict(artifact=artifact, artifact_hash=research.digest(artifact),
                runtime_hash=p.runtime_hash(), policy="SYNTHETIC SOFTWARE FIXTURE; NO QUALIFICATION")
    path = tmp_path/"synthetic-ncaaf.sqlite3"
    model_id = store.insert(dict(schema=1, kind="model", created_at="2026-10-28T12:00:00Z", data=data), path)
    model = next(r for r in store.records(path) if r["id"] == model_id)
    raw = event()
    raw["sport_key"] = "americanfootball_ncaaf"
    raw["bookmakers"][0]["key"] = book
    spread, total = (10.5, 50.5) if half else (10, 50)
    raw["bookmakers"][0]["markets"] = [
        dict(key="spreads", period="full_game", settlement_rules="synthetic:full-game-ot-win-push-loss",
             outcomes=[dict(name="Alabama", point=-spread, price=-110), dict(name="Georgia", point=spread, price=-110)]),
        dict(key="totals", period="full_game", settlement_rules="synthetic:full-game-ot-win-push-loss",
             outcomes=[dict(name="Over", point=total, price=-110), dict(name="Under", point=total, price=-110)])]
    get = Mock(return_value=response([raw]))
    capture_id, info = p.capture(inputs(), model, "synthetic-key", get=get, path=path)
    assert info == {"saved_games": 1, "skipped_games": 0} and get.call_count == 1
    records = contract.read_records(path)
    return records, capture_id, path


def selection(records, capture_id, kind):
    capture = next(r for r in records if r["id"] == capture_id)
    e = capture["data"]["events"][0]
    q = next(q for q in e["models"]["constant"]["candidates"] if q["market_type"] == kind)
    return dict(capture_id=capture_id, event_id=e["event_id"], model_name="constant", kind=kind,
                line=q["point"], book=q["book"], price=q["price"], quote_time=q["recorded_at"])


def review(records, chosen):
    return dict(binding=contract.review_binding(records, **chosen),
        rights="ACCEPTED_FOR_PRIVATE_RESEARCH", review_id="SYNTHETIC-REVIEW",
        operator="synthetic operator", product="synthetic sportsbook", listing_id="synthetic listing",
        settlement="full_game_including_overtime_binary_win_push_loss",
        effective_from="2026-10-01T00:00:00Z", effective_until="2026-11-01T00:00:00Z",
        reviewed_at="2026-10-28T00:00:00Z")


@pytest.mark.parametrize("kind", sorted(contract.KINDS))
@pytest.mark.parametrize("half", [False, True])
def test_actual_capture_private_export_replay_preserves_orientation_push_and_clocks(tmp_path, monkeypatch, kind, half):
    records, capture_id, path = captured(tmp_path, monkeypatch, half=half)
    before = path.read_bytes()
    chosen = selection(records, capture_id, kind)
    packet = contract.export_observation(records, source_review=review(records, chosen),
                                         evidence_label="SYNTHETIC", **chosen)
    packet = json.loads(json.dumps(packet))  # Actual owner JSON serialization.
    result = contract.replay(packet)
    assert result["evidence_label"] == "SYNTHETIC"
    assert result["probabilities"]["push"] == 0 if half else result["probabilities"]["push"] > 0
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    assert result["probabilities"] == packet["payload"]["probabilities"]
    assert result["live_stake"] == 0 and not result["scientific_acceptance"] and not result["wagering_authority"]
    payload = packet["payload"]
    assert payload["original_blend"] is None and payload["ui_refresh"] is None
    assert payload["original_inference_time"] is None
    assert payload["clock_missingness"] == "NCAAF_PROSPECTIVE_V1_PER_MODEL_INFERENCE_CLOCK_NOT_RETAINED"
    assert payload["original_capture_finished_at"] == NOW.isoformat()
    assert payload["target_contract"]["selection"] == kind.split("_")[1]
    assert payload["target_contract"]["signed_line"] == chosen["line"]
    assert payload["original_quote"]["recorded_at"] == chosen["quote_time"]
    assert payload["feature_order"] == research.FEATURES
    assert len(payload["dependency_receipts"]) == 4
    assert not payload["historical_publication_time_verified"]
    assert path.read_bytes() == before
    assert "synthetic-key" not in json.dumps(packet)


def resign(records):
    for record in records:
        record["id"] = contract._hash({k: v for k, v in record.items() if k != "id"})
    return next(r["id"] for r in records if r["kind"] == "capture")


@pytest.mark.parametrize("change,reason", [
    ("feature", "NCAAF_ALTERED_ORDERED_FEATURES"),
    ("future_dependency", "NCAAF_FUTURE_OR_STALE_DEPENDENCY"),
    ("stale_dependency", "NCAAF_FUTURE_OR_STALE_DEPENDENCY"),
    ("duplicate_quote", "NCAAF_EXACT_OFFER_AMBIGUOUS_OR_MISSING"),
    ("duplicate_game", "NCAAF_EVENT_AMBIGUOUS"),
    ("reverse", "NCAAF_CANONICAL_EVENT_CONFLICT"),
    ("revised_start", "NCAAF_CANONICAL_EVENT_CONFLICT"),
    ("regulation", "NCAAF_PERIOD_OR_SETTLEMENT_MISSING"),
    ("rules", "NCAAF_PERIOD_OR_SETTLEMENT_MISSING"),
    ("provider", "NCAAF_PROVIDER_BINDING_MISSING"),
    ("future_quote", "NCAAF_QUOTE_CAPTURE_START_CLOCK_CONFLICT"),
    ("missing_stats", "NCAAF_ORIGINAL_FEATURES_UNAVAILABLE"),
    ("extra_secret", "NCAAF_INPUT_NOT_ALLOWLISTED"),
])
def test_negative_capture_bindings(tmp_path, monkeypatch, change, reason):
    records, capture_id, _ = captured(tmp_path, monkeypatch)
    chosen = selection(records, capture_id, "spread_away")
    c = next(r for r in records if r["id"] == capture_id)
    e = c["data"]["events"][0]
    q = next(q for q in e["models"]["constant"]["candidates"] if q["market_type"] == "spread_away")
    state = c["data"]["inputs"]
    if change == "feature": e["features"]["home_ppg"] += 1
    elif change == "future_dependency": state["batches"][1]["retrieved_at"] = "2026-10-29T13:00:00Z"
    elif change == "stale_dependency": state["batches"][1]["retrieved_at"] = "2026-10-27T12:00:00Z"
    elif change == "duplicate_quote": e["models"]["constant"]["candidates"].append(deepcopy(q))
    elif change == "duplicate_game": c["data"]["events"].append(deepcopy(e))
    elif change == "reverse": e["home_id"], e["away_id"] = e["away_id"], e["home_id"]
    elif change == "revised_start": q["event_start_utc"] = "2026-10-30T13:00:00Z"
    elif change == "regulation": q["period"] = "regulation"
    elif change == "rules": q["rules_source"] = ""
    elif change == "provider": q["provider_event_id"] = "another-event"
    elif change == "future_quote": q["recorded_at"] = chosen["quote_time"] = "2026-10-29T13:00:00Z"
    elif change == "missing_stats": state["batches"][1]["records"] = []
    elif change == "extra_secret": state["batches"][1]["records"][0]["apiKey"] = "must never export"
    chosen["capture_id"] = resign(records)
    with pytest.raises(ValueError, match=reason):
        contract.review_binding(records, **chosen)


@pytest.mark.parametrize("change,reason", [
    ("runtime", "NCAAF_FROZEN_RUNTIME_MISMATCH"),
    ("winner", "NCAAF_ARTIFACT_TARGET_CONFLICT"),
    ("future", "NCAAF_FUTURE_MODEL"),
    ("reader", "NCAAF_ARTIFACT_READER_MISMATCH"),
    ("lineage", "NCAAF_TRAINING_LINEAGE_MISSING"),
])
def test_artifact_conflicts(tmp_path, monkeypatch, change, reason):
    records, capture_id, _ = captured(tmp_path, monkeypatch)
    chosen = selection(records, capture_id, "total_over")
    m = next(r for r in records if r["kind"] == "model")
    c = next(r for r in records if r["kind"] == "capture")
    if change == "runtime": m["data"]["runtime_hash"] = "older-runtime"
    elif change == "winner": m["data"]["artifact"]["models"]["constant"] = {"home_won": {}}
    elif change == "future": m["created_at"] = "2026-10-30T00:00:00Z"
    elif change == "reader": m["data"]["artifact"]["source_hash"] = "0"*64
    elif change == "lineage": m["data"]["artifact"]["train_hash"] = ""
    m["data"]["artifact_hash"] = contract._hash(m["data"]["artifact"])
    m["id"] = contract._hash({k: v for k, v in m.items() if k != "id"})
    c["data"]["model_id"] = m["id"]
    chosen["capture_id"] = resign(records)
    with pytest.raises(ValueError, match=reason): contract.review_binding(records, **chosen)


@pytest.mark.parametrize("change,reason", [
    ("missing", "NCAAF_EXACT_SOURCE_REVIEW_MISSING_OR_CONFLICT"),
    ("listing", "NCAAF_SOURCE_PAYOFF_OR_RIGHTS_UNSUPPORTED"),
    ("rights", "NCAAF_SOURCE_PAYOFF_OR_RIGHTS_UNSUPPORTED"),
    ("clock", "NCAAF_SOURCE_REVIEW_CLOCK_CONFLICT"),
])
def test_separate_source_review_required(tmp_path, monkeypatch, change, reason):
    records, capture_id, _ = captured(tmp_path, monkeypatch)
    chosen = selection(records, capture_id, "total_under")
    r = review(records, chosen)
    if change == "missing": r = {}
    elif change == "listing": r["listing_id"] = ""
    elif change == "rights": r["rights"] = "ASSERTED_FEED_TERMS"
    elif change == "clock": r["effective_until"] = "2026-10-29T00:00:00Z"
    with pytest.raises(ValueError, match=reason):
        contract.export_observation(records, source_review=r, evidence_label="SYNTHETIC", **chosen)


def test_novig_binary_payoff_rejected(tmp_path, monkeypatch):
    records, capture_id, _ = captured(tmp_path, monkeypatch, book="novig")
    chosen = selection(records, capture_id, "total_over")
    with pytest.raises(ValueError, match="NCAAF_SOURCE_PAYOFF_OR_RIGHTS_UNSUPPORTED"):
        contract.export_observation(records, source_review=review(records, chosen), evidence_label="SYNTHETIC", **chosen)


@pytest.mark.parametrize("change,reason", [
    ("corrupt", "NCAAF_PACKET_INTEGRITY"),
    ("probability", "NCAAF_REPLAY_PROBABILITY_CONFLICT"),
    ("authority", "NCAAF_EXPORT_CONTRACT_CONFLICT"),
    ("ev", "NCAAF_REPLAY_PAYOFF_CONFLICT"),
])
def test_corrupt_or_resigned_packet_fails(tmp_path, monkeypatch, change, reason):
    records, capture_id, _ = captured(tmp_path, monkeypatch)
    chosen = selection(records, capture_id, "spread_home")
    packet = contract.export_observation(records, source_review=review(records, chosen), evidence_label="SYNTHETIC", **chosen)
    if change == "corrupt": packet["sha256"] = "0"*64
    elif change == "authority": packet["payload"]["wagering_authority"] = True
    else:
        pld = packet["payload"]
        rows = pld["records"]
        c = next(r for r in rows if r["kind"] == "capture")
        q = next(q for q in c["data"]["events"][0]["models"]["constant"]["candidates"] if q["market_type"] == "spread_home")
        q["win" if change == "probability" else "ev"] = .9
        pld["selection"]["capture_id"] = resign(rows)
        packet = contract.export_observation(rows, source_review=review(rows, pld["selection"]), evidence_label="SYNTHETIC", **pld["selection"])
    if change == "authority": packet["sha256"] = contract._hash(packet["payload"])
    with pytest.raises(ValueError, match=reason): contract.replay(packet)


def test_missing_store_not_initialized_and_default_board_stays_unavailable(tmp_path):
    path = tmp_path/"absent.sqlite3"
    with pytest.raises(ValueError, match="NCAAF_EXISTING_STORE_INACCESSIBLE"): contract.read_records(path)
    assert not path.exists()
    frame = pd.DataFrame([dict(League="NCAAF", market_type="spread_away", spread_line=10, ml_feature_eligible=True)])
    result = predict_market_probabilities(frame)
    assert pd.isna(result.iloc[0]["ml_probability"])
    assert result.iloc[0]["ml_unavailable_reason"] == "No market-specific model configured for NCAAF"


def test_live_wal_requires_coherent_owner_snapshot(tmp_path, monkeypatch):
    _, _, path = captured(tmp_path, monkeypatch)
    wal = Path(str(path)+"-wal")
    wal.write_bytes(b"uncheckpointed synthetic WAL")
    with pytest.raises(ValueError, match="NCAAF_STORE_SNAPSHOT_REQUIRED"): contract.read_records(path)
    assert wal.read_bytes() == b"uncheckpointed synthetic WAL"


@pytest.mark.parametrize("kind,line", [("moneyline_home", 0), ("spread_away", 10.25), ("total_over", -1), ("spread_home", True)])
def test_unsupported_target_and_lines(kind, line):
    with pytest.raises(ValueError): contract.target_contract(kind, line)
