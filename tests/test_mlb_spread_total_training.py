"""Synthetic fixtures exercise code only; they are never committed as model evidence."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import pytest
from app_core import mlb_spread_total_model as m


def receipt(day, market="spread_home", *, number=0):
    start = datetime(2024, 6, 1, 18, tzinfo=timezone.utc)+timedelta(days=day)
    at = start-timedelta(hours=1)
    payload = {"schema_version": m.SCHEMA, "provider_namespace": "mlb", "provider_event_id": str(1000+day),
        "season": 2024, "home_team_id": "H", "away_team_id": "A", "game_start_utc": start.isoformat(),
        "prediction_cutoff": at.isoformat(), "captured_at": at.isoformat(),
        "quote": {"market_type": market, "line": (-1.5 if market == "spread_home" else 1.5) if market.startswith("spread") else 8.5,
                  "decimal_odds": 1.91, "sportsbook": "test-only", "observed_at": at.isoformat()},
        "prior_games": [], "baselines": {name: {"probability": .5, "source_version": "test-only",
            "probability_semantics": m.SEMANTICS, "generated_at": at.isoformat()}
            for name in ("deterministic", "configured_blend")}}
    for i in range(10):
        end = start-timedelta(days=i+2)
        payload["prior_games"].append({"provider_namespace": "mlb", "game_id": str(i), "season": 2024, "status": "FINAL",
            "home_id": "H", "away_id": "A", "home_score": 6+i%3+(day%3), "away_score": 2+i%4,
            "completed_at": end.isoformat(), "available_at": end.isoformat()})
    return {"payload": payload, "sha256": m.digest(payload)}


def records():
    result = []
    for day in range(60):
        for market in m.TARGETS:
            snapshot = receipt(day, market)
            p = snapshot["payload"]
            outcome = {k: p[k] for k in ("provider_namespace", "provider_event_id", "season", "home_team_id", "away_team_id", "game_start_utc")}
            outcome.update(status="FINAL", home_score=7 if day%2 else 2, away_score=3,
                           available_at=(m.timestamp(p["game_start_utc"])+timedelta(hours=4)).isoformat())
            result.append({"snapshot": snapshot, "outcome": outcome})
    return result


@pytest.fixture
def trained(tmp_path):
    return m.train(records(), tmp_path/"models", source="SYNTHETIC TEST ONLY",
                   train_through="2024-06-20", validation_through="2024-07-10")


def test_chronological_slate_isolation_and_real_cutoff(trained):
    rows = m.prepare_rows(records())
    split = m.split_rows(rows, "2024-06-20", "2024-07-10")
    for a, b in (("train", "validation"), ("validation", "holdout")):
        assert max(r["start"] for r in split[a]) < min(r["start"] for r in split[b])
        assert not {r["slate"] for r in split[a]} & {r["slate"] for r in split[b]}
    manifest = m.load_mlb_spread_total_model(trained)["manifest"]
    assert manifest["model_trained_through"] == max(r["outcome_at"] for r in split["train"])
    assert m.timestamp(manifest["model_trained_through"]) <= m.timestamp(manifest["model_available_at"])
    assert manifest["maturity"] == "RESEARCH"
    assert manifest["production_eligible"] is False and manifest["recommended_stake"] == 0
    assert manifest["training_rows"] == {"spread": 20, "total": 20}
    assert manifest["validation_metrics"]["spread"]["sample_count"] == 40


@pytest.mark.parametrize("change", ["target", "future", "quote", "captured", "namespace", "duplicate", "hash"])
def test_feature_leakage_rejected(change):
    r = receipt(5)
    p = r["payload"]
    if change == "target": p["prior_games"][0]["game_id"] = p["provider_event_id"]
    if change == "future": p["prior_games"][0]["available_at"] = p["game_start_utc"]
    if change == "quote": p["quote"]["observed_at"] = p["game_start_utc"]
    if change == "captured": p["captured_at"] = p["game_start_utc"]
    if change == "namespace": p["provider_namespace"] = "odds_api"
    if change == "duplicate": p["prior_games"].append(p["prior_games"][0])
    if change != "hash": r["sha256"] = m.digest(p)
    else: r["sha256"] = "wrong"
    with pytest.raises(ValueError): m.receipt_features(r)


@pytest.mark.parametrize("market,line,h,a,expected", [
    ("spread_home", -2, 5, 3, "PUSH"), ("spread_home", -1.5, 5, 3, "WIN"),
    ("spread_away", 1.5, 5, 3, "LOSS"), ("spread_away", 2, 5, 3, "PUSH"),
    ("total_over", 8, 5, 3, "PUSH"), ("total_over", 7.5, 5, 3, "WIN"),
    ("total_under", 7.5, 5, 3, "LOSS"), ("total_under", 8, 5, 3, "PUSH")])
def test_exact_target_labels(market,line,h,a,expected):
    assert m.label(market,line,h,a) == expected
    assert m.label(market,line,None,None,"VOID") == "VOID"


def test_model_version_reproducibility_and_no_overwrite(trained, tmp_path, monkeypatch):
    original = m.load_mlb_spread_total_model(trained)["manifest"]
    assert m.model_version(original) == original["model_version"]
    changed = deepcopy(original); changed["config"]["ridge"] = 5
    assert m.model_version(changed) != original["model_version"]
    changed = deepcopy(original); changed["artifact_hashes"]["estimators.json"] = "other"
    assert m.model_version(changed) != original["model_version"]
    # Stable completion receipt/config/data produce exactly the same fitted bundle.
    monkeypatch.setattr(m, "utcnow", lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    a = m.train(records(), tmp_path/"a", source="test",train_through="2024-06-20",validation_through="2024-07-10")
    b = m.train(records(), tmp_path/"b", source="test",train_through="2024-06-20",validation_through="2024-07-10")
    assert a.name == b.name
    with pytest.raises(FileExistsError):
        m.train(records(), tmp_path/"a",source="test",train_through="2024-06-20",validation_through="2024-07-10")


def test_holdout_outcomes_do_not_change_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "utcnow", lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))
    original = records(); modified = deepcopy(original)
    for r in modified[164:]: r["outcome"]["home_score"] += 10
    a = m.train(original,tmp_path/"a",source="test",train_through="2024-06-20",validation_through="2024-07-10")
    b = m.train(modified,tmp_path/"b",source="test",train_through="2024-06-20",validation_through="2024-07-10")
    assert (a/"estimators.json").read_bytes() == (b/"estimators.json").read_bytes()


def test_no_pit_data_fails_closed(tmp_path):
    with pytest.raises(KeyError):
        m.train([{"home_ppg":5,"home_score":3}],tmp_path/"models",source="legacy",train_through="2024-06-20",validation_through="2024-07-10")
    assert not (tmp_path/"models").exists()


def test_late_labels_and_duplicates_fail():
    data=records(); data[0]["outcome"]["available_at"]="2025-01-01T00:00:00Z"
    # Same event across targets must have identical outcome facts first.
    with pytest.raises(ValueError): m.prepare_rows(data)
    with pytest.raises(ValueError): m.prepare_rows(records()+records()[:1])
    with pytest.raises(ValueError): m.label("h2h",0,3,1)


def test_cli_rejects_legacy_dataset_without_writing_artifact(tmp_path):
    from scripts.train_mlb_spread_total_model import main
    data=tmp_path/"legacy.json";data.write_text(json.dumps([{"home_score":5}]))
    with pytest.raises(SystemExit) as error:
        main(["--dataset",str(data),"--source","legacy","--train-through","2024-06-20",
              "--validation-through","2024-07-10","--output",str(tmp_path/"model")])
    assert error.value.code==2
    assert not (tmp_path/"model").exists()


def test_push_and_void_excluded_from_fit(tmp_path):
    data=records()
    for r in data:
        day=int(r["snapshot"]["payload"]["provider_event_id"])-1000
        if day in {19,39,59}:
            r["outcome"].update(status="VOID")
        elif day in {18,38,58}:
            p=r["snapshot"]["payload"]
            p["quote"]["line"]=(1 if p["quote"]["market_type"]=="spread_home" else -1) if p["quote"]["market_type"].startswith("spread") else 5
            r["snapshot"]["sha256"]=m.digest(p)
    # Voids and pushes never become negative labels to satisfy the minimum.
    rows=m.prepare_rows(data)
    assert {r["outcome"] for r in rows}>={"WIN","LOSS","PUSH","VOID"}
    with pytest.raises(ValueError,match="insufficient point-in-time"):
        m.train(data,tmp_path/"model",source="test",train_through="2024-06-20",validation_through="2024-07-10")
