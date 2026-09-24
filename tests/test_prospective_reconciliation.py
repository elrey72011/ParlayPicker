"""Historical research lineage never acquires prospective wager authority."""

from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import sqlite3

import pytest

from app_core import mlb_pregame_receipts as mlb
from app_core import mlb_prospective_store as mlb_legacy
from app_core import ncaaf_prospective_store as ncaaf
from app_core import nfl_market_store as nfl
from app_core.prospective_evidence import EvidenceConflict, read_records
from app_core.prospective_reconciliation import (
    reconcile_sport, reconciliation_readiness,
)
from app_core.prospective_remote import sync as sync_canonical


NOW = datetime.now(timezone.utc).replace(microsecond=0)


def _bytes(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _count(path, table):
    with sqlite3.connect(path) as db:
        return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_nfl_reconciliation_is_read_only_idempotent_and_deduplicates_identity(tmp_path):
    source = tmp_path / "nfl-market.sqlite3"
    canonical = tmp_path / "prospective-evidence.sqlite3"
    event = {"event_id": "odds-8", "home": "Detroit Lions", "away": "New Orleans Saints",
             "start": (NOW + timedelta(hours=2)).isoformat(), "quotes": [
                 {"market": "spreads", "selection": "Detroit Lions", "point": -3,
                  "odds_american": -110, "book": "book-a", "recorded_at": NOW.isoformat()},
                 {"market": "totals", "selection": "Over", "point": 43.5,
                  "odds_american": -105, "book": "book-a", "recorded_at": NOW.isoformat()}]}
    data = {"sport": "NFL", "protocol": "nfl-market-v1", "events": [event]}
    for minute in (0, 1):
        nfl.insert({"schema": 1, "kind": "capture",
                    "created_at": (NOW + timedelta(minutes=minute)).isoformat(),
                    "data": data}, source)
    before = _bytes(source)
    first = reconcile_sport("NFL", canonical, source)
    assert _bytes(source) == before
    assert first["source_records"] == 2
    assert first["source_market_rows"] == 4
    assert first["canonical_research_identities"] == 2
    assert first["duplicates"] == 2
    assert first["new_source_artifacts"] == 2
    assert first["new_reconciled_facts"] == 4
    assert first["canonical_predictions"] == 0
    assert _count(canonical, "prospective_reconciled_fact") == 4
    assert read_records(canonical, "prospective_prediction") == []
    second = reconcile_sport("NFL", canonical, source)
    assert second["new_source_artifacts"] == second["new_reconciled_facts"] == 0
    assert _count(canonical, "prospective_reconciled_fact") == 4
    readiness = reconciliation_readiness(canonical)
    spread = next(row for row in readiness if row["sport"] == "NFL" and
                  row["market_family"] == "SPREAD")
    assert spread["reconciled_research_rows"] == 1
    assert spread["reconciled_games"] == 1
    assert spread["canonical_predictions"] == 0
    assert spread["recommended_stake"] == 0
    assert first["markets"]["SPREAD"]["failure_counts"]["identity_failure"] == 2


def test_ncaaf_score_append_enriches_same_identity_without_backdating(tmp_path):
    source = tmp_path / "ncaaf-prospective.sqlite3"
    canonical = tmp_path / "prospective-evidence.sqlite3"
    start = NOW - timedelta(days=2)
    captured = start - timedelta(hours=2)
    model = ncaaf.insert({"schema": 1, "kind": "model",
        "created_at": (captured - timedelta(days=1)).isoformat(),
        "data": {"artifact": {"models": {}}, "runtime_hash": "runtime"}}, source)
    event = {"cfbd_id": 9, "event_id": "odds-9", "home_id": 1, "away_id": 2,
             "home": "Alabama", "away": "Georgia", "start": start.isoformat(),
             "features": {"prior_games": 3}, "models": {"constant": {"selected": {
                 "market_type": "spread_home", "point": -3, "price": -110,
                 "decimal_odds": 1.9090909091, "book": "draftkings",
                 "recorded_at": captured.isoformat(), "win": .54, "push": .03,
                 "loss": .43}}}}
    ncaaf.insert({"schema": 1, "kind": "capture", "created_at": captured.isoformat(),
                  "data": {"model_id": model, "captured_at": captured.isoformat(),
                           "events": [event], "production_eligible": False}}, source)
    before = _bytes(source)
    initial = reconcile_sport("NCAAF", canonical, source)
    assert _bytes(source) == before
    assert initial["new_reconciled_facts"] == 1
    assert initial["markets"]["SPREAD"]["historical_pregame_prediction_facts"] == 1
    assert initial["canonical_predictions"] == 0
    ncaaf.insert({"schema": 1, "kind": "scores",
        "created_at": (start + timedelta(hours=4)).isoformat(),
        "data": {"scores": [{"cfbd_id": 9, "home_id": 1, "away_id": 2,
                              "home_score": 28, "away_score": 21}]}}, source)
    after_score = _bytes(source)
    enriched = reconcile_sport("NCAAF", canonical, source)
    assert _bytes(source) == after_score
    assert enriched["new_source_artifacts"] == 1
    assert enriched["new_reconciled_facts"] == 1
    assert enriched["markets"]["SPREAD"]["settled_games"] == 1
    assert _count(canonical, "prospective_reconciled_fact") == 2
    assert reconcile_sport("NCAAF", canonical, source)["new_reconciled_facts"] == 0
    spread = next(row for row in reconciliation_readiness(canonical)
                  if row["sport"] == "NCAAF" and row["market_family"] == "SPREAD")
    assert spread["reconciled_research_rows"] == 1
    assert spread["reconciled_settled_games"] == 1
    assert spread["canonical_predictions"] == 0


def test_same_market_identity_with_changed_capture_price_fails_closed(tmp_path):
    source = tmp_path / "nfl-market.sqlite3"
    canonical = tmp_path / "canonical.sqlite3"
    base = {"sport": "NFL", "protocol": "nfl-market-v1", "events": [{
        "event_id": "odds-identity", "home": "Home", "away": "Away",
        "start": (NOW + timedelta(hours=2)).isoformat(),
        "quotes": [{"market": "spreads", "selection": "Home", "point": -2.5,
                    "odds_american": -110, "book": "book-a",
                    "recorded_at": NOW.isoformat()}]}]}
    nfl.insert({"schema": 1, "kind": "capture", "created_at": NOW.isoformat(),
                "data": base}, source)
    assert reconcile_sport("NFL", canonical, source)["new_reconciled_facts"] == 1
    changed = {**base, "events": [{**base["events"][0],
        "quotes": [{**base["events"][0]["quotes"][0], "odds_american": -105}]}]}
    nfl.insert({"schema": 1, "kind": "capture",
                "created_at": (NOW + timedelta(minutes=1)).isoformat(),
                "data": changed}, source)
    with pytest.raises(EvidenceConflict, match="changed capture facts"):
        reconcile_sport("NFL", canonical, source)
    assert _count(canonical, "prospective_reconciled_fact") == 1


def _mlb_source(path, *, price=2.1):
    start = NOW + timedelta(days=30)
    game = {"gamePk": 100, "season": str(start.year), "gameDate": start.isoformat(),
            "status": {"abstractGameState": "Preview"},
            "teams": {"home": {"team": {"id": 112, "name": "Chicago Cubs"}},
                      "away": {"team": {"id": 134, "name": "Pittsburgh Pirates"}}}}
    quote_payload = {"id": "odds-100", "home_team": "Chicago Cubs",
        "away_team": "Pittsburgh Pirates", "bookmakers": [{"key": "novig",
            "markets": [{"key": "spreads", "last_update": NOW.isoformat(),
                "outcomes": [{"name": "Chicago Cubs", "point": -1.5, "price": 110},
                             {"name": "Pittsburgh Pirates", "point": 1.5,
                              "price": -120}]}]}]}
    schedule_ref = mlb.persist_observation({"source": "mlb_statsapi",
        "endpoint": "api/v1/schedule", "observed_at": NOW.isoformat(),
        "payload": {"dates": [{"games": [game]}]}}, path)
    quote_ref = mlb.persist_observation({"source": "odds_api",
        "observed_at": NOW.isoformat(), "payload": quote_payload}, path)
    prior = [{"provider_namespace": "mlb", "game_id": str(i), "season": start.year,
              "home_id": "mlb:112", "away_id": "mlb:134", "home_score": 5,
              "away_score": 2, "status": "FINAL",
              "completed_at": (NOW - timedelta(days=i)).isoformat(),
              "available_at": (NOW - timedelta(days=i) + timedelta(hours=1)).isoformat(),
              "observation_hash": f"prior-{i}"} for i in range(1, 11)]
    quote = {"market_type": "spread_home", "line": -1.5,
             "decimal_odds": price, "sportsbook": "Novig",
             "observed_at": NOW.isoformat(), "provider_event_id": "odds-100",
             "provider_namespace": "odds_api", "provider_updated_at": NOW.isoformat()}
    snapshot = mlb.build_receipt(game, quote, prior, captured_at=NOW.isoformat(),
        source_refs={"schedule": schedule_ref, "quotes": quote_ref})
    mlb.save_receipt(snapshot, path)
    return snapshot


def test_mlb_receipt_counts_and_same_immutable_id_conflict(tmp_path, monkeypatch):
    monkeypatch.setattr(mlb, "now", lambda: NOW + timedelta(minutes=1))
    first = tmp_path / "first" / "mlb-pregame-receipts.sqlite3"
    second = tmp_path / "second" / "mlb-pregame-receipts.sqlite3"
    canonical = tmp_path / "prospective-evidence.sqlite3"
    _mlb_source(first)
    original_hash = _bytes(first)
    report = reconcile_sport("MLB", canonical, first)
    assert _bytes(first) == original_hash
    assert report["source_receipts"] == 1
    assert report["source_games"] == 1
    assert report["source_market_rows"] == 1
    assert report["markets"]["RUN_LINE"]["independent_settled_event_line_units"] == 0
    assert report["canonical_predictions"] == 0
    assert reconcile_sport("MLB", canonical, first)["new_reconciled_facts"] == 0
    _mlb_source(second, price=2.05)
    with pytest.raises(EvidenceConflict, match="immutable identity conflict"):
        reconcile_sport("MLB", canonical, second)
    assert _count(canonical, "prospective_reconciled_fact") == 1


def test_absent_legacy_source_does_not_create_source_or_canonical(tmp_path):
    source = tmp_path / "missing.sqlite3"
    canonical = tmp_path / "canonical.sqlite3"
    report = reconcile_sport("NFL", canonical, source)
    assert report["source_status"] == "LOCAL_ABSENT_REMOTE_UNKNOWN"
    assert not source.exists() and not canonical.exists()


def test_unpriced_older_mlb_forecasts_are_preserved_without_market_predictions(tmp_path):
    older = tmp_path / "mlb-prospective.sqlite3"
    receipts = tmp_path / "mlb-pregame-receipts.sqlite3"
    canonical = tmp_path / "canonical.sqlite3"
    model = mlb_legacy.insert({"schema": 1, "kind": "model",
        "created_at": (NOW - timedelta(days=3)).isoformat(),
        "data": {"runtime_hash": "legacy-runtime", "train_year": 2023}}, older)
    capture_at = NOW - timedelta(days=2)
    mlb_legacy.insert({"schema": 1, "kind": "capture",
        "created_at": capture_at.isoformat(), "data": {
            "model_id": model, "observed_at": capture_at.isoformat(),
            "events": [{"game_id": 99, "start": (NOW - timedelta(days=1)).isoformat(),
                "home_id": 112, "away_id": 134,
                "features": {"home_ppg": 4.1, "away_ppg": 3.7},
                "forecasts": {"team_only": {"margin": 0.4, "total": 7.8}}}]}}
        , older)
    before = _bytes(older)
    report = reconcile_sport("MLB", canonical, receipts)
    assert _bytes(older) == before and not receipts.exists()
    assert report["source_status"] == "MLB_RECEIPTS_ABSENT_PROSPECTIVE_PRESENT"
    assert report["legacy_forecast_source_records"] == 2
    assert report["legacy_unpriced_forecast_rows"] == 2
    assert report["source_games"] == 1
    assert report["source_receipts"] == 0
    assert report["canonical_predictions"] == 0
    assert read_records(canonical, "prospective_prediction") == []
    assert reconcile_sport("MLB", canonical, receipts)["new_reconciled_facts"] == 0


def test_reconciled_ledger_restores_from_canonical_remote(tmp_path):
    source = tmp_path / "nfl-market.sqlite3"
    first = tmp_path / "first.sqlite3"
    second = tmp_path / "second.sqlite3"
    nfl.insert({"schema": 1, "kind": "capture", "created_at": NOW.isoformat(),
        "data": {"sport": "NFL", "protocol": "nfl-market-v1", "events": [{
            "event_id": "remote-1", "home": "Home", "away": "Away",
            "start": (NOW + timedelta(hours=2)).isoformat(),
            "quotes": [{"market": "totals", "selection": "Over", "point": 42.5,
                        "odds_american": -110, "book": "book-a",
                        "recorded_at": NOW.isoformat()}]}]}}, source)
    reconcile_sport("NFL", first, source)

    class Cloud:
        def __init__(self):
            self.objects = {}
            self.parallel_calls = 0

        def read_objects(self, *, Prefix):
            return [(key, raw) for key, raw in self.objects.items() if key.startswith(Prefix)]

        def put_object(self, *, Bucket, Key, Body, ContentType, IfNoneMatch):
            if Key in self.objects:
                raise AssertionError("unexpected duplicate upload")
            self.objects[Key] = Body

        def get_object(self, *, Bucket, Key):
            return {"Body": BytesIO(self.objects[Key])}

        def run_parallel(self, operation, items, progress=None):
            self.parallel_calls += 1
            result = []
            for item in items:
                result.append(operation(self, item))
                progress(len(result), len(items))
            return result

    cloud = Cloud()
    uploaded = sync_canonical(first, cloud, "folder")
    assert uploaded["new_records_verified"] == 2
    assert cloud.parallel_calls == 1
    restored = sync_canonical(second, cloud, "folder")
    assert restored["remote_records_read"] == 2
    assert reconciliation_readiness(second) == reconciliation_readiness(first)
    assert read_records(second, "prospective_prediction") == []
