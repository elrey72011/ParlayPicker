"""Legacy prospective evidence stays readable and unpromoted."""
from datetime import datetime, timedelta, timezone
import hashlib
import pytest

from app_core import mlb_pregame_receipts as mlb
from app_core import ncaaf_prospective_store as ncaaf
from app_core import nfl_market_store as nfl
from app_core.prospective_legacy_view import CANONICAL_FIELDS, legacy_evidence

NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)
START = NOW + timedelta(days=30)

class _SyntheticStoreClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)



def unchanged(path, sport):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = legacy_evidence(sport, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert all(set(CANONICAL_FIELDS) <= row.keys() for row in rows)
    assert all(row["production_eligible"] is False and row["recommended_stake"] == 0
               and "MISSING_SPORT_MARKET_VALIDATION_ARTIFACT" in row["blockers"] for row in rows)
    return rows


def test_mlb_receipt_projection_is_read_only_and_quote_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(mlb, "now", lambda: NOW)
    path = tmp_path / "mlb-pregame-receipts.sqlite3"
    game = {"gamePk": 100, "season": "2026", "gameDate": START.isoformat(),
            "status": {"abstractGameState": "Preview"},
            "teams": {"home": {"team": {"id": 112, "name": "Chicago Cubs"}},
                      "away": {"team": {"id": 134, "name": "Pittsburgh Pirates"}}}}
    quote_payload = {
        "id": "odds-100", "home_team": "Chicago Cubs", "away_team": "Pittsburgh Pirates",
        "bookmakers": [{"key": "novig", "markets": [{"key": "spreads",
            "last_update": NOW.isoformat(), "outcomes": [
                {"name": "Chicago Cubs", "point": -1.5, "price": 110},
                {"name": "Pittsburgh Pirates", "point": 1.5, "price": -120}]}]}]}
    schedule_ref = mlb.persist_observation({"source": "mlb_statsapi", "endpoint": "api/v1/schedule",
        "observed_at": NOW.isoformat(), "payload": {"dates": [{"games": [game]}]}}, path)
    quote_ref = mlb.persist_observation({"source": "odds_api", "observed_at": NOW.isoformat(),
                                         "payload": quote_payload}, path)
    prior = [{"provider_namespace": "mlb", "game_id": str(i), "season": 2026,
              "home_id": "mlb:112", "away_id": "mlb:134", "home_score": 5,
              "away_score": 2, "status": "FINAL",
              "completed_at": (NOW - timedelta(days=i)).isoformat(),
              "available_at": (NOW - timedelta(days=i) + timedelta(hours=1)).isoformat(),
              "observation_hash": f"prior-{i}"} for i in range(1, 11)]
    quote = {"market_type": "spread_home", "line": -1.5, "decimal_odds": 2.1,
             "sportsbook": "Novig", "observed_at": NOW.isoformat(),
             "provider_event_id": "odds-100", "provider_namespace": "odds_api",
             "provider_updated_at": NOW.isoformat()}
    snapshot = mlb.build_receipt(game, quote, prior, captured_at=NOW.isoformat(),
        source_refs={"schedule": schedule_ref, "quotes": quote_ref})
    mlb.save_receipt(snapshot, path)
    rows = unchanged(path, "MLB")
    assert len(rows) == 1
    assert rows[0]["market_family"] == "RUN_LINE"
    assert rows[0]["home_team_id"] == "mlb:112"
    assert rows[0]["quote_verified"] is True
    assert rows[0]["model_id"] is None and rows[0]["result_outcome"] is None


def test_ncaaf_research_model_and_scores_do_not_become_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(ncaaf, "datetime", _SyntheticStoreClock)
    path = tmp_path / "ncaaf-prospective.sqlite3"
    model_id = ncaaf.save("model", {"artifact": {"models": {}}, "runtime_hash": "runtime"}, path)
    event = {"cfbd_id": 9, "event_id": "odds-9", "home_id": 1, "away_id": 2,
             "home": "Alabama", "away": "Georgia", "start": START.isoformat(),
             "features": {"prior_games": 3}, "models": {"constant": {"selected": {
                 "market_type": "spread_home", "point": -3, "price": -110,
                 "decimal_odds": 1.9090909091, "book": "draftkings",
                 "recorded_at": NOW.isoformat(), "win": .54, "push": .03, "loss": .43}}}}
    ncaaf.save("capture", {"model_id": model_id, "captured_at": NOW.isoformat(),
                           "events": [event], "production_eligible": False}, path)
    ncaaf.save("scores", {"scores": [{"cfbd_id": 9, "home_id": 1, "away_id": 2,
                                       "home_score": 28, "away_score": 21}]}, path)
    rows = unchanged(path, "NCAAF")
    assert len(rows) == 1
    assert rows[0]["market_family"] == "SPREAD"
    assert rows[0]["model_id"] == model_id and rows[0]["calibration_id"] is None
    assert rows[0]["result_outcome"] == "WIN"
    assert rows[0]["quote_verified"] is False


def test_nfl_market_tracking_has_distinct_quotes_and_model_blocker(tmp_path, monkeypatch):
    monkeypatch.setattr(nfl, "datetime", _SyntheticStoreClock)
    path = tmp_path / "nfl-market.sqlite3"
    nfl.save("capture", {"sport": "NFL", "protocol": "nfl-market-v1", "events": [{
        "event_id": "odds-8", "home": "Detroit Lions", "away": "New Orleans Saints",
        "start": START.isoformat(), "quotes": [
            {"market": "spreads", "selection": "Detroit Lions", "point": -3,
             "odds_american": -110, "book": "book-a", "recorded_at": NOW.isoformat()},
            {"market": "spreads", "selection": "Detroit Lions", "point": -3,
             "odds_american": -105, "book": "book-b", "recorded_at": NOW.isoformat()}]}]}, path)
    rows = unchanged(path, "NFL")
    assert len(rows) == 2 and len({row["observation_id"] for row in rows}) == 2
    assert all(row["home_team_id"] is None and row["model_id"] is None
               and "MISSING_STABLE_TEAM_IDS" in row["blockers"] for row in rows)


def test_missing_legacy_path_is_not_created(tmp_path):
    path = tmp_path / "absent.sqlite3"
    assert legacy_evidence("NFL", path) == []
    assert not path.exists()


@pytest.mark.parametrize("store", [ncaaf, nfl], ids=["ncaaf", "nfl"])
def test_native_legacy_capture_save_rejects_started_events(tmp_path, monkeypatch, store):
    class AfterStart(datetime):
        @classmethod
        def now(cls, tz=None):
            value = START + timedelta(seconds=1)
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(store, "datetime", AfterStart)
    payload = {"events": [{"start": START.isoformat()}]}
    if store is nfl:
        payload.update(sport="NFL", protocol="nfl-market-v1")
    path = tmp_path / "synthetic-started.sqlite3"
    record_id = store.save("capture", payload, path)
    from app_core.prospective_legacy_view import _records
    retained = next(value for key, value in _records(path, "records") if key == record_id)
    assert retained["data"]["events"] == []


@pytest.mark.parametrize("wall_year", [2025, 2027])
@pytest.mark.parametrize("store,body", [
    (ncaaf, test_ncaaf_research_model_and_scores_do_not_become_validation),
    (nfl, test_nfl_market_tracking_has_distinct_quotes_and_model_blocker),
], ids=["ncaaf", "nfl"])
def test_legacy_synthetic_chronology_is_calendar_independent_and_local(
        tmp_path, monkeypatch, wall_year, store, body):
    class WallClock(datetime):
        @classmethod
        def now(cls, tz=None):
            value = NOW.replace(year=wall_year)
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(store, "datetime", WallClock)
    with pytest.MonkeyPatch.context() as local:
        body(tmp_path, local)
        assert store.datetime is _SyntheticStoreClock
    assert store.datetime is WallClock


@pytest.mark.parametrize("scenario", ["empty", "valid", "missing_table", "database_error", "hash_error"])
def test_legacy_reader_explicitly_closes_its_owned_connection(tmp_path, monkeypatch, scenario):
    import json
    import sqlite3
    from app_core import prospective_legacy_view as view
    path = tmp_path / "SYNTHETIC-owned-reader.sqlite3"
    native_connect = sqlite3.connect
    if scenario == "database_error":
        path.write_bytes(b"SYNTHETIC not a SQLite database")
    else:
        db = native_connect(path)
        try:
            if scenario != "missing_table":
                db.execute("CREATE TABLE records (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            if scenario in {"valid", "hash_error"}:
                raw = json.dumps({"SYNTHETIC": True}, sort_keys=True)
                key = hashlib.sha256(raw.encode()).hexdigest() if scenario == "valid" else "SYNTHETIC-wrong-hash"
                db.execute("INSERT INTO records VALUES (?, ?)", (key, raw))
                db.commit()
        finally:
            db.close()
    before = path.read_bytes()
    owned = []
    calls = []
    def observed_connect(*args, **kwargs):
        calls.append((args, kwargs))
        connection = native_connect(*args, **kwargs)
        owned.append(connection)  # Deliberately keep a reference: GC is not the contract.
        return connection
    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    try:
        if scenario == "missing_table":
            with pytest.raises(ValueError, match="legacy evidence table missing"):
                view._records(path, "records")
        elif scenario == "database_error":
            with pytest.raises(sqlite3.DatabaseError):
                view._records(path, "records")
        elif scenario == "hash_error":
            with pytest.raises(ValueError, match="legacy evidence hash mismatch"):
                view._records(path, "records")
        else:
            rows = view._records(path, "records")
            assert len(rows) == (1 if scenario == "valid" else 0)
        assert len(owned) == len(calls) == 1
        assert calls[0][0][0] == path.resolve().as_uri() + "?mode=ro" and calls[0][1] == {"uri": True}
        assert path.read_bytes() == before
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            owned[0].execute("SELECT 1")
        path.unlink()  # Exact test-owned file; must work while the closed object is retained.
        assert not path.exists()
    finally:
        for connection in owned:
            connection.close()


def test_legacy_owned_connection_closes_after_interruption(tmp_path, monkeypatch):
    import sqlite3
    from app_core import prospective_legacy_view as view
    path = tmp_path / "SYNTHETIC-interrupted-reader.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE records (id TEXT, payload TEXT)")
    connection.commit()
    connection.close()
    native = sqlite3.connect
    events = []
    class Interrupted:
        def __init__(self, *args, **kwargs):
            self.db = native(*args, **kwargs)
        def __enter__(self):
            self.db.__enter__()
            events.append("transaction-enter")
            return self
        def execute(self, *args):
            raise KeyboardInterrupt()
        def __exit__(self, *error):
            events.append(("transaction-exit", error[0]))
            return self.db.__exit__(*error)
        def close(self):
            events.append("close")
            self.db.close()
    monkeypatch.setattr(sqlite3, "connect", Interrupted)
    with pytest.raises(KeyboardInterrupt):
        view._records(path, "records")
    assert events == ["transaction-enter", ("transaction-exit", KeyboardInterrupt), "close"]
    path.unlink()
    assert not path.exists()
