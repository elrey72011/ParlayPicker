"""Legacy prospective evidence stays readable and unpromoted."""
from datetime import datetime, timedelta, timezone
import hashlib

from app_core import mlb_pregame_receipts as mlb
from app_core import ncaaf_prospective_store as ncaaf
from app_core import nfl_market_store as nfl
from app_core.prospective_legacy_view import CANONICAL_FIELDS, legacy_evidence

NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)
START = NOW + timedelta(days=30)


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


def test_ncaaf_research_model_and_scores_do_not_become_validation(tmp_path):
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


def test_nfl_market_tracking_has_distinct_quotes_and_model_blocker(tmp_path):
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
