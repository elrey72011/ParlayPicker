"""Six-sport source view keeps native market tracking out of validation."""
from datetime import datetime, timedelta, timezone
import hashlib
import pytest

from app_core.odds_market_store import for_sport
from app_core.odds_research_adapter import PROTOCOL, SPORT_KEYS, digest, quotes
from app_core.prospective_source_view import all_source_readiness, source_evidence

AT = datetime.now(timezone.utc).replace(microsecond=0)
START = AT + timedelta(hours=1)


def fixture(sport, market, *, mismatched_id=False):
    home, away = "Home Team", "Away Team"
    raw = {"id": sport.lower() + "-game", "sport_key": SPORT_KEYS[sport],
           "home_team": home, "away_team": away, "commence_time": START.isoformat(),
           "bookmakers": [{"key": "book-a", "markets": [{"key": market,
               "last_update": AT.isoformat(), "outcomes": (
                   [{"name": home, "point": -1.5, "price": -110},
                    {"name": away, "point": 1.5, "price": 100}]
                   if market == "spreads" else
                   [{"name": "Over", "point": 5.5, "price": -110},
                    {"name": "Under", "point": 5.5, "price": 100}])}]}]}
    accepted, rejected = quotes(sport, raw, AT)
    assert len(accepted) == 2 and not rejected
    participants = [{"id": sport + ":1", "full_name": home},
                    {"id": sport + ":2", "full_name": away}]
    meta = lambda value: {"observed_at": AT.isoformat(), "source_id": "provider",
                          "source_hash": digest(value), "raw_source": value}
    event = {"event_id": raw["id"], "home": home, "away": away,
             "start": START.isoformat(), "provider_namespace": "THE_ODDS_API",
             "provider_event_id": raw["id"],
             "home_team_id": sport + (":wrong" if mismatched_id else ":1"),
             "away_team_id": sport + ":2", "quotes": accepted,
             "response_received_at": AT.isoformat(), "source_id": raw["id"],
             "source_hash": digest(raw), "raw_source": raw,
             "discovery_source": meta({k: raw[k] for k in
                 ("id", "sport_key", "home_team", "away_team", "commence_time")})}
    return {"sport": sport, "protocol": PROTOCOL,
            "participants_source": meta(participants), "events": [event]}


@pytest.mark.parametrize("sport,market,family", [
    ("NBA", "spreads", "SPREAD"),
    ("NCAAB", "totals", "TOTAL"),
    ("NHL", "spreads", "PUCK_LINE"),
])
def test_native_projection_replays_identity_and_exact_quotes(tmp_path, sport, market, family):
    path = tmp_path / (sport.lower() + "-market.sqlite3")
    store = for_sport(sport)
    store.save("capture", fixture(sport, market), path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = source_evidence(sport, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert len(rows) == 2 and {row["market_family"] for row in rows} == {family}
    assert all(row["quote_verified"] and row["identity_verified"]
               and row["home_team_id"] == sport + ":1"
               and row["model_id"] is None and row["production_eligible"] is False
               and row["recommended_stake"] == 0 for row in rows)


def test_invalid_participant_binding_remains_explicit_blocker(tmp_path):
    path = tmp_path / "nba-market.sqlite3"
    for_sport("NBA").save("capture", fixture("NBA", "spreads", mismatched_id=True), path)
    rows = source_evidence("NBA", path)
    assert len(rows) == 2
    assert all(row["identity_verified"] is False and row["home_team_id"] is None
               and "MISSING_REPLAYABLE_TEAM_IDENTITY" in row["blockers"] for row in rows)


def test_missing_native_store_is_not_created(tmp_path):
    path = tmp_path / "absent.sqlite3"
    assert source_evidence("NHL", path) == []
    assert not path.exists()


def test_native_score_projection_uses_append_only_latest_revision(tmp_path):
    sport = "NBA"
    path = tmp_path / "nba-market.sqlite3"
    store = for_sport(sport)
    store.save("capture", fixture(sport, "spreads"), path)

    def score(home_score, version, previous):
        raw = {"id": "nba-game", "sport_key": SPORT_KEYS[sport],
               "home_team": "Home Team", "away_team": "Away Team",
               "commence_time": START.isoformat(), "completed": True,
               "scores": [{"name": "Home Team", "score": str(home_score)},
                          {"name": "Away Team", "score": "90"}]}
        return {"sport": sport, "protocol": PROTOCOL, "event": {
            "event_id": "nba-game", "home": "Home Team", "away": "Away Team",
            "start": START.isoformat(), "home_score": home_score,
            "away_score": 90, "available_at": (START + timedelta(hours=version)).isoformat(),
            "result_source": "THE_ODDS_API", "grading_version": version,
            "revises_score_record_id": previous,
            "source_hash": digest(raw), "raw_source": raw}}

    first = store.save("scores", score(100, 1, None), path)
    second = store.save("scores", score(101, 2, first), path)
    rows = source_evidence(sport, path)
    assert len(rows) == 2
    assert all(row["result_home_score"] == 101 and
               row["result_outcome"] == "NEEDS_REVIEW" for row in rows)
    assert first != second
    restored = tmp_path / "restored-nba-market.sqlite3"
    for record in reversed(store.records(path)):
        store.insert({key: value for key, value in record.items() if key != "id"}, restored)
    assert all(row["result_home_score"] == 101 for row in source_evidence(sport, restored))


def test_native_score_projection_rejects_broken_revision_chain(tmp_path):
    sport = "NHL"
    path = tmp_path / "nhl-market.sqlite3"
    store = for_sport(sport)
    store.save("capture", fixture(sport, "spreads"), path)
    store.save("scores", {"sport": sport, "protocol": PROTOCOL, "event": {
        "event_id": "nhl-game", "grading_version": 2,
        "revises_score_record_id": "missing"}}, path)
    with pytest.raises(ValueError, match="revision chain"):
        source_evidence(sport, path)


def test_all_sport_source_inventory_remains_descriptive_and_read_only(tmp_path):
    missing = all_source_readiness(tmp_path)
    assert len(missing) == 12
    assert all(row["local_source_status"] == "ABSENT" and
               row["research_quote_rows"] == 0 and
               row["production_eligible"] is False and
               row["recommended_stake"] == 0 for row in missing)
    assert list(tmp_path.iterdir()) == []

    path = tmp_path / "nba-market.sqlite3"
    for_sport("NBA").save("capture", fixture("NBA", "spreads"), path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    inventory = all_source_readiness(tmp_path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    spread = next(row for row in inventory if (row["sport"], row["market_family"]) ==
                  ("NBA", "SPREAD"))
    total = next(row for row in inventory if (row["sport"], row["market_family"]) ==
                 ("NBA", "TOTAL"))
    assert spread["local_source_status"] == "PRESENT"
    assert spread["research_quote_rows"] == 2
    assert spread["research_captured_events"] == 1
    assert spread["replay_verified_quote_rows"] == 2
    assert spread["research_settled_events"] == 0
    assert total["research_quote_rows"] == 0
    assert all(row["research_only"] and not row["production_eligible"] for row in inventory)
