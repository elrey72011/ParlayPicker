from copy import deepcopy
from app_core.football_identity_capture import attach


def fixture():
    game = {"home_team": "Kansas City", "away_team": "Denver", "commence_time": "2026-09-21T00:20:00Z"}
    event = {"id": "1", "date": game["commence_time"], "competitions": [{"competitors": [
        {"homeAway": "home", "team": {"id": "12", "displayName": "Kansas City Chiefs"}},
        {"homeAway": "away", "team": {"id": "7", "displayName": "Denver Broncos"}}]}]}
    return game, event


def test_unique_pregame_identity_preserves_original_and_does_not_create_features():
    game, event = fixture()
    before = deepcopy(game)
    row = attach([game], "NFL", [event], "2026-09-20T12:00:00Z")[0]
    assert row["home_team_id"] == "espn:nfl:12"
    assert row["provider_ids"] == {"espn": "1"}
    assert row["football_identity_source_hash"]
    assert "features_generated_at" not in row and "identity_verified" not in row
    assert game == before


def test_ambiguous_conflicting_and_started_events_fail_closed():
    game, event = fixture()
    other = dict(event, id="2")
    assert "home_team_id" not in attach([game], "NFL", [event,other], "2026-09-20T12:00:00Z")[0]
    row = attach([dict(game,provider_ids={"espn":"99"})], "NFL", [event], "2026-09-20T12:00:00Z")[0]
    assert row["football_identity_status"] == "CONFLICT"
    assert "home_team_id" not in row
    assert attach([game], "NFL", [event], "2026-09-21T00:20:00Z")[0]["football_identity_status"] == "NOT_PREGAME"


def test_wrong_kickoff_and_opponent_do_not_match():
    game, event = fixture()
    assert "home_team_id" not in attach([dict(game,home_team="Dallas")], "NFL", [event], "2026-09-20T12:00:00Z")[0]
    assert "home_team_id" not in attach([dict(game,commence_time="2026-09-21T00:25:00Z")], "NFL", [event], "2026-09-20T12:00:00Z")[0]
