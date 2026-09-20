from datetime import datetime, timezone
from core.football_inventory import build

NOW = datetime(2026, 9, 20, tzinfo=timezone.utc)


def row(**updates):
    r = dict(sport="NFL", matchup_id="game", snapshot_id="a", selected_as_best_pick=True,
             prediction_generated_at="2026-09-19T10:00:00Z", capture_recorded_at="2026-09-19T10:01:00Z",
             game_start_utc="2026-09-19T17:00:00Z", odds_recorded_at="2026-09-19T09:59:00Z",
             quote_binding_verified=True, market_type="spread_home", candidate_outcome="LOSS",
             result_source="ESPN", result_provider_event_id="1", outcome_recorded_at="2026-09-19T22:00:00Z")
    return dict(r, **updates)


def test_late_capture_excluded_and_first_selection_not_replaced():
    result = build([row(), row(snapshot_id="b", capture_recorded_at="2026-09-19T11:00:00Z", candidate_outcome="WIN"),
                    row(matchup_id="late", capture_recorded_at="2026-09-19T18:00:00Z")], NOW)
    sport = result["sports"]["NFL"]
    assert sport["unique_pregame_selections"] == 1
    assert sport["records"][0]["snapshot_id"] == "a"
    assert sport["exclusions"]["unverified_or_late_capture"] == 1
    assert sport["settled_research_records"] == 1
    assert result["training_authorized"] is False


def test_conflict_missing_settlement_and_sport_isolation():
    result = build([row(), row(snapshot_id="conflict"), row(sport="NCAAF", result_source=None), row(sport="MLB")], NOW)
    assert result["sports"]["NFL"]["unique_pregame_selections"] == 0
    assert result["sports"]["NCAAF"]["settled_research_records"] == 0
    assert result["sports"]["NCAAF"]["cohorts"] == {"UNKNOWN": 1}
    assert "MLB" not in result["sports"]
