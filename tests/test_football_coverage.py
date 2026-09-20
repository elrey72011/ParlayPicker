from copy import deepcopy
from core.football_coverage import summarize


def test_missing_model_and_quote_are_visible_without_theover_requirement():
    report = {"games": [{"league": "NCAAF", "matchup_id": "a", "snapshot_id": "s",
                        "production_probability": .51, "independent_model_probability": None,
                        "verified_quote_candidates": 0}], "candidates": []}
    before = deepcopy(report)
    result = summarize(report)
    row = result["sports"]["NCAAF"]["rows"][0]
    assert row["status"] == "LIMITED_EVIDENCE"
    assert row["cohort"] == "UNKNOWN"
    assert "no_verified_candidate_quotes" in row["reasons"]
    assert "independent_model_unavailable" in row["reasons"]
    assert result["wager_authority"] is False
    assert report == before


def test_recorded_classification_and_sport_isolation():
    game = {"league": "NCAAF", "matchup_id": "a", "snapshot_id": "s", "production_probability": .51}
    candidate = {"matchup_id": "a", "snapshot_id": "s", "selected": True,
                 "home_classification": "FBS", "away_classification": "FCS"}
    result = summarize({"games": [game, dict(game, league="MLB")], "candidates": [candidate]})
    assert set(result["sports"]) == {"NCAAF"}
    assert result["sports"]["NCAAF"]["cohorts"] == {"FBS/FCS": 1}
    assert result["sports"]["NCAAF"]["repeated_probabilities"] == []
