from copy import deepcopy
from datetime import datetime, timedelta, timezone
import math
import pytest
from app_core.mlb_research import prepare, fit, digest, evaluate, probabilities, run_research


def checkpoint():
    state = {"schema_version": 1, "games": {}, "excluded": {}, "schedules": {}}
    for year in (2023, 2024, 2025):
        schedule = []
        for i in range(112):
            gid = year * 1000 + i
            start = datetime(year, 4, 1, tzinfo=timezone.utc) + timedelta(days=i)
            game = dict(game_id=gid, season=year, cutoff=start.isoformat(), started_at=start.isoformat(),
                        completed_at=(start + timedelta(hours=3)).isoformat(), home_id=1, away_id=2,
                        home_score=2 + i % 3, away_score=1 + i % 2)
            if game["home_score"] == game["away_score"]: game["home_score"] += 1
            state["games"][str(gid)] = {"record": game}
            schedule.append({"gamePk": gid})
        state["schedules"][str(year)] = {"payload": {"dates": [{"games": schedule}]}}
    return state


def test_holdout_outcomes_cannot_change_fit():
    state = checkpoint()
    splits, _ = prepare(state)
    artifact = fit(splits[2023], splits[2024])
    altered = deepcopy(state)
    for entry in altered["games"].values():
        if entry["record"]["season"] == 2025:
            entry["record"]["home_score"] += 20
    changed, _ = prepare(altered)
    assert digest(fit(changed[2023], changed[2024])) == digest(artifact)
    before = digest(artifact)
    metrics, predictions = evaluate(artifact, splits[2025])
    assert digest(artifact) == before
    assert all(m["margin"]["n"] == 102 for m in metrics.values())
    assert predictions


def test_missing_collection_and_invalid_chronology_fail():
    state = checkpoint()
    del state["games"]["2023000"]
    with pytest.raises(ValueError, match="pending"):
        prepare(state)
    state = checkpoint()
    state["games"]["2023000"]["record"]["completed_at"] = "2022-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="chronology"):
        prepare(state)


def test_probabilities_push_and_total_support():
    p = probabilities(0, 4, 0)
    assert p["push"] > 0 and sum(p.values()) == pytest.approx(1)
    p = probabilities(8, 4, 8.5, total=True)
    assert p["push"] == 0 and sum(p.values()) == pytest.approx(1)
    assert probabilities(8, 4, -1, total=True)["over"] == pytest.approx(1)


def test_research_only_and_finite_metrics():
    result = run_research(checkpoint())
    assert result["report"]["production_eligible"] is False
    assert result["report"]["artifact_hash"] == digest(result["artifact"])
    for model in result["report"]["holdout_metrics"].values():
        assert math.isfinite(model["margin"]["log_loss"])
        assert 0 <= model["margin"]["brier"] <= 1
