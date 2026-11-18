from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import math
from unittest.mock import patch
import numpy as np
import pytest
from app_core.ncaaf_research import (prepare, fit, evaluate, digest, probabilities, run_research, FEATURES)
from app_core.ncaaf_history import new_collection, checkpoint_bytes


def sample():
    rng = np.random.default_rng(8)
    features, targets = [], []
    for year in (2023, 2024, 2025):
        for i in range(60):
            values = rng.normal([28, 25, 22, 24, 350, 330], [5, 5, 4, 4, 30, 30])
            row = dict(zip(FEATURES[:-1], values.tolist()))
            row.update(game_id=year*100+i, season=year,
                kickoff=(datetime(year, 9, 1, tzinfo=timezone.utc)+timedelta(days=i)).isoformat(),
                neutral_site=bool(i%5 == 0), scoring_features_available=True,
                home_prior_games=3, away_prior_games=3, home_yards_games=3, away_yards_games=3)
            features.append(row)
            targets.append(dict(game_id=row["game_id"], home_margin=float(round(values[0]-values[1]+rng.normal(0,10))),
                                total_points=float(round(values[0]+values[1]+rng.normal(0,10)))))
    return features, targets


def test_holdout_targets_cannot_change_artifact():
    features, targets = sample()
    splits, _ = prepare(features, targets)
    frozen = fit(splits[2023], splits[2024])
    changed = deepcopy(targets)
    for r in changed:
        if r["game_id"] >= 202500:
            r["home_margin"] += 200
            r["total_points"] += 200
    other, _ = prepare(features, changed)
    assert digest(fit(other[2023], other[2024])) == digest(frozen)
    assert evaluate(frozen, splits[2025])[0] != evaluate(frozen, other[2025])[0]


def test_scaler_train_only_and_calibration_changes_bias():
    features, targets = sample()
    splits, _ = prepare(features, targets)
    artifact = fit(splits[2023], splits[2024])
    m = artifact["models"]["ridge"]["margin"]
    assert m["x_mean"] == pytest.approx(np.mean([[r[k] for k in FEATURES] for r in splits[2023]], axis=0))
    changed = deepcopy(splits[2024])
    for r in changed:
        r["y_margin"] += 10
    other = fit(splits[2023], changed)["models"]["ridge"]["margin"]
    assert other["coefficients"] == m["coefficients"]
    assert other["bias"] == pytest.approx(m["bias"] + 10)


@pytest.mark.parametrize("line", [0, 0.5, -3, 54, 54.5])
@pytest.mark.parametrize("total", [True, False])
def test_discrete_probabilities_sum_and_push(line, total):
    p = probabilities(30.0 if total else 0.0, 12.0, line, total=total)
    assert sum(p.values()) == pytest.approx(1)
    assert all(0 <= v <= 1 for v in p.values())
    assert p["push"] == 0 if line % 1 else p["push"] >= 0
    if total:
        assert probabilities(0., 12., -0.5, total=True)["over"] == pytest.approx(1)


def test_symmetric_margin_and_invalid_sigma():
    p = probabilities(0., 10., 0)
    assert p["over"] == pytest.approx(p["under"])
    assert p["push"] > 0
    with pytest.raises(ValueError):
        probabilities(0, 0, 0)


def test_coverage_exclusions_duplicates_and_dates():
    features, targets = sample()
    features[0]["neutral_site"] = None
    _, coverage = prepare(features, targets)
    assert coverage[0]["excluded"] == 1
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(features, targets+[targets[0]])
    features[60]["kickoff"] = features[1]["kickoff"]
    with pytest.raises(ValueError, match="overlap"):
        prepare(features, targets)


def test_metrics_ties_and_reliability_counts():
    features, targets = sample()
    splits, _ = prepare(features, targets)
    artifact = fit(splits[2023], splits[2024])
    splits[2025][0]["y_margin"] = 0
    metrics, predictions = evaluate(artifact, splits[2025])
    for m in metrics.values():
        a = m["margin"]
        assert a["ties_excluded"] >= 1
        assert a["decided_games"] + a["ties_excluded"] == 60
        assert sum(b["n"] for b in a["reliability"]) == a["decided_games"]
        assert 0 <= a["brier"] <= 1 and math.isfinite(a["log_loss"])
        assert 0 <= m["total"]["nominal_90_interval_coverage"] <= 1
    assert len(predictions) == 360


def test_incomplete_collection_rejected():
    with pytest.raises(ValueError, match="Complete"):
        run_research(new_collection(2026))


def test_ui_does_not_train_on_rerun_or_display_stale_results():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_string("from app.ui.ncaaf_research import render_ncaaf_research\nfrom app_core.ncaaf_history import new_collection\nrender_ncaaf_research(new_collection(2026))")
    state = new_collection(2026)
    result = {"report": {"checkpoint_hash": hashlib.sha256(checkpoint_bytes(state)).hexdigest()}, "artifact": {}, "predictions": []}
    with patch("app.ui.ncaaf_research.pending_requests", return_value=[]), patch(
        "app.ui.ncaaf_research.run_research", return_value=result
    ) as run, patch("app.ui.ncaaf_research.markdown_report", return_value="Research result"):
        app.run()
        run.assert_not_called()
        app.button(key="ncaaf_research_run").click().run()
        assert run.call_count == 1 and not app.exception
        app.run()
        assert run.call_count == 1
        app.session_state["ncaaf_research_result"] = {"report": {"checkpoint_hash": "different"}}
        app.run()
        assert not app.exception
        assert "Research result" not in [m.value for m in app.markdown]
