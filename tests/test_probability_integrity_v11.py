from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from core import probability_calibration as pc
from core.probability_semantics import conditional_probabilities


def test_duplicate_predictor_counterexample_is_a_constant_function():
    probabilities = [0.6] * 200
    outcomes = [1] * 120 + [0] * 80

    knots = pc.fit_isotonic_calibration(probabilities, outcomes)
    actual = pc.apply_calibration(
        pd.Series([0.599999, 0.6, 0.600001]), knots
    ).tolist()

    assert len(knots) == 1
    assert knots[0] == pytest.approx([0.6, 0.6])
    assert actual == pytest.approx([0.6, 0.6, 0.6])


def test_pooled_plateau_preserves_original_x_coordinates():
    probabilities = [0.1, 0.2, 0.3, 0.4]
    outcomes = [0, 1, 0, 1]

    knots = pc.fit_isotonic_calibration(probabilities, outcomes)

    assert knots == [
        [0.1, 0.0],
        [0.2, 0.5],
        [0.3, 0.5],
        [0.4, 1.0],
    ]
    assert pc.apply_calibration(pd.Series(probabilities), knots).tolist() == [
        0.0, 0.5, 0.5, 1.0
    ]


def test_fit_is_permutation_stable_and_supports_positive_weights():
    observations = [
        (0.2, 1, 1.0), (0.2, 0, 3.0), (0.4, 1, 2.0),
        (0.6, 0, 1.0), (0.8, 1, 4.0),
    ]
    expected = pc.fit_isotonic_calibration(
        [row[0] for row in observations],
        [row[1] for row in observations],
        sample_weights=[row[2] for row in observations],
    )
    random.Random(2351).shuffle(observations)
    actual = pc.fit_isotonic_calibration(
        [row[0] for row in observations],
        [row[1] for row in observations],
        sample_weights=[row[2] for row in observations],
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("probabilities", "outcomes", "weights", "match"),
    [
        ([0.2, 0.3], [1], None, "equal lengths"),
        ([0.2, float("nan")], [1, 0], None, "missing"),
        ([0.2, float("inf")], [1, 0], None, "finite"),
        ([-0.1, 0.3], [1, 0], None, r"\[0, 1\]"),
        ([0.2, 0.3], [1, 2], None, "binary"),
        ([0.2, 0.3], [1, 0], [1.0], "sample_weights"),
        ([0.2, 0.3], [1, 0], [1.0, 0.0], "positive"),
    ],
)
def test_fit_rejects_malformed_observations(
    probabilities, outcomes, weights, match
):
    with pytest.raises(ValueError, match=match):
        pc.fit_isotonic_calibration(
            probabilities, outcomes, sample_weights=weights
        )


@pytest.mark.parametrize(
    "knots",
    [
        [[0.2, 0.4], [0.2, 0.5]],
        [[0.2, 0.6], [0.4, 0.5]],
        [[-0.1, 0.5], [0.4, 0.6]],
        [[0.2, float("nan")]],
        [[0.2, 0.4, 0.6]],
    ],
)
def test_apply_rejects_ambiguous_or_invalid_knots(knots):
    with pytest.raises(ValueError):
        pc.apply_calibration(pd.Series([0.3]), knots)


def test_missing_empty_table_keeps_existing_pass_through_contract():
    probabilities = pd.Series([0.2, 0.8])
    assert pc.apply_calibration(probabilities, []).equals(probabilities)


def test_apply_clips_to_endpoint_values_and_preserves_missing_values():
    result = pc.apply_calibration(
        pd.Series([0.0, 0.2, 0.5, 0.8, 1.0, np.nan]),
        [[0.2, 0.3], [0.8, 0.7]],
    )
    assert result.iloc[:5].tolist() == pytest.approx([0.3, 0.3, 0.5, 0.7, 0.7])
    assert pd.isna(result.iloc[5])


@pytest.mark.parametrize("seed", range(10))
def test_runtime_pav_matches_sklearn_reference(seed):
    rng = np.random.default_rng(seed)
    x = np.round(rng.uniform(0.05, 0.95, 250), 1)
    y = rng.integers(0, 2, size=len(x))
    weights = rng.integers(1, 6, size=len(x)).astype(float)
    query = np.linspace(0.0, 1.0, 301)

    expected = IsotonicRegression(out_of_bounds="clip").fit(
        x, y, sample_weight=weights
    ).predict(query)
    knots = pc.fit_isotonic_calibration(
        x.tolist(), y.tolist(), sample_weights=weights.tolist()
    )
    actual = pc.apply_calibration(pd.Series(query), knots).to_numpy()

    assert actual == pytest.approx(expected, abs=1e-12)


def test_same_date_production_artifact_is_rejected_with_structured_reason(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "same-date.json"
    pc.save_calibration(
        [[0.2, 0.3], [0.8, 0.7]],
        artifact,
        meta={
            "validation": {
                "promotable": True,
                "train_end": "2026-09-20T23:00:00Z",
                "test_start": "2026-09-20T23:30:00Z",
            }
        },
    )
    monkeypatch.setattr(pc, "DEFAULT_CALIBRATION_PATH", artifact)

    report = pc.inspect_calibration_artifact()

    assert report["acceptance_state"] == "REJECTED"
    assert "HOLDOUT_NOT_STRICTLY_FUTURE" in report["rejection_reasons"]
    assert pc.load_calibration() is None
    assert pc.load_calibration(artifact).acceptance["acceptance_state"] == "RESEARCH_ONLY"


def test_malformed_artifact_is_unavailable_instead_of_partially_loaded(tmp_path):
    artifact = tmp_path / "bad.json"
    artifact.write_text(json.dumps({"knots": [[0.5, 0.6], [0.5, 0.7]], "meta": {}}))

    report = pc.inspect_calibration_artifact(artifact)

    assert report["acceptance_state"] == "REJECTED"
    assert report["rejection_reasons"][0].startswith("KNOTS_INVALID:")
    assert pc.load_calibration(artifact) is None


def test_unconditional_push_mass_converts_to_conditional_probability_exactly():
    converted = conditional_probabilities(
        {
            "probability_semantics": "win_unconditional_with_push",
            "calibrated_probability": 0.54,
            "push_probability": 0.10,
            "market_probability": 0.45,
            "market_push_probability": 0.10,
        }
    )
    assert converted == pytest.approx((0.60, 0.50))


def test_unknown_probability_semantics_cannot_enter_conditional_scoring():
    assert conditional_probabilities(
        {
            "probability_semantics": "UNKNOWN",
            "calibrated_probability": 0.60,
            "market_probability": 0.50,
            "push_probability": 0.0,
            "market_push_probability": 0.0,
        }
    ) is None


def test_force_cannot_write_the_live_calibration_path(tmp_path, monkeypatch):
    from scripts import fit_calibration

    live = tmp_path / "live-calibration.json"
    monkeypatch.setattr(fit_calibration, "DEFAULT_CALIBRATION_PATH", live)

    result = fit_calibration.main(
        [str(tmp_path / "missing-exports"), str(live), "--force"]
    )

    assert result == 2
    assert not live.exists()
