"""Implausible Platt fits fall back to the identity; v1 calibrations are refitted (owner 2026-11-08)."""
import random

from app_core import pinnacle_model as pm
from app_core import prospective_calibration as pc
from tests.test_pinnacle_model import _pin_event, _research_market, clock  # noqa: F401  (clock is a fixture)
from tests.test_prospective_research_models import _at


def test_a_reversing_fit_on_near_even_prices_falls_back_to_the_identity():
    rng = random.Random(7)
    # 40 spread games priced 48-52% whose results happen to run against the price: the slope is
    # not identified and the fit reverses it (the NCAAB SPREAD case: a = -0.24, b = -13.2).
    pairs = [(p, int(p < 0.5) if rng.random() < 0.8 else int(p >= 0.5))
             for p in (rng.uniform(0.48, 0.52) for _ in range(40))]
    assert pc.platt_fit(pairs)[1] < 0
    a, b, fallback = pc.guarded_fit(pairs)
    assert (a, b) == (0.0, 1.0) and fallback["reason"] == "IMPLAUSIBLE_FIT_IDENTITY_USED"
    assert fallback["fitted_b"] < 0
    assert pc.calibrated({"win": 0.52, "push": 0.0, "loss": 0.48}, {"a": a, "b": b})["win"] == 0.52


def test_a_plausible_fit_is_kept():
    rng = random.Random(3)
    pairs = []
    for _ in range(4000):
        p = rng.uniform(0.2, 0.8)
        true = 1 / (1 + 2.718281828 ** (-(0.1 + 0.8 * pc._logit(p))))
        pairs.append((p, int(rng.random() < true)))
    a, b, fallback = pc.guarded_fit(pairs)
    assert fallback is None and 0.6 < b < 1.0 and abs(a - 0.1) < 0.15


def test_only_current_version_calibrations_count_as_present():
    rows = [{"model_id": "m", "calibration_version": "platt-logit-v1"},
            {"model_id": "m", "calibration_version": pc.CALIBRATION_VERSION},
            {"model_id": "other", "calibration_version": pc.CALIBRATION_VERSION}]
    assert pc.current_calibrations(rows, "m") == [rows[1]]
    assert pc.CALIBRATION_VERSION == "platt-logit-v2"


def test_a_v1_pinnacle_calibration_is_refitted_and_the_draft_binds_the_new_one(tmp_path, clock, monkeypatch):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _research_market(path, clock)
    model = pm.register(path, "NBA", "SPREAD", now=clock[0])
    for index in range(72, 72 + 48):
        _pin_event(path, clock, index)
        clock[0] = _at(index, 1)
        pc.predict_candidate(path, "NBA", "SPREAD", now=clock[0])    # the research cycle's own predictions
    clock[0] = _at(121)
    pc.settle_scope(path, "NBA", "SPREAD")
    monkeypatch.setattr(pc, "CALIBRATION_VERSION", "platt-logit-v1")
    old = pm.fit_calibration(path, "NBA", "SPREAD", model, now=clock[0])
    assert old["status"] == "FITTED"
    assert pm.fit_calibration(path, "NBA", "SPREAD", model, now=clock[0])["status"] == "PRESENT"
    monkeypatch.setattr(pc, "CALIBRATION_VERSION", "platt-logit-v2")
    clock[0] = _at(122)
    new = pm.fit_calibration(path, "NBA", "SPREAD", model, now=clock[0])
    assert new["status"] == "FITTED" and new["calibration_id"] != old["calibration_id"]
    assert pm.fit_calibration(path, "NBA", "SPREAD", model, now=clock[0])["status"] == "PRESENT"
