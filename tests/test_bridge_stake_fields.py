"""Go-live Part 2: the stake fields a bridged pick needs, derived from stored evidence. SYNTHETIC."""
from datetime import datetime

import pytest

from app_core import bridge_stake_fields as bsf
from app_core import market_skill_validation as ms
from app_core import prospective_evidence as pe
from app_core import qualified_bridge as qb
from tests.test_market_skill_validation import _plan
from tests.test_prospective_evidence import (  # noqa: F401  (clock is a fixture)
    _trained_market, at, clock, digest, event, prediction, quote)
from tests.test_qualified_bridge import _upcoming

DEPLOYMENT = {"sport": "NBA", "market_family": "SPREAD", "deployment_state": "PROVISIONAL_VALIDATED",
              "validation_state": "PROVISIONAL_VALIDATED", "validation_id": "v", "artifact_id": "a",
              "model_id": "model", "model_version": "test-model-v1", "calibration_id": "calibration",
              "calibration_version": "test-cal-v1",
              "validation_report": {"validation": {"effective_observations": 210},
                                    "holdout": {"effective_observations": 190}},
              "deployment_criteria": {"validated_policy": {"uncertainty_quantile": 0.25},
                                      "regime_brake": {"window_bets": 50, "min_mean_clv": 0.0}}}


def _snapshot_prediction(path, event_id, quote_id, observation_id, stamp, *, mean=.55, loss=.45):  # noqa: F811
    snapshot = {"protocol": "test", "quote_id": quote_id, "event_id": event_id}
    return pe.insert_prediction(path, dict(
        observation_id=observation_id, event_id=event_id, sport="NBA", market_family="SPREAD", selection="Home",
        quote_id=quote_id, model_id="model", calibration_id="calibration", feature_version="test-features-v1",
        feature_snapshot_id=pe._sha(pe._json(snapshot)), feature_snapshot=snapshot, feature_frozen_at=at(stamp),
        policy_version="synthetic-policy-v1", prediction_timestamp=at(stamp), mean_probability=mean,
        conservative_probability=0.40, push_probability=0, loss_probability=loss,
        probability_semantics="win_push_loss", evidence_snapshot_id=f"evidence-{observation_id}",
        evidence_hash=digest(observation_id), runtime_hash=digest("runtime"), source_commit="synthetic-commit"))


def test_unvalidated_markets_get_fail_closed_fields(tmp_path, clock):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _trained_market(path, clock)
    _upcoming(path, clock)
    row = qb.rows(path, {("NBA", "SPREAD")}, now=datetime.fromisoformat(at(8, 10)))[0]
    assert (row["model_validated"], row["calibration_validated"], row["critical_feature_error"]) == (False, False, True)
    assert row["evidence_effective_sample_size"] == 0.0 and row["current_regime_conflict"] is None
    assert row["evidence_snapshot_id"] == "upcoming-pred"
    assert row["conservative_probability"] == 0.58                          # the stored value, untouched


def test_validated_market_fields_come_from_the_deployment_and_the_stored_snapshot(tmp_path, clock, monkeypatch):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _trained_market(path, clock)
    _upcoming(path, clock)
    _snapshot_prediction(path, "upcoming", "upcoming-quote", "with-snapshot", 8, mean=.55, loss=.45)
    monkeypatch.setattr(pe, "deployment_state", lambda *a: DEPLOYMENT)
    context = bsf.market_context(path, "NBA", "SPREAD", now=datetime.fromisoformat(at(8, 10)))
    assert context["validated"] and context["effective_n"] == 400 and context["z"] == pytest.approx(0.6745, abs=1e-4)
    assert context["regime"] == {"bets": 0, "mean_clv": None, "conflict": False}     # no plan, no live bets
    records = {p["observation_id"]: p for p in pe.read_records(path, "prospective_prediction")}
    quote_row = pe.load_record(path, "prospective_quote", "upcoming-quote")
    event_row = pe.load_record(path, "prospective_event", "upcoming")
    good = bsf.fields(context, records["with-snapshot"], quote_row, event_row)
    assert good["model_validated"] and good["calibration_validated"] and good["critical_feature_error"] is False
    assert good["evidence_effective_sample_size"] == 400
    assert good["calibration_uncertainty"] == pytest.approx(0.6745 * (0.55 * 0.45 / 400) ** 0.5, abs=1e-5)
    assert good["conservative_probability"] == pytest.approx(0.55 - good["calibration_uncertainty"])
    assert good["current_regime_conflict"] is False and good["evidence_snapshot_id"] == "evidence-with-snapshot"
    # The fixture's other prediction stored no feature snapshot: a critical feature error.
    assert bsf.fields(context, records["upcoming-pred"], quote_row, event_row)["critical_feature_error"] is True
    # Another model version: not validated.
    other = dict(records["with-snapshot"], model_version="other")
    assert bsf.fields(context, other, quote_row, event_row)["model_validated"] is False
    # A Pinnacle model's snapshot must hold Pinnacle's probability.
    pinnacle = dict(records["with-snapshot"], model_id="pinnacle-model:NBA:SPREAD:x")
    assert bsf.fields(context, pinnacle, quote_row, event_row)["critical_feature_error"] is True


def _live_game(path, clock, name, start, mean):  # noqa: F811
    clock(hours=start - 2)
    event(path, event_id=name, start=start, clock_hour=start - 2)
    quote(path, event_id=name, quote_id=f"{name}-home", stamp=start - 2)
    quote(path, event_id=name, quote_id=f"{name}-away", stamp=start - 2, selection="Away", line=2.5, raw_line=-2.5)
    prediction(path, event_id=name, quote_id=f"{name}-home", observation_id=f"{name}-pred", stamp=start - 2,
               mean=mean, loss=1 - mean)


@pytest.mark.parametrize("close_odds, conflict", [(1.80, False), (2.05, True)])
def test_drift_brake_reads_the_live_plan_bets_clv(tmp_path, clock, monkeypatch, close_odds, conflict):  # noqa: F811
    path = tmp_path / "SYNTHETIC.sqlite3"
    _trained_market(path, clock)
    clock(hours=8)
    _plan(path)                                                     # holdout ends at hour 25
    _live_game(path, clock, "live-bet", 30, 0.60)                   # EV +14.5% at -110: a plan bet
    _live_game(path, clock, "live-pass", 34, 0.51)                  # EV -2.6%: no bet
    clock(hours=40)
    monkeypatch.setattr(pe, "_comparable_close", lambda db, q, as_of: {"decimal_odds": close_odds})
    criteria = {"regime_brake": {"window_bets": 50, "min_mean_clv": 0.0},
                "market_skill_methodology": ms.methodology(0.02)}
    result = bsf.regime(path, "NBA", "SPREAD", DEPLOYMENT, criteria, now=datetime.fromisoformat(at(40)))
    assert result["bets"] == 1 and result["conflict"] is conflict
    assert result["mean_clv"] == pytest.approx(1.909090909 / close_odds - 1)
