"""Synthetic gate controls cannot authorize a production market."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone

import pytest

from activation_fixture import NOW, setup
from core.market_policy import SPORT_MARKET_FAMILIES, sport_market_family
from core.sport_market_activation import activate_market, verify_market_activation
from core.sport_market_gate import _canonical_quote_matches, market_gate
from core.sport_policy import SportPolicy


def fixture(tmp_path):
    row, policy, config = setup(tmp_path / "ledger.sqlite3")
    return row, policy, config, config["_test_only_market_deployments"]["NFL:SPREAD"], config["market_activations"]["NFL:SPREAD"]


def test_all_twelve_sport_market_scopes_are_distinct():
    assert sum(len(families) for families in SPORT_MARKET_FAMILIES.values()) == 12
    assert {sport_market_family(sport, "spread_home") for sport in ("NFL", "NBA", "MLB", "NHL")} == {
        "SPREAD", "RUN_LINE", "PUCK_LINE"}
    for sport, families in SPORT_MARKET_FAMILIES.items():
        assert sport_market_family(sport, "total_under") == "TOTAL"
        assert sport_market_family(sport, "spread_away") == families[0]
        assert sport_market_family(sport, "moneyline_home") is None


def test_exact_verified_scope_needs_independent_owner_activation(tmp_path):
    row, policy, config, state, activation = fixture(tmp_path)
    approved, blockers = market_gate(row, state, activation, config["exposure"], NOW, allow_test_only=True)
    assert approved == policy and blockers == []
    approved, blockers = market_gate(row, state, None, config["exposure"], NOW, allow_test_only=True)
    assert approved is None and "owner_market_activation_or_exposure_missing" in blockers


@pytest.mark.parametrize("change,blocker", [
    ({"market_type": "total_over", "line": 42.5, "selection": "Over 42.5"}, "exact_market_deployment_missing"),
    ({"sport": "NBA"}, "exact_market_deployment_missing"),
    ({"model_id": "different"}, "exact_market_model_id_mismatch"),
    ({"model_version": "different"}, "exact_market_model_version_mismatch"),
    ({"calibration_id": "different"}, "exact_market_calibration_id_mismatch"),
    ({"calibration_version": "different"}, "exact_market_calibration_version_mismatch"),
    ({"validation_id": "different"}, "candidate_validation_id_mismatch"),
    ({"provider_event_id": None}, "exact_provider_event_identity_missing"),
    ({"model_available_at": (NOW + timedelta(seconds=1)).isoformat()}, "market_model_calibration_chronology_invalid"),
    ({"calibration_available_at": (NOW + timedelta(seconds=1)).isoformat()}, "market_model_calibration_chronology_invalid"),
    ({"quote_time": (NOW + timedelta(seconds=1)).isoformat()}, "market_model_calibration_chronology_invalid"),
    ({"quote_time": (NOW - timedelta(hours=1)).isoformat()}, "exact_pregame_quote_missing_or_stale"),
])
def test_other_scope_or_incomplete_evidence_cannot_pass(tmp_path, change, blocker):
    row, _, config, state, activation = fixture(tmp_path)
    approved, blockers = market_gate(dict(row, **change), state, activation,
                                     config["exposure"], NOW, allow_test_only=True)
    assert approved is None and blocker in blockers


def test_owner_record_is_explicit_hash_bound_and_limited(tmp_path):
    row, policy, config, state, activation = fixture(tmp_path)
    import json
    assert verify_market_activation(activation, json.loads(json.dumps(state)),
                                    config["exposure"], now=NOW, allow_test_only=True) == policy
    with pytest.raises(ValueError, match="confirmation"):
        activate_market(state, policy, config["exposure"], owner_id="owner",
                        expires_at=(NOW + timedelta(days=1)).isoformat(), now=NOW)
    assert verify_market_activation(activation, state, config["exposure"],
                                    now=NOW, allow_test_only=True) == policy
    tampered = dict(activation, bankroll=10000)
    with pytest.raises(ValueError, match="hash"):
        verify_market_activation(tampered, state, config["exposure"], now=NOW, allow_test_only=True)
    broader = deepcopy(config["exposure"])
    broader["total_cap"] = .5
    from core.exposure_ledger import digest
    broader["snapshot_hash"] = digest({k: v for k, v in broader.items() if k != "snapshot_hash"})
    with pytest.raises(ValueError, match="limit"):
        verify_market_activation(activation, state, broader, now=NOW, allow_test_only=True)
    with pytest.raises(ValueError, match="Test"):
        verify_market_activation(activation, state, config["exposure"], now=NOW)
    changed = dict(state, validated_policy=dict(state["validated_policy"], sport_exposure_cap=.9))
    with pytest.raises(ValueError, match="policy"):
        verify_market_activation(activation, changed, config["exposure"], now=NOW, allow_test_only=True)


def test_sport_policy_cannot_stand_in_for_unreviewed_market(tmp_path):
    row, policy, config, state, activation = fixture(tmp_path)
    unvalidated = dict(state, validation_state="UNVALIDATED", deployment_state="UNVALIDATED")
    approved, blockers = market_gate(row, unvalidated, activation, config["exposure"], NOW,
                                     allow_test_only=True)
    assert approved is None and "exact_market_not_validated" in blockers
    with pytest.raises(ValueError, match="reviewed"):
        activate_market(unvalidated, replace(policy, deployment_state="UNVALIDATED"),
                        config["exposure"], owner_id="owner",
                        expires_at=(NOW + timedelta(days=1)).isoformat(), confirm=True, now=NOW)


@pytest.mark.parametrize("sport,family", [("NBA", "SPREAD"), ("NCAAB", "SPREAD"),
                                          ("NHL", "PUCK_LINE")])
def test_new_sport_fallback_book_requires_exact_retained_quote(tmp_path, monkeypatch, sport, family):
    from app_core import prospective_evidence as evidence
    from core.exposure_ledger import digest
    base = datetime(2030, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(evidence, "_clock", lambda: base)
    path = tmp_path / "prospective.sqlite3"
    start = base + timedelta(hours=1)
    market = {"key": "spreads", "last_update": base.isoformat(), "outcomes": [
        {"name": "Home", "point": -2.5, "price": -110},
        {"name": "Away", "point": 2.5, "price": -110}]}
    provider_response = {"id": "provider-event", "sport_key": evidence.ODDS_API_SPORT_KEYS[sport],
        "home_team": "Home", "away_team": "Away", "commence_time": start.isoformat(),
        "bookmakers": [{"key": "draftkings", "markets": [market]}]}
    source_id = f"provider-event:draftkings:spreads:Home:{base.isoformat()}"
    evidence.insert_event(path, dict(event_id="event", sport=sport, game_id="game",
        provider_namespace="THE_ODDS_API", provider_event_id="provider-event",
        home_team="Home", away_team="Away", home_team_id="home-id", away_team_id="away-id",
        scheduled_start=start.isoformat(), observed_at=base.isoformat(),
        source_id="event-source", raw_source=provider_response))
    evidence.insert_quote(path, dict(quote_id="quote", event_id="event", sport=sport,
        market_family=family, selection="Home", line=-2.5, american_odds=-110,
        decimal_odds=1.909090909, sportsbook="draftkings", quote_timestamp=base.isoformat(),
        quote_source="THE_ODDS_API", quote_verified=True, source_id=source_id,
        raw_source=market))
    saved = evidence.load_record(path, "prospective_quote", "quote")
    now = base + timedelta(minutes=2)
    row = dict(sport=sport, game_id="game", market_type="spread_home", selection="Home",
        line=-2.5, odds_american=-110, book="DraftKings", exact_quote_verified=True,
        identity_verified=True, team_ids=["home-id", "away-id"],
        provider_namespace="THE_ODDS_API", provider_event_id="provider-event",
        prospective_event_id="event", prospective_quote_id="quote",
        quote_source_id=source_id, quote_source_hash=saved["source_hash"],
        quote_time=base.isoformat(), start=start.isoformat(),
        prediction_generated_at=(base + timedelta(minutes=1)).isoformat(),
        model_trained_through=(base - timedelta(days=1)).isoformat(),
        model_available_at=(base - timedelta(hours=1)).isoformat(),
        calibration_available_at=(base - timedelta(hours=1)).isoformat(),
        evidence_frozen_at=(base - timedelta(minutes=1)).isoformat(),
        model_id="synthetic-model", model_version="v1",
        calibration_id="synthetic-calibration", calibration_version="c1",
        sport_policy_version="synthetic-policy")
    assert _canonical_quote_matches(row, family, path)
    exposure = dict(as_of=now.isoformat(), bankroll=1000., unit_value=10., currency="USD",
                    committed={}, ledger_hash="synthetic-ledger", total_cap=.05,
                    daily_cap=.05, weekly_cap=.05, game_cap=.02, team_cap=.02)
    exposure["snapshot_hash"] = digest(exposure)
    policy = SportPolicy(sport, "synthetic-policy", validation_id="SYNTHETIC-VALIDATION",
                         deployment_state="PROVISIONAL_VALIDATED", provisional_allowed=True,
                         provisional_stake_cap=.0025, kelly_fraction=.1, sport_exposure_cap=.03)
    state = dict(sport=sport, market_family=family, validation_state="PROVISIONAL_VALIDATED",
                 deployment_state="PROVISIONAL_VALIDATED", validation_id=policy.validation_id,
                 artifact_id="synthetic-artifact", model_id=row["model_id"],
                 model_version=row["model_version"], calibration_id=row["calibration_id"],
                 calibration_version=row["calibration_version"],
                 validated_policy=dict(asdict(policy), validation_id=""))
    activation = activate_market(state, policy, exposure, owner_id="SYNTHETIC-OWNER",
        expires_at=(now + timedelta(days=1)).isoformat(), confirm=True, now=now)
    approved, blockers = market_gate(row, state, activation, exposure, now,
                                     allow_test_only=True, evidence_path=path)
    assert approved == policy and blockers == []
    for change in ({"quote_source_hash": "wrong"}, {"line": -3.5},
                   {"odds_american": -120}, {"book": "FanDuel"},
                   {"provider_event_id": "other"}, {"selection": "Away"}):
        changed = dict(row, **change)
        approved, blockers = market_gate(changed, state, activation, exposure, now,
                                         allow_test_only=True, evidence_path=path)
        assert approved is None and "canonical_quote_source_binding_missing" in blockers
    # Even if normalized columns are rewritten to match a candidate, the
    # retained provider market still proves the original -2.5 line.
    real_load = evidence.load_record
    def altered_normalized_quote(db_path, table, record_id):
        record = real_load(db_path, table, record_id)
        return dict(record, line=-3.5) if table == "prospective_quote" else record
    monkeypatch.setattr(evidence, "load_record", altered_normalized_quote)
    assert not _canonical_quote_matches(dict(row, line=-3.5), family, path)
