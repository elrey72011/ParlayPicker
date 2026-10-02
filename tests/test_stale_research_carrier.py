"""Stale carrier facts cannot override current retained-input declarations."""
from copy import deepcopy
import json
import socket

import pandas as pd
import pytest

from app_core.research_display import preserve_source_semantics
from core.streamlit_pipeline import _coerce_export_to_canonical
from test_research_probability_producer import forecast, real_path, assert_pass
from test_research_probability_browser import inspect_browser
from test_research_probability_display import NOW


def carried(raw):
    return preserve_source_semantics(pd.DataFrame([raw])).iloc[0].to_dict()


def assert_no_authority(result):
    assert_pass(result)
    for rows in result["package"]["games"].values():
        for row in rows:
            assert row["status"] == "PASS"
            assert row["win_estimate"] is None and row["ev"] is None
            contract = row.get("wager_contract")
            if contract:
                assert contract["production_bet_amount"] == 0
                assert contract["conservative_probability"] is None
                assert contract["conservative_ev"] is None


@pytest.mark.parametrize("changes,reason", [
    ({}, "AVAILABLE"),
    ({"probability_semantics": "unsupported"}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"probability_semantics": float("nan")}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability": .1}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability": -.1}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability": float("nan")}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability": float("inf")}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability": True}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability": False}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"inference_status": "FAILED"}, "INFERENCE_FAILED"),
    ({"model_status": "FAILED"}, "INFERENCE_FAILED"),
    ({"inference_status": "UNAVAILABLE"}, "INFERENCE_UNAVAILABLE"),
    ({"model_status": "missing"}, "INFERENCE_UNAVAILABLE"),
    ({"inference_status": "unknown"}, "INFERENCE_UNAVAILABLE"),
    ({"model_status": "unknown"}, "INFERENCE_UNAVAILABLE"),
    ({"inference_status": True}, "INFERENCE_UNAVAILABLE"),
    ({"model_status": float("nan")}, "INFERENCE_UNAVAILABLE"),
])
def test_current_negative_facts_survive_authority_capture(monkeypatch, tmp_path, changes, reason):
    raw = carried(forecast())
    raw.update(changes)
    result = real_path(monkeypatch, tmp_path, raw)
    assert_no_authority(result)
    display = result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"] == reason
    if reason != "AVAILABLE":
        assert display["probability"] is None and display["ev"] is None


@pytest.mark.parametrize("saved", ["not JSON", "{}", "null", "[]",
    '{"version":true,"fields":{}}',
    json.dumps({"version": True, "fields": {
        "probability_semantics": {"state": "VALUE", "value": "win_unconditional_with_push"},
        "push_probability": {"state": "VALUE", "value": 0.0},
        "inference_status": {"state": "VALUE", "value": "success"},
        "model_status": {"state": "VALUE", "value": "success"}}})])
def test_malformed_carrier_is_not_repaired(monkeypatch, tmp_path, saved):
    raw = forecast(research_source_semantics=saved)
    result = real_path(monkeypatch, tmp_path, raw)
    assert_no_authority(result)
    assert result["captured"].iloc[0].research_source_semantics == saved
    assert result["package"]["games"]["overall"][0]["research_display"]["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"


@pytest.mark.parametrize("changes,reason", [
    ({"probability_semantics": "unsupported"}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"push_probability": True}, "UNSUPPORTED_PROBABILITY_SEMANTICS"),
    ({"inference_status": "FAILED"}, "INFERENCE_FAILED"),
    ({"inference_status": "unknown"}, "INFERENCE_UNAVAILABLE"),
    ({"model_status": "unknown"}, "INFERENCE_UNAVAILABLE"),
])
def test_original_rejection_stays_sticky_through_repeated_normalization(monkeypatch, tmp_path, changes, reason):
    raw = carried(forecast(**changes))
    saved = raw["research_source_semantics"]
    raw.update(probability_semantics="win_unconditional_with_push", push_probability=0.0,
               inference_status="success", model_status="success")
    for _ in range(3):
        raw = carried(raw)
        assert raw["research_source_semantics"] == saved
    result = real_path(monkeypatch, tmp_path, raw)
    assert_no_authority(result)
    assert result["package"]["games"]["overall"][0]["research_display"]["availability_reason"] == reason


def test_matching_and_missing_current_facts_preserve_carrier_bytes(monkeypatch, tmp_path):
    raw = carried(forecast())
    saved = raw["research_source_semantics"]
    for field in ("probability_semantics", "push_probability", "inference_status", "model_status"):
        raw.pop(field)
    for _ in range(3):
        raw = carried(raw)
        assert raw["research_source_semantics"] == saved
    result = real_path(monkeypatch, tmp_path, raw)
    assert_no_authority(result)
    # The carrier itself is preserved. With the current probability contract
    # absent, the real producer does not export a normalized probability/basis.
    # Retained metadata cannot manufacture that missing producer output.
    display = result["package"]["games"]["overall"][0]["research_display"]
    assert display["probability"] is None
    assert display["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"


def test_boolean_semantics_export_rejects_without_fabricated_capture(monkeypatch, tmp_path):
    from test_research_probability_display import source, package_for
    raw = carried(source())
    raw["probability_semantics"] = True
    _, package = package_for(monkeypatch, raw)
    display = package["games"]["overall"][0]["research_display"]
    assert display["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("current", ["win_conditional_on_decision", "win_unconditional_with_push"])
def test_conditional_export_normalization_and_conflicting_source_relabel(monkeypatch, tmp_path, current):
    from test_research_probability_display import source, package_for, QUOTE
    raw = source(best_pick="Home -2", spread_line=-2.0, odds_american=100,
        best_available_probability=.575, probability_semantics="win_conditional_on_decision",
        push_probability=.1, provider_quotes=json.dumps([dict(book="novig", market_type="spread_home",
            point=-2, price=100, recorded_at=QUOTE)]))
    raw["wager_contract"].update(selection="Home -2", line=-2.0, odds=100)
    raw = carried(raw)
    raw["probability_semantics"] = current
    _, package = package_for(monkeypatch, raw)
    display = package["games"]["overall"][0]["research_display"]
    if current == "win_conditional_on_decision":
        # The real exporter converts the original conditional mass. Retaining
        # its source semantics must preserve that legitimate transformation.
        assert display["probability"] == pytest.approx(.5175)
        assert display["push_probability"] == .1 and display["ev"] == pytest.approx(.135)
    else:
        # Changing only the source label leaves .575 as unconditional input to
        # the exporter, conflicting with the retained conditional contract.
        assert display["probability"] is None and display["ev"] is None
        assert display["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] == display["probability"]
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


def test_reprocessed_non_probability_first_cannot_override_current_rejection(monkeypatch, tmp_path):
    from test_research_probability_display import source, package_for
    raw = carried(source(best_available_selection_policy="", production_win_probability=.6,
        production_expected_value=.6*(1+100/110)-1))
    raw["probability_semantics"] = "unsupported"
    _, package = package_for(monkeypatch, raw)
    display = package["games"]["overall"][0]["research_display"]
    assert display["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"
    assert display["probability"] is None and display["ev"] is None
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


def test_canonical_upload_reconciles_before_semantics_column_is_projected_away(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("Real transport prohibited")))
    raw = carried(forecast())
    raw["probability_semantics"] = "unsupported"
    frame = _coerce_export_to_canonical(pd.DataFrame([raw]), ["MLB"])
    for _ in range(3):
        recorded = json.loads(frame.iloc[0].research_source_semantics)["fields"]["probability_semantics"]
        assert recorded == {"state": "VALUE", "value": "unsupported"}
        frame = _coerce_export_to_canonical(frame, ["MLB"])


@pytest.mark.parametrize("target", [None, "", "spread_cover", "home_win", "total_under"])
def test_upload_target_alias_preserves_only_supplied_values(monkeypatch, target):
    monkeypatch.setattr(socket.socket, "connect", lambda *args, **kwargs:
                        (_ for _ in ()).throw(AssertionError("Real transport prohibited")))
    raw = forecast()
    if target is None:
        raw.pop("ml_target")
    else:
        raw["ml_target"] = target
    normalized = _coerce_export_to_canonical(pd.DataFrame([raw]), ["MLB"])
    if target is None:
        assert pd.isna(normalized.iloc[0].ml_target)
    else:
        assert normalized.iloc[0].ml_target == target


def test_missing_target_cannot_mask_explicit_unsupported_carrier(monkeypatch, tmp_path):
    from test_research_probability_display import source, package_for
    raw = carried(source(ml_target="", probability_semantics="unsupported"))
    raw.pop("wager_contract")
    raw["probability_semantics"] = "win_unconditional_with_push"
    _, package = package_for(monkeypatch, raw)
    row = package["games"]["overall"][0]
    assert row["research_display"]["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"
    assert row["win_estimate"] is None and row["ev"] is None
    from app_core.public_history import original_estimate
    assert original_estimate(row) == {}


@pytest.mark.parametrize("inference,reason,state", [
    ("success", "AVAILABLE", "RECORDED"),
    (None, "AVAILABLE", "UNKNOWN"),
    ("FAILED", "INFERENCE_FAILED", "FAILED"),
    ("unrecognized-run-status", "INFERENCE_UNAVAILABLE", "UNAVAILABLE"),
    ("unknown", "INFERENCE_UNAVAILABLE", "UNAVAILABLE"),
    ("Market Score Model", "INFERENCE_UNAVAILABLE", "UNAVAILABLE"),
])
def test_existing_producer_model_type_is_not_an_inference_verdict(
        monkeypatch, tmp_path, inference, reason, state):
    raw = carried(forecast(model_status="Market Score Model", inference_status=inference))
    result = real_path(monkeypatch, tmp_path, raw)
    assert_no_authority(result)
    display = result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"] == reason
    assert display["inference_status"] == state
    assert (display["probability"] is not None) == (reason == "AVAILABLE")
    browser = inspect_browser(result["package"], tmp_path / "browser", NOW, rendered_html=result["html"])
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


def test_unknown_model_type_still_rejects_recorded_inference(monkeypatch, tmp_path):
    result = real_path(monkeypatch, tmp_path, carried(forecast(
        model_status="unrecognized-model", inference_status="success")))
    assert_no_authority(result)
    display = result["package"]["games"]["overall"][0]["research_display"]
    assert display["availability_reason"] == "INFERENCE_UNAVAILABLE"


def test_known_model_type_cannot_erase_original_failure(monkeypatch, tmp_path):
    raw = carried(forecast(model_status="FAILED", inference_status="success"))
    raw["model_status"] = "Market Score Model"
    result = real_path(monkeypatch, tmp_path, raw)
    assert_no_authority(result)
    assert result["package"]["games"]["overall"][0]["research_display"]["availability_reason"] == "INFERENCE_FAILED"


def conditional_source_without_target(semantics="win_conditional_on_decision"):
    from test_research_probability_display import source, QUOTE
    raw = source(ml_target="", best_pick="Home -2", spread_line=-2.0, odds_american=100,
        best_available_probability=.575, probability_semantics=semantics,
        push_probability=.1, provider_quotes=json.dumps([dict(book="novig", market_type="spread_home",
            point=-2, price=100, recorded_at=QUOTE)]))
    raw.pop("wager_contract")
    return carried(raw)


@pytest.mark.parametrize("probability,ev,valid", [
    (.575, .25, True), (.6, .3, False), (.575, .3, False),
])
def test_legacy_unconditional_source_mass_and_ev_must_match(monkeypatch, tmp_path, probability, ev, valid):
    from test_research_probability_display import package_for
    from app_core.public_board import build_package, validate_package
    from app_core.public_history import original_estimate
    raw = conditional_source_without_target("win_unconditional_with_push")
    frames, _ = package_for(monkeypatch, raw)
    for frame in frames[:2]:
        frame["research_source_semantics"] = raw["research_source_semantics"]
        frame["best_available_probability"] = raw["best_available_probability"]
        frame["win_probability"] = probability
        frame["ev"] = ev
    package = build_package(*frames); validate_package(package)
    row = package["games"]["overall"][0]
    if valid:
        assert row["win_estimate"] == .575 and row["ev"] == .25
    else:
        assert row["win_estimate"] is None and row["ev"] is None
        assert original_estimate(row) == {}
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("token", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["probability_semantics", "push_probability", "inference_status", "model_status"])
def test_nonfinite_carrier_values_survive_current_conflict_as_malformed(
        monkeypatch, tmp_path, token, field):
    raw = forecast(quote_bookmaker="novig")
    decoded = json.loads(carried(raw)["research_source_semantics"])
    decoded["fields"][field] = {"state": "VALUE", "value": token}
    saved = json.dumps(decoded, sort_keys=True, separators=(",", ":"))
    raw["research_source_semantics"] = saved
    raw["model_status" if field == "inference_status" else "inference_status"] = "FAILED"
    result = upload_path(monkeypatch, tmp_path, raw)
    assert result["normalized"].iloc[0].research_source_semantics == saved
    assert result["captured"].iloc[0].research_source_semantics == saved
    assert_no_authority(result)
    row = result["package"]["games"]["overall"][0]
    assert row["research_display"]["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"
    from app_core.public_history import original_estimate
    assert original_estimate(row) == {}


@pytest.mark.parametrize("probability_first", [True, False])
@pytest.mark.parametrize("probability,valid", [(False, False), (True, False), (0.0, True)])
def test_missing_target_cannot_convert_boolean_source_into_legacy_zero_or_one(
        monkeypatch, tmp_path, probability_first, probability, valid):
    from test_research_probability_display import source, package_for
    from app_core.public_history import original_estimate
    values = {"ml_target": ""}
    if probability_first:
        values["best_available_probability"] = probability
    else:
        values.update(best_available_selection_policy="", production_win_probability=probability,
                      production_expected_value=float(probability)*(1+100/110)-1)
    raw = source(**values); raw.pop("wager_contract")
    frames, package = package_for(monkeypatch, raw)
    row = package["games"]["overall"][0]
    if valid:
        assert row["win_estimate"] == 0.0 and row["ev"] == -1.0
        assert original_estimate(row)["original_win_estimate"] == 0.0
    else:
        assert row["research_display"]["availability_reason"] == "INVALID_PROBABILITY"
        assert row["win_estimate"] is None and row["ev"] is None
        assert original_estimate(row) == {}
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("target", ["", "spread_cover"])
@pytest.mark.parametrize("ev,valid", [(False, False), (True, False), (0.0, True)])
def test_boolean_producer_ev_cannot_hide_behind_missing_metadata(monkeypatch, tmp_path, target, ev, valid):
    from test_research_probability_display import source, package_for
    from app_core.public_history import original_estimate
    probability = 110/210
    raw = source(best_available_selection_policy="", ml_target=target,
        production_win_probability=probability, production_expected_value=ev)
    raw.pop("wager_contract")
    frames, package = package_for(monkeypatch, raw)
    row = package["games"]["overall"][0]
    if valid:
        assert row["ev"] == 0.0 if not target else row["research_display"]["ev"] == 0.0
    else:
        assert row["research_display"]["value_reason"] == "INVALID_RECORDED_EV"
        assert row["ev"] is None
        if not target:
            assert row["win_estimate"] is None and original_estimate(row) == {}
        else:
            assert row["research_display"]["probability"] == pytest.approx(probability)
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("probability_first", [True, False])
@pytest.mark.parametrize("value,reason", [
    (None, "ESTIMATE_NOT_RECORDED"), (float("nan"), "ESTIMATE_NOT_RECORDED"),
    (float("inf"), "NONFINITE_PROBABILITY"), (-float("inf"), "NONFINITE_PROBABILITY"),
    ("invalid", "NONFINITE_PROBABILITY"), (-.1, "INVALID_PROBABILITY"), (1.1, "INVALID_PROBABILITY"),
])
def test_missing_target_does_not_hide_invalid_probability_or_keep_orphaned_ev(
        monkeypatch, tmp_path, probability_first, value, reason):
    from test_research_probability_display import source, package_for
    from app_core.public_history import original_estimate
    values = {"ml_target": ""}
    if probability_first:
        values["best_available_probability"] = value
    else:
        values.update(best_available_selection_policy="", production_win_probability=value,
                      production_expected_value=0.0)
    raw = source(**values); raw.pop("wager_contract")
    frames, package = package_for(monkeypatch, raw)
    row = package["games"]["overall"][0]
    assert row["research_display"]["availability_reason"] == reason
    assert row["win_estimate"] is None and row["ev"] is None
    assert original_estimate(row) == {}
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("current,valid", [
    ("win_conditional_on_decision", True), ("win_unconditional_with_push", False),
])
def test_missing_target_preserves_real_conversion_but_rejects_source_relabel(
        monkeypatch, tmp_path, current, valid):
    from test_research_probability_display import package_for
    from app_core.public_history import original_estimate
    from app_core.release_preflight import evaluate_release
    raw = conditional_source_without_target()
    raw["probability_semantics"] = current
    frames, package = package_for(monkeypatch, raw)
    row = package["games"]["overall"][0]
    if valid:
        assert frames[0].iloc[0].win_probability == pytest.approx(.5175)
        assert row["win_estimate"] == pytest.approx(.5175)
        assert row["ev"] == pytest.approx(.135)
        assert row["research_display"]["availability_reason"] == "MODEL_TARGET_NOT_RECORDED"
    else:
        assert row["research_display"]["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"
        assert row["win_estimate"] is None and row["ev"] is None
        assert original_estimate(row) == {}
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    # A missing model target still leaves the card unavailable even when the
    # valid legacy record retains its correctly converted historical values.
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0
    assert row["status"] == "PASS" and not frames[0].iloc[0].Bettable
    assert frames[0].iloc[0].Play_Stake == 0
    assert evaluate_release(package, at=NOW)["actionable_row_count"] == 0


@pytest.mark.parametrize("push", [.1, True, False, float("nan"), float("inf"), -.1, 1.1])
def test_missing_target_non_probability_first_cannot_drop_invalid_source_push(
        monkeypatch, tmp_path, push):
    from test_research_probability_display import source, package_for
    from app_core.public_history import original_estimate
    raw = source(best_available_selection_policy="", ml_target="", production_win_probability=.6,
        production_expected_value=.6*(1+100/110)-1, push_probability=push)
    raw.pop("wager_contract")
    frames, package = package_for(monkeypatch, raw)
    row = package["games"]["overall"][0]
    assert row["research_display"]["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"
    assert row["win_estimate"] is None and row["ev"] is None
    assert original_estimate(row) == {}
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("raw_probability,probability,ev,retained_raw,valid", [
    (.575, .575, .135, True, False), (.575, .5175, .135, True, True),
    (.575, .5175, .135, False, False), (.575, .5175, .25, True, False),
    (.4, .36, -.18, True, True),
])
def test_legacy_semantic_conversion_needs_retained_raw_mass(
        monkeypatch, tmp_path, raw_probability, probability, ev, retained_raw, valid):
    from test_research_probability_display import package_for
    from app_core.public_board import build_package, validate_package
    from app_core.public_history import original_estimate
    raw = conditional_source_without_target()
    raw["best_available_probability"] = raw_probability
    frames, _ = package_for(monkeypatch, raw)
    # A retained legacy export can include original source facts. A carrier by
    # itself contains only declarations, and must not invent its missing raw p.
    for frame in frames[:2]:
        frame["research_source_semantics"] = raw["research_source_semantics"]
        if retained_raw:
            frame["best_available_probability"] = raw["best_available_probability"]
        frame["win_probability"] = probability
        frame["ev"] = ev
    package = build_package(*frames)
    validate_package(package)
    row = package["games"]["overall"][0]
    if valid:
        assert row["win_estimate"] == pytest.approx(probability) and row["ev"] == pytest.approx(ev)
    else:
        assert row["win_estimate"] is None and row["ev"] is None
        assert original_estimate(row) == {}
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("changes", [
    {"probability_semantics": "unsupported"},
    {"push_probability": True},
    {"push_probability": float("nan")},
    {"push_probability": .1},
    {"inference_status": "FAILED"},
    {"inference_status": "unknown"},
    {"model_status": "unknown"},
    {"model_status": "unknown-model"},
])
def test_legacy_exception_cannot_mask_explicit_post_export_rejection(monkeypatch, changes):
    from test_research_probability_display import source, package_for
    from app_core.public_board import pick_record
    from app_core.public_history import original_estimate
    raw = carried(source(ml_target=""))
    raw.pop("wager_contract")
    frames, _ = package_for(monkeypatch, raw)
    row = frames[0].iloc[0].to_dict()
    assert json.loads(row["research_display"])["availability_reason"] == "MODEL_TARGET_NOT_RECORDED"
    row.update(changes)
    public = pick_record(row)
    assert public["status"] == "PASS"
    assert public["win_estimate"] is None and public["ev"] is None
    assert original_estimate(public) == {}


def upload_path(monkeypatch, tmp_path, raw):
    def prohibited(*args, **kwargs):
        raise AssertionError("Real transport is prohibited in carrier regression")
    monkeypatch.setattr(socket.socket, "connect", prohibited)
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)
    normalized = _coerce_export_to_canonical(pd.DataFrame([raw]).rename(columns={"game_time_est": "game time"}), ["MLB"])
    result = real_path(monkeypatch, tmp_path, normalized.iloc[0].to_dict())
    result["normalized"] = normalized
    assert_pass(result)
    return result


@pytest.mark.parametrize("current,available", [("win_unconditional_with_push", True), ("unsupported", False)])
def test_retained_upload_to_captured_public_browser(monkeypatch, tmp_path, current, available):
    # Synthetic retained upload. The normalizer's existing
    # allowlist supports quote_bookmaker; supply the exact fake quote's book
    # before normalization, rather than repairing a contract after export.
    # Existing incomplete line provenance leaves the final contract's line
    # unresolved; research cannot borrow that different selection's authority.
    raw = carried(forecast(quote_bookmaker="novig"))
    raw["probability_semantics"] = current
    original = deepcopy(raw)
    result = upload_path(monkeypatch, tmp_path, raw)
    display = result["package"]["games"]["overall"][0]["research_display"]
    browser = inspect_browser(result["package"], tmp_path / "browser", NOW, rendered_html=result["html"])
    assert (display["availability_reason"] == "AVAILABLE") == available
    assert (browser["initial"]["shown"][0]["probability"] is not None) == available
    if not available:
        assert display["availability_reason"] == "UNSUPPORTED_PROBABILITY_SEMANTICS"
        assert display["probability"] is None and display["ev"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0
    assert browser["initial"]["saved"] == [dict(probability=None, ev=None, status="PASS", stake=0)]
    from app_core.public_history import original_estimate, selections
    from app_core.top_ten_history import ranked_picks
    from app_core.public_parlays import build_research_parlays
    from app_core.production_parlays import build_production_parlays
    rows = result["package"]["games"]["overall"]
    assert original_estimate(rows[0]) == {}
    assert ranked_picks(result["package"], NOW) == []
    assert build_research_parlays(rows, NOW) == build_production_parlays(rows, NOW) == []
    publication = dict(package=result["package"], confirmed_at=NOW.isoformat(),
                       package_hash="synthetic-upload-test")
    assert all(row["group"] == "Research" for row in selections([publication]))
    assert raw == original


@pytest.mark.parametrize("changes", [
    {"prospective_quote_id": "different-exact-quote"}, {"period": "first_half"},
    {"best_pick": "Home -2.5"},
])
def test_missing_target_cannot_hide_explicit_identity_conflicts(monkeypatch, tmp_path, changes):
    from test_research_probability_display import source, package_for
    from app_core.public_history import original_estimate
    raw = source(ml_target="", **changes); raw.pop("wager_contract")
    frames, package = package_for(monkeypatch, raw)
    row = package["games"]["overall"][0]
    assert row["research_display"]["availability_reason"] == "ESTIMATE_IDENTITY_MISMATCH"
    assert row["win_estimate"] is None and row["ev"] is None
    assert original_estimate(row) == {}
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("field,value", [
    ("event_id", "other-event"), ("candidate_id", "other-candidate"),
    ("export_run_id", "other-export"), ("sport", "NFL"),
    ("market", "total_over"), ("selection", "Home -2.5"), ("line", -2.5),
    ("period", "first_half"), ("rules", "other-rules"), ("model_target", "home_win"),
    ("sportsbook", "other-book"), ("odds", -125.0), ("quote_id", "other-quote"),
    ("quote_time", "2026-10-01T19:20:00Z"), ("analysis_time", "2026-10-01T19:21:00Z"),
    ("start", "2026-10-01T23:00:00Z"), (None, None),
])
def test_legacy_missing_metadata_cannot_borrow_a_copied_display_identity(
        monkeypatch, tmp_path, field, value):
    from test_research_probability_display import source, package_for
    from app_core.public_board import build_package, validate_package
    from app_core.public_history import original_estimate
    raw = source(ml_target=""); raw.pop("wager_contract")
    frames, _ = package_for(monkeypatch, raw)
    for frame in frames:
        for index in frame.index:
            saved = json.loads(frame.at[index, "research_display"])
            if field is not None:
                saved["identity"][field] = value
            frame.at[index, "research_display"] = json.dumps(saved)
    package = build_package(*frames); validate_package(package)
    row = package["games"]["overall"][0]
    if field is None:
        assert row["win_estimate"] == pytest.approx(.6)
        assert row["ev"] == pytest.approx(.6*(1+100/110)-1)
    else:
        assert row["research_display"]["availability_reason"] == "ESTIMATE_IDENTITY_MISMATCH"
        assert row["win_estimate"] is None and row["ev"] is None
        assert original_estimate(row) == {}
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("conflicting_event", [False, True])
def test_legacy_unrecorded_start_enrichment_preserves_only_the_same_record(
        monkeypatch, tmp_path, conflicting_event):
    from test_research_probability_display import source, package_for, START
    from app_core.public_board import build_package, validate_package
    from app_core.public_history import original_estimate
    raw = source(ml_target="", game_time_est="", game_start_utc="", start="")
    raw.pop("wager_contract")
    frames, _ = package_for(monkeypatch, raw)
    for frame in frames:
        if not frame.empty:
            frame["start"] = START
            if conflicting_event:
                for index in frame.index:
                    saved = json.loads(frame.at[index, "research_display"])
                    saved["identity"]["event_id"] = "different-event"
                    frame.at[index, "research_display"] = json.dumps(saved)
    package = build_package(*frames); validate_package(package)
    row = package["games"]["overall"][0]
    assert row["research_display"]["availability_reason"] == "ESTIMATE_IDENTITY_MISMATCH"
    assert row["research_display"]["probability"] is None
    if conflicting_event:
        assert row["win_estimate"] is None and row["ev"] is None
        assert original_estimate(row) == {}
    else:
        assert row["win_estimate"] == pytest.approx(.6)
        assert row["ev"] == pytest.approx(.6*(1+100/110)-1)
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("probability,ev,valid", [
    (.6, .8, False), (.6, .6*(1+100/110)-1, True),
    (.4, .4*(1+100/110)-1, True), (.4, .8, False),
    (110/210, 0.0, True), (.6, None, True),
])
def test_actual_source_ev_basis_checked_before_projection_without_reinjection(
        monkeypatch, tmp_path, probability, ev, valid):
    from test_research_probability_display import source, package_for
    from app_core.public_history import original_estimate
    raw = source(best_available_selection_policy="", ml_target="",
        production_win_probability=probability, production_expected_value=ev,
        probability_semantics="win_unconditional_with_push", push_probability=0.0)
    raw.pop("wager_contract")
    assert "research_source_semantics" not in raw
    original = deepcopy(raw)
    frames, package = package_for(monkeypatch, raw)
    # The actual export drops these source facts. Do not inject a carrier, raw
    # probability, or source declarations after this boundary to make it pass.
    assert "production_win_probability" not in frames[0].columns
    assert "research_source_semantics" not in frames[0].columns or frames[0].iloc[0].research_source_semantics is None
    row = package["games"]["overall"][0]
    assert row["research_display"]["availability_reason"] == "MODEL_TARGET_NOT_RECORDED"
    if valid:
        assert row["win_estimate"] == pytest.approx(probability)
        assert row["ev"] is None if ev is None else row["ev"] == pytest.approx(ev)
    else:
        assert row["research_display"]["value_reason"] == "PRICE_VALUE_MISMATCH"
        assert row["win_estimate"] is None and row["ev"] is None
        assert original_estimate(row) == {}
    assert raw == original
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


@pytest.mark.parametrize("probability,ev,declared,valid,reason", [
    (.6, True, False, False, "INVALID_RECORDED_EV"),
    (.6, False, False, False, "INVALID_RECORDED_EV"),
    (.6, float("nan"), False, False, "INVALID_RECORDED_EV"),
    (.6, float("inf"), False, False, "INVALID_RECORDED_EV"),
    (.6, -float("inf"), False, False, "INVALID_RECORDED_EV"),
    (.6, "invalid", False, False, "INVALID_RECORDED_EV"),
    (.6, .8, True, False, "PRICE_VALUE_MISMATCH"),
    (.6, .6*(1+100/110)-1, True, True, None),
    (.4, .4*(1+100/110)-1, True, True, None),
    (110/210, 0.0, True, True, None), (.6, None, False, True, None),
    (.6, .8, False, True, None),
])
def test_ordinary_export_ev_types_and_declared_basis_do_not_need_a_carrier(
        monkeypatch, tmp_path, probability, ev, declared, valid, reason):
    from test_research_probability_display import source, package_for
    from app_core.public_board import build_package, validate_package
    from app_core.public_history import original_estimate
    raw = source(ml_target="", best_available_probability=probability); raw.pop("wager_contract")
    frames, _ = package_for(monkeypatch, raw)
    for frame in frames:
        frame.drop(columns=[field for field in ["research_source_semantics", "probability_semantics", "push_probability"]
                            if field in frame.columns], inplace=True)
        if not frame.empty:
            frame["ev"] = ev
            if declared:
                frame["probability_semantics"] = "win_unconditional_with_push"
                frame["push_probability"] = 0.0
    package = build_package(*frames); validate_package(package)
    row = package["games"]["overall"][0]
    if valid:
        assert row["win_estimate"] == pytest.approx(probability)
        assert row["ev"] is None if ev is None else row["ev"] == pytest.approx(ev)
    else:
        assert row["research_display"]["value_reason"] == reason
        assert row["win_estimate"] is None and row["ev"] is None
        assert original_estimate(row) == {}
    assert row["status"] == "PASS" and frames[0].iloc[0].Play_Stake == 0
    browser = inspect_browser(package, tmp_path / "browser", NOW)
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0
