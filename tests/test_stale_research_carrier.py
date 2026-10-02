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
