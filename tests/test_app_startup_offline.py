"""The whole application starting offline after the clean-up of 18 November 2026; synthetic settings, no provider or inference.

Takes over from the start-up tests that were written around the sidebar's legacy tools, which the owner removed: the app
still starts without touching a database or importing a prediction, the private controls still answer only to the
owner's token, and the removed controls are not there to be switched on.
"""
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest
from streamlit.testing.v1 import AppTest
from scripts.benchmark_drive_history_loading import blocked_network

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "SYNTHETIC-owner-for-startup-regressions-only"
REMOVED_LABELS = ("Show legacy prospective and market tools", "Run Player Props", "🔄 Sync Historical Rosters",
                  "Grade Uploaded Player Props", "Check NCAAF Data Access", "Collect next NCAAF batch",
                  "Refresh / Backfill Final Scores", "Enable Sweet Spot Filter")


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(tmp_path / "SYNTHETIC-evidence"))
    monkeypatch.setenv("PARLAYPICKER_PUBLISH_TOKEN", TOKEN)
    monkeypatch.setenv("PARLAYPICKER_NFL_ACQUISITION_ENABLED", "false")
    monkeypatch.setenv("PARLAYPICKER_NFL_INFERENCE_ENABLED", "false")
    from app_core import market_probability_model, ncaaf_research, nfl_scenario_research
    from core import streamlit_pipeline
    for module, name in [(market_probability_model, "predict_market_probabilities"), (ncaaf_research, "centers"),
                         (nfl_scenario_research, "compute"), (streamlit_pipeline, "run_analysis_pipeline")]:
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError("SYNTHETIC_PREDICTION_FORBIDDEN")))
    with blocked_network():
        yield


def _app(tmp_path, name="SYNTHETIC-main"):
    script = tmp_path / (name + ".py")
    script.write_text("import streamlit_app\nstreamlit_app.main()\n", encoding="utf-8")
    return AppTest.from_file(str(script), default_timeout=30).run()


def _no_database(*args, **kwargs):
    raise AssertionError("PASSIVE_DATABASE_ACCESS_FORBIDDEN")


def _labels(at):
    return {x.label for kind in (at.checkbox, at.button, at.selectbox, at.expander) for x in kind}


def test_the_app_starts_with_three_sections_and_touches_no_database(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite3, "connect", _no_database)
    at = _app(tmp_path)
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Today", "Pick Details", "Workspace", "Full Pick Board", "Parlays", "Portfolio",
                                             "Diagnostics", "Preview & Publish"]
    labels = _labels(at)
    assert {"Refresh picks", "Select Sport", "Settings & research", "Use reviewed private NCAAF inputs for Refresh"} <= labels
    assert not labels & set(REMOVED_LABELS)
    assert [button.label for button in at.sidebar.button] == ["Refresh picks"]
    # A second run of the same session is as quiet as the first.
    at.run()
    assert not at.exception and not _labels(at) & set(REMOVED_LABELS)
    from app_core import market_probability_model, ncaaf_research, nfl_scenario_research
    assert not market_probability_model.predict_market_probabilities.called
    assert not ncaaf_research.centers.called and not nfl_scenario_research.compute.called


def test_the_private_controls_answer_only_to_the_owners_token_and_start_no_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(sqlite3, "connect", _no_database)
    at = _app(tmp_path)
    at.session_state["SYNTHETIC_unrelated"] = "keep"
    at.text_input(key="publication_token").input("wrong").run()
    assert not at.exception and "Show saved NCAAF Refresh results" not in _labels(at)
    at.text_input(key="publication_token").input(TOKEN).run()
    assert not at.exception
    assert {"Show saved NCAAF Refresh results", "Enable guided private NFL workflow"} <= _labels(at)
    assert at.session_state["SYNTHETIC_unrelated"] == "keep"
    at.text_input(key="publication_token").input("wrong").run()
    assert not at.exception and "Show saved NCAAF Refresh results" not in _labels(at)
    assert at.session_state["SYNTHETIC_unrelated"] == "keep" and not _labels(at) & set(REMOVED_LABELS)


@pytest.mark.parametrize("sport", ["NCAAF", "NFL", "NCAAB"])
def test_a_sport_switch_a_rerun_and_a_fresh_session_stay_quiet(monkeypatch, tmp_path, sport):
    monkeypatch.setattr(sqlite3, "connect", _no_database)
    at = _app(tmp_path)
    next(x for x in at.selectbox if x.label == "Select Sport").set_value(sport).run()
    assert not at.exception and not _labels(at) & set(REMOVED_LABELS)
    at.run()
    assert not at.exception
    fresh = _app(tmp_path, "SYNTHETIC-fresh-" + sport)
    assert not fresh.exception and not _labels(fresh) & set(REMOVED_LABELS)
    assert not fresh.session_state.filtered_state.get("ncaaf_private_refresh", False)
    assert not fresh.session_state.filtered_state.get("nfl_workflow_enabled", False)
