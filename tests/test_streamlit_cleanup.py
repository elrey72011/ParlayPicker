"""The Streamlit clean-up of 18 November 2026 (owner): what the app no longer shows, and what it still does. SYNTHETIC."""
import importlib.util
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
# Panels nothing used any more: the Strategy Lab, Odds and Analysis tabs, the Results tab and the legacy sidebar tools.
REMOVED = ("strategy_lab_dashboard", "odds_dashboard", "analysis_dashboard", "mlb_prospective", "ncaaf_prospective",
           "ncaaf_history", "nfl_market", "ncaaf_research", "ncaaf_data_access", "results_dashboard", "threshold_dashboard",
           "pick_accuracy", "scoreboard_panel")


def test_the_removed_panels_are_gone_and_no_production_code_names_them():
    for name in REMOVED:
        assert importlib.util.find_spec(f"app.ui.{name}") is None, name
    named = re.compile(r"app\.ui\.(?:%s)\b|from app\.ui import [^\n]*\b(?:%s)\b" % ("|".join(REMOVED), "|".join(REMOVED)))
    production = [ROOT / "streamlit_app.py", *sorted((ROOT / "app").rglob("*.py")), *sorted((ROOT / "app_core").glob("*.py")),
                  *sorted((ROOT / "core").glob("*.py"))]
    assert len(production) > 100
    for path in production:
        assert not named.search(path.read_text(encoding="utf-8")), path.name


def test_the_page_names_its_sections_once_and_carries_none_of_the_removed_ones():
    import streamlit_app as app
    assert app.MAIN_TABS == ["Today", "Pick Details", "Workspace"]
    assert app.WORKSPACE_TABS == ["Full Pick Board", "Parlays", "Portfolio", "Diagnostics", "Preview & Publish"]
    page = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert page.count("st.tabs(MAIN_TABS") == 1 and page.count("st.tabs(WORKSPACE_TABS") == 1
    for gone in ('"Enable Sweet Spot Filter"', 'Performance Tracker"', "Ranked Games — Play Card", "Export All-Games Play Card",
                 "Export Sweet Spot Card", "Export Analysis", "_prepare_sweet_spot_card", "render_strategy_lab", "render_odds_table",
                 "render_analysis(", "run_player_props", "run_prop_analysis", "strikeout_prop_card", "prop_parlays_df",
                 "props_diagnostics", "tab_performance", "run_performance_pipeline", "render_results_dashboard", "render_scoreboard",
                 "Player Props", "perf_tracker_uploads"):
        assert gone not in page, gone
    # What the owner uses stays where it was.
    for kept in ("Export Candidate Selection Audit", "Export Best Picks (Compact)", "Export All Best-Available Picks", "Research Card",
                 "render_price_gap_candidates()", "render_daily_dashboard(", "render_readiness_dashboard(", "render_publish_panel(",
                 "render_draftkings_builders(", "render_memory_readout()", "Best Parlays", "Portfolio Allocation", "Pipeline Debug",
                 "Kalshi Match Failures"):
        assert kept in page, kept


def _navigation(remembered):
    import streamlit as st
    from streamlit_app import _render_main_tabs
    if remembered:
        st.session_state["main_navigation"] = remembered
    st.session_state["sections"] = len(_render_main_tabs())


@pytest.mark.parametrize("remembered, opens_on", [("", "Today"), ("Pick Details", "Pick Details"), ("Workspace", "Workspace"),
                                                 ("Results", "Today"), ("Performance Recap", "Today"), ("Best Picks", "Workspace")])
def test_a_session_that_remembers_a_removed_section_opens_on_one_that_exists(remembered, opens_on):
    at = AppTest.from_function(_navigation, kwargs={"remembered": remembered}).run()
    assert not at.exception
    assert at.session_state["sections"] == 3 and [tab.label for tab in at.tabs] == ["Today", "Pick Details", "Workspace"]
    assert at.session_state["main_navigation"] == opens_on


def _sidebar():
    import streamlit as st
    from app.ui.sidebar_controls import render_sidebar
    st.session_state["controls"] = sorted(render_sidebar())


def test_the_sidebar_keeps_refresh_its_settings_and_the_two_theover_uploads():
    at = AppTest.from_function(_sidebar).run()
    assert not at.exception
    assert [button.label for button in at.button] == ["Refresh picks"]
    assert [box.label for box in at.checkbox] == ["Enable ML Predictions", "Require Gemini Review for Bets",
                                                  "Display Debug Information", "Show Kalshi Diagnostics"]
    assert [upload.proto.label for upload in at.get("file_uploader")] == ["Upload TheOver Spreads (CSV or workbook)",
                                                                          "Upload TheOver Totals (CSV or workbook)"]
    assert [expander.label for expander in at.expander] == ["Settings & research"]
    assert at.session_state["controls"] == ["bankroll", "coverage_date", "run_analysis_counter", "schedule_end", "schedule_start",
                                            "show_debug", "show_kalshi_diagnostics", "sports", "theover_spreads", "theover_totals",
                                            "use_gemini", "use_ml"]
    text = " ".join(str(getattr(element, "value", "")) for element in list(at.caption) + list(at.subheader))
    assert "prop" not in text.lower() and "Data Maintenance" not in text
    sidebar = (ROOT / "app" / "ui" / "sidebar_controls.py").read_text(encoding="utf-8")
    for gone in ("Run Player Props", "prop_results_log", "Grade Uploaded Player Props", "legacy prospective", "Sync Historical Rosters",
                 "NCAAF Data Access", "render_ncaaf_history"):
        assert gone not in sidebar, gone
