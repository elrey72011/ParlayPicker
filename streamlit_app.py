from __future__ import annotations

from typing import Any

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import pandas as pd
import streamlit as st

from app.ui.analysis_dashboard import render_analysis
from app.ui.data_diagnostics import show_data_diagnostics
from app.ui.debug_panel import render_debug, render_debug_panel
from app.ui.kalshi_diagnostics import render_kalshi_diagnostics
from app.ui.layout import setup_page
from app.ui.odds_dashboard import render_odds_table
from app.ui.parlay_dashboard import render_parlays
from app.ui.portfolio_dashboard import render_portfolio
from app.ui.sidebar_controls import render_sidebar
from app.ui.strategy_lab_dashboard import render_strategy_lab
from core.streamlit_pipeline import (
    generate_parlays,
    optimize_portfolio_allocation,
    run_analysis_pipeline,
    run_bankroll_simulation,
)
from core.team_normalizer import normalize_team
from core.theover_loader import load_theover_csv


if hasattr(st, "session_state"):
    if "analysis_df" not in st.session_state:
        st.session_state.analysis_df = None

    if "parlays_df" not in st.session_state:
        st.session_state.parlays_df = None


def _legacy_module():
    """Lazy import to preserve backward compatibility for utility function imports."""
    import app.legacy_streamlit_app as legacy

    return legacy


def select_best_spread_pick(*args: Any, **kwargs: Any):
    return _legacy_module().select_best_spread_pick(*args, **kwargs)


def select_best_total_pick(*args: Any, **kwargs: Any):
    return _legacy_module().select_best_total_pick(*args, **kwargs)


def map_kalshi_prob_for_pick(*args: Any, **kwargs: Any):
    return _legacy_module().map_kalshi_prob_for_pick(*args, **kwargs)


def calculate_best_pick_metrics(*args: Any, **kwargs: Any):
    return _legacy_module().calculate_best_pick_metrics(*args, **kwargs)


def main() -> None:
    setup_page()
    controls = render_sidebar()

    if "analysis_df" not in st.session_state:
        st.session_state["analysis_df"] = None
    if "parlays_df" not in st.session_state:
        st.session_state["parlays_df"] = None
    if "portfolio_df" not in st.session_state:
        st.session_state["portfolio_df"] = None
    if "odds_df" not in st.session_state:
        st.session_state["odds_df"] = None
    if "theover_df" not in st.session_state:
        st.session_state["theover_df"] = None
    if "kalshi_df" not in st.session_state:
        st.session_state["kalshi_df"] = None
    if "gemini_df" not in st.session_state:
        st.session_state["gemini_df"] = None
    if "simulation_results" not in st.session_state:
        st.session_state["simulation_results"] = None

    if controls["run_analysis"]:
        spreads_df = load_theover_csv(controls.get("theover_spreads"))
        totals_df = load_theover_csv(controls.get("theover_totals"))

        for upload_df in (spreads_df, totals_df):
            for team_col in ["home_team", "away_team"]:
                if team_col in upload_df.columns:
                    upload_df[team_col] = upload_df[team_col].apply(normalize_team)

        st.write("TheOver spreads rows:", len(spreads_df))
        st.write("TheOver totals rows:", len(totals_df))

        analysis_df = run_analysis_pipeline(
            sports=controls["sports"],
            max_rows=int(controls["max_rows"]),
            use_ml=bool(controls["use_ml"]),
            spreads_df=spreads_df,
            totals_df=totals_df,
        )

        st.session_state.analysis_df = analysis_df

        if analysis_df.empty:
            st.warning("No rows found for the selected sports.")
            st.session_state["analysis_df"] = None
            st.session_state["parlays_df"] = None
            st.session_state["portfolio_df"] = None
            st.session_state["odds_df"] = None
            st.session_state["theover_df"] = None
            st.session_state["kalshi_df"] = None
            st.session_state["gemini_df"] = None
            st.session_state["simulation_results"] = None
            return

        if controls["use_vertex"]:
            from core.vertex_master_analyzer import VertexMasterAnalyzer

            analyzer = VertexMasterAnalyzer()
            analysis_df = analyzer.run_master_analysis(analysis_df)

        if controls["use_gemini"]:
            from integrations.gemini_client import run_gemini_analysis

            analysis_df = run_gemini_analysis(analysis_df)

        if "gemini_analysis" not in analysis_df.columns:
            analysis_df["gemini_analysis"] = ""

        parlays_df = generate_parlays(analysis_df)
        st.session_state.parlays_df = parlays_df
        portfolio_df = optimize_portfolio_allocation(analysis_df, bankroll=float(controls["bankroll"]))
        simulation_results = run_bankroll_simulation(portfolio_df, bankroll=float(controls["bankroll"]))

        odds_df = analysis_df.copy()
        theover_df = (
            load_theover_csv(controls["theover_spreads"])
            if controls["theover_spreads"]
            else None
        )
        if controls["theover_totals"]:
            totals_loaded = load_theover_csv(controls["theover_totals"])
            theover_df = (
                pd.concat([theover_df, totals_loaded], ignore_index=True)
                if theover_df is not None
                else totals_loaded
            )

        kalshi_cols = [c for c in analysis_df.columns if "kalshi" in c.lower()]
        kalshi_df = analysis_df[kalshi_cols].dropna(how="all") if kalshi_cols else analysis_df.iloc[0:0]

        gemini_df = (
            analysis_df[analysis_df.get("gemini_analysis", "").astype(str).str.len() > 0]
            if "gemini_analysis" in analysis_df.columns
            else analysis_df.iloc[0:0]
        )

        st.session_state["analysis_df"] = analysis_df
        st.session_state["parlays_df"] = parlays_df
        st.session_state["portfolio_df"] = portfolio_df
        st.session_state["odds_df"] = odds_df
        st.session_state["theover_df"] = theover_df
        st.session_state["kalshi_df"] = kalshi_df
        st.session_state["gemini_df"] = gemini_df
        st.session_state["simulation_results"] = simulation_results
    analysis_df = st.session_state["analysis_df"]
    parlays_df = st.session_state["parlays_df"]
    portfolio_df = st.session_state["portfolio_df"]
    odds_df = st.session_state["odds_df"]
    theover_df = st.session_state["theover_df"]
    kalshi_df = st.session_state["kalshi_df"]
    gemini_df = st.session_state["gemini_df"]
    simulation_results = st.session_state["simulation_results"]

    if analysis_df is None:
        st.info("Configure filters in the sidebar and click **Run Master Analysis**.")
        return

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Odds", "Analysis", "Parlays", "Portfolio", "Debug", "Strategy Lab"])

    with tab1:
        render_odds_table(analysis_df)

    with tab2:
        if st.session_state.analysis_df is not None:
            render_analysis(st.session_state.analysis_df)
        if st.session_state["analysis_df"] is not None:
            analysis_export_df = st.session_state["analysis_df"].copy()
            if "best_pick" in analysis_export_df.columns:
                export_priority = [
                    "league",
                    "away_team",
                    "home_team",
                    "best_pick",
                    "market_type",
                    "expected_value",
                    "edge",
                    "market_probability",
                    "calibrated_probability",
                    "decimal_odds",
                ]
                ordered_cols = [c for c in export_priority if c in analysis_export_df.columns]
                trailing_cols = [c for c in analysis_export_df.columns if c not in ordered_cols]
                analysis_export_df = analysis_export_df[ordered_cols + trailing_cols]
            analysis_csv = analysis_export_df.to_csv(index=False)
            st.download_button(
                "Export Analysis",
                analysis_csv,
                "analysis_export.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("Top EV picks")
        render_parlays(parlays_df)
        if st.session_state["parlays_df"] is not None:
            parlays_csv = st.session_state["parlays_df"].to_csv(index=False)
            st.download_button(
                "Export Parlays",
                parlays_csv,
                "parlays_export.csv",
                mime="text/csv",
            )

    with tab4:
        render_portfolio(portfolio_df)

    with tab5:
        show_data_diagnostics(
            odds_df=odds_df,
            theover_df=theover_df if theover_df is not None else analysis_df.iloc[0:0],
            kalshi_df=kalshi_df,
            gemini_df=gemini_df,
        )

        odds_matches = len(odds_df) if odds_df is not None else 0
        theover_matches = len(theover_df) if theover_df is not None else 0
        kalshi_matches = len(kalshi_df) if kalshi_df is not None else 0

        if controls["show_debug"]:
            parlay_count = len(parlays_df) if parlays_df is not None else 0
            render_debug(analysis_df, odds_matches, theover_matches, kalshi_matches, parlay_count)
            render_debug_panel(analysis_df, odds_matches, theover_matches, kalshi_matches, parlay_count)
        else:
            st.info("Enable 'Display Debug Information' in the sidebar to inspect debug data.")

        if controls["show_kalshi_diagnostics"]:
            render_kalshi_diagnostics(analysis_df)

    with tab6:
        render_strategy_lab(
            analysis_df=analysis_df,
            portfolio_df=portfolio_df if portfolio_df is not None else analysis_df.iloc[0:0],
            parlays_df=parlays_df if parlays_df is not None else analysis_df.iloc[0:0],
            simulation_results=simulation_results or {},
        )


if __name__ == "__main__":
    main()
