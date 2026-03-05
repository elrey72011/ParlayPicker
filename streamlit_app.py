from __future__ import annotations

from typing import Any

import streamlit as st

from app.ui.analysis_dashboard import render_analysis
from app.ui.debug_panel import render_debug_panel
from app.ui.kalshi_diagnostics import render_kalshi_diagnostics
from app.ui.layout import setup_page
from app.ui.odds_dashboard import render_odds_table
from app.ui.parlay_dashboard import render_parlays
from app.ui.portfolio_dashboard import render_portfolio
from app.ui.sidebar_controls import render_sidebar
from core.streamlit_pipeline import (
    generate_parlays_table,
    optimize_portfolio_allocation,
    run_analysis_pipeline,
)
from core.theover_loader import load_theover_csv


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

    if not controls["run_analysis"]:
        st.info("Configure filters in the sidebar and click **Run Master Analysis**.")
        return

    spreads_df = load_theover_csv(controls["theover_spreads"])
    totals_df = load_theover_csv(controls["theover_totals"])

    analysis_df = run_analysis_pipeline(
        sports=controls["sports"],
        max_rows=int(controls["max_rows"]),
        use_ml=bool(controls["use_ml"]),
        spreads_df=spreads_df,
        totals_df=totals_df,
    )

    if analysis_df.empty:
        st.warning("No rows found for the selected sports.")
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

    parlays_df = generate_parlays_table(analysis_df)
    portfolio_df = optimize_portfolio_allocation(analysis_df)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Odds", "Analysis", "Parlays", "Portfolio", "Debug"])

    with tab1:
        render_odds_table(analysis_df)

    with tab2:
        render_analysis(analysis_df)

    with tab3:
        render_parlays(parlays_df)

    with tab4:
        render_portfolio(portfolio_df)

    with tab5:
        if controls["show_debug"]:
            render_debug_panel(analysis_df)
        else:
            st.info("Enable 'Display Debug Information' in the sidebar to inspect debug data.")

        if controls["show_kalshi_diagnostics"]:
            render_kalshi_diagnostics(analysis_df)


if __name__ == "__main__":
    main()
