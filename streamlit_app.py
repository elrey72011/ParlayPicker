from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from app.ui.analysis_dashboard import render_analysis
from app.ui.debug_panel import render_debug, render_debug_panel
from app.ui.export_tools import render_exports
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

    spreads_df = load_theover_csv(controls.get("theover_spreads"))
    totals_df = load_theover_csv(controls.get("theover_totals"))

    st.write("TheOver spreads rows:", len(spreads_df))
    st.write("TheOver totals rows:", len(totals_df))

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

    odds_df = analysis_df.copy()
    theover_df = (
        load_theover_csv(controls["theover_spreads"])
        if controls["theover_spreads"]
        else None
    )
    if controls["theover_totals"]:
        totals_loaded = load_theover_csv(controls["theover_totals"])
        theover_df = pd.concat([theover_df, totals_loaded], ignore_index=True) if theover_df is not None else totals_loaded

    kalshi_cols = [c for c in analysis_df.columns if "kalshi" in c.lower()]
    kalshi_df = analysis_df[kalshi_cols].dropna(how="all") if kalshi_cols else analysis_df.iloc[0:0]

    gemini_df = (
        analysis_df[analysis_df.get("gemini_analysis", "").astype(str).str.len() > 0]
        if "gemini_analysis" in analysis_df.columns
        else analysis_df.iloc[0:0]
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Odds", "Analysis", "Parlays", "Portfolio", "Debug"])

    with tab1:
        render_odds_table(analysis_df)

    with tab2:
        render_analysis(analysis_df)
        render_exports(analysis_df, filename="analysis_export.csv")

    with tab3:
        st.subheader("Recommended Parlays")
        render_parlays(parlays_df)
        render_exports(parlays_df, filename="parlays_export.csv")

    with tab4:
        render_portfolio(portfolio_df)

    with tab5:
        st.subheader("Data Source Status")
        st.write("TheOdds rows:", len(odds_df))
        st.write("TheOver rows:", len(theover_df) if theover_df is not None else 0)
        st.write("Kalshi rows:", len(kalshi_df))
        st.write("Gemini rows:", len(gemini_df))

        if controls["show_debug"]:
            render_debug(analysis_df)
            render_debug_panel(analysis_df)
        else:
            st.info("Enable 'Display Debug Information' in the sidebar to inspect debug data.")

        if controls["show_kalshi_diagnostics"]:
            render_kalshi_diagnostics(analysis_df)


if __name__ == "__main__":
    main()
