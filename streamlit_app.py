from __future__ import annotations

from typing import Any

import streamlit as st

from app.ui.analysis_dashboard import render_analysis
from app.ui.layout import setup_page
from app.ui.odds_dashboard import render_odds_table
from app.ui.parlay_dashboard import render_parlays
from app.ui.portfolio_dashboard import render_portfolio
from app.ui.realtime_dashboard import render_realtime_edges
from app.ui.sidebar_controls import render_sidebar
from core.streamlit_pipeline import (
    build_realtime_edges,
    generate_parlays_table,
    optimize_portfolio_allocation,
    run_analysis_pipeline,
)


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

    sports = controls["sports"]
    analysis_df = run_analysis_pipeline(sports=sports, max_rows=int(controls["max_rows"]))

    if analysis_df.empty:
        st.warning("No rows found for the selected sports.")
        return

    render_odds_table(analysis_df)
    render_analysis(analysis_df)

    parlays_df = generate_parlays_table(analysis_df)
    render_parlays(parlays_df)

    if controls["include_realtime"]:
        realtime_edges = build_realtime_edges(analysis_df)
        render_realtime_edges(realtime_edges)

    if controls["include_portfolio"]:
        portfolio_df = optimize_portfolio_allocation(analysis_df)
        render_portfolio(portfolio_df)


if __name__ == "__main__":
    main()
