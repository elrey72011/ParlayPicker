import pandas as pd
import streamlit as st

from app_core.performance_pipeline import run_performance_pipeline
from app_core.strategy_lab_realized import (
    REALIZED_MODE_ACTIONABLE_ONLY,
    REALIZED_MODE_ACTIONABLE_PLUS_HIGH_VARIANCE,
    REALIZED_MODE_ALL_GRADED,
    REALIZED_MODE_CUSTOM,
    REALIZED_STRATEGY_MODE_ORDER,
    STATUS_BUCKET_ORDER,
    build_realized_strategy_lab,
)


def _render_theoretical_strategy_lab(
    analysis_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    parlays_df: pd.DataFrame,
    simulation_results: dict,
) -> None:
    left, right = st.columns(2)

    with left:
        st.markdown("**Edge distribution**")
        if "edge" in analysis_df.columns:
            st.bar_chart(analysis_df["edge"].dropna())
        else:
            st.write("No edge data available.")

    with right:
        st.markdown("**Kelly bet sizes**")
        if portfolio_df is not None and not portfolio_df.empty and "recommended_bet" in portfolio_df.columns:
            st.bar_chart(portfolio_df.set_index(portfolio_df.index.astype(str))["recommended_bet"])
        else:
            st.write("No Kelly sizing available.")

    st.markdown("**Monte Carlo bankroll curves**")
    curves = simulation_results.get("bankroll_curves", []) if simulation_results else []
    if curves:
        sample_curves = curves[:50]
        curves_df = pd.DataFrame(sample_curves).T
        st.line_chart(curves_df)
        st.write(
            {
                "expected_bankroll": simulation_results.get("expected_bankroll", 0.0),
                "risk_of_ruin": simulation_results.get("risk_of_ruin", 0.0),
                "max_drawdown": simulation_results.get("max_drawdown", 0.0),
            }
        )
    else:
        st.write("No simulation output available.")

    st.markdown("**Top +EV bets**")
    if "expected_value" in analysis_df.columns:
        st.dataframe(analysis_df.nlargest(15, "expected_value"), width="stretch")

    st.markdown("**Best parlays**")
    if parlays_df is not None and not parlays_df.empty:
        st.dataframe(parlays_df.head(15), width="stretch")
    else:
        st.write("No positive EV parlays generated.")


def _render_realized_strategy_lab(analysis_df: pd.DataFrame) -> None:
    st.caption("Realized results are sourced from the same graded performance pipeline used by Prior Day Performance recap.")

    graded_df = run_performance_pipeline()
    if graded_df is None or graded_df.empty:
        st.info("No graded recap source found from the performance pipeline.")
        return

    selected_mode = st.selectbox(
        "Realized strategy mode",
        REALIZED_STRATEGY_MODE_ORDER,
        index=0,
        help="Default is Actionable only (core strategy). Broader modes include exploratory statuses.",
    )
    custom_statuses = None
    if selected_mode == REALIZED_MODE_CUSTOM:
        custom_statuses = st.multiselect(
            "Custom statuses",
            STATUS_BUCKET_ORDER,
            default=[STATUS_BUCKET_ORDER[0]],
        )

    realized_df, summary, diagnostics = build_realized_strategy_lab(
        graded_df=graded_df,
        strategy_source_df=analysis_df,
        strategy_mode=selected_mode,
        custom_statuses=custom_statuses,
    )

    st.markdown("### Realized graded performance")
    st.caption(f"Selected mode: **{summary['Strategy Mode']}**")
    if selected_mode in {REALIZED_MODE_ALL_GRADED, REALIZED_MODE_CUSTOM}:
        st.warning(
            "You are viewing a broad exploratory strategy. Results include non-core bets and may not reflect the intended production card."
        )
    if selected_mode == REALIZED_MODE_CUSTOM and custom_statuses:
        if any(status in {"No Play", "Below Threshold"} for status in custom_statuses):
            st.warning(
                "You are viewing a broad exploratory strategy. Results include non-core bets and may not reflect the intended production card."
            )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Staked", f"{summary['Total Staked']:.2f}")
    col2.metric("Gross Returned", f"{summary['Gross Returned']:.2f}")
    col3.metric("Net P/L", f"{summary['Net P/L']:+.2f}")
    col4.metric("ROI", f"{summary['ROI']:.2%}")

    col5, col6, col7, col8, col9 = st.columns(5)
    col5.metric("Wins", int(summary["Win Count"]))
    col6.metric("Losses", int(summary["Loss Count"]))
    col7.metric("Pushes", int(summary["Push Count"]))
    col8.metric("Hit Rate", f"{summary['Hit Rate']:.2%}")
    col9.metric("Mismatches", int(summary["Mismatch Count"]))

    st.markdown("**Mode comparison (realized)**")
    st.dataframe(diagnostics["mode_comparison"], width="stretch")

    st.markdown("**Status bucket summary (realized)**")
    st.dataframe(diagnostics["status_bucket_summary"], width="stretch")

    st.markdown("**Realized bankroll rows**")
    cols = [
        "league",
        "home_team",
        "away_team",
        "Strategy Pick",
        "Recap Pick",
        "Status Bucket",
        "Result",
        "Stake",
        "American Odds",
        "Decimal Odds",
        "Gross Return",
        "Net Profit",
        "Running Bankroll",
        "Excluded Reason",
    ]
    show_cols = [c for c in cols if c in realized_df.columns]
    st.dataframe(realized_df[show_cols], width="stretch")

    with st.expander("Validation diagnostics"):
        st.write(
            {
                "ungraded_rows": diagnostics["ungraded_count"],
                "missing_odds_rows": diagnostics["missing_odds_count"],
                "mismatch_rows": diagnostics["mismatch_count"],
                "outside_selected_mode_rows": diagnostics["outside_mode_count"],
                "excluded_rows": len(diagnostics["excluded_rows"]),
            }
        )

        if not diagnostics["pick_mismatches"].empty:
            st.markdown("**Pick mismatches (excluded)**")
            st.dataframe(diagnostics["pick_mismatches"], width="stretch")
        if not diagnostics["missing_scores"].empty:
            st.markdown("**Rows missing actual scores (excluded)**")
            st.dataframe(diagnostics["missing_scores"], width="stretch")
        if not diagnostics["missing_odds"].empty:
            st.markdown("**Rows missing odds needed for payout (excluded)**")
            st.dataframe(diagnostics["missing_odds"], width="stretch")
        if not diagnostics["outside_mode_rows"].empty:
            st.markdown("**Rows excluded by strategy mode filter**")
            st.dataframe(diagnostics["outside_mode_rows"], width="stretch")


def render_strategy_lab(
    analysis_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    parlays_df: pd.DataFrame,
    simulation_results: dict,
) -> None:
    st.subheader("Strategy Lab")

    if analysis_df is None or analysis_df.empty:
        st.info("Run analysis to populate Strategy Lab insights.")
        return

    theoretical_tab, realized_tab = st.tabs(["Theoretical", "Realized"])
    with theoretical_tab:
        _render_theoretical_strategy_lab(analysis_df, portfolio_df, parlays_df, simulation_results)
    with realized_tab:
        _render_realized_strategy_lab(analysis_df)
