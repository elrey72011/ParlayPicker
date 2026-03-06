import pandas as pd
import streamlit as st


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
        st.dataframe(analysis_df.nlargest(15, "expected_value"), use_container_width=True)

    st.markdown("**Best parlays**")
    if parlays_df is not None and not parlays_df.empty:
        st.dataframe(parlays_df.head(15), use_container_width=True)
    else:
        st.write("No positive EV parlays generated.")
