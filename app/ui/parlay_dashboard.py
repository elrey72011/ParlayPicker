import pandas as pd
import streamlit as st


def render_parlays(parlays: pd.DataFrame) -> None:
    if parlays is None or parlays.empty:
        st.info("No +EV parlays available.")
        return

    st.subheader("Smart Parlays Portfolio")

    # Configure columns for Enhanced Parlay Visualizer
    column_config = {
        "parlay_legs": st.column_config.TextColumn(
            "Parlay Legs",
            help="The individual picks in this parlay",
            width="large",
        ),
        "combined_probability": st.column_config.NumberColumn(
            "Model Win %",
            help="True probability of the parlay hitting",
            format="%.1f%%",
        ),
        "combined_decimal_odds": st.column_config.NumberColumn(
            "Best Odds (Decimal)",
            help="The highest payout odds found across bookmakers",
            format="%.2f",
        ),
        "parlay_ev": st.column_config.NumberColumn(
            "Parlay EV",
            help="Expected Value of the entire parlay",
            format="%.3f",
        ),
        "ev_boost_pct": st.column_config.NumberColumn(
            "EV Boost / Edge",
            help="Additional edge over market price",
            format="%.1f%%",
        ),
        "is_high_correlation": st.column_config.CheckboxColumn(
            "High Correlation",
            help="Indicates if legs share a game, increasing true probability",
        ),
        "Conviction_Score": st.column_config.ProgressColumn(
            "Conviction",
            help="High-reliability score based on historical calibration",
            format="%.2f",
            min_value=0.0,
            max_value=1.0,
        ),
        "best_payout_book": st.column_config.TextColumn(
            "Best Bookmaker",
            help="The bookmaker offering the best payout for this exact parlay",
        ),
        "risk_tier": st.column_config.TextColumn(
            "Risk Tier",
            help="Strategic categorization of this parlay",
        ),
        "recommended_bet": st.column_config.NumberColumn(
            "Rec. Bet ($)",
            help="Recommended bet size based on Simultaneous Kelly Optimization",
            format="$%.2f",
        ),
        "unified_bankroll_recommendation": st.column_config.TextColumn(
            "RR Recommendation",
            help="Total package allocation for Round Robin strategies",
            width="large",
        ),
    }

    # Formatting conversions for display purposes if columns exist
    display_df = parlays.copy()
    if "combined_probability" in display_df.columns:
        display_df["combined_probability"] = display_df["combined_probability"] * 100
    if "ev_boost_pct" in display_df.columns:
        display_df["ev_boost_pct"] = display_df["ev_boost_pct"] * 100

    st.dataframe(
        display_df,
        column_config=column_config,
        width="stretch",
        hide_index=True,
    )
