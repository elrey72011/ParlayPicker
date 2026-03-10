import pandas as pd
import streamlit as st


def render_kalshi_diagnostics(df: pd.DataFrame) -> None:
    st.subheader("Kalshi Diagnostics")
    if df is None or df.empty or "kalshi_match_status" not in df.columns:
        st.info("No Kalshi diagnostic rows found in analysis output.")
        return

    status = df["kalshi_match_status"].astype(str).str.lower()
    matched = df[status.eq("matched")].copy()
    misses = df[status.ne("matched")].copy()

    st.write("Matched rows:", len(matched))
    st.write("Miss/error rows:", len(misses))

    match_cols = [
        c for c in [
            "league",
            "away_team",
            "home_team",
            "game_date",
            "kalshi_market_ticker",
            "kalshi_market_title",
            "kalshi_probability",
            "kalshi_line",
            "kalshi_match_status",
        ]
        if c in df.columns
    ]
    if not matched.empty and match_cols:
        st.dataframe(matched[match_cols].head(50), width="stretch")

    if "kalshi_match_reason" in df.columns:
        st.markdown("#### Miss reasons")
        reason_counts = (
            misses["kalshi_match_reason"].fillna("unknown").astype(str).value_counts().rename_axis("reason").reset_index(name="rows")
            if not misses.empty
            else pd.DataFrame(columns=["reason", "rows"])
        )
        st.dataframe(reason_counts, width="stretch")

    miss_cols = [
        c for c in [
            "league",
            "away_team",
            "home_team",
            "game_date",
            "kalshi_match_status",
            "kalshi_match_reason",


        ]
        if c in df.columns
    ]
    if not misses.empty and miss_cols:
        st.markdown("#### Miss details")
        st.dataframe(misses[miss_cols].head(100), width="stretch")
