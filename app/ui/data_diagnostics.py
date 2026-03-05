import streamlit as st


def show_data_diagnostics(
    odds_df,
    theover_df,
    kalshi_df,
    gemini_df,
):
    st.subheader("Data Source Diagnostics")

    data = {
        "Source": [
            "TheOdds",
            "TheOver",
            "Kalshi",
            "Gemini",
        ],
        "Rows": [
            len(odds_df),
            len(theover_df),
            len(kalshi_df),
            len(gemini_df),
        ],
    }

    st.dataframe(data)
