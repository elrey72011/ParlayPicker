import streamlit as st


def _safe_len(df):
    if df is not None and not df.empty:
        return len(df)
    return 0


def show_data_diagnostics(
    odds_df,
    theover_df,
    kalshi_df,
    gemini_df,
):
    st.subheader("Data Source Diagnostics")

    st.write("TheOdds rows:", _safe_len(odds_df))
    st.write("TheOver rows:", _safe_len(theover_df))
    st.write("Kalshi rows:", _safe_len(kalshi_df))
    st.write("Gemini rows:", _safe_len(gemini_df))

    data = {
        "Source": [
            "TheOdds",
            "TheOver",
            "Kalshi",
            "Gemini",
        ],
        "Rows": [
            _safe_len(odds_df),
            _safe_len(theover_df),
            _safe_len(kalshi_df),
            _safe_len(gemini_df),
        ],
    }

    st.dataframe(data)
