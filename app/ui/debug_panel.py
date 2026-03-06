import pandas as pd
import streamlit as st


def show_pipeline_debug(
    df: pd.DataFrame,
    odds_matches: int | None = None,
    theover_matches: int | None = None,
    kalshi_matches: int | None = None,
) -> None:
    st.subheader("Pipeline Debug")
    st.write("Games loaded:", len(df))
    st.write("EV calculated:", df["expected_value"].notna().sum() if "expected_value" in df.columns else 0)
    st.write("Positive EV bets:", (df["expected_value"] > 0).sum() if "expected_value" in df.columns else 0)
    st.write("TheOdds matches:", odds_matches if odds_matches is not None else len(df))
    st.write("TheOver matches:", theover_matches if theover_matches is not None else 0)
    st.write("Kalshi matches:", kalshi_matches if kalshi_matches is not None else 0)
    st.write("Columns:", list(df.columns))
    st.dataframe(df.head(20), use_container_width=True)


def render_debug(
    df: pd.DataFrame,
    odds_matches: int | None = None,
    theover_matches: int | None = None,
    kalshi_matches: int | None = None,
) -> None:
    show_pipeline_debug(df, odds_matches, theover_matches, kalshi_matches)


def render_debug_panel(
    df: pd.DataFrame,
    odds_matches: int | None = None,
    theover_matches: int | None = None,
    kalshi_matches: int | None = None,
) -> None:
    with st.expander("Debug Info"):
        show_pipeline_debug(df, odds_matches, theover_matches, kalshi_matches)
