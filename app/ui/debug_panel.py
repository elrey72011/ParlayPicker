import pandas as pd
import streamlit as st


def show_pipeline_debug(
    df: pd.DataFrame,
    odds_matches: int | None = None,
    theover_matches: int | None = None,
    kalshi_matches: int | None = None,
    parlay_count: int | None = None,
) -> None:
    st.subheader("Pipeline Debug")
    st.write("Total games:", len(df))
    st.write("Odds rows:", odds_matches if odds_matches is not None else len(df))
    st.write("TheOver matches:", theover_matches if theover_matches is not None else 0)
    st.write("Kalshi matches:", kalshi_matches if kalshi_matches is not None else 0)
    st.write("Parlay count:", parlay_count if parlay_count is not None else 0)

    merge_keys = ""
    if "debug_merge_keys" in df.columns and not df.empty:
        merge_keys = str(df["debug_merge_keys"].iloc[0])
    st.write("Merge keys:", merge_keys or "league, home_team, away_team, game_date")

    ml_loaded = False
    if "debug_model_loaded" in df.columns and not df.empty:
        ml_loaded = bool(df["debug_model_loaded"].iloc[0])
    st.write("ML model loaded:", ml_loaded)

    st.write("ML predictions:", df["ml_prob"].notna().sum() if "ml_prob" in df.columns else 0)
    st.write("Positive EV picks:", (df["expected_value"] > 0).sum() if "expected_value" in df.columns else 0)

    if "expected_value" in df.columns:
        ev = pd.to_numeric(df["expected_value"], errors="coerce").dropna()
        if not ev.empty:
            st.write(
                "EV distribution:",
                {
                    "min": float(ev.min()),
                    "median": float(ev.median()),
                    "max": float(ev.max()),
                },
            )

    st.dataframe(df.head(20), width="stretch")


def render_debug(
    df: pd.DataFrame,
    odds_matches: int | None = None,
    theover_matches: int | None = None,
    kalshi_matches: int | None = None,
    parlay_count: int | None = None,
) -> None:
    show_pipeline_debug(df, odds_matches, theover_matches, kalshi_matches, parlay_count)


def render_debug_panel(
    df: pd.DataFrame,
    odds_matches: int | None = None,
    theover_matches: int | None = None,
    kalshi_matches: int | None = None,
    parlay_count: int | None = None,
) -> None:
    with st.expander("Debug Info"):
        show_pipeline_debug(df, odds_matches, theover_matches, kalshi_matches, parlay_count)
