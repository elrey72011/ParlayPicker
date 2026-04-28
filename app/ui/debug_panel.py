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
    if "nba_stats_fetch_status" in df.columns and not df.empty:
        st.write("NBA stats fetch status:", str(df["nba_stats_fetch_status"].iloc[0]))
    if "nba_stats_fetch_retries_used" in df.columns and not df.empty:
        st.write("NBA stats fetch retries used:", int(df["nba_stats_fetch_retries_used"].iloc[0]))
    if "stats_source_counts" in df.columns and not df.empty:
        st.write("Stats source counts:", str(df["stats_source_counts"].iloc[0]))
    if "stats_unresolved_count_by_league_detail" in df.columns and not df.empty:
        st.write("Stats unresolved count by league:", str(df["stats_unresolved_count_by_league_detail"].iloc[0]))
    if "fallback_summary_by_league" in df.columns and not df.empty:
        st.write("Fallback summary by league:", str(df["fallback_summary_by_league"].iloc[0]))
    if "fallback_heavy_slate_flag" in df.columns and not df.empty:
        st.write("Fallback-heavy slate:", bool(df["fallback_heavy_slate_flag"].iloc[0]))
    if "run_health_warning" in df.columns and not df.empty and str(df["run_health_warning"].iloc[0]).strip():
        st.warning(str(df["run_health_warning"].iloc[0]))

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
