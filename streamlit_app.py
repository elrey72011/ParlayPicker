from __future__ import annotations

from typing import Any

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app.ui.analysis_dashboard import render_analysis
try:
    from app.ui.data_diagnostics import show_data_diagnostics
except Exception:  # pragma: no cover
    def show_data_diagnostics(**_: Any) -> None:
        st.info("Data diagnostics module unavailable in this environment.")
from app.ui.debug_panel import render_debug, render_debug_panel
from app.ui.kalshi_diagnostics import render_kalshi_diagnostics
from app.ui.layout import setup_page
from app.ui.odds_dashboard import render_odds_table
from app.ui.parlay_dashboard import render_parlays
from app.ui.sidebar_controls import render_sidebar
from app.ui.strategy_lab_dashboard import render_strategy_lab
from core.streamlit_pipeline import (
    generate_parlays,
    optimize_portfolio_allocation,
    run_analysis_pipeline,
    run_bankroll_simulation,
)
from core.team_normalizer import normalize_team
from core.theover_loader import load_theover_csv

try:
    from app_core.kalshi_integrator import enrich_with_kalshi_markets
except Exception:  # pragma: no cover
    def enrich_with_kalshi_markets(df: pd.DataFrame) -> pd.DataFrame:
        return df



def _safe_str_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="string")
    if col in df.columns:
        return df[col].fillna(default).astype("string")
    return pd.Series([default] * len(df), index=df.index, dtype="string")


def _safe_numeric_series(df: pd.DataFrame, col: str, default: float | int | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="float64")
    if col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
    else:
        s = pd.Series([pd.NA] * len(df), index=df.index, dtype="Float64")
    if default is not None:
        s = s.fillna(default)
    return s


def main() -> None:
    setup_page()

    stable_defaults = {
        "analysis_df": pd.DataFrame(),
        "best_picks_df": pd.DataFrame(),
        "diagnostics": {},
        "parlays_df": pd.DataFrame(),
        "portfolio_df": pd.DataFrame(),
    }
    for key, default in stable_defaults.items():
        st.session_state.setdefault(key, default)

    for leg_count in (2, 3, 4, 5):
        key = f"parlays_{leg_count}_df"
        if key not in st.session_state:
            st.session_state[key] = pd.DataFrame()
    if "odds_df" not in st.session_state:
        st.session_state["odds_df"] = pd.DataFrame()
    if "theover_df" not in st.session_state:
        st.session_state["theover_df"] = pd.DataFrame()
    if "kalshi_df" not in st.session_state:
        st.session_state["kalshi_df"] = pd.DataFrame()
    if "gemini_df" not in st.session_state:
        st.session_state["gemini_df"] = pd.DataFrame()
    if "simulation_results" not in st.session_state:
        st.session_state["simulation_results"] = {}
    if "pipeline_status" not in st.session_state:
        st.session_state["pipeline_status"] = "idle"

    controls = render_sidebar()
    run_clicked = bool(controls["run_analysis"])

    if run_clicked:
        spreads_df = load_theover_csv(controls.get("theover_spreads"))
        totals_df = load_theover_csv(controls.get("theover_totals"))

        for upload_df in (spreads_df, totals_df):
            for team_col in ["home_team", "away_team"]:
                if team_col in upload_df.columns:
                    upload_df[team_col] = upload_df[team_col].apply(normalize_team)

        st.write("TheOver spreads rows:", len(spreads_df))
        st.write("TheOver totals rows:", len(totals_df))

        analysis_df, best_picks_df, diagnostics = run_analysis_pipeline(
            sports=controls["sports"],
            max_rows=int(controls["max_rows"]),
            use_ml=bool(controls["use_ml"]),
            spreads_df=spreads_df,
            totals_df=totals_df,
        )

        if isinstance(best_picks_df, pd.DataFrame) and not best_picks_df.empty:
            if "game_date" not in best_picks_df.columns:
                raise ValueError("best_picks_df missing game_date before Kalshi enrichment")
            best_picks_df = enrich_with_kalshi_markets(best_picks_df)

            with_date = best_picks_df["game_date"].notna()
            tried_empty = (
                best_picks_df["kalshi_tried_tickers"].astype(str).str.strip().isin(["", "[]", "null", "None"])
                if "kalshi_tried_tickers" in best_picks_df.columns
                else pd.Series([True] * len(best_picks_df), index=best_picks_df.index)
            )
            if bool((with_date & tried_empty).any()):
                diagnostics["kalshi_ticker_warning"] = "kalshi_tried_tickers empty for rows with non-null game_date"

        if not analysis_df.empty and "market_type" not in analysis_df.columns:
            raise ValueError("analysis_df missing market_type before best-pick construction")
        allowed_rows = int(diagnostics.get("allowed_market_type_rows", 0))
        if not analysis_df.empty and allowed_rows > 0 and (best_picks_df is None or best_picks_df.empty):
            diagnostics["best_picks_warning"] = "best_picks_df empty while analysis_df has valid spread/total rows"

        attempted = int(len(best_picks_df)) if isinstance(best_picks_df, pd.DataFrame) else 0
        matched = int(best_picks_df["kalshi_match_status"].astype(str).str.lower().eq("matched").sum()) if attempted and "kalshi_match_status" in best_picks_df.columns else int(diagnostics.get("kalshi_matches", 0))
        diagnostics["kalshi_attempted"] = attempted
        diagnostics["kalshi_matches"] = matched
        diagnostics["kalshi_match_rate"] = float(matched / max(attempted, 1))
        diagnostics["match_rate"] = diagnostics["kalshi_match_rate"]
        diagnostics["kalshi_missing_date_rows"] = int(best_picks_df["kalshi_match_reason"].astype(str).eq("missing_date").sum()) if attempted and "kalshi_match_reason" in best_picks_df.columns else 0
        diagnostics["kalshi_missing_team_code_rows"] = int(best_picks_df["kalshi_match_reason"].astype(str).eq("missing_team_code").sum()) if attempted and "kalshi_match_reason" in best_picks_df.columns else 0
        diagnostics["kalshi_no_market_rows"] = int(best_picks_df["kalshi_match_reason"].astype(str).eq("no_market_for_tickers").sum()) if attempted and "kalshi_match_reason" in best_picks_df.columns else 0

        st.session_state.analysis_df = analysis_df
        st.session_state["diagnostics"] = diagnostics

        if analysis_df.empty:
            st.warning("No rows found for the selected sports.")
            st.session_state["analysis_df"] = pd.DataFrame()
            st.session_state["parlays_df"] = pd.DataFrame()
            st.session_state["portfolio_df"] = pd.DataFrame()
            for leg_count in (2, 3, 4, 5):
                st.session_state[f"parlays_{leg_count}_df"] = pd.DataFrame()
            st.session_state["odds_df"] = pd.DataFrame()
            st.session_state["theover_df"] = pd.DataFrame()
            st.session_state["kalshi_df"] = pd.DataFrame()
            st.session_state["gemini_df"] = pd.DataFrame()
            st.session_state["simulation_results"] = {}
            st.session_state["best_picks_df"] = pd.DataFrame()
            return


        if controls["use_gemini"]:
            from integrations.gemini_client import run_gemini_analysis

            analysis_df = run_gemini_analysis(analysis_df)

        if "gemini_analysis" not in analysis_df.columns:
            analysis_df["gemini_analysis"] = ""

        parlays_df = generate_parlays(best_picks_df) if best_picks_df is not None and not best_picks_df.empty else pd.DataFrame(columns=["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"])
        st.session_state.parlays_df = parlays_df
        parlay_columns = ["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"]
        for leg_count in (2, 3, 4, 5):
            parlay_slice = parlays_df[_safe_numeric_series(parlays_df, "legs").eq(leg_count)].copy()
            if not parlay_slice.empty:
                parlay_slice = parlay_slice[parlay_columns]
            else:
                parlay_slice = pd.DataFrame(columns=parlay_columns)
            st.session_state[f"parlays_{leg_count}_df"] = parlay_slice
        portfolio_source_df = best_picks_df if best_picks_df is not None and not best_picks_df.empty else pd.DataFrame()
        portfolio_df = optimize_portfolio_allocation(
            portfolio_source_df,
            bankroll=float(controls["bankroll"]),
        ) if not portfolio_source_df.empty else pd.DataFrame()
        if isinstance(portfolio_df, pd.DataFrame) and not portfolio_df.empty:
            if "best_pick" not in portfolio_df.columns:
                portfolio_df["best_pick"] = pd.NA
            if "best_pick" in portfolio_source_df.columns:
                empty_best_pick = portfolio_df["best_pick"].isna() | portfolio_df["best_pick"].astype(str).str.strip().eq("")
                join_keys = [
                    c
                    for c in ["league", "home_team", "away_team", "game_date"]
                    if c in portfolio_df.columns and c in portfolio_source_df.columns
                ]
                if join_keys:
                    best_lookup = portfolio_source_df[join_keys + ["best_pick"]].drop_duplicates()
                    recovered = portfolio_df.loc[empty_best_pick, join_keys].merge(best_lookup, on=join_keys, how="left")
                    portfolio_df.loc[empty_best_pick, "best_pick"] = recovered["best_pick"].values
        required_portfolio_cols = {"calibrated_probability", "decimal_odds", "recommended_bet"}
        if portfolio_df is not None and not portfolio_df.empty and required_portfolio_cols.issubset(set(portfolio_df.columns)):
            simulation_results = run_bankroll_simulation(portfolio_df, bankroll=float(controls["bankroll"]))
        else:
            diagnostics["bankroll_simulation_skipped"] = True
            simulation_results = {}

        odds_df = analysis_df.copy()
        theover_frames = []
        if controls["theover_spreads"]:
            theover_frames.append(load_theover_csv(controls["theover_spreads"]))
        if controls["theover_totals"]:
            theover_frames.append(load_theover_csv(controls["theover_totals"]))

        valid_theover_frames = []
        for frame in theover_frames:
            if frame is None:
                continue
            if not isinstance(frame, pd.DataFrame):
                continue
            if frame.empty:
                continue
            if frame.dropna(how="all").empty:
                continue
            valid_theover_frames.append(frame)
        theover_df = pd.concat(valid_theover_frames, ignore_index=True) if valid_theover_frames else pd.DataFrame()

        if "kalshi_probability" in analysis_df.columns:
            kalshi_df = analysis_df[analysis_df["kalshi_probability"].notna()].copy()
        else:
            kalshi_df = analysis_df.iloc[0:0]

        gemini_df = (
            analysis_df[_safe_str_series(analysis_df, "gemini_analysis").str.len() > 0]
            if "gemini_analysis" in analysis_df.columns
            else analysis_df.iloc[0:0]
        )

        st.session_state["pipeline_status"] = "using stored results"
        st.session_state["analysis_df"] = analysis_df
        st.session_state["parlays_df"] = parlays_df
        st.session_state["portfolio_df"] = portfolio_df
        st.session_state["odds_df"] = odds_df
        st.session_state["theover_df"] = theover_df
        st.session_state["kalshi_df"] = kalshi_df
        st.session_state["gemini_df"] = gemini_df
        st.session_state["simulation_results"] = simulation_results
        st.session_state["best_picks_df"] = best_picks_df
    analysis_df = st.session_state["analysis_df"]
    parlays_df = st.session_state["parlays_df"]
    portfolio_df = st.session_state["portfolio_df"]
    odds_df = st.session_state["odds_df"]
    theover_df = st.session_state["theover_df"]
    kalshi_df = st.session_state["kalshi_df"]
    gemini_df = st.session_state["gemini_df"]
    simulation_results = st.session_state["simulation_results"]
    best_picks_df = st.session_state["best_picks_df"]
    diagnostics = st.session_state.get("diagnostics", {})

    pipeline_status = st.session_state.get("pipeline_status", "idle")
    st.caption(f"Pipeline status: {pipeline_status}")

    if analysis_df is None or analysis_df.empty:
        st.info("Configure filters in the sidebar and click **Run Master Analysis**.")
        return

    games_count = int(diagnostics.get("total_games", 0))
    bet_rows = int(diagnostics.get("bet_rows", len(analysis_df)))
    best_rows = int(diagnostics.get("best_picks", len(best_picks_df) if isinstance(best_picks_df, pd.DataFrame) else 0))
    kalshi_matches = int(diagnostics.get("kalshi_matches", 0))
    match_rate = float(diagnostics.get("match_rate", diagnostics.get("kalshi_match_rate", kalshi_matches / max(1, best_rows))))
    totals_games = int(diagnostics.get("theover_totals_games", 0))
    spreads_games = int(diagnostics.get("theover_spreads_games", 0))
    totals_bet_games = totals_games
    spreads_bet_games = spreads_games
    date_fill_attempted = int(diagnostics.get("date_fill_total_rows", 0))
    date_fill_filled = int(diagnostics.get("date_fill_success_rows", 0))
    date_fill_rate = float(diagnostics.get("date_fill_success_rate", 0.0))
    positive_ev_picks = int(diagnostics.get("positive_ev_picks", 0))
    odds_base_loaded = bool(diagnostics.get("odds_schedule_loaded", False))
    with st.container():
        m1, m2, m3, m4, m5, m6, m7, m8, m9 = st.columns(9)
        m1.metric("Total games", games_count)
        m2.metric("Bet rows", bet_rows)
        m3.metric("Best picks", best_rows)
        m4.metric("Kalshi matches", kalshi_matches)
        m5.metric("Match rate", f"{match_rate:.0%}")
        m6.metric("TheOver totals games", f"{totals_bet_games}/{games_count}")
        m7.metric("TheOver spreads games", f"{spreads_bet_games}/{games_count}")
        m8.metric("Date fill success", f"{date_fill_filled}/{date_fill_attempted} ({date_fill_rate:.0%})")
        m9.metric("Positive EV picks", positive_ev_picks)
        st.progress(max(0.0, min(1.0, match_rate)), text=f"Kalshi match rate: {match_rate:.0%}")
        st.caption(f"Merge keys used: {diagnostics.get('merge_keys_used', [])}")
        st.caption(f"Odds/base schedule loaded: {odds_base_loaded}")
        st.caption(f"Stale base schedule: {bool(diagnostics.get('stale_base_schedule', False))}")
        if diagnostics.get("stale_base_schedule") and diagnostics.get("has_normalized_bet_rows", False):
            st.warning("Pipeline warning: stale base schedule relative to uploaded bet rows.")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Odds", "Analysis", "Best Picks", "Parlays", "Portfolio", "Debug", "Strategy Lab"])

    with tab1:
        render_odds_table(analysis_df)

    with tab2:
        if analysis_df is not None and not analysis_df.empty:
            render_analysis(analysis_df)
        if analysis_df is not None and not analysis_df.empty:
            analysis_export_df = analysis_df.copy()
            if "best_pick" in analysis_export_df.columns:
                export_priority = [
                    "league", "home_team", "away_team", "game_date", "matchup",
                    "market", "pick", "pickteam", "line", "winprobability", "theover_probability",
                    "market_type", "spread_line", "total_line",
                    "expected_value", "edge", "calibrated_probability",
                    "kalshi_probability", "kalshi_market_title", "kalshi_market_ticker", "kalshi_event_ticker", "kalshi_line",
                ]
                ordered_cols = [c for c in export_priority if c in analysis_export_df.columns]
                trailing_cols = [c for c in analysis_export_df.columns if c not in ordered_cols]
                analysis_export_df = analysis_export_df[ordered_cols + trailing_cols]
            analysis_csv = analysis_export_df.to_csv(index=False)
            st.download_button(
                "Export Analysis",
                analysis_csv,
                "analysis_export.csv",
                mime="text/csv",
            )

    with tab3:
        st.subheader("Best Picks")
        if best_picks_df is None or best_picks_df.empty:
            st.info("No eligible spread/total best picks found.")
            with st.expander("Best Picks Debug Diagnostics", expanded=True):
                st.json({
                    "market_type_counts": diagnostics.get("market_type_counts", {}),
                    "allowed_market_type_rows": diagnostics.get("allowed_market_type_rows", 0),
                    "positive_ev_rows": diagnostics.get("positive_ev_rows", 0),
                    "best_pick_nonempty_rows": diagnostics.get("best_pick_nonempty_rows", 0),
                    "bet_rows": diagnostics.get("bet_rows", 0),
                })
        else:
            display_df = best_picks_df.copy()
            rename_map = {
                "league": "League",
                "away_team": "Away Team",
                "home_team": "Home Team",
                "game_date": "Game Date",
                "best_pick": "Best Pick",
                "calibrated_probability": "Prob",
                "expected_value": "EV",
                "edge": "Edge",
                "kalshi_match_status": "Kalshi Status",
            }
            display_df = display_df.rename(columns=rename_map)
            preferred = ["League", "Home Team", "Away Team", "Game Date", "Best Pick", "Prob", "EV", "Edge", "Kalshi Status"]
            ordered = [c for c in preferred if c in display_df.columns] + [c for c in display_df.columns if c not in preferred]
            display_df = display_df[ordered]
            st.dataframe(display_df, width="stretch")
            export_cols = [c for c in ["league", "home_team", "away_team", "game_date", "best_pick", "calibrated_probability", "expected_value", "edge", "odds_american", "market_probability", "ml_probability"] if c in best_picks_df.columns]
            best_picks_export = best_picks_df[export_cols] if export_cols else best_picks_df
            best_picks_csv = best_picks_export.to_csv(index=False)
            st.download_button(
                "Export Best Picks",
                best_picks_csv,
                "best_picks_export.csv",
                mime="text/csv",
            )

    with tab4:
        st.subheader("Best Parlays")
        parlay_columns = ["parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "legs"]
        base_parlays_df = parlays_df if parlays_df is not None else pd.DataFrame(columns=parlay_columns)
        tabs_2, tabs_3, tabs_4, tabs_5 = st.tabs(["2-Leg Parlays", "3-Leg Parlays", "4-Leg Parlays", "5-Leg Parlays"])

        for leg_count, parlay_tab in zip((2, 3, 4, 5), (tabs_2, tabs_3, tabs_4, tabs_5)):
            with parlay_tab:
                filtered = base_parlays_df[_safe_numeric_series(base_parlays_df, "legs").eq(leg_count)].copy()
                filtered = filtered[parlay_columns] if not filtered.empty else pd.DataFrame(columns=parlay_columns)
                if filtered.empty:
                    st.info(f"Not enough eligible spread/total picks to build {leg_count}-leg parlays yet.")
                    continue

                render_parlays(filtered)
                parlay_csv = filtered.to_csv(index=False)
                st.download_button(
                    f"Download {leg_count}-Leg Parlays CSV",
                    parlay_csv,
                    f"parlays_{leg_count}_leg_export.csv",
                    mime="text/csv",
                    key=f"download_{leg_count}_leg_parlays",
                )

    with tab5:
        st.subheader("Portfolio Allocation")
        portfolio_display = portfolio_df.copy() if portfolio_df is not None else pd.DataFrame()
        if not portfolio_display.empty:
            if "best_pick" not in portfolio_display.columns:
                portfolio_display["best_pick"] = ""
            portfolio_display["best_pick"] = _safe_str_series(portfolio_display, "best_pick").str.strip()

            if portfolio_display["best_pick"].str.len().eq(0).all():
                st.warning("Portfolio built, but best_pick strings are missing upstream.")

            display_first_columns = [
                "league",
                "away_team",
                "home_team",
                "best_pick",
                "calibrated_probability",
                "expected_value",
                "edge",
                "recommended_bet",
            ]
            ordered_columns = [c for c in display_first_columns if c in portfolio_display.columns]
            trailing_columns = [c for c in portfolio_display.columns if c not in ordered_columns]
            portfolio_display = portfolio_display[ordered_columns + trailing_columns]

            league_s = _safe_str_series(portfolio_display, "league").str.upper()
            pick_s = _safe_str_series(portfolio_display, "best_pick")
            bet_s = pd.to_numeric(_safe_str_series(portfolio_display, "recommended_bet", "0"), errors="coerce").fillna(0.0)

            portfolio_display["allocation_label"] = (
                league_s + " | " + pick_s + " | $" + bet_s.map(lambda x: f"{x:,.2f}")
            )

        st.dataframe(portfolio_display, width="stretch")
        st.download_button(
            "Export Portfolio",
            portfolio_display.to_csv(index=False),
            "portfolio_export.csv",
            mime="text/csv",
            key="export_portfolio_csv",
        )

    with tab6:
        show_data_diagnostics(
            odds_df=odds_df,
            theover_df=theover_df if theover_df is not None else analysis_df.iloc[0:0],
            kalshi_df=kalshi_df,
            gemini_df=gemini_df,
        )

        odds_matches = len(odds_df) if odds_df is not None else 0
        theover_matches = len(theover_df) if theover_df is not None else 0
        kalshi_matches = len(kalshi_df) if kalshi_df is not None else 0

        total_analysis_rows = len(analysis_df) if analysis_df is not None else 0
        kalshi_non_null_rows = (
            int(analysis_df["kalshi_probability"].notna().sum())
            if analysis_df is not None and "kalshi_probability" in analysis_df.columns
            else 0
        )
        kalshi_matched_rows = (
            int(analysis_df["kalshi_match_status"].astype(str).str.lower().eq("matched").sum())
            if analysis_df is not None and "kalshi_match_status" in analysis_df.columns
            else 0
        )
        kalshi_miss_rows = (
            int(analysis_df["kalshi_match_status"].astype(str).str.lower().eq("no_match").sum())
            if analysis_df is not None and "kalshi_match_status" in analysis_df.columns
            else 0
        )

        st.markdown("### Kalshi Merge Diagnostics")
        st.write("analysis_df total rows:", total_analysis_rows)
        st.write("rows with non-null kalshi_probability:", kalshi_non_null_rows)
        st.write('rows with kalshi_match_status == "matched":', kalshi_matched_rows)
        st.write('rows with kalshi_match_status == "no_match":', kalshi_miss_rows)

        if controls["show_debug"]:
            parlay_count = len(parlays_df) if parlays_df is not None else 0
            render_debug(analysis_df, odds_matches, theover_matches, kalshi_matches, parlay_count)
            render_debug_panel(analysis_df, odds_matches, theover_matches, kalshi_matches, parlay_count)
        else:
            st.info("Enable 'Display Debug Information' in the sidebar to inspect debug data.")

        if controls["show_kalshi_diagnostics"]:
            render_kalshi_diagnostics(analysis_df)

    with tab7:
        render_strategy_lab(
            analysis_df=analysis_df,
            portfolio_df=portfolio_df if portfolio_df is not None else analysis_df.iloc[0:0],
            parlays_df=parlays_df if parlays_df is not None else analysis_df.iloc[0:0],
            simulation_results=simulation_results or {},
        )


if __name__ == "__main__":
    main()
