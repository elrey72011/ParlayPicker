from __future__ import annotations

import traceback
import warnings
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
from app.ui.sidebar_controls import render_sidebar
from app.ui.strategy_lab_dashboard import render_strategy_lab
from core.streamlit_pipeline import (
    generate_parlays,
    optimize_portfolio_allocation,
    run_analysis_pipeline,
    run_bankroll_simulation,
    CANONICAL_BET_COLUMNS,
    VALID_MARKETS,
    MIN_EDGE_THRESHOLD,
)
from core.team_normalizer import normalize_team
from core.theover_loader import load_theover_csv

try:
    from app_core.kalshi_integrator import enrich_with_kalshi_markets as _enrich_kalshi_raw
except Exception:  # pragma: no cover
    _enrich_kalshi_raw = None  # type: ignore[assignment]

try:
    from app_core.odds_api import OddsAPIAuthError
except ImportError:
    class OddsAPIAuthError(Exception): pass


KALSHI_ENRICH_TIMEOUT_SECONDS = 60


def _enrich_with_kalshi_safe(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Run Kalshi enrichment with a hard timeout.
    Returns (enriched_df, error_message_or_None).
    """
    if _enrich_kalshi_raw is None:
        return df, None

    import concurrent.futures

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_enrich_kalshi_raw, df)
    try:
        result = future.result(timeout=KALSHI_ENRICH_TIMEOUT_SECONDS)
        return result, None
    except concurrent.futures.TimeoutError:
        future.cancel()
        return df, f"Kalshi enrichment timed out (>{KALSHI_ENRICH_TIMEOUT_SECONDS}s) — skipped."
    except Exception as e:
        return df, f"Kalshi enrichment failed: {e}"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _safe_str_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype="string")
    if col in df.columns:
        return df[col].fillna(default).astype("string")
    return pd.Series([default] * len(df), index=df.index, dtype="string")






def _should_run_pipeline(state: dict[str, Any], run_counter: int) -> bool:
    """Run once per monotonically increasing sidebar run counter."""
    last_processed = int(state.get("last_processed_run_counter", 0))
    if run_counter <= last_processed:
        return False
    state["last_processed_run_counter"] = run_counter
    return True

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




def _recompute_consensus_from_kalshi(df: pd.DataFrame) -> pd.DataFrame:
    """Set consensus based on Kalshi availability and probability gap, and update blends."""
    if df is None or df.empty:
        return df
    out = df.copy()

    # Recalculate blended probability and EV/Edge since we might have new Kalshi probabilities
    from core.streamlit_pipeline import compute_blended_probability

    kalshi_prob = _safe_numeric_series(out, "kalshi_probability")
    market_prob = _safe_numeric_series(out, "market_probability")

    # Handle the two variations of model prob stored depending on df origin
    if "model_probability" in out.columns:
        model_prob = _safe_numeric_series(out, "model_probability")
    else:
        # Fallback to ml or theover prob
        ml = _safe_numeric_series(out, "ml_probability")
        theover = _safe_numeric_series(out, "theover_probability")
        model_prob = ml.where(ml.notna(), theover)

    blended = compute_blended_probability(
        p_market=market_prob,
        p_kalshi=kalshi_prob,
        p_ml=model_prob,
        league=_safe_str_series(out, "league"),
        market_type=_safe_str_series(out, "market_type")
    )

    # Update core metrics
    out["calibrated_probability"] = blended
    decimal_odds = _safe_numeric_series(out, "decimal_odds")
    if decimal_odds.isna().all() and "odds_american" in out.columns:
        from core.streamlit_pipeline import american_to_decimal
        # FIXED: No default -110, preserve NaN for missing odds
        decimal_odds = _safe_numeric_series(out, "odds_american").apply(american_to_decimal)

        if decimal_odds.isna().all():
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("⚠️ All odds are missing - using default 1.91 (-110 equivalent)")
            decimal_odds = pd.Series([1.91] * len(out), index=out.index)

    out["expected_value"] = blended * (decimal_odds - 1) - (1 - blended)
    out["edge"] = blended - market_prob

    status = _safe_str_series(out, "kalshi_match_status").str.lower()

    out["consensus_agreement"] = "⚪ No Kalshi"
    matched = status.eq("matched") & kalshi_prob.notna()
    gap = blended - kalshi_prob

    out.loc[matched, "consensus_agreement"] = "⚖️ Neutral"
    out.loc[matched & gap.ge(0.03), "consensus_agreement"] = "✅ Agrees"
    out.loc[matched & gap.le(-0.03), "consensus_agreement"] = "❌ Disagrees"

    # Debug log for probability blend verification (first 5 picks)
    if not out.empty and "market_probability" in out.columns:
        import logging
        logger = logging.getLogger(__name__)
        debug_sample = out.head(5)
        for idx, row in debug_sample.iterrows():
            logger.info(
                f"Blend Debug | Pick: {row.get('best_pick', '')} | "
                f"Market: {row.get('market_probability')}, "
                f"Kalshi: {row.get('kalshi_probability')}, "
                f"ML: {row.get('ml_probability')} | "
                f"Blended: {row.get('calibrated_probability')}"
            )

    # After recalculating edge, filter picks that fall below minimum threshold
    # TEMP DISABLED [2026-03-08]: Filter after Kalshi enrichment to see match diagnostics
    # if "edge" in out.columns and "best_pick" in out.columns:
    #     # Only apply dropping to best picks (not all analysis rows)
    #     if len(out) > 0 and pd.notna(out["best_pick"].iloc[0]):
    #         out = out[out["edge"] >= MIN_EDGE_THRESHOLD].copy().reset_index(drop=True)
    #         if "parlay_rank" in out.columns:
    #             out["parlay_rank"] = range(1, len(out) + 1)

    return out

def _merge_kalshi_into_analysis(analysis_df: pd.DataFrame, best_picks_df: pd.DataFrame) -> pd.DataFrame:
    if analysis_df is None or analysis_df.empty or best_picks_df is None or best_picks_df.empty:
        return analysis_df
    kalshi_cols = [
        "kalshi_probability",
        "kalshi_market_title",
        "kalshi_event_ticker",
        "kalshi_match_status",
        "kalshi_match_reason",

    ]
    available_cols = [c for c in kalshi_cols if c in best_picks_df.columns]
    if not available_cols:
        return analysis_df

    merge_keys = ["league", "home_team", "away_team", "game_date"]

    left = analysis_df.copy()
    right = best_picks_df[merge_keys + available_cols].drop_duplicates().copy()

    if "game_date" in merge_keys:
        left["game_date"] = pd.to_datetime(left["game_date"], errors="coerce", utc=True)
        right["game_date"] = pd.to_datetime(right["game_date"], errors="coerce", utc=True)

    merged = left.merge(right, on=merge_keys, how="left", suffixes=("", "_best"))
    for col in available_cols:
        best_col = f"{col}_best"
        if best_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[best_col]) if col in merged.columns else merged[best_col]
            merged = merged.drop(columns=[best_col])
    return merged


def _run_pipeline(controls: dict) -> tuple[dict, list[str], list[str]]:
    """Run the full analysis pipeline. Returns (state_updates, warnings, errors).
    Contains NO st.* calls.
    """
    deferred_warnings: list[str] = []
    deferred_errors: list[str] = []

    spreads_df, err = load_theover_csv(controls.get("theover_spreads"))
    if err:
        deferred_warnings.append(err)

    totals_df, err = load_theover_csv(controls.get("theover_totals"))
    if err:
        deferred_warnings.append(err)

    for upload_df in (spreads_df, totals_df):
        if upload_df is None or upload_df.empty:
            continue
        for team_col in ["home_team", "away_team"]:
            if team_col in upload_df.columns:
                upload_df[team_col] = upload_df[team_col].apply(normalize_team)

    analysis_df, best_picks_df, diagnostics = run_analysis_pipeline(
        sports=controls["sports"],
        max_rows=10_000,
        use_ml=bool(controls["use_ml"]),
        spreads_df=spreads_df,
        totals_df=totals_df,
    )

    parlay_columns = ["parlay_type", "parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "kelly_fraction_1_8", "legs", "leg1_game", "leg2_game", "leg3_game", "leg4_game", "leg5_game"]
    empty_per_leg = {f"parlays_{lc}_df": pd.DataFrame(columns=parlay_columns) for lc in (2, 3, 4, 5)}

    empty_state: dict = {
        "analysis_df": pd.DataFrame(),
        "parlays_df": pd.DataFrame(),
        "portfolio_df": pd.DataFrame(),
        "odds_df": pd.DataFrame(),
        "theover_df": pd.DataFrame(),
        "kalshi_df": pd.DataFrame(),
        "gemini_df": pd.DataFrame(),
        "simulation_results": {},
        "best_picks_df": pd.DataFrame(),
        "diagnostics": diagnostics,
        "pipeline_status": "idle",
        "pipeline_running": False,
        **empty_per_leg,
    }

    if analysis_df is None or analysis_df.empty:
        deferred_warnings.append("No rows found for the selected sports.")
        return empty_state, deferred_warnings, deferred_errors

    # Kalshi enrichment with hard timeout
    if isinstance(analysis_df, pd.DataFrame) and not analysis_df.empty:
        if "game_date" not in analysis_df.columns or analysis_df["game_date"].isna().all():
            deferred_warnings.append("game_date missing from analysis_df — Kalshi matching skipped.")
        else:
            analysis_df, kalshi_err = _enrich_with_kalshi_safe(analysis_df)
            if kalshi_err:
                deferred_warnings.append(kalshi_err)

    analysis_df = _recompute_consensus_from_kalshi(analysis_df)

    from core.streamlit_pipeline import build_best_picks_df
    best_picks_df = build_best_picks_df(analysis_df)

    if "gemini_analysis" not in analysis_df.columns:
        analysis_df["gemini_analysis"] = ""

    # Gemini Integration for Top Picks
    if controls.get("use_gemini") and not best_picks_df.empty:
        import os
        from gemini_integration import GeminiAnalyzer
        gcp_project = os.environ.get("GCP_PROJECT_ID", st.secrets.get("GCP_PROJECT_ID", ""))
        if gcp_project:
            try:
                analyzer = GeminiAnalyzer(project_id=gcp_project)
                games_to_analyze = []
                for _, row in best_picks_df.iterrows():
                    game = {
                        "home_team": row.get("home_team"),
                        "away_team": row.get("away_team"),
                        "sport_key": row.get("league"),
                        "commence_time": str(row.get("game_date")),
                        "best_moneyline": None,
                        "best_spread": float(row.get("spread_line")) if pd.notna(row.get("spread_line")) else None,
                        "context_data": {
                            "pick": row.get("best_pick"),
                            "expected_value": row.get("expected_value"),
                            "edge": row.get("edge"),
                            "ml": {
                                "model_used": "XGBoost",
                                "confidence": row.get("ml_probability"),
                            } if "ml_probability" in row and pd.notna(row.get("ml_probability")) else None
                        }
                    }
                    games_to_analyze.append(game)

                if games_to_analyze:
                    gemini_results = analyzer.analyze_games_batch(games_to_analyze)
                    gemini_explanations = []
                    gemini_risks = []
                    for res in gemini_results:
                        gemini_explanations.append(res.get("confidence_explanation", ""))
                        risks = res.get("risk_notes", "")
                        if isinstance(risks, list):
                            risks = ", ".join(risks)
                        gemini_risks.append(risks)
                    best_picks_df["gemini_explanation"] = gemini_explanations
                    best_picks_df["gemini_risk_notes"] = gemini_risks

                    # Update analysis_df to reflect these rows were analyzed (for diagnostics tab)
                    for idx, res in enumerate(gemini_results):
                        home = games_to_analyze[idx]["home_team"]
                        away = games_to_analyze[idx]["away_team"]
                        mask = (analysis_df["home_team"] == home) & (analysis_df["away_team"] == away)
                        analysis_df.loc[mask, "gemini_analysis"] = res.get("confidence_explanation", "Analyzed")
            except Exception as e:
                deferred_warnings.append(f"Gemini analysis failed: {e}")

    attempted = int(len(analysis_df)) if isinstance(analysis_df, pd.DataFrame) else 0
    matched = int(analysis_df.get("kalshi_match_status", pd.Series(dtype="string")).astype(str).str.lower().eq("matched").sum())

    diagnostics["kalshi_attempted"] = attempted
    diagnostics["kalshi_matches"] = matched
    diagnostics["kalshi_match_rate"] = float(matched / max(attempted, 1))
    diagnostics["match_rate"] = diagnostics["kalshi_match_rate"]
    diagnostics["kalshi_missing_date_rows"] = int(analysis_df["kalshi_match_reason"].astype(str).eq("missing_date").sum()) if attempted and "kalshi_match_reason" in analysis_df.columns else 0
    diagnostics["kalshi_missing_team_code_rows"] = int(analysis_df["kalshi_match_reason"].astype(str).eq("missing_team_code").sum()) if attempted and "kalshi_match_reason" in analysis_df.columns else 0
    diagnostics["kalshi_no_market_rows"] = int(analysis_df["kalshi_match_reason"].astype(str).eq("no_market_for_tickers").sum()) if attempted and "kalshi_match_reason" in analysis_df.columns else 0
    diagnostics["best_picks"] = int(len(best_picks_df)) if isinstance(best_picks_df, pd.DataFrame) else 0
    diagnostics["positive_ev_rows"] = int((_safe_numeric_series(analysis_df, "expected_value", 0.0) > 0).sum()) if not analysis_df.empty else 0
    diagnostics["positive_ev_picks"] = int((_safe_numeric_series(best_picks_df, "expected_value", 0.0) > 0).sum()) if not best_picks_df.empty else 0
    diagnostics["best_pick_nonempty_rows"] = int(_safe_str_series(best_picks_df, "best_pick").str.strip().str.len().gt(0).sum()) if not best_picks_df.empty else 0

    parlays_df = generate_parlays(best_picks_df)
    per_leg: dict = {}
    for lc in (2, 3, 4, 5):
        parlay_slice = parlays_df[_safe_numeric_series(parlays_df, "legs").eq(lc)].copy()
        per_leg[f"parlays_{lc}_df"] = parlay_slice[parlay_columns] if not parlay_slice.empty else pd.DataFrame(columns=parlay_columns)

    portfolio_df = optimize_portfolio_allocation(best_picks_df, bankroll=float(controls["bankroll"]))

    required_portfolio_cols = {"calibrated_probability", "decimal_odds", "recommended_bet"}
    if portfolio_df is not None and not portfolio_df.empty and required_portfolio_cols.issubset(set(portfolio_df.columns)):
        simulation_results = run_bankroll_simulation(portfolio_df, bankroll=float(controls["bankroll"]))
    else:
        diagnostics["bankroll_simulation_skipped"] = True
        simulation_results = {}

    odds_df = analysis_df.copy()
    theover_frames = [spreads_df, totals_df]
    valid_theover_frames = [
        f for f in theover_frames
        if f is not None and isinstance(f, pd.DataFrame) and not f.empty and not f.dropna(how="all").empty
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        theover_df = pd.concat(valid_theover_frames, ignore_index=True) if valid_theover_frames else pd.DataFrame()

    kalshi_df = analysis_df[analysis_df["kalshi_probability"].notna()].copy() if "kalshi_probability" in analysis_df.columns else analysis_df.iloc[0:0]

    gemini_df = (
        analysis_df[_safe_str_series(analysis_df, "gemini_analysis").str.len() > 0]
        if "gemini_analysis" in analysis_df.columns
        else analysis_df.iloc[0:0]
    )

    state_updates = {
        "pipeline_status": "using stored results",
        "pipeline_running": False,
        "analysis_df": analysis_df,
        "parlays_df": parlays_df,
        "portfolio_df": portfolio_df,
        "odds_df": odds_df,
        "theover_df": theover_df,
        "kalshi_df": kalshi_df,
        "gemini_df": gemini_df,
        "simulation_results": simulation_results,
        "diagnostics": diagnostics,
        "best_picks_df": best_picks_df,
        **per_leg,
    }
    return state_updates, deferred_warnings, deferred_errors


def main() -> None:
    setup_page()

    stable_defaults = {
        "analysis_df": pd.DataFrame(),
        "best_picks_df": pd.DataFrame(),
        "diagnostics": {},
        "parlays_df": pd.DataFrame(),
        "portfolio_df": pd.DataFrame(),
        "odds_df": pd.DataFrame(),
        "theover_df": pd.DataFrame(),
        "kalshi_df": pd.DataFrame(),
        "gemini_df": pd.DataFrame(),
        "simulation_results": {},
        "pipeline_status": "idle",
        "pipeline_running": False,
    }
    for key, default in stable_defaults.items():
        st.session_state.setdefault(key, default)
    for leg_count in (2, 3, 4, 5):
        st.session_state.setdefault(f"parlays_{leg_count}_df", pd.DataFrame())

    controls = render_sidebar()

    run_counter = int(controls.get("run_analysis_counter", 0))
    should_run = _should_run_pipeline(st.session_state, run_counter)

    # Only run pipeline once per button click; always reset flag on completion or crash
    if should_run and not st.session_state.get("pipeline_running", False):
        st.session_state["pipeline_running"] = True
        try:
            with st.spinner("Running analysis..."):
                state_updates, pipe_warnings, pipe_errors = _run_pipeline(controls)
            st.session_state.update(state_updates)
            for msg in pipe_warnings:
                st.warning(msg)
            for msg in pipe_errors:
                st.error(msg)
        except OddsAPIAuthError as e:
            st.session_state["pipeline_running"] = False
            st.error('The Odds API key is invalid, revoked, or missing. Please verify your credentials in Streamlit secrets.')
            st.stop()
        except Exception:
            st.error(f"Pipeline crashed:\n```\n{traceback.format_exc()}\n```")
        finally:
            # CRITICAL: Always release the pipeline lock so future runs are not blocked
            st.session_state["pipeline_running"] = False

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
    if not st.session_state["analysis_df"].empty:
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
    date_fill_attempted = int(diagnostics.get("date_fill_total_rows", 0))
    date_fill_filled = int(diagnostics.get("date_fill_success_rows", 0))
    date_fill_rate = float(diagnostics.get("date_fill_success_rate", 0.0))
    positive_ev_rows = int(diagnostics.get("positive_ev_rows", 0))
    consensus_agrees = (
        int(best_picks_df["consensus_agreement"].astype(str).eq("✅ Agrees").sum())
        if isinstance(best_picks_df, pd.DataFrame) and "consensus_agreement" in best_picks_df.columns
        else 0
    )
    odds_base_loaded = bool(diagnostics.get("odds_schedule_loaded", False))

    with st.container():
        m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11 = st.columns(11)
        m1.metric("Total games", games_count)
        m2.metric("Bet rows", bet_rows)
        m3.metric("Best picks", best_rows)
        m4.metric("Kalshi matches", kalshi_matches)
        m5.metric("Match rate", f"{match_rate:.0%}")
        m6.metric("TheOver totals games", f"{totals_games}/{games_count}")
        m7.metric("TheOver spreads games", f"{spreads_games}/{games_count}")
        m8.metric("Date fill success", f"{date_fill_filled}/{date_fill_attempted} ({date_fill_rate:.0%})")
        m9.metric("Positive EV rows", positive_ev_rows)
        m10.metric("Consensus ✅", consensus_agrees)

        kalshi_hits = analysis_df["kalshi_probability"].notna().sum() if analysis_df is not None and not analysis_df.empty and "kalshi_probability" in analysis_df.columns else 0
        total_analysis_len = len(analysis_df) if analysis_df is not None and not analysis_df.empty else 1
        m11.metric("Kalshi Matches", f"{kalshi_hits}/{len(analysis_df) if analysis_df is not None else 0} ({kalshi_hits/total_analysis_len*100:.0f}%)")

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
                    "league", "home_team", "away_team", "game_date", "game_time_est", "matchup",
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
            st.warning(f"⚠️ No picks meet {MIN_EDGE_THRESHOLD*100:.1f}% edge threshold")
            st.dataframe(pd.DataFrame(columns=best_picks_df.columns if best_picks_df is not None and not best_picks_df.empty else ["league", "pick", "edge"]))
            with st.expander("Best Picks Debug Diagnostics", expanded=True):
                st.json({
                    "market_type_counts": diagnostics.get("market_type_counts", {}),
                    "allowed_market_type_rows": diagnostics.get("allowed_market_type_rows", 0),
                    "positive_ev_rows": diagnostics.get("positive_ev_rows", 0),
                    "best_pick_nonempty_rows": diagnostics.get("best_pick_nonempty_rows", 0),
                    "bet_rows": diagnostics.get("bet_rows", 0),
                })
        else:
            st.success(f"✅ {len(best_picks_df)} picks found")
            display_df = best_picks_df.copy()
            rename_map = {
                "league": "League",
                "away_team": "Away Team",
                "home_team": "Home Team",
                "game_date": "Game Date",
                "game_time_est": "Game Time (ET)",
                "best_pick": "Best Pick",
                "calibrated_probability": "Prob",
                "expected_value": "EV",
                "edge": "Edge",
                "consensus_agreement": "Consensus",
                "kalshi_match_status": "Kalshi Status",
                "ml_probability": "ML Prob",
            }
            display_df = display_df.rename(columns=rename_map)
            preferred = ["parlay_rank", "League", "Home Team", "Away Team", "Game Date", "Game Time (ET)", "Best Pick", "Prob", "ML Prob", "EV", "Edge", "Consensus", "Kalshi Status"]
            ordered = [c for c in preferred if c in display_df.columns] + [c for c in display_df.columns if c not in preferred]
            display_df = display_df[ordered]
            st.dataframe(display_df, width="stretch")
            export_prep_df = best_picks_df.copy()

            csv_rename_map = {
                "home_team": "Home",
                "away_team": "Away",
                "game_date": "Local Date",
                "game_time_est": "Commence (Local)",
                "ml_probability": "WinProbability"
            }
            export_prep_df = export_prep_df.rename(columns=csv_rename_map)

            target_export_cols = [
                "parlay_rank", "league", "Home", "Away", "Local Date",
                "Commence (Local)", "best_pick", "calibrated_probability", "expected_value",
                "edge", "consensus_agreement", "odds_american", "market_probability",
                "kalshi_probability", "WinProbability", "gemini_explanation", "gemini_risk_notes"
            ]

            final_export_cols = [c for c in target_export_cols if c in export_prep_df.columns]
            best_picks_export = export_prep_df[final_export_cols].copy()

            # Phase 3: Synchronize Kalshi missing strings
            if "kalshi_probability" in best_picks_export.columns:
                best_picks_export["kalshi_probability"] = best_picks_export["kalshi_probability"].fillna("⚪ No Kalshi")

            # Apply explicit secondary sorts before export as requested
            sort_cols = ["expected_value", "Commence (Local)", "league", "Home"]
            available_sort_cols = [c for c in sort_cols if c in best_picks_export.columns]
            if available_sort_cols:
                asc = [False] + [True] * (len(available_sort_cols) - 1)
                best_picks_export = best_picks_export.sort_values(available_sort_cols, ascending=asc).reset_index(drop=True)
                if "parlay_rank" in best_picks_export.columns:
                    best_picks_export["parlay_rank"] = range(1, len(best_picks_export) + 1)

            if "Home" in best_picks_export.columns and not best_picks_export.empty:
                if not best_picks_export["Home"].notna().all():
                    st.warning("Warning: Some rows in the Best Picks export have a missing 'Home' team.")

            if "WinProbability" in best_picks_export.columns and not best_picks_export.empty:
                null_pct = best_picks_export["WinProbability"].isna().mean()
                if null_pct > 0.10:
                    st.warning(f"Warning: {null_pct:.1%} of Best Picks are missing ML Probability. Check upstream ML data flow.")

            best_picks_csv = best_picks_export.to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "Export Best Picks",
                best_picks_csv,
                "best_picks_export.csv",
                mime="text/csv",
            )

    with tab4:
        st.subheader("Best Parlays")
        parlay_columns = ["parlay_type", "parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev", "kelly_fraction_1_8", "legs", "leg1_game", "leg2_game", "leg3_game", "leg4_game", "leg5_game"]
        base_parlays_df = parlays_df if parlays_df is not None else pd.DataFrame(columns=parlay_columns)

        view_mode = st.radio("Parlay View", ["Ranked Parlays", "Top Combinations"], horizontal=True)
        selected_type = "ranked" if view_mode == "Ranked Parlays" else "top_combo"
        filtered = base_parlays_df[_safe_str_series(base_parlays_df, "parlay_type").eq(selected_type)].copy()

        if filtered.empty:
            st.info("No parlays available for this view yet.")
        elif selected_type == "ranked":
            ranked = filtered.sort_values("parlay_ev", ascending=False).reset_index(drop=True)
            for idx, row in ranked.iterrows():
                st.markdown(f"### Parlay #{idx + 1} ({int(row.get('legs', 0))}-Leg)")
                st.markdown(f"- **Combined Probability:** {float(row.get('combined_probability', 0.0)):.2%}")
                st.markdown(f"- **Combined Decimal Odds:** {float(row.get('combined_decimal_odds', 0.0)):.3f}")
                st.markdown(f"- **Parlay EV:** {float(row.get('parlay_ev', 0.0)):.3f}")
                st.markdown(f"- **1/8th Kelly Sizing:** {float(row.get('kelly_fraction_1_8', 0.0)):.2%}")
                legs = [leg.strip() for leg in str(row.get("parlay_legs", "")).split("|") if leg.strip()]
                for leg in legs:
                    st.markdown(f"- {leg}")
                st.divider()
        else:
            top_combo = filtered.sort_values("parlay_ev", ascending=False).head(10).reset_index(drop=True)
            table_df = top_combo[["combined_probability", "combined_decimal_odds", "parlay_ev", "kelly_fraction_1_8", "legs"]].copy()
            table_df["Parlay"] = ["<br>".join([leg.strip() for leg in str(v).split("|") if leg.strip()]) for v in top_combo["parlay_legs"]]
            table_df = table_df[["Parlay", "combined_probability", "combined_decimal_odds", "parlay_ev", "kelly_fraction_1_8", "legs"]]
            st.write(table_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        parlay_csv = base_parlays_df.to_csv(index=False)
        st.download_button(
            "Download Parlays CSV",
            parlay_csv,
            "parlays_export.csv",
            mime="text/csv",
            key="download_parlays_csv",
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
                "league", "away_team", "home_team", "game_time_est", "best_pick",
                "calibrated_probability", "expected_value", "edge", "recommended_bet",
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
        kalshi_matches_tab = len(kalshi_df) if kalshi_df is not None else 0
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
            render_debug(analysis_df, odds_matches, theover_matches, kalshi_matches_tab)
            render_debug_panel(analysis_df, odds_matches, theover_matches, kalshi_matches_tab)
        else:
            st.info("Enable 'Display Debug Information' in the sidebar to inspect debug data.")
        if controls["show_kalshi_diagnostics"]:
            render_kalshi_diagnostics(analysis_df)
            if analysis_df is not None and not analysis_df.empty and "kalshi_match_status" in analysis_df.columns:
                failures_df = analysis_df[
                    analysis_df["kalshi_match_status"].astype(str).str.lower().ne("matched")
                ].copy()
                failure_cols = [
                    "league",
                    "home_team",
                    "away_team",
                    "kalshi_match_status",
                    "kalshi_match_reason",
                ]
                visible_cols = [c for c in failure_cols if c in failures_df.columns]
                with st.expander("Kalshi Match Failures", expanded=False):
                    if failures_df.empty or not visible_cols:
                        st.info("No unmatched Kalshi rows found.")
                    else:
                        st.dataframe(failures_df[visible_cols], width="stretch")

    with tab7:
        render_strategy_lab(
            analysis_df=analysis_df,
            portfolio_df=portfolio_df if portfolio_df is not None else analysis_df.iloc[0:0],
            parlays_df=parlays_df if parlays_df is not None else analysis_df.iloc[0:0],
            simulation_results=simulation_results or {},
        )


if __name__ == "__main__":
    main()
