Exit code: 0
Wall time: 1.1 seconds
Total output lines: 2469
Output:
from __future__ import annotations

import traceback
import warnings
from typing import Any

import logging
import sys

logger = logging.getLogger(__name__)
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from core.streamlit_pipeline import (
    generate_parlays,
    optimize_portfolio_allocation,
    run_analysis_pipeline,
    run_bankroll_simulation,
    CANONICAL_BET_COLUMNS,
    VALID_MARKETS,
    MIN_EDGE_THRESHOLD,
    ensure_best_pick_export_columns,
    REQUIRED_BEST_PICK_EXPORT_COLUMNS,
    apply_no_bet_pick_quality,
)
from core.team_normalizer import normalize_team
from core.theover_loader import load_theover_csv
from app_core.weights_config import (
    ENABLE_EMPTY_CARD_RECOVERY,
    EMPTY_CARD_RECOVERY_MAX_PICKS,
    EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EV,
    EMPTY_CARD_RECOVERY_MIN_PRODUCTION_EDGE,
    EMPTY_CARD_RECOVERY_MIN_PRODUCTION_WIN_PROB,
    EMPTY_CARD_RECOVERY_EXCLUDE_MARKET_TYPES,
    EMPTY_CARD_RECOVERY_EXCLUDE_SOURCES,
    EMPTY_CARD_RECOVERY_MAX_KELLY_TOTAL_PCT,
    EMPTY_CARD_RECOVERY_MAX_KELLY_PER_PICK_PCT,
    ALLOW_MLB_TOTAL_OVER_EMPTY_CARD_RECOVERY,
)

try:
    from app_core.kalshi_integrator import enrich_with_kalshi_markets as _enrich_kalshi_raw
except Exception:  # pragma: no cover
    _enrich_kalshi_raw = None  # type: ignore[assignment]

try:
    from app_core.odds_api import OddsAPIAuthError
except ImportError:
    class OddsAPIAuthError(Exception): pass


KALSHI_ENRICH_TIMEOUT_SECONDS = 450


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
        failing_game_ids: list[str] = []
        try:
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "game_id" in df.columns:
                    failing_game_ids = (
                        df["game_id"].dropna().astype(str).head(5).tolist()
                    )
                elif {"league", "home_team", "away_team", "game_date"}.issubset(df.columns):
                    preview = (
                        df[["league", "home_team", "away_team", "game_date"]]
                        .head(3)
                        .to_dict(orient="records")
                    )
                    logger.error("Kalshi enrichment failed on rows preview: %s", preview)
        except Exception:
            failing_game_ids = []

        logger.error(
            "Kalshi enrichment failed: %s | sample_game_ids=%s",
            e,
            failing_game_ids,
            exc_info=True,
        )
        return df, f"Kalshi enrichment failed: {e}\n{traceback.format_exc()}"
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _safe_str_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    if df is None:
        return pd.Series(dtype="string")
    if df.empty:
        return pd.Series(index=df.index, dtype="string")
    if col in df.columns:
        series = df[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype("object")
        return series.astype("string").fillna(default)
    return pd.Series([default] * len(df), index=df.index, dtype="string")






_NO_BET_STAGE_REASONS = {
    "game_already_started": "⏱️ Game already started — live odds, not pre-game lines",
    "baseline_guardrail": "Market priced better — negative EV at this price",
    "actionable_threshold": "Failed the confidence threshold",
    "min_win_probability_floor": "Below the 55% win-probability floor",
    "empirical_proven_losing_bucket": "Proven-losing bucket — history says pass",
    "empirical_tier_overlay": "Small-sample bucket — not enough history to trust",
    "divergence_viability_floor": "Signals diverge and the pick failed the viability floor",
    "kalshi_wrong_game_title": "Kalshi data mismatch (wrong game)",
    "line_provenance_unresolved": "Could not verify the betting line",
    "extreme_price_guard": "Odds moved to an extreme price",
}


# User-facing status vocabulary (owner, 4 Jul): "No Play" reads like the game is
# off the board when it means "no edge at this price", and most of the card
# carried it. Everything the user READS gets these labels; the internal gate
# names ("No Play"/"Below Threshold"/...) stay untouched inside the pipeline,
# where grading history and gate logic key off them.
STATUS_DISPLAY_LABELS = {
    "No Play": "No Edge",
    "Below Threshold": "Near Miss",
    "High Variance/Speculative": "Unproven",
    "Fallback / Low Confidence": "Low Data",
}


def apply_status_display_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map internal status names to the user-facing vocabulary (display/export only)."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "Pick_Status" in out.columns:
        out["Pick_Status"] = out["Pick_Status"].astype(str).replace(STATUS_DISPLAY_LABELS)
    for col in ("Pick_Quality", "Pick Quality"):
        if col in out.columns:
            s = out[col].astype(str)
            for old, new in STATUS_DISPLAY_LABELS.items():
                s = s.str.replace(old, new, regex=False)
            out[col] = s
    return out


def _friendly_no_bet_reason(row: pd.Series | dict) -> str:
    """One plain-English line for why a non-Actionable pick isn't bettable.

    Display-only: collapses the status ladder (No Play / Below Threshold /
    High Variance) into a single 'why', keyed on status_blocker_stage with the
    raw Status_Reason as fallback. Exports and grading keep the raw statuses.
    """
    get = row.get if hasattr(row, "get") else lambda k, d="": d
    stage = str(get("status_blocker_stage", "") or "").strip()
    if stage in _NO_BET_STAGE_REASONS:
        return _NO_BET_STAGE_REASONS[stage]
    reason = str(get("Status_Reason", "") or get("status_blocker_reason", "") or "").strip()
    return reason if reason and reason.lower() != "nan" else "Did not qualify"


def _should_run_pipeline(state: dict[str, Any], run_counter: int, controls: dict[str, Any] | None = None) -> bool:
    """Run once per monotonically increasing sidebar run counter.

    Also guard against accidental duplicate reruns in the same click cycle by
    de-duplicating identical control signatures for the same run counter.
    """
    last_processed = int(state.get("last_processed_run_counter", 0))
    if run_counter <= last_processed:
        return False
    if controls is not None:
        signature = (
            int(run_counter),
            tuple(sorted(str(s) for s in controls.get("sports", []))),
            bool(controls.get("use_ml")),
            bool(controls.get("use_gemini")),
            float(controls.get("bankroll", 0.0)),
            bool(controls.get("theover_spreads") is not None),
            bool(controls.get("theover_totals") is not None),
        )
        if signature == state.get("last_pipeline_signature"):
            logger.info("PIPELINE DIAGNOSTIC: Suppressing duplicate pipeline invocation for identical run signature.")
            return False
        state["last_pipeline_signature"] = signature
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


def _et_floor_day(series: pd.Series) -> pd.Series:
    """Normalize any datetime-like series to ET day boundaries for deterministic joins."""
    return (
        pd.to_datetime(series, errors="coerce", utc=True)
        .dt.tz_convert("America/New_York")
        .dt.floor("D")
    )


def _compose_model_probability(out: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build model probability using market-aware fallback logic.

    Returns a tuple of (model_probability, ml_probability, theover_probability).
    """
    ml = _safe_numeric_series(out, "ml_probability")
    theover = _safe_numeric_series(out, "theover_probability")

    # Apply robust normalization if data format shifts, convert to probability [0, 1]
    # The previous logic only divided by 100 if theover > 1.
    theover = theover.where(theover <= 1.0, theover / 100.0)

    market_type = _safe_str_series(out, "market_type").str.lower()

    # Reject known broken XGBoost baseline default score when feature matrix collapses.
    is_broken_ml = (ml > 0.19063) & (ml < 0.19064)

    # Log a WARNING with matchup_ids for future retraining tracking
    if is_broken_ml.any():
        broken_matchups = out.loc[is_broken_ml, "matchup_id"].unique() if "matchup_id" in out.columns else []
        logger.warning(f"⚠️ Trapped broken XGBoost scores (0.19063-0.19064) for matchups: {broken_matchups}. Discarding ML predictions and forcing Statistical Fallback.")

    ml_clean = ml.where(~is_broken_ml, pd.NA)

    spread_model = ml_clean.where(ml_clean.notna(), theover)

    # Bayesian Updating for Totals Markets
    # Replace hard hierarchy with a weighted average: 0.6 * theover_probability + 0.4 * ml_clean
    total_model = (0.6 * theover) + (0.4 * ml_clean)

    # If one of them is NA, fallback to the other
    total_model = total_model.where(total_model.notna(), theover.where(theover.notna(), ml_clean))

    model_probability = pd.Series(
        pd.NA,
        index=out.index,
        dtype="Float64",
    )
    is_spread = market_type.str.startswith("spread")
    model_probability = model_probability.where(~is_spread, spread_model)
    model_probability = model_probability.where(is_spread, total_model)
    return model_probability.astype("float64"), ml, theover




def _recompute_consensus_from_kalshi(df: pd.DataFrame, require_ml: bool = False) -> pd.DataFrame:
    """Set consensus based on Kalshi availability and probability gap, and update blends."""
    if df is None or df.empty:
        return df
    out = df.copy()

    # Recalculate blended probability and EV/Edge since we might have new Kalshi probabilities
    from core.streamlit_pipeline import compute_blended_probability, _scoped_theover_blend_fade

    kalshi_prob = _safe_numeric_series(out, "kalshi_probability")
    market_prob = _safe_numeric_series(out, "market_probability")

    ml = _safe_numeric_series(out, "ml_probability")
    is_broken_ml = (ml > 0.19063) & (ml < 0.19064)

    if is_broken_ml.any():
        broken_matchups = out.loc[is_broken_ml, "matchup_id"].unique() if "matchup_id" in out.columns else []
        logger.warning(f"⚠️ Consensus Step: Trapped broken XGBoost scores (0.19063-0.19064) for matchups: {broken_matchups}.")

    ml_valid = ml.where(~is_broken_ml, pd.NA)

    # Handle the two variations of model prob stored depending on df origin
    if "model_probability" in out.columns:
        model_prob = _safe_numeric_series(out, "model_probability")
    else:
        # Fallback to market-aware ml/theover composition used in the analysis pipeline
        model_prob, _, _ = _compose_model_probability(out)

    if require_ml and ml_valid.notna().sum() == 0:
        raise ValueError("ML predictions failed to merge with the analysis dataframe.")

    # Extract the missing features required for the new Tiered Weight system
    theover_prob = _safe_numeric_series(out, "theover_probability")
    sentiment_prob = _safe_numeric_series(out, "sentiment_diff", default=0.5)

    # FADE genuine-but-cold TheOver sources (model_hit_rate_flipped) in this re-blend,
    # mirroring run_analysis_pipeline. This post-Kalshi refresh recomputes
    # calibrated_probability/EV/blend_in_theover, so it must apply the same fade or it
    # would re-inject TheOver's full Under value and undo the tempering before selection.
    _src_col = out["win_prob_source"] if "win_prob_source" in out.columns else None
    # Mirror run_analysis_pipeline: MLB-tuned fade shrink only on MLB totals (shared helper).
    theover_prob_blend = _scoped_theover_blend_fade(
        theover_prob.to_numpy(dtype=float),
        _src_col,
        _safe_str_series(out, "league"),
        _safe_str_series(out, "market_type"),
        theover_prob.index,
    )

    blended = compute_blended_probability(
        p_market=market_prob,
        p_kalshi=kalshi_prob,
        p_ml=model_prob,
        p_theover=theover_prob_blend,
        p_sentiment=sentiment_prob,
        league=_safe_str_series(out, "league"),
        market_type=_safe_str_series(out, "market_type")
    )

    # Update core metrics
    out["calibrated_probability"] = blended

    # Refresh blend-input metadata to reflect the final post-Kalshi blend.
    # blend_in_* and blend_tier were recorded in run_analysis_pipeline before
    # live Kalshi was available; now that kalshi_prob is populated, update them
    # so the export/backtest columns accurately describe the actual blend used.
    import numpy as _np
    out["blend_in_kalshi"] = kalshi_prob
    out["blend_in_market"] = market_prob
    out["blend_in_ml"] = model_prob
    out["blend_in_theover"] = theover_prob_blend
    out["blend_tier"] = _np.where(
        kalshi_prob.fillna(0.0) >= 0.55, 1, 2
    )

    # Check if the Hard Safety Net was used (e.g., probability is exactly 0.5 for all and there's a note)
    # Since ml_valid might be filled with 0.5 from fallback:
    if "ml_probability" in out.columns and (out["ml_probability"] == 0.5).all() and len(out) > 0:
        logger.warning(
            "Hard Safety Net (Neutral Fallback 0.5) triggered for predictions. "
            "Please check the ML engine logs for the specific missing features that caused the matrix to be mostly empty."
        )

    from core.streamlit_pipeline import american_to_decimal, american_to_prob, get_opposing_odds_from_exchange

    # Always re-derive decimal_odds and market_probability from odds_american to ensure
    # we don't carry stale values after odds patching (e.g. fallback_novig).
    if "odds_american" in out.columns:
        odds_american = _safe_numeric_series(out, "odds_american")
        decimal_odds = odds_american.apply(american_to_decimal)

        # Re-derive market_probability if needed for edge
        implied_prob = odds_american.apply(american_to_prob)
        opposing_implied = odds_american.apply(get_opposing_odds_from_exchange).apply(american_to_prob)
        market_prob = implied_prob / (implied_prob + opposing_implied)
        out["market_probability"] = market_prob
    else:
        decimal_odds = _safe_numeric_series(out, "decimal_odds")

    if decimal_odds.isna().all():
        logger.warning("⚠️ All odds are missing - using default 1.91 (-110 equivalent)")
        decimal_odds = pd.Series([1.91] * len(out), index=out.index)
    # Row-level fallback: ensure every bet row has actionable decimal odds.
    decimal_odds = pd.to_numeric(decimal_odds, errors="coerce").fillna(1.91)

    out["decimal_odds"] = decimal_odds

    # Re-apply the market-anchored MLB total de-bias (#1919/#1921) here too. This
    # post-Kalshi refresh re-blends calibrated_probability and recomputes EV/edge from
    # scratch, so without re-running the correction it silently UNDOES the de-bias that
    # run_analysis_pipeline applied — the card then rebuilds on the raw model over-lean
    # and direction selection never rebalances (the 14 Jun all-Over card: model P(over)
    # ~0.55 vs de-vig market ~0.46 across 15 games, de-bias 0.0747 never reaching the
    # card). Same shared helper as both pipeline paths, so the three cannot drift. Runs
    # before EV/edge/consensus below so all of them see the corrected probability.
    from core.streamlit_pipeline import apply_mlb_total_market_debias
    blended, _mlb_total_debias = apply_mlb_total_market_debias(blended, out)
    out["calibrated_probability"] = blended
    if abs(_mlb_total_debias) > 1e-9:
        out["mlb_total_market_debias"] = _mlb_total_debias

    out["expected_value"] = blended * (decimal_odds - 1) - (1 - blended)
    out["edge"] = blended - market_prob

    status = _safe_str_series(out, "kalshi_match_status").str.lower()

    out["consensus_agreement"] = "No Kalshi"
    valid_kalshi = (kalshi_prob.notna() & (kalshi_prob > 0.0)).fillna(False)
    gap = blended - kalshi_prob

    out.loc[valid_kalshi, "consensus_agreement"] = "Neutral"
    # "Agrees": Kalshi also favors pick direction (P(pick) >= 50%) AND model is more confident.
    # kalshi_probability is pre-oriented to the pick side (P(Under) for Under rows, P(Over)
    # for Over rows) by kalshi_integrator. A value < 0.50 means Kalshi says the OTHER side
    # is more likely — that is Disagrees, not Agrees, regardless of the probability gap.
    out.loc[(valid_kalshi & gap.ge(0.03) & kalshi_prob.ge(0.50)).fillna(False), "consensus_agreement"] = "Agrees"
    # "Disagrees": Kalshi says other direction (P(pick) < 50%) OR Kalshi more confident than model.
    out.loc[(valid_kalshi & (gap.le(-0.03) | kalshi_prob.lt(0.50))).fillna(False), "consensus_agreement"] = "Disagrees"

    # Debug log for probability blend verification (first 5 picks)
    if not out.empty and "market_probability" in out.columns:
        debug_sample = out.head(5)
        for idx, row in debug_sample.iterrows():
            logger.info(
                f"Blend Debug | Pick: {row.get('best_pick', '')} | "
                f"Market: {row.get('market_probability')}, "
                f"Kalshi: {row.get('kalshi_probability')}, "
                f"ML: {row.get('ml_probability')} | "
                f"Blended: {row.get('calibrated_probability')}"
            )

    # Do not filter out rows from the master analysis_df based on edge or expected value.
    # The frontend grids must display the entire master schedule.
    # We leave the strict edge/EV filtering strictly for best_picks_df construction.

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

    def _mk_matchup_id(df: pd.DataFrame) -> pd.Series:
        home = df["home_team"].astype(str).str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True).str.strip()
        away = df["away_team"].astype(str).str.lower().str.replace(r"[^a-z0-9\s]", "", regex=True).str.strip()
        team_a = home.where(home <= away, away)
        team_b = away.where(home <= away, home)
        date = pd.to_datetime(df["game_date"], errors="coerce", utc=True)
        date_key = pd.Series([""] * len(df), index=df.index, dtype="string")
        valid = date.notna()
        if valid.any():
            date_key.loc[valid] = date[valid].dt.tz_convert("America/New_…25343 tokens truncated…, exc)

            # ── MLB player props (separate softer-market card) ──
            prop_card = st.session_state.get("strikeout_prop_card")
            if prop_card is not None and not prop_card.empty:
                st.subheader("⚾ MLB Player Props — Pitchers · Batters")
                st.caption(
                    "Pitcher strikeouts plus batter hits/total bases that cleared the +EV and "
                    "minimum-edge gates. New batter markets start at a $1 maximum and stay out "
                    "of strict parlays until 20 graded results at 55%+."
                )
                st.dataframe(prop_card, width="stretch")
                st.download_button(
                    "Export MLB Player Props",
                    prop_card.to_csv(index=False, encoding="utf-8-sig"),
                    "mlb_player_props_export.csv",
                    mime="text/csv",
                )
            elif st.session_state.get("diagnostics", {}).get("strikeout_prop_error"):
                _prop_err_type = st.session_state.get("diagnostics", {}).get(
                    "strikeout_prop_feed_error_type", "unexpected error"
                )
                st.caption(
                    f"⚾ Strikeout props unavailable after retry ({_prop_err_type}). "
                    "Run the slate again; the game card remains valid."
                )
            elif st.session_state.get("diagnostics", {}).get("strikeout_prop_feed_status") in {
                "event_list_failed", "prop_fetch_failed"
            }:
                _prop_stage = st.session_state.get("diagnostics", {}).get("strikeout_prop_feed_status")
                st.caption(
                    f"⚾ Pitcher-prop feed did not respond after retry ({_prop_stage}). "
                    "Run the slate again; no stale props were used."
                )
            else:
                st.caption("⚾ No MLB player props cleared the edge bar today.")

            # Compact export (.xlsx): a readable Excel table with only the columns
            # needed to scan a slate left-to-right, matching the Strategy Lab layout.
            # Win Amount is auto-computed from the Kelly stake and odds; W/L is left
            # blank for manual entry; Total Amount is a P&L formula that resolves
            # once W/L is filled in (+Win Amount on a win, -stake on a loss).
            from io import BytesIO
            import openpyxl
            from openpyxl.worksheet.table import Table, TableStyleInfo
            from openpyxl.utils import get_column_letter
            from openpyxl.styles import Font

            compact_cols = [
                "WinProbability", "expected_value", "edge",
                "Conviction_Score", "market_probability", "kalshi_probability", "ml_probability",
                "effective_expected_value", "effective_edge", "effective_win_probability",
                "consensus_agreement", "Play_Tier", "Play_Stake", "Pick_Status", "Pick_Quality", "league", "Home", "Away",
                "Commence (Local)", "odds_american", "best_pick", "gemini_pick", "Kelly_Bet_Size",
            ]
            available_compact_cols = [c for c in compact_cols if c in best_picks_export.columns]
            compact_export = best_picks_export[available_compact_cols].copy()
            # Surface the Novig price for the pick as "Novig Line" (the line/total itself is
            # already encoded in best_pick). Sits between "Commence (Local)" and "best_pick"
            # per its position in compact_cols above. odds_american is the Novig price only
            # when odds_source is a genuine Novig quote (odds_api / novig); for uploaded or
            # synthetic-fallback (fallback_novig = -110) rows it is NOT a Novig price, so
            # blank those cells rather than mislabel a non-Novig price as Novig.
            if "odds_american" in compact_export.columns:
                compact_export = compact_export.rename(columns={"odds_american": "Novig Line"})
                if "odds_source" in best_picks_export.columns:
                    _is_novig = best_picks_export["odds_source"].astype(str).str.strip().str.lower().isin({"odds_api", "novig"})
                    compact_export.loc[~_is_novig, "Novig Line"] = pd.NA

            # Win Amount and W/L are left blank for manual entry. Total Amount is a
            # per-row P&L formula (filled below) that resolves once they're entered.
            compact_export["Win Amount"] = ""
            compact_export["W/L"] = ""
            compact_export["Total Amount"] = ""

            final_cols = list(compact_export.columns)
            pct_cols = {
                "WinProbability", "expected_value", "edge", "Conviction_Score",
                "market_probability", "kalshi_probability", "ml_probability",
                "effective_expected_value", "effective_edge", "effective_win_probability",
            }
            money_cols = {"Kelly_Bet_Size", "Win Amount", "Total Amount"}
            odds_cols = {"Novig Line"}  # American odds, shown with explicit sign (e.g. +109 / -114)

            def _col_num(x):
                v = pd.to_numeric(x, errors="coerce")
                return float(v) if pd.notna(v) else None

            def _letter(name):
                return get_column_letter(final_cols.index(name) + 1)

            wl_L = _letter("W/L")
            win_L = _letter("Win Amount")
            kelly_L = _letter("Kelly_Bet_Size") if "Kelly_Bet_Size" in final_cols else None

            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Best Picks"
            ws.append(final_cols)

            for i, (_, row) in enumerate(compact_export.iterrows()):
                excel_row = i + 2  # row 1 is the header
                values = []
                for col in final_cols:
                    if col == "Total Amount":
                        values.append(
                            f'=IF({wl_L}{excel_row}="W",{win_L}{excel_row},'
                            f'IF({wl_L}{excel_row}="L",-{kelly_L}{excel_row},""))'
                            if kelly_L else None
                        )
                    elif col == "W/L":
                        values.append(None)
                    elif col in pct_cols or col in money_cols or col in odds_cols:
                        values.append(_col_num(row[col]))
                    else:
                        v = row[col]
                        values.append(None if pd.isna(v) else v)
                ws.append(values)

            last_row = len(compact_export) + 1
            money_fmt = "#,##0"
            pnl_fmt = "#,##0;(#,##0)"  # negatives shown in parentheses, e.g. (2,500)
            for idx, col in enumerate(final_cols, start=1):
                letter = get_column_letter(idx)
                if col in pct_cols:
                    fmt = "0.0%"
                elif col == "Total Amount":
                    fmt = pnl_fmt
                elif col in money_cols:
                    fmt = money_fmt
                elif col in odds_cols:
                    fmt = "+0;-0"  # American odds: +109 / -114
                else:
                    fmt = None
                if fmt:
                    for r in range(2, last_row + 1):
                        ws[f"{letter}{r}"].number_format = fmt
                ws.column_dimensions[letter].width = max(12, min(30, len(col) + 2))

            if len(compact_export) > 0:
                ref = f"A1:{get_column_letter(len(final_cols))}{last_row}"
                table = Table(displayName="BestPicks", ref=ref)
                table.tableStyleInfo = TableStyleInfo(
                    name="TableStyleMedium2", showRowStripes=True, showColumnStripes=False
                )
                ws.add_table(table)

                # Summary rows below the table: "Actionable Totals" (Actionable tier
                # only) and "Totals" (all picks). All values are live formulas over the
                # data range, so they update as Win Amount / W/L are filled in. Net P&L
                # counts a win as +Win Amount and anything not yet won as -stake (money
                # at risk), matching the Strategy Lab convention.
                label_col = "best_pick" if "best_pick" in final_cols else final_cols[0]
                status_L = _letter("Pick_Status") if "Pick_Status" in final_cols else None
                stake_rng = f"{kelly_L}2:{kelly_L}{last_row}"
                win_rng = f"{win_L}2:{win_L}{last_row}"
                wl_rng = f"{wl_L}2:{wl_L}{last_row}"

                def _write_summary(excel_row, label, kelly_f, win_f, wl_f, tot_f):
                    c = ws.cell(row=excel_row, column=final_cols.index(label_col) + 1, value=label)
                    c.font = Font(bold=True)
                    kc = ws.cell(row=excel_row, column=final_cols.index("Kelly_Bet_Size") + 1, value=kelly_f)
                    kc.number_format = money_fmt
                    kc.font = Font(bold=True)
                    wc = ws.cell(row=excel_row, column=final_cols.index("Win Amount") + 1, value=win_f)
                    wc.number_format = money_fmt
                    wc.font = Font(bold=True)
                    rc = ws.cell(row=excel_row, column=final_cols.index("W/L") + 1, value=wl_f)
                    rc.font = Font(bold=True)
                    tc = ws.cell(row=excel_row, column=final_cols.index("Total Amount") + 1, value=tot_f)
                    tc.number_format = pnl_fmt
                    tc.font = Font(bold=True)

                if kelly_L and status_L:
                    act = f'{status_L}2:{status_L}{last_row}'
                    _write_summary(
                        last_row + 2,
                        "Actionable Totals",
                        f'=SUMIF({act},"Actionable",{stake_rng})',
                        f'=SUMIF({act},"Actionable",{win_rng})',
                        f'=COUNTIFS({act},"Actionable",{wl_rng},"W")&"-"&COUNTIFS({act},"Actionable",{wl_rng},"L")',
                        f'=SUMIFS({win_rng},{act},"Actionable",{wl_rng},"W")'
                        f'-SUMIF({act},"Actionable",{stake_rng})'
                        f'+SUMIFS({stake_rng},{act},"Actionable",{wl_rng},"W")',
                    )

                # 🏆 Pick of the Day line so the export carries the day's single
                # best cross-board pick (games + props) alongside the table.
                if _potd is not None:
                    _potd_cell = ws.cell(
                        row=last_row + 4 if kelly_L else last_row + 2,
                        column=final_cols.index(label_col) + 1,
                        value=(
                            f"🏆 Pick of the Day: {_potd['pick']} "
                            f"({_potd['win_probability']:.1%} win prob, ${_potd['stake']:.2f}"
                            f"{', BELOW FLOOR — bet light' if _potd['below_floor'] else ''})"
                        ),
                    )
                    _potd_cell.font = Font(bold=True)

                if kelly_L:
                    _write_summary(
                        last_row + 3,
                        "Totals",
                        f'=SUM({stake_rng})',
                        f'=SUM({win_rng})',
                        f'=COUNTIF({wl_rng},"W")&"-"&COUNTIF({wl_rng},"L")',
                        f'=SUMIF({wl_rng},"W",{win_rng})-SUM({stake_rng})+SUMIF({wl_rng},"W",{stake_rng})',
                    )

            buf = BytesIO()
            wb.save(buf)
            st.download_button(
                "Export Best Picks (Compact)",
                buf.getvalue(),
                "best_picks_compact.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="export_best_picks_compact",
            )

    with tab4:
        st.subheader("Best Parlays")
        base_parlays_df = parlays_df if parlays_df is not None and not parlays_df.empty else pd.DataFrame()

        view_mode = st.radio("Parlay View", ["Ranked Parlays", "Top Combinations"], horizontal=True)

        if base_parlays_df.empty:
            st.info("No parlays available for this view yet.")
        elif view_mode == "Ranked Parlays":
            ranked = base_parlays_df.sort_values("parlay_ev", ascending=False).reset_index(drop=True)
            for idx, row in ranked.iterrows():
                tier = row.get("risk_tier", "")
                tier_label = f" — {tier}" if tier else ""
                st.markdown(f"### Parlay #{idx + 1} ({int(row.get('legs', 0))}-Leg{tier_label})")
                st.markdown(f"- **Combined Probability:** {float(row.get('combined_probability', 0.0)):.2%}")
                st.markdown(f"- **Combined Decimal Odds:** {float(row.get('combined_decimal_odds', 0.0)):.3f}")
                st.markdown(f"- **Parlay EV:** {float(row.get('parlay_ev', 0.0)):.3f}")
                kf = row.get("kelly_fraction", 0.0)
                st.markdown(f"- **Kelly Fraction (1/8):** {float(kf) if pd.notna(kf) else 0.0:.2%}")
                legs = [leg.strip() for leg in str(row.get("parlay_legs", "")).split("|") if leg.strip()]
                for leg in legs:
                    st.markdown(f"- {leg}")
                st.divider()
        else:
            top_combo = base_parlays_df.sort_values("parlay_ev", ascending=False).head(10).reset_index(drop=True)
            table_cols = [c for c in ["combined_probability", "combined_decimal_odds", "parlay_ev", "kelly_fraction", "legs", "risk_tier"] if c in top_combo.columns]
            table_df = top_combo[table_cols].copy()
            table_df.insert(0, "Parlay", ["<br>".join([leg.strip() for leg in str(v).split("|") if leg.strip()]) for v in top_combo["parlay_legs"]])
            st.write(table_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # Rearrange columns for the export
        parlay_export_columns = [
            "risk_tier", "group_id", "parlay_legs", "combined_probability", "combined_decimal_odds",
            "parlay_ev", "legs", "combined_market_prob", "ev_boost_pct", "is_high_correlation",
            "best_payout_book", "Conviction_Score", "min_leg_prob", "kelly_fraction", "recommended_bet"
        ]
        export_parlays_df = base_parlays_df.copy() if not base_parlays_df.empty else pd.DataFrame(columns=parlay_export_columns)

        for col in parlay_export_columns:
            if col not in export_parlays_df.columns:
                export_parlays_df[col] = pd.NA

        export_parlays_df = export_parlays_df[parlay_export_columns]

        parlay_csv = export_parlays_df.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            "Export Strategic Parlays",
            parlay_csv,
            "smart_parlays_export.csv",
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
            best_picks_df=best_picks_df,
            portfolio_df=portfolio_df if portfolio_df is not None else analysis_df.iloc[0:0],
            parlays_df=parlays_df if parlays_df is not None else analysis_df.iloc[0:0],
            simulation_results=simulation_results or {},
        )


if __name__ == "__main__":
    main()

