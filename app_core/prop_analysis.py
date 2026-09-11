"""Explicit prop-only analysis; does not run or replace the game pipeline."""
import logging
import pandas as pd
from app_core.stage_timing import StageTimer

logger = logging.getLogger(__name__)


def run_prop_analysis(controls, session_state, best_picks_df=None, progress=None):
    best_picks_df = best_picks_df if isinstance(best_picks_df, pd.DataFrame) else pd.DataFrame()
    diagnostics = {}
    timer = StageTimer(progress)
    gemini_gate_enabled = bool(controls.get("use_gemini"))
    started = pd.Timestamp.now(tz="UTC")
    timer.start("Player props and prop reviews")
    strikeout_prop_card = pd.DataFrame()
    try:
        from app_core.weights_config import (
            ENABLE_NFL_PLAYER_PROPS,
            ENABLE_STRIKEOUT_PROPS_PRODUCTION,
            STRIKEOUT_PROP_KELLY_PER_PICK_PCT,
            STRIKEOUT_PROP_KELLY_TOTAL_PCT,
            STRIKEOUT_PROP_KELLY_FRACTION,
        )
        if ENABLE_STRIKEOUT_PROPS_PRODUCTION or ENABLE_NFL_PLAYER_PROPS:
            from datetime import datetime
            import pytz
            from app_core.odds_api import TheOddsAPIClient
            from app_core.prop_runner import build_prop_card
            from core.streamlit_pipeline import _get_odds_api_key

            _prop_key = _get_odds_api_key()
            if not _prop_key:
                raise ValueError("Odds API key is missing")
            if _prop_key:
                _prop_date = datetime.now(pytz.timezone("America/New_York")).strftime("%Y-%m-%d")
                _selected_prop_sports = {
                    str(value).strip().upper()
                    for value in controls.get("sports", [])
                }
                _prop_client = TheOddsAPIClient(api_key=_prop_key, markets="h2h")
                _prop_frames = []
                _nfl_game_count = int(
                    best_picks_df.get(
                        "league", pd.Series("", index=best_picks_df.index)
                    ).fillna("").astype(str).str.upper().eq("NFL").sum()
                )
                diagnostics["nfl_prop_requested"] = bool(
                    ENABLE_NFL_PLAYER_PROPS and "NFL" in _selected_prop_sports
                )
                diagnostics["nfl_selected_game_count"] = _nfl_game_count
                if ENABLE_STRIKEOUT_PROPS_PRODUCTION and "MLB" in _selected_prop_sports:
                    timer.start("MLB player props")
                    _mlb_prop_card = build_prop_card(
                        _prop_client,
                        _prop_date,
                        int(_prop_date[:4]),
                        float(controls["bankroll"]),
                        kelly_per_pick_pct=STRIKEOUT_PROP_KELLY_PER_PICK_PCT,
                        kelly_total_pct=STRIKEOUT_PROP_KELLY_TOTAL_PCT,
                        kelly_fraction=STRIKEOUT_PROP_KELLY_FRACTION,
                        prop_results_log=controls.get("prop_results_log"),
                        diagnostics=diagnostics,
                    )
                    if not _mlb_prop_card.empty:
                        _prop_frames.append(_mlb_prop_card)
                if ENABLE_NFL_PLAYER_PROPS and "NFL" in _selected_prop_sports:
                    from app_core.nfl_prop_pipeline import build_nfl_prop_card

                    timer.start("NFL player props")
                    _nfl_prop_card = build_nfl_prop_card(
                        _prop_client,
                        _prop_date,
                        int(_prop_date[:4]),
                        diagnostics=diagnostics,
                    )
                    if not _nfl_prop_card.empty:
                        _prop_frames.append(_nfl_prop_card)
                if _prop_frames:
                    strikeout_prop_card = pd.concat(
                        _prop_frames, ignore_index=True, sort=False
                    )
                    from app_core.nfl_prop_pipeline import attach_nfl_prop_coverage

                    strikeout_prop_card = attach_nfl_prop_coverage(
                        strikeout_prop_card,
                        _selected_prop_sports,
                        diagnostics,
                        nfl_game_count=_nfl_game_count,
                    )
                    if gemini_gate_enabled:
                        from integrations.gemini_client import run_gemini_prop_analysis

                        logger.info(
                            "Firing Gemini API for %s player props...",
                            len(strikeout_prop_card),
                        )
                        timer.start("Gemini prop reviews")
                        strikeout_prop_card = run_gemini_prop_analysis(
                            strikeout_prop_card,
                            session_state,
                        )
                    from app_core.gemini_bet_gate import apply_gemini_bet_gate

                    strikeout_prop_card = apply_gemini_bet_gate(
                        strikeout_prop_card,
                        enabled=gemini_gate_enabled,
                        product="prop",
                        diagnostics=diagnostics,
                    )
                    if gemini_gate_enabled:
                        # Re-label rows after the secondary gate zeroes held stakes.
                        from app_core.prop_runner import apply_prop_stake_status

                        strikeout_prop_card = apply_prop_stake_status(
                            strikeout_prop_card
                        )
                _prop_stake_status = strikeout_prop_card.get(
                    "Stake_Status", pd.Series("", index=strikeout_prop_card.index)
                ).astype(str).str.strip()
                diagnostics["strikeout_prop_actionable_count"] = int(
                    _prop_stake_status.eq("Funded").sum()
                )
                diagnostics["strikeout_prop_research_count"] = int(
                    _prop_stake_status.isin(
                        ["Research / No Stake", "Qualified / No Stake"]
                    ).sum()
                )

    except Exception as exc:  # never let the prop slice break the main card
        logger.warning("strikeout prop card build failed: %s", exc)
        diagnostics["strikeout_prop_error"] = str(exc)
        diagnostics["strikeout_prop_feed_status"] = "unexpected_error"
        diagnostics["strikeout_prop_feed_error_type"] = type(exc).__name__
        _prop_error_detail = type(exc).__name__
        if isinstance(exc, SyntaxError):
            _syntax_file = str(getattr(exc, "filename", "") or "").replace("\\", "/").rsplit("/", 1)[-1]
            _syntax_line = getattr(exc, "lineno", None)
            _prop_error_detail = f"SyntaxError in {_syntax_file or 'unknown file'}:{_syntax_line or '?'}"
        diagnostics["strikeout_prop_error_detail"] = _prop_error_detail

    timer.finish()
    diagnostics["stage_seconds"] = timer.timings
    if diagnostics.get("strikeout_prop_error"):
        raise RuntimeError("Player prop analysis failed (" + diagnostics["strikeout_prop_error_detail"] + "). Previous props have been retained.")
    from app_core.prop_runner import stamp_prop_export
    from core.streamlit_pipeline import PIPELINE_BUILD
    strikeout_prop_card = stamp_prop_export(strikeout_prop_card, PIPELINE_BUILD,
        export_run_id=started.strftime("%Y%m%dT%H%M%S.%fZ"))
    if not strikeout_prop_card.empty:
        if "prediction_generated_at" not in strikeout_prop_card:
            strikeout_prop_card["prediction_generated_at"] = started.isoformat()
        else:
            missing = strikeout_prop_card["prediction_generated_at"].isna() | strikeout_prop_card["prediction_generated_at"].astype(str).str.strip().eq("")
            strikeout_prop_card.loc[missing, "prediction_generated_at"] = started.isoformat()
    timer.start("Build player-prop parlays")
    from app_core.best_duos import build_tiered_prop_parlays
    prop_parlays = build_tiered_prop_parlays(pd.DataFrame(), strikeout_prop_card,
                                           bankroll=float(controls["bankroll"]))
    timer.finish()
    return {"prop_parlays_df": prop_parlays, "strikeout_prop_card": strikeout_prop_card, "props_diagnostics": diagnostics,
            "props_analyzed_at": started.isoformat(), "props_run_sports": list(controls.get("sports", []))}
