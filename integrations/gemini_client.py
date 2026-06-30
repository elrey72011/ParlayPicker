from __future__ import annotations

from typing import Any
import pandas as pd


def run_gemini_analysis(df: pd.DataFrame, session_state: Any = None) -> pd.DataFrame:
    """Best-effort Gemini annotation that preserves UI behavior when Gemini is unavailable."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    result = df.copy()

    try:
        from app_core.llm_assistant import generate_batch_confidence_explanation

        # Clean data for LLM reasoning - scrub ALL potential date columns
        cols_to_drop = ["Local Date", "Commence (Local)", "Commence (UTC)", "Commence UTC", "game_date", "game_time_est"]
        llm_payload = result.drop(columns=cols_to_drop, errors="ignore")

        # Inject a fallback game_id if one doesn't exist
        if "game_id" not in llm_payload.columns:
            llm_payload["game_id"] = [str(i) for i in range(len(llm_payload))]

        # Ensure is_live_data exists in payload correctly derived from available fields
        if "is_live_data" not in llm_payload.columns:
            llm_payload["is_live_data"] = False

        # Fix: fallback status should strictly check row-specific states
        if "stats_quality" in llm_payload.columns:
            llm_payload["is_live_data"] = llm_payload["stats_quality"].isin(["REAL", "ESPN"])
        elif "used_stale_features" in llm_payload.columns:
            llm_payload["is_live_data"] = ~llm_payload["used_stale_features"]

        games_list = llm_payload.to_dict('records') # Convert to List[Dict]

        # Call with session_state
        analyses = generate_batch_confidence_explanation(games_list, session_state)

        # Unpack dictionary into the two expected export columns based on game_id
        # Use result.index to ensure alignment
        analyses_results = [analyses.get(str(gid), {}) for gid in llm_payload["game_id"]]

        explanations = []
        risk_notes = []
        picks = []
        for res in analyses_results:
            expl = res.get("explanation")
            if expl is None or expl == "":
                expl = "Gemini analysis unavailable"

            risk = res.get("risk_notes")
            if risk is None or risk == "":
                # The user specified defaulting gemini_risk_notes to "Gemini analysis unavailable" if it's missing
                risk = "Gemini analysis unavailable"

            pick = res.get("recommended_bet")
            if pick is None or pick == "":
                pick = "No Gemini pick"

            explanations.append(str(expl))
            risk_notes.append(str(risk))
            picks.append(str(pick))

        result["gemini_explanation"] = explanations
        result["gemini_risk_notes"] = risk_notes
        result["gemini_pick"] = picks

    except Exception as exc:  # pragma: no cover
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Gemini integration mapping failed: {exc}", exc_info=True)
        result["gemini_explanation"] = "Gemini analysis unavailable"
        result["gemini_risk_notes"] = "Gemini analysis unavailable"
        result["gemini_pick"] = "No Gemini pick"

    return result
