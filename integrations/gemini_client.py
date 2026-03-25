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

        games_list = llm_payload.to_dict('records') # Convert to List[Dict]

        # Call with session_state
        analyses = generate_batch_confidence_explanation(games_list, session_state)

        # Unpack dictionary into the two expected export columns based on game_id
        explanations = []
        risk_notes = []

        for idx, row in llm_payload.iterrows():
            g_id = str(row.get("game_id", ""))
            analysis = analyses.get(g_id, {})
            explanations.append(str(analysis.get("explanation", "Gemini analysis unavailable")))
            risk_notes.append(str(analysis.get("risk_notes", "")))

        # Assign back to result dataframe directly to avoid index/alignment issues
        result["gemini_explanation"] = explanations
        result["gemini_risk_notes"] = risk_notes

    except Exception as exc:  # pragma: no cover
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Gemini integration mapping failed: {exc}", exc_info=True)
        result["gemini_explanation"] = "Gemini analysis unavailable"
        result["gemini_risk_notes"] = ""

    return result
