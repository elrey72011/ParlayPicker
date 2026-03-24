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
        cols_to_drop = ["Local Date", "Commence (Local)", "Commence (UTC)", "Commence UTC"]
        llm_payload = result.drop(columns=cols_to_drop, errors="ignore")

        # Inject a fallback game_id if one doesn't exist
        if "game_id" not in llm_payload.columns:
            llm_payload["game_id"] = [str(i) for i in range(len(llm_payload))]

        games_list = llm_payload.to_dict('records') # Convert to List[Dict]

        # Call with session_state
        analyses = generate_batch_confidence_explanation(games_list, session_state)

        # Unpack dictionary into the two expected export columns
        def unpack_gemini(row_id):
            analysis = analyses.get(str(row_id), {})
            return pd.Series({
                "gemini_explanation": str(analysis.get("explanation", "Gemini analysis unavailable")),
                "gemini_risk_notes": str(analysis.get("risk_notes", ""))
            })

        # Apply unpacking
        temp_cols = pd.Series(llm_payload["game_id"]).apply(unpack_gemini)

        # Merge back to result
        result["gemini_explanation"] = temp_cols["gemini_explanation"]
        result["gemini_risk_notes"] = temp_cols["gemini_risk_notes"]

    except Exception as exc:  # pragma: no cover
        result["gemini_explanation"] = "Gemini analysis unavailable"
        result["gemini_risk_notes"] = ""

    return result
