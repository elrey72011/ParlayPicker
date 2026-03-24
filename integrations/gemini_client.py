from __future__ import annotations

import pandas as pd


def run_gemini_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Best-effort Gemini annotation that preserves UI behavior when Gemini is unavailable."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    result = df.copy()

    try:
        from app_core.llm_assistant import initialize_gemini, generate_batch_confidence_explanation

        model = initialize_gemini()
        if model is None:
            raise RuntimeError("Gemini model failed to initialize.")

        # Clean data for LLM reasoning
        llm_payload = result.drop(columns=["Local Date", "Commence (Local)"], errors="ignore")
        games_list = llm_payload.to_dict('records') # Convert to List[Dict]

        analyses = generate_batch_confidence_explanation(games_list, model)

        # Since `generate_batch_confidence_explanation` actually returns a dict, let's process it correctly
        if isinstance(analyses, dict) and analyses:
            # Map back to result if `game_id` is present
            if "game_id" in result.columns:
                result["gemini_analysis"] = result["game_id"].astype(str).map(analyses)
            else:
                # If no game_id, try using index or order
                result["gemini_analysis"] = [analyses.get(str(i), {}) for i in range(len(result))]
        elif isinstance(analyses, list) and len(analyses) == len(result):
            result["gemini_analysis"] = analyses
        else:
            result["gemini_analysis"] = "Gemini analysis unavailable"
    except Exception as exc:  # pragma: no cover
        result["gemini_analysis"] = f"Gemini disabled/error: {exc}"

    return result
