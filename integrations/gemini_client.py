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

        analyses = generate_batch_confidence_explanation(result, model)
        if isinstance(analyses, list) and len(analyses) == len(result):
            result["gemini_analysis"] = analyses
        else:
            result["gemini_analysis"] = "Gemini analysis unavailable"
    except Exception as exc:  # pragma: no cover
        result["gemini_analysis"] = f"Gemini disabled/error: {exc}"

    return result
