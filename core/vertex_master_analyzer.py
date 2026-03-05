from __future__ import annotations

import pandas as pd

try:
    from complete_workflow_implementation import run_ml_predictions, run_master_analysis
except Exception:  # pragma: no cover
    run_ml_predictions = None
    run_master_analysis = None


class VertexMasterAnalyzer:
    def run_master_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        if run_ml_predictions and run_master_analysis:
            ml_df = run_ml_predictions(df)
            return run_master_analysis(df, ml_df, df)

        result = df.copy()
        if "vertex_analysis" not in result.columns:
            result["vertex_analysis"] = "Vertex unavailable in this environment"
        return result
