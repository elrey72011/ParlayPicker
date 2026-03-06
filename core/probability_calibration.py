from __future__ import annotations

import pandas as pd


def calibrate_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Blend model and market probabilities to reduce overconfidence."""
    if df is None or df.empty or "model_probability" not in df.columns:
        return df

    calibrated = df.copy()
    calibrated["calibrated_probability"] = (
        pd.to_numeric(calibrated["model_probability"], errors="coerce").fillna(0.5) * 0.9
        + pd.to_numeric(calibrated.get("market_probability", 0.5), errors="coerce").fillna(0.5) * 0.1
    ).clip(0.0, 1.0)

    return calibrated
