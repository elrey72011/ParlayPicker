import pandas as pd

from core.streamlit_pipeline import ensure_best_pick_export_columns


def test_compact_alias_preserves_validation_probability():
    source = pd.DataFrame({'WinProbability': [.62, .71]})
    result = ensure_best_pick_export_columns(source)
    assert result.calibrated_probability.tolist() == [.62, .71]
    assert result.WinProbability.tolist() == [.62, .71]
    assert 'calibrated_probability' not in source


def test_existing_canonical_probability_is_not_overridden():
    source = pd.DataFrame({'WinProbability': [.9], 'calibrated_probability': [.6]})
    result = ensure_best_pick_export_columns(source)
    assert result.calibrated_probability.tolist() == [.6]
