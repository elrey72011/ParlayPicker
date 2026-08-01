import numpy as np
import pandas as pd
import pytest

from core.streamlit_pipeline import compute_blended_probability


def _blend(*, market, model, kalshi=np.nan, theover=np.nan, market_type="total_under"):
    return float(
        compute_blended_probability(
            p_market=pd.Series([market]),
            p_kalshi=pd.Series([kalshi]),
            p_ml=pd.Series([model]),
            p_theover=pd.Series([theover]),
            p_sentiment=pd.Series([0.5]),
            league=pd.Series(["WNBA"]),
            market_type=pd.Series([market_type]),
        ).iloc[0]
    )


def test_wnba_cold_start_model_cannot_split_fallback_weight_with_market():
    probability = _blend(market=0.48, model=0.62)

    # Present-signal renormalization: market 0.70, model 0.15.
    expected = (0.48 * 0.70 + 0.62 * 0.15) / (0.70 + 0.15)
    assert probability == pytest.approx(expected)
    assert probability < 0.51


def test_wnba_cold_start_weights_apply_to_spreads_too():
    probability = _blend(
        market=0.52,
        model=0.68,
        market_type="spread_home",
    )

    expected = (0.52 * 0.70 + 0.68 * 0.15) / (0.70 + 0.15)
    assert probability == pytest.approx(expected)


def test_wnba_kalshi_agreement_keeps_model_as_supporting_vote():
    probability = _blend(market=0.48, model=0.62, kalshi=0.60)

    expected = (0.60 * 0.35 + 0.48 * 0.40 + 0.62 * 0.10) / (
        0.35 + 0.40 + 0.10
    )
    assert probability == pytest.approx(expected)
