import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from streamlit_app import _recompute_consensus_from_kalshi


def test_consensus_no_kalshi_when_unmatched():
    df = pd.DataFrame(
        {
            "calibrated_probability": [0.61],
            "kalshi_probability": [pd.NA],
            "kalshi_match_status": ["miss"],
        }
    )
    out = _recompute_consensus_from_kalshi(df)
    assert out.loc[0, "consensus_agreement"] == "⚪ No Kalshi"


def test_consensus_uses_gap_when_matched():
    df = pd.DataFrame(
        {
            "calibrated_probability": [0.60, 0.45, 0.52],
            "kalshi_probability": [0.54, 0.50, 0.51],
            "kalshi_match_status": ["matched", "matched", "matched"],
        }
    )
    out = _recompute_consensus_from_kalshi(df)
    assert out.loc[0, "consensus_agreement"] == "✅ Agrees"
    assert out.loc[1, "consensus_agreement"] == "❌ Disagrees"
    assert out.loc[2, "consensus_agreement"] == "⚖️ Neutral"
