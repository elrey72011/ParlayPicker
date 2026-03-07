import os
import sys
import time

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit_app


def test_enrich_with_kalshi_safe_returns_quickly_on_timeout(monkeypatch):
    def _slow_enrich(df):
        time.sleep(0.2)
        out = df.copy()
        out["kalshi_match_status"] = "matched"
        return out

    monkeypatch.setattr(streamlit_app, "_enrich_kalshi_raw", _slow_enrich)
    monkeypatch.setattr(streamlit_app, "KALSHI_ENRICH_TIMEOUT_SECONDS", 0.01)

    df = pd.DataFrame({"league": ["NBA"]})

    start = time.monotonic()
    out, err = streamlit_app._enrich_with_kalshi_safe(df)
    elapsed = time.monotonic() - start

    assert elapsed < 0.15
    assert out.equals(df)
    assert "timed out" in (err or "")
