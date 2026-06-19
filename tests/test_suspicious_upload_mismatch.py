"""upload_line_market_mismatch should flag a genuine upload/live line mismatch,
not the no-upload default (19 Jun).

Every live_unfiltered row defaults upload_market_match=False; the old guard read that
as a "mismatch" and mislabeled every no-upload pick (e.g. the Houston/Cleveland block
reason). Now it fires only when an upload WAS matched and its line disagrees.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import build_best_picks_df


def _hi_ev_row(idx=1):
    h, a = f"Home{idx}", f"Away{idx}"
    return {
        "league": "MLB", "home_team": h, "away_team": a, "game_date": "2026-06-19",
        "matchup_id": f"2026-06-19|{h}|{a}", "market_type": "spread_home",
        "expected_value": 0.45, "edge": 0.07, "calibrated_probability": 0.62,
        "ml_probability": 0.62, "model_probability": 0.62, "kalshi_probability": 0.55,
        "odds_american": -110, "spread_line": -1.5, "total_line": pd.NA,
        "line_source": "live", "live_spread_line": -1.5, "live_total_line": pd.NA,
        "is_live_data": True, "used_stale_features": False, "odds_source": "odds_api",
    }


def test_no_upload_row_not_flagged_upload_mismatch():
    # The exact misfire: a live row with no uploaded counterpart.
    row = _hi_ev_row()
    row["upload_market_match"] = False
    row["line_delta"] = pd.NA
    out = build_best_picks_df(pd.DataFrame([row]))
    r = out.iloc[0]
    assert "upload_line_market_mismatch" not in str(r["suspicious_data_reasons"])
    assert r["status_blocker_stage"] != "suspicious_data_guardrail"


def test_matched_upload_with_consistent_line_not_flagged():
    row = _hi_ev_row()
    row["upload_market_match"] = True
    row["line_delta"] = 0.0
    out = build_best_picks_df(pd.DataFrame([row]))
    assert "upload_line_market_mismatch" not in str(out.iloc[0]["suspicious_data_reasons"])


def test_matched_upload_with_divergent_line_is_flagged():
    row = _hi_ev_row()
    row["upload_market_match"] = True
    row["line_delta"] = 1.5  # uploaded line disagrees with the live line by 1.5
    out = build_best_picks_df(pd.DataFrame([row]))
    assert "upload_line_market_mismatch" in str(out.iloc[0]["suspicious_data_reasons"])
