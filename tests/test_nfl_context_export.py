import pandas as pd
from pregame_selection_fixture import build_pregame_best_picks_df

from core.streamlit_pipeline import (
    BEST_PICK_COLUMNS,
    REQUIRED_BEST_PICK_EXPORT_COLUMNS,
    build_best_picks_df,
)
from test_best_available_candidate_audit import _candidate


def test_nfl_context_fields_survive_best_pick_and_csv_boundaries():
    expected = {
        "feature_home_last_game_summary",
        "feature_away_last_game_summary",
        "injury_home_summary",
        "injury_away_summary",
        "injury_context_status",
        "injury_probability_adjustment",
        "nfl_context_status",
        "nfl_context_model_used",
    }
    assert expected.issubset(BEST_PICK_COLUMNS)
    assert expected.issubset(REQUIRED_BEST_PICK_EXPORT_COLUMNS)


def test_nfl_context_values_survive_candidate_audit_and_best_pick(monkeypatch):
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)
    row = _candidate("spread_home", probability=0.57, ev=0.04, league="NFL")
    row.update(
        ml_probability_source="score-distribution-v1:nfl",
        feature_home_last_game_summary="L 7-27 at PHI (2026-09-13)",
        feature_away_last_game_summary="W 28-20 vs DAL (2026-09-13)",
        injury_home_summary="Puka Nacua (WR) Questionable",
        injury_away_summary="",
        injury_context_status="available",
        injury_probability_adjustment=-0.0165,
        nfl_context_status="complete",
        nfl_context_model_used=True,
    )
    diagnostics = {}

    best = build_pregame_best_picks_df(pd.DataFrame([row]), diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]

    assert best.iloc[0]["feature_home_last_game_summary"].startswith("L 7-27")
    assert "Puka Nacua" in best.iloc[0]["injury_home_summary"]
    assert best.iloc[0]["nfl_context_status"] == "complete"
    assert audit.iloc[0]["feature_away_last_game_summary"].startswith("W 28-20")
    assert audit.iloc[0]["injury_probability_adjustment"] == -0.0165
