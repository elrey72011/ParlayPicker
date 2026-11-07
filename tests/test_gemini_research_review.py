"""Gemini role #1 (owner 2026-11-07): research review of the top picks, never a stake."""
import pandas as pd

from app_core import gemini_research_review as grr


def _best():
    return pd.DataFrame({
        "matchup_id": ["g1", "g2", "g3", "g4"], "market_type": ["total_under", "spread_home", "spread_away", "total_over"],
        "best_pick": ["Under 152.5", "Uic -10.5", "Stony Brook +12.5", "Over 140.5"],
        "pinnacle_ev": [0.025, 0.02, float("nan"), float("nan")], "expected_value": [-0.01, -0.02, -0.004, 0.01],
        "blend_in_kalshi": [float("nan"), 0.48, float("nan"), float("nan")], "kalshi_probability": [0.75, 0.48, 0.6, 0.5],
        "candidate_context": ["CURRENT_PREGAME", "CURRENT_PREGAME", "CURRENT_PREGAME", "POST_START_DIAGNOSTIC"],
        "Play_Stake": [0.0, 0.0, 0.0, 0.0], "gemini_review_status": ["SKIPPED"] * 4})


def test_selects_pinnacle_value_first_then_ev_pregame_only():
    assert list(grr.select(_best(), 2)) == [0, 1]
    assert 3 not in list(grr.select(_best(), 10))           # started game never reviewed


def test_review_writes_only_research_columns_and_passes_gated_kalshi():
    seen = {}

    def fake_run(targets, session_state, analysis, eligible_only, extra_columns):
        seen["kalshi"] = targets["kalshi_probability"].tolist()
        seen["extra"] = extra_columns
        out = targets.copy()
        out["gemini_pick"] = ["Under 152.5", "Robert Morris +10.5", "Stony Brook +12.5"][:len(out)]
        out["gemini_confidence"] = ["MEDIUM", "LOW", "HIGH"][:len(out)]
        out["gemini_flags"] = ""; out["gemini_explanation"] = "x"; out["gemini_risk_notes"] = "y"
        out["gemini_reviewed"] = True; out["gemini_reviewed_at"] = "t"; out["gemini_review_model"] = "m"
        return out
    best = _best()
    out, diag = grr.review(best, enabled=True, setting=lambda *a: "3", run=fake_run)
    assert diag == {"status": "REVIEWED", "selected": 3, "reviewed": 3}
    assert pd.isna(seen["kalshi"][0]) and seen["kalshi"][1] == 0.48       # blend's value, not the raw 0.75
    assert "pinnacle_ev" in seen["extra"]
    assert out["gemini_research_agrees"].tolist()[:3] == [1.0, 0.0, 1.0]
    assert pd.isna(out.loc[3, "gemini_research_pick"])
    assert out["Play_Stake"].tolist() == [0.0] * 4 and out["gemini_review_status"].tolist() == ["SKIPPED"] * 4
    assert out["kalshi_probability"].tolist() == best["kalshi_probability"].tolist()


def test_off_switches_and_failures_never_break_refresh():
    assert grr.review(_best(), enabled=False)[1]["status"] == "OFF"
    assert grr.review(_best(), enabled=True, setting=lambda *a: "0")[1]["status"] == "OFF"

    def boom(*a, **k):
        raise RuntimeError("down")
    out, diag = grr.review(_best(), enabled=True, setting=lambda *a: "12", run=boom)
    assert diag["status"] == "ERROR" and out["gemini_research_pick"].isna().all()


def test_board_label_and_export_columns():
    from app.ui.daily_dashboard import gemini_research_labels
    from core.streamlit_pipeline import BEST_PICK_COLUMNS
    board = pd.DataFrame({"matchup_id": ["g1", "g2"], "market_type": ["total_under", "spread_home"],
                          "best_pick": ["Under 152.5", "Uic -10.5"], "gemini_research_agrees": [1.0, 0.0],
                          "gemini_research_pick": ["Under 152.5", "Robert Morris +10.5"],
                          "gemini_research_confidence": ["MEDIUM", "LOW"]})
    result = pd.DataFrame({"matchup_id": ["g1", "g2"], "market_type": ["total_under", "spread_home"],
                           "pick": ["Under 152.5", "Uic -10.5"]})
    assert gemini_research_labels(result, board).tolist() == ["Agrees · MEDIUM", "Disagrees (Robert Morris +10.5) · LOW"]
    for column in grr.COLUMNS:
        assert column in BEST_PICK_COLUMNS


def test_gate_payload_unchanged_without_extra_columns():
    import inspect
    from integrations.gemini_client import run_gemini_analysis
    assert inspect.signature(run_gemini_analysis).parameters["extra_columns"].default == ()
