"""Best Available line integrity must be settled before scoring and auditing."""

import pandas as pd

from core.streamlit_pipeline import build_best_picks_df


def _candidate(market_type: str, *, win_prob: float, ev: float, edge: float) -> dict:
    is_total = market_type.startswith("total")
    is_home = market_type.endswith("home")
    return {
        "league": "MLB",
        "home_team": "Texas",
        "away_team": "Seattle",
        "game_date": "2026-07-27",
        "game_time_est": "2026-07-27 6:40 PM ET",
        "matchup_id": "2026-07-27|texas|seattle",
        "market_type": market_type,
        "candidate_source": "upload_matched" if is_total else "live_market_only",
        "expected_value": ev,
        "edge": edge,
        "calibrated_probability": win_prob,
        "ml_probability": win_prob,
        "model_probability": win_prob,
        "odds_american": -110,
        "market_probability": 0.50,
        "kalshi_probability": 0.55,
        "spread_line": pd.NA if is_total else (-1.5 if is_home else 1.5),
        "total_line": 13.5 if is_total else pd.NA,
        "line_source": "live_odds",
        "live_spread_line": pd.NA if is_total else (-1.5 if is_home else 1.5),
        "live_total_line": 13.5 if is_total else pd.NA,
        "uploaded_total_line": 8.0 if is_total else pd.NA,
        "upload_total_line": 8.0 if is_total else pd.NA,
        "is_live_data": True,
        "used_stale_features": False,
        "odds_source": "odds_api",
    }


def test_corrupt_live_totals_cannot_beat_valid_spreads_before_selection():
    df = pd.DataFrame([
        _candidate("spread_home", win_prob=0.58, ev=0.01, edge=0.01),
        _candidate("spread_away", win_prob=0.56, ev=0.00, edge=0.00),
        # These corrupt 13.5 candidates would dominate on probability/EV if they
        # reached ranking, then be rewritten to 8.0 after selection.
        _candidate("total_over", win_prob=0.90, ev=0.40, edge=0.25),
        _candidate("total_under", win_prob=0.88, ev=0.35, edge=0.22),
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]
    selected = audit[audit["best_available_selected"].astype(bool)]

    assert len(out) == 1
    assert out.iloc[0]["market_type"] in {"spread_home", "spread_away"}
    assert not audit["market_type"].astype(str).str.startswith("total").any()
    assert len(audit) == 2
    assert int(audit["best_available_candidate_count"].iloc[0]) == 2
    assert diagnostics["preselection_invalid_total_candidate_count"] == 2
    assert diagnostics["preselection_dropped_total_candidate_count"] == 2
    assert selected.iloc[0]["market_type"] == out.iloc[0]["market_type"]
    assert selected.iloc[0]["best_pick"] == out.iloc[0]["best_pick"]
    assert float(selected.iloc[0]["best_available_score"]) == float(
        out.iloc[0]["best_available_score"]
    )


def test_lone_repaired_total_is_value_neutral_and_audit_matches_export():
    df = pd.DataFrame([
        _candidate("total_under", win_prob=0.88, ev=0.35, edge=0.22),
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]
    selected = audit[audit["best_available_selected"].astype(bool)].iloc[0]
    row = out.iloc[0]

    assert row["best_pick"] == "Under 8.0"
    assert row["market_line_source_detail"] == "upload_total_fallback_after_rejected_live"
    assert float(row["expected_value"]) == 0.0
    assert float(row["edge"]) == 0.0
    assert float(row["best_available_score"]) == 0.0
    assert "Research-only upload line fallback" in row["best_available_selection_reason"]
    assert selected["best_pick"] == row["best_pick"]
    assert float(selected["expected_value"]) == 0.0
    assert float(selected["edge"]) == 0.0
    assert float(selected["best_available_score"]) == 0.0
    assert selected["best_available_rejection_reason"] == (
        "selected_research_only_after_upload_line_repair"
    )
    assert diagnostics["preselection_invalid_total_candidate_count"] == 1
    assert diagnostics["preselection_dropped_total_candidate_count"] == 0
    assert diagnostics["preselection_retained_only_candidate_count"] == 1
    assert diagnostics["postvalidation_candidate_audit_sync_count"] == 1
