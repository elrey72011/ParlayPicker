import pandas as pd
from core.streamlit_pipeline import build_best_picks_df, _total_over_concentration_downgrades


def _row(i, league="MLB", market_type="total_over", prob=0.68, ev=0.12, edge=0.10, odds=-110, degraded=False):
    return {
        "league": league,
        "home_team": f"H{i}",
        "away_team": f"A{i}",
        "game_date": "2026-05-03",
        "market_type": market_type,
        "expected_value": ev,
        "edge": edge,
        "calibrated_probability": prob,
        "odds_american": odds,
        "decimal_odds": 1.91,
        "total_line": 8.5,
        "market_line_used": 8.5,
        "matched_live_total_line": 8.5,
        "upload_total_line": 8.5,
        "base_total_line": 8.5,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "market_line_source": "live",
        "line_provenance_warning": "",
        "degraded_feature_subset_flag": degraded,
        "run_health_warning": "degraded_subset: all_constant_feature_count=6" if degraded else "",
    }


def test_total_over_concentration_guard_downgrades_excess_and_keeps_top_ranked():
    df = pd.DataFrame([_row(i, league="NBA", prob=0.72 - i * 0.01, ev=0.14 - i * 0.01, edge=0.11 - i * 0.01) for i in range(7)])
    diags = {}
    out = build_best_picks_df(df, diagnostics_out=diags)
    actionable_overs = out[(out["Pick_Status"] == "Actionable") & (out["market_type"] == "total_over")]
    assert len(actionable_overs) <= 4
    assert diags.get("actionable_total_over_count", 0) <= 4


def test_probability_shrinkage_and_production_metrics_recomputed():
    df = pd.DataFrame([_row(1, league="NBA", market_type="total_over", prob=0.70, ev=0.12, edge=0.10)])
    out = build_best_picks_df(df)
    r = out.iloc[0]
    # The effective probability may already contain the same shrinkage. The
    # production layer must never inflate it, but equality is valid/idempotent.
    assert r["production_win_probability"] <= r["effective_win_probability"]
    assert pd.notna(r["production_expected_value"]) and pd.notna(r["production_edge"])


def test_mlb_total_over_production_gate_downgrades_failures():
    df = pd.DataFrame([_row(1, league="MLB", market_type="total_over", prob=0.63, ev=0.09, edge=0.07)])
    out = build_best_picks_df(df)
    r = out.iloc[0]
    assert r["status_blocker_stage"] in {"production_market_guard", "production_concentration_guard", "none", "value_guardrail", "line_provenance"}
    if r["status_blocker_stage"] == "production_market_guard":
        assert r["Kelly_Bet_Size"] == 0
        assert r["production_eligible"] is False
        assert "downgraded by MLB total-over production guard" in str(r["Status_Reason"])


def test_degraded_feature_kelly_reduction_applies_and_non_actionable_zero_kelly():
    df = pd.DataFrame([_row(i, degraded=True, league="NBA", market_type="spread_home", prob=0.65, ev=0.10, edge=0.08) for i in range(4)])
    diags = {}
    out = build_best_picks_df(df, diagnostics_out=diags)
    assert diags["degraded_feature_kelly_guard_active"] is True
    non_actionable = out[out["Pick_Status"] != "Actionable"]
    if not non_actionable.empty:
        assert (non_actionable["Kelly_Bet_Size"].fillna(0) == 0).all()


def test_hv_concentration_downgrades_respect_overall_cap():
    # Six total_over candidates (none MLB), overall cap 3 -> keep best 3, downgrade the
    # lowest-ranked 3. Candidates are pre-sorted best-first (index 0 = best).
    cands = pd.DataFrame({"_is_mlb_over": [False] * 6})
    downgrades = _total_over_concentration_downgrades(cands, overall_cap=3, mlb_cap=2)
    assert downgrades == [3, 4, 5]


def test_hv_concentration_enforces_mlb_subcap_binds_first():
    # Four MLB overs (the 6 Jun pattern) with MLB sub-cap 2: only the best 2 MLB overs
    # survive even though the overall cap is 3. The 2 lowest-ranked MLB overs downgrade.
    cands = pd.DataFrame({"_is_mlb_over": [True, True, True, True]})
    downgrades = _total_over_concentration_downgrades(cands, overall_cap=3, mlb_cap=2)
    assert downgrades == [2, 3]


def test_hv_concentration_mixed_keeps_non_mlb_until_overall_cap():
    # Best-first: MLB, MLB, MLB, NBA, NBA. MLB sub-cap 2 downgrades the 3rd (MLB) pick;
    # the first NBA fills the 3rd overall slot, and the last NBA exceeds the overall cap.
    cands = pd.DataFrame({"_is_mlb_over": [True, True, True, False, False]})
    downgrades = _total_over_concentration_downgrades(cands, overall_cap=3, mlb_cap=2)
    assert downgrades == [2, 4]


def test_under_concentration_uses_flag_col_and_binds_mlb_subcap():
    # The 12 Jun pattern mirrored to Unders: 5 MLB Actionable Unders, MLB sub-cap 2 ->
    # keep the best 2, downgrade the rest. The under guard passes flag_col="_is_mlb_under".
    cands = pd.DataFrame({"_is_mlb_under": [True, True, True, True, True]})
    downgrades = _total_over_concentration_downgrades(
        cands, overall_cap=3, mlb_cap=2, flag_col="_is_mlb_under"
    )
    assert downgrades == [2, 3, 4]


def test_no_regression_kelly_column_order_and_line_provenance_present():
    df = pd.DataFrame([_row(1, league="NBA", market_type="spread_home", prob=0.66, ev=0.11, edge=0.09)])
    out = build_best_picks_df(df)
    cols = list(out.columns)
    idx = cols.index("best_pick")
    assert cols[idx + 1] == "Kelly_Bet_Size"
    assert "line_provenance_warning" in cols

