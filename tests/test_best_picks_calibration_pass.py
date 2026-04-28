import pandas as pd

from core.streamlit_pipeline import build_best_picks_df


def _row(
    *,
    idx: int,
    league: str,
    market_type: str,
    win_prob: float,
    ev: float,
    edge: float,
    kalshi_probability=None,
):
    home = f"Home{idx}"
    away = f"Away{idx}"
    return {
        "league": league,
        "home_team": home,
        "away_team": away,
        "game_date": "2026-04-24",
        "matchup_id": f"2026-04-24|{home}|{away}",
        "market_type": market_type,
        "expected_value": ev,
        "edge": edge,
        "calibrated_probability": win_prob,
        "ml_probability": win_prob,
        "model_probability": win_prob,
        "odds_american": -110,
        "spread_line": -3.5 if "spread" in market_type else pd.NA,
        "total_line": 220.5 if "total" in market_type else pd.NA,
        "is_live_data": True,
        "used_stale_features": False,
        "odds_source": "odds_api",
        "kalshi_probability": kalshi_probability,
    }


def test_total_under_requires_stronger_bar_than_generic_totals():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.54),
            _row(idx=2, league="NFL", market_type="total_under", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.54),
        ]
    )
    out = build_best_picks_df(df)
    assert out.loc[out["market_type"] == "total_over", "Pick_Status"].iloc[0] == "Actionable"
    assert out.loc[out["market_type"] == "total_under", "Pick_Status"].iloc[0] == "Below Threshold"


def test_nba_totals_are_no_longer_overpenalized_vs_weak_mlb_spreads():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.58, ev=0.05, edge=0.06, kalshi_probability=0.53),
            _row(idx=2, league="MLB", market_type="spread_home", win_prob=0.52, ev=0.02, edge=0.03, kalshi_probability=0.48),
        ]
    )
    out = build_best_picks_df(df)
    assert out.loc[out["league"] == "NBA", "Pick_Status"].iloc[0] == "Actionable"
    assert out.loc[out["league"] == "MLB", "Pick_Status"].iloc[0] == "Below Threshold"


def test_no_kalshi_totals_are_harder_than_kalshi_backed_totals():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=None),
            _row(idx=2, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=0.55),
        ]
    )
    out = build_best_picks_df(df)
    no_kalshi_row = out[out["kalshi_probability"].isna()].iloc[0]
    kalshi_row = out[out["kalshi_probability"].notna()].iloc[0]
    assert no_kalshi_row["consensus_agreement"] == "No Kalshi"
    assert no_kalshi_row["Pick_Status"] == "Below Threshold"
    assert kalshi_row["Pick_Status"] == "Actionable"


def test_agrees_does_not_auto_promote_in_standard_mode():
    df = pd.DataFrame(
        [
            # Agrees (gap +0.04) but still weak totals profile
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.57, ev=0.02, edge=0.03, kalshi_probability=0.53),
            # Neutral (gap ~0) same weak thresholds
            _row(idx=2, league="NFL", market_type="total_over", win_prob=0.57, ev=0.02, edge=0.03, kalshi_probability=0.57),
        ]
    )
    out = build_best_picks_df(df)
    statuses = out.sort_values("home_team")["Pick_Status"].astype(str).tolist()
    assert statuses == ["Below Threshold", "Below Threshold"]


def test_overs_and_sides_not_penalized_like_unders():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.54, ev=0.07, edge=0.07, kalshi_probability=0.48),
            _row(idx=2, league="MLB", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
            _row(idx=3, league="MLB", market_type="total_under", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
        ]
    )
    out = build_best_picks_df(df)
    assert out.loc[out["market_type"] == "spread_home", "Pick_Status"].iloc[0] == "Actionable"
    assert out.loc[out["market_type"] == "total_over", "Pick_Status"].iloc[0] == "Actionable"
    assert out.loc[out["market_type"] == "total_under", "Pick_Status"].iloc[0] == "Below Threshold"


def test_diagnostics_blocked_rows_and_shadow_cards_populate():
    df = pd.DataFrame(
        [
            # Under-specific block: base would pass, stricter under bar blocks it.
            _row(idx=1, league="NFL", market_type="total_under", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
            # NBA total penalty block.
            _row(idx=2, league="NBA", market_type="total_over", win_prob=0.58, ev=0.03, edge=0.039, kalshi_probability=0.54),
            # No Kalshi total penalty block.
            _row(idx=3, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=None),
            # Actionable side to keep card non-empty with non-total representation.
            _row(idx=4, league="MLB", market_type="spread_home", win_prob=0.54, ev=0.07, edge=0.07, kalshi_probability=0.48),
        ]
    )
    diagnostics = {"is_fallback_heavy": True}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert not out.empty
    assert diagnostics["blocked_by_under_specific_thresholds"] >= 1
    assert diagnostics["blocked_by_nba_total_penalty"] >= 1
    assert diagnostics["blocked_by_no_kalshi_total_penalty"] >= 1
    assert "shadow_card_counts" in diagnostics
    shadow = diagnostics["shadow_card_counts"]
    for key in [
        "current_card",
        "overs_only_plus_sides_card",
        "no_unders_card",
        "no_nba_totals_card",
        "no_kalshi_totals_card",
    ]:
        assert key in shadow
    assert "actionable_counts_by_league_family" in diagnostics


def test_mlb_spread_finalist_penalty_can_demote_weak_spread_winner():
    matchup_id = "2026-04-24|home1|away1"
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.60, ev=0.20, edge=0.20, kalshi_probability=0.50),
            _row(idx=1, league="MLB", market_type="total_over", win_prob=0.60, ev=0.19, edge=0.19, kalshi_probability=0.50),
        ]
    )
    df["matchup_id"] = matchup_id
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert len(out) == 1
    assert out.iloc[0]["market_type"] == "total_over"
    assert diagnostics["demoted_by_mlb_spread_finalist_score_penalty"] >= 1


def test_nba_side_bonus_can_promote_borderline_side():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="spread_home", win_prob=0.52, ev=0.01, edge=0.015, kalshi_probability=0.48),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert diagnostics["promoted_by_nba_side_bonus"] >= 1


def test_nba_over_bonus_can_promote_borderline_over():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.58, ev=0.03, edge=0.04, kalshi_probability=0.55),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert out.iloc[0]["Pick_Status"] == "Actionable"
    assert diagnostics["promoted_by_nba_over_bonus"] >= 1


def test_mlb_over_explicit_actionable_gate_blocks_weak_over():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="total_over", win_prob=0.56, ev=0.05, edge=0.05, kalshi_probability=0.53),
            _row(idx=2, league="MLB", market_type="total_over", win_prob=0.58, ev=0.05, edge=0.05, kalshi_probability=0.53),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics).sort_values("home_team")
    assert out.iloc[0]["Pick_Status"] == "Below Threshold"
    assert "MLB over actionable gate" in out.iloc[0]["Status_Reason"]
    assert out.iloc[1]["Pick_Status"] == "Actionable"
    assert diagnostics["blocked_by_mlb_over_promotion_gate"] >= 1


def test_new_diagnostics_populate_without_regressing_existing_total_protections():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.53, ev=0.03, edge=0.03, kalshi_probability=0.48),
            _row(idx=2, league="NBA", market_type="spread_home", win_prob=0.52, ev=0.01, edge=0.015, kalshi_probability=0.48),
            _row(idx=3, league="NBA", market_type="total_over", win_prob=0.58, ev=0.03, edge=0.04, kalshi_probability=0.55),
            _row(idx=4, league="NFL", market_type="total_over", win_prob=0.60, ev=0.04, edge=0.05, kalshi_probability=None),
            _row(idx=5, league="NFL", market_type="total_under", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert not out.empty
    assert diagnostics["blocked_by_mlb_spread_penalty"] >= 1
    assert diagnostics["promoted_by_nba_side_bonus"] >= 1
    assert diagnostics["promoted_by_nba_over_bonus"] >= 1
    assert diagnostics["blocked_by_no_kalshi_total_penalty"] >= 1
    assert diagnostics["blocked_by_under_specific_thresholds"] >= 1


def test_high_ev_alone_is_not_auto_blocked_as_suspicious_data():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="spread_home", win_prob=0.61, ev=0.41, edge=0.06, kalshi_probability=0.55),
        ]
    )
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "Actionable"
    assert "strong EV/edge" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "none"
    assert row["suspicious_data_flag"] is False or row["suspicious_data_flag"] == False


def test_suspicious_data_rows_still_blocked_with_explicit_reason_and_diagnostics():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="spread_home", win_prob=0.62, ev=0.45, edge=0.07, kalshi_probability=0.55),
        ]
    )
    df.loc[0, "line_source"] = "synthetic"
    df.loc[0, "line_delta"] = 12.0
    df.loc[0, "market_probability"] = 0.20
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert row["suspicious_data_flag"] is True or row["suspicious_data_flag"] == True
    assert "No Play: blocked due to suspicious data" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "suspicious_data_guardrail"
    assert diagnostics["blocked_by_suspicious_data"] >= 1


def test_divergence_cannot_preserve_negative_ev_row():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.56, ev=-0.01, edge=0.02, kalshi_probability=0.80),
        ]
    )
    df.loc[0, "ml_probability"] = 0.50
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert "divergence override denied" in row["Status_Reason"]
    assert row["status_metric_basis"] == "raw"
    assert row["status_blocker_stage"] == "divergence_viability_floor"


def test_divergence_cannot_preserve_negative_edge_row():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.56, ev=0.03, edge=-0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[0, "ml_probability"] = 0.50
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "No Play"
    assert "divergence override denied" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "divergence_viability_floor"


def test_divergence_can_preserve_minimally_viable_row():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.56, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[0, "ml_probability"] = 0.50
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["Pick_Status"] == "High Variance/Speculative"
    assert "capped due to divergence" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "divergence_guardrail"


def test_divergence_viability_diagnostics_populate():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NBA", market_type="total_over", win_prob=0.56, ev=-0.01, edge=0.02, kalshi_probability=0.80),
            _row(idx=2, league="NBA", market_type="total_over", win_prob=0.56, ev=0.03, edge=-0.01, kalshi_probability=0.80),
            _row(idx=3, league="NBA", market_type="total_over", win_prob=0.56, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[:, "ml_probability"] = 0.50
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert not out.empty
    assert diagnostics["divergence_rows_preserved"] >= 1
    assert diagnostics["divergence_rows_blocked_by_viability_floor"] >= 2
    assert diagnostics["divergence_rows_negative_ev"] >= 1
    assert diagnostics["divergence_rows_negative_edge"] >= 1
    assert "No Play" in diagnostics["divergence_rows_by_pick_status"]


def test_effective_metric_transparency_when_blocked_by_effective_thresholds():
    df = pd.DataFrame(
        [
            _row(idx=1, league="NFL", market_type="total_over", win_prob=0.60, ev=0.036, edge=0.05, kalshi_probability=0.55),
        ]
    )
    diagnostics = {"is_fallback_heavy": True}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    row = out.iloc[0]
    assert row["Pick_Status"] == "Below Threshold"
    assert row["status_metric_basis"] == "effective"
    assert float(row["effective_expected_value"]) < float(row["expected_value"])
    assert "Effective EV" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "fallback_heavy_guardrail"
    assert row["status_blocker_reason"] == row["Status_Reason"]


def test_run_health_fields_are_export_visible_when_present():
    df = pd.DataFrame(
        [
            _row(idx=1, league="MLB", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
        ]
    )
    df["nba_stats_fetch_status"] = "cached"
    df["nba_stats_fetch_source"] = "cached"
    df["nba_stats_fetch_retries_used"] = 3
    df["stats_source_counts"] = "{'live': 0, 'cached': 1, 'fallback': 0, 'failed': 0}"
    df["fallback_summary_by_league"] = "{'NBA': 12}"
    df["fallback_heavy_slate_flag"] = True
    df["run_health_warning"] = "Run health warning: fallback usage is elevated"
    df["degraded_feature_subset_flag"] = True
    df["degraded_feature_subset_reason"] = "degraded_subset:all_constant_feature_count=10"
    out = build_best_picks_df(df)
    row = out.iloc[0]
    assert row["nba_stats_fetch_status"] == "cached"
    assert row["nba_stats_fetch_source"] == "cached"
    assert bool(row["fallback_heavy_slate_flag"]) is True
    assert "Run health warning" in row["run_health_warning"]
    assert bool(row["degraded_feature_subset_flag"]) is True
    assert "degraded_subset" in str(row["degraded_feature_subset_reason"])


def test_high_ev_clean_row_promotes_to_actionable_not_high_variance():
    df = pd.DataFrame(
        [
            _row(idx=10, league="NBA", market_type="total_over", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=0.59),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    row = out.iloc[0]
    assert row["Pick_Status"] == "Actionable"
    assert "strong EV/edge" in row["Status_Reason"]
    assert diagnostics["promoted_high_ev_to_actionable_no_uncertainty"] >= 1
    assert diagnostics["high_variance_due_only_high_ev"] == 0


def test_high_ev_can_be_capped_for_real_uncertainty_reason():
    df = pd.DataFrame(
        [
            _row(idx=11, league="NBA", market_type="total_over", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=None),
        ]
    )
    df["degraded_feature_subset_flag"] = True
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    row = out.iloc[0]
    assert row["Pick_Status"] == "High Variance/Speculative"
    assert "capped due to" in row["Status_Reason"]
    assert row["status_blocker_stage"] == "variance_uncertainty_guardrail"


def test_high_variance_inflation_diagnostics_populate():
    df = pd.DataFrame(
        [
            _row(idx=21, league="NBA", market_type="total_over", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=0.59),
            _row(idx=22, league="NBA", market_type="total_over", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=None),
            _row(idx=23, league="NBA", market_type="total_over", win_prob=0.56, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[df["home_team"] == "Home22", "degraded_feature_subset_flag"] = True
    df.loc[df["home_team"] == "Home23", "ml_probability"] = 0.50
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    assert not out.empty
    assert diagnostics["high_variance_due_only_high_ev"] == 0
    assert diagnostics["promoted_high_ev_to_actionable_no_uncertainty"] >= 1
    assert diagnostics["high_variance_capped_due_to_degraded_subset"] >= 1
    assert diagnostics["high_variance_capped_due_to_no_kalshi"] >= 1
    assert diagnostics["high_variance_capped_due_to_divergence"] >= 1
    assert "High Variance/Speculative" in diagnostics["final_pick_status_counts"]


def test_side_balance_does_not_promote_side_blocked_by_degraded_subset_reason():
    df = pd.DataFrame(
        [
            _row(idx=30, league="NBA", market_type="total_over", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.59),
            _row(idx=31, league="NBA", market_type="spread_home", win_prob=0.64, ev=0.45, edge=0.18, kalshi_probability=None),
        ]
    )
    # Side remains high variance due to degraded subset uncertainty and should not be balance-promoted.
    df.loc[df["market_type"] == "spread_home", "degraded_feature_subset_flag"] = True
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    actionable = out[out["Pick_Status"] == "Actionable"]
    assert not actionable.empty
    assert actionable["market_type"].str.contains("spread|h2h", case=False, regex=True, na=False).sum() == 0
    assert diagnostics["side_balance_promotions"] == 0


def test_transparency_fields_are_populated_for_every_export_row():
    df = pd.DataFrame(
        [
            _row(idx=40, league="NFL", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55),
            _row(idx=41, league="NFL", market_type="total_under", win_prob=0.56, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    df.loc[df["home_team"] == "Home41", "ml_probability"] = 0.50
    out = build_best_picks_df(df)
    required = [
        "status_metric_basis",
        "effective_expected_value",
        "effective_edge",
        "effective_win_probability",
        "status_blocker_reason",
        "status_blocker_stage",
    ]
    for col in required:
        assert col in out.columns
    assert out["status_metric_basis"].notna().all()
    assert out["effective_expected_value"].notna().all()
    assert out["effective_edge"].notna().all()
    assert out["effective_win_probability"].notna().all()
    assert out["status_blocker_stage"].notna().all()

from core.streamlit_pipeline import ensure_best_pick_export_columns, REQUIRED_BEST_PICK_EXPORT_COLUMNS


def test_best_pick_export_guarantees_required_transparency_columns():
    df = pd.DataFrame([_row(idx=501, league="NFL", market_type="total_over", win_prob=0.60, ev=0.05, edge=0.05, kalshi_probability=0.55)])
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    for col in REQUIRED_BEST_PICK_EXPORT_COLUMNS:
        assert col in out.columns
    assert diagnostics["best_pick_export_required_columns_ok"] is True
    assert diagnostics["best_pick_export_missing_columns"] == []


def test_missing_export_columns_are_backfilled_and_logged(caplog):
    partial = pd.DataFrame({"best_pick": ["Over 220.5"], "expected_value": [0.02]})
    with caplog.at_level("WARNING"):
        out = ensure_best_pick_export_columns(partial, diagnostics_out={})
    for col in REQUIRED_BEST_PICK_EXPORT_COLUMNS:
        assert col in out.columns
    assert "best_pick_export_missing_columns" in caplog.text


def test_side_balance_guard_promotes_viable_side_when_actionable_is_totals_only():
    df = pd.DataFrame(
        [
            _row(idx=510, league="NBA", market_type="total_over", win_prob=0.58, ev=0.04, edge=0.04, kalshi_probability=0.56),
            _row(idx=511, league="NBA", market_type="spread_home", win_prob=0.54, ev=0.03, edge=0.03, kalshi_probability=0.30),
        ]
    )
    # Keep side out of initial Actionable via divergence cap, then allow balance guard promotion.
    df.loc[df["market_type"] == "spread_home", "ml_probability"] = 0.64
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    actionable = out[out["Pick_Status"].astype(str) == "Actionable"]
    assert actionable["market_type"].astype(str).str.contains("spread|h2h", case=False, regex=True, na=False).any()
    assert int(diagnostics["side_promoted_by_balance_guard_count"]) >= 1
    assert diagnostics["side_balance_guard_reason"] == "promoted_viable_side_candidate"


def test_side_balance_guard_does_not_promote_weak_sides():
    df = pd.DataFrame(
        [
            _row(idx=520, league="NBA", market_type="total_over", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.59),
            _row(idx=521, league="NBA", market_type="spread_home", win_prob=0.51, ev=0.005, edge=0.005, kalshi_probability=0.58),
        ]
    )
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    actionable = out[out["Pick_Status"].astype(str) == "Actionable"]
    assert actionable["market_type"].astype(str).str.contains("spread|h2h", case=False, regex=True, na=False).sum() == 0
    assert diagnostics["side_promoted_by_balance_guard_count"] == 0


def test_totals_only_actionable_allowed_when_no_viable_sides_exist():
    df = pd.DataFrame([
        _row(idx=530, league="NBA", market_type="total_over", win_prob=0.64, ev=0.08, edge=0.07, kalshi_probability=0.59),
        _row(idx=531, league="NBA", market_type="total_under", win_prob=0.62, ev=0.06, edge=0.06, kalshi_probability=0.57),
    ])
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    actionable = out[out["Pick_Status"].astype(str) == "Actionable"]
    assert not actionable.empty
    assert actionable["market_type"].astype(str).str.contains("total", case=False, regex=True, na=False).all()
    assert diagnostics["viable_side_candidates_count"] == 0


def test_no_regression_mlb_spread_suspicious_and_divergence_guardrails():
    df = pd.DataFrame(
        [
            _row(idx=540, league="MLB", market_type="spread_home", win_prob=0.52, ev=0.02, edge=0.03, kalshi_probability=0.48),
            _row(idx=541, league="NBA", market_type="total_over", win_prob=0.64, ev=0.50, edge=0.20, kalshi_probability=0.59),
            _row(idx=542, league="NBA", market_type="spread_home", win_prob=0.50, ev=0.01, edge=0.01, kalshi_probability=0.80),
        ]
    )
    # suspicious high EV total with bad market status should remain blocked.
    df.loc[df["home_team"] == "Home541", "market_status"] = "suspended"
    diagnostics = {}
    out = build_best_picks_df(df, diagnostics_out=diagnostics)

    mlb_spread_row = out[out["home_team"] == "Home540"].iloc[0]
    suspicious_row = out[out["home_team"] == "Home541"].iloc[0]
    divergence_row = out[out["home_team"] == "Home542"].iloc[0]

    assert mlb_spread_row["Pick_Status"] != "Actionable"
    assert suspicious_row["status_blocker_stage"] == "suspicious_data_guardrail"
    assert divergence_row["status_blocker_stage"] in {"divergence_guardrail", "divergence_viability_floor"}
