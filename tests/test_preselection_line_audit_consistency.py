"""Best Available line integrity must be settled before scoring and auditing."""

import pandas as pd

from core.streamlit_pipeline import build_best_picks_df


def _candidate(
    market_type: str,
    *,
    win_prob: float,
    ev: float,
    edge: float,
    **overrides,
) -> dict:
    is_total = market_type.startswith("total")
    is_home = market_type.endswith("home")
    row = {
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
    row.update(overrides)
    return row


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


def test_reoriented_novig_spread_is_trusted_through_final_validation():
    df = pd.DataFrame([
        _candidate(
            "spread_away",
            win_prob=0.72,
            ev=0.20,
            edge=0.16,
            spread_line=1.5,
            live_spread_line=1.5,
            line_source="novig_theover_moneyline_reoriented",
            orientation_source="exact_match|theover_moneyline_favorite",
            odds_american=-178,
            odds_source="novig",
            # Raw live fields still reflect Novig's pre-remap team attachment:
            # they say the away team is favored even though the independently
            # oriented spread correctly assigns Seattle +1.5.
            game_home_ml_price=120,
            game_away_ml_price=-125,
        ),
        _candidate(
            "spread_home",
            win_prob=0.30,
            ev=-0.10,
            edge=0.0,
            spread_line=-1.5,
            live_spread_line=-1.5,
            line_source="novig_theover_moneyline_reoriented",
            orientation_source="exact_match|theover_moneyline_favorite",
            odds_american=160,
            odds_source="novig",
        ),
        _candidate(
            "total_under",
            win_prob=0.55,
            ev=0.01,
            edge=0.01,
            total_line=8.5,
            live_total_line=8.5,
            uploaded_total_line=8.5,
            upload_total_line=8.5,
        ),
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]
    selected = audit[audit["best_available_selected"].astype(bool)].iloc[0]
    row = out.iloc[0]

    assert row["market_type"] == "spread_away"
    assert row["best_pick"] == "Seattle +1.5"
    assert float(row["market_line_used"]) == 1.5
    assert row["market_line_source_detail"] == "novig_theover_moneyline_reoriented"
    assert bool(row["line_consistency_flag"])
    assert bool(row["line_event_identity_match_flag"])
    assert bool(row["best_available_ranking_verified"])
    assert bool(row["final_pick_valid"])
    assert row["final_pick_valid_reason"] == "validated_live_line"
    assert row["status_blocker_stage"] != "spread_orientation_guardrail"
    assert "spread_orientation_fault" not in str(row["Status_Reason"])
    assert bool(selected["best_available_ranking_verified"])
    assert bool(selected["final_pick_valid"])
    assert diagnostics["preselection_invalid_spread_candidate_count"] == 0


def test_invalid_spread_is_removed_and_valid_total_becomes_fallback():
    df = pd.DataFrame([
        _candidate(
            "spread_away",
            win_prob=0.90,
            ev=0.40,
            edge=0.30,
            spread_line=pd.NA,
            live_spread_line=pd.NA,
            line_source="rejected_live_orientation",
            odds_american=pd.NA,
        ),
        _candidate(
            "total_under",
            win_prob=0.58,
            ev=0.03,
            edge=0.03,
            total_line=8.5,
            live_total_line=8.5,
            uploaded_total_line=8.5,
            upload_total_line=8.5,
        ),
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]
    selected = audit[audit["best_available_selected"].astype(bool)].iloc[0]
    row = out.iloc[0]

    assert len(audit) == 1
    assert not audit["market_type"].astype(str).str.startswith("spread").any()
    assert row["market_type"] == "total_under"
    assert row["best_pick"] == "Under 8.5"
    assert bool(row["final_pick_valid"])
    assert selected["market_type"] == "total_under"
    assert bool(selected["final_pick_valid"])
    assert diagnostics["preselection_invalid_spread_candidate_count"] == 1
    assert diagnostics["preselection_dropped_spread_candidate_count"] == 1
    assert diagnostics["preselection_dropped_line_candidate_count"] == 1


def test_moneyline_reversed_team_bound_spreads_are_removed_before_ranking():
    df = pd.DataFrame([
        _candidate(
            "spread_home",
            win_prob=0.90,
            ev=0.40,
            edge=0.30,
            spread_line=1.5,
            live_spread_line=1.5,
            line_source="fanduel_standard_spread_consensus",
            orientation_source="exact_match|standard_spread_consensus",
            odds_american=-220,
            opposing_odds_american=180,
            game_home_ml_price=-117,
            game_away_ml_price=115,
        ),
        _candidate(
            "spread_away",
            win_prob=0.88,
            ev=0.35,
            edge=0.25,
            spread_line=-1.5,
            live_spread_line=-1.5,
            line_source="fanduel_standard_spread_consensus",
            orientation_source="exact_match|standard_spread_consensus",
            odds_american=180,
            opposing_odds_american=-220,
            game_home_ml_price=-117,
            game_away_ml_price=115,
        ),
        _candidate(
            "total_under",
            win_prob=0.58,
            ev=0.03,
            edge=0.03,
            total_line=8.5,
            live_total_line=8.5,
            uploaded_total_line=8.5,
            upload_total_line=8.5,
            opposing_odds_american=-105,
        ),
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]

    assert out.iloc[0]["market_type"] == "total_under"
    assert out.iloc[0]["best_pick"] == "Under 8.5"
    assert not audit["market_type"].astype(str).str.startswith("spread").any()
    assert diagnostics["preselection_mlb_spread_orientation_fault_count"] == 2
    assert diagnostics["preselection_dropped_spread_candidate_count"] == 2


def test_plain_live_spread_still_uses_raw_moneyline_orientation_guard():
    df = pd.DataFrame([
        _candidate(
            "spread_away",
            win_prob=0.72,
            ev=0.20,
            edge=0.16,
            spread_line=1.5,
            live_spread_line=1.5,
            line_source="live_odds",
            orientation_source="exact_match",
            odds_american=-135,
            odds_source="novig",
            game_home_ml_price=120,
            game_away_ml_price=-125,
        ),
    ])

    row = build_best_picks_df(df).iloc[0]

    assert row["best_pick"] == "Seattle +1.5"
    assert row["status_blocker_stage"] == "spread_orientation_guardrail"
    assert "spread_orientation_fault" in str(row["Status_Reason"])


def test_team_bound_live_spread_cannot_bypass_final_orientation_guard():
    df = pd.DataFrame([
        _candidate(
            "spread_home",
            win_prob=0.72,
            ev=0.20,
            edge=0.16,
            spread_line=1.5,
            live_spread_line=1.5,
            line_source="novig_team_bound_quote",
            orientation_source="exact_match|odds_api_team_binding",
            odds_american=-194,
            opposing_odds_american=182,
            odds_source="novig",
            game_home_ml_price=-117,
            game_away_ml_price=115,
        ),
    ])

    row = build_best_picks_df(df).iloc[0]

    assert row["best_pick"] == "Texas +1.5"
    assert row["status_blocker_stage"] == "spread_orientation_guardrail"
    assert "spread_orientation_fault" in str(row["Status_Reason"])


def test_extreme_juice_spread_cannot_win_best_available_ranking():
    df = pd.DataFrame([
        _candidate(
            "spread_home",
            win_prob=0.90,
            ev=0.40,
            edge=0.30,
            spread_line=5.5,
            live_spread_line=5.5,
            line_source="novig_moneyline_verified",
            orientation_source="exact_match|novig_moneyline_favorite",
            odds_american=-1150,
            odds_source="novig",
        ),
        _candidate(
            "total_under",
            win_prob=0.58,
            ev=0.03,
            edge=0.03,
            total_line=8.5,
            live_total_line=8.5,
            uploaded_total_line=8.5,
            upload_total_line=8.5,
        ),
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]

    assert len(out) == 1
    assert out.iloc[0]["market_type"] == "total_under"
    assert out.iloc[0]["best_pick"] == "Under 8.5"
    assert not audit["market_type"].astype(str).str.startswith("spread").any()
    assert diagnostics["preselection_invalid_extreme_spread_price_count"] == 1
    assert diagnostics["preselection_invalid_spread_candidate_count"] == 1
    assert diagnostics["preselection_dropped_spread_candidate_count"] == 1


def test_mismatched_display_total_row_is_rejected_before_audit():
    df = pd.DataFrame([
        _candidate("spread_home", win_prob=0.58, ev=0.02, edge=0.02),
        _candidate("spread_away", win_prob=0.54, ev=0.01, edge=0.01),
        _candidate(
            "total_over",
            win_prob=0.90,
            ev=0.40,
            edge=0.25,
            total_line=19.5,
            live_total_line=8.0,
            uploaded_total_line=8.0,
            upload_total_line=8.0,
        ),
        _candidate(
            "total_under",
            win_prob=0.55,
            ev=0.02,
            edge=0.02,
            total_line=8.0,
            live_total_line=8.0,
            uploaded_total_line=8.0,
            upload_total_line=8.0,
        ),
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    audit_frame = diagnostics["candidate_audit_df"]

    assert len(out) == 1
    assert out.iloc[0]["market_type"] == "total_under"
    assert "total_over" not in set(audit_frame["market_type"])
    assert "total_under" in set(audit_frame["market_type"])
    assert diagnostics["preselection_total_display_live_mismatch_count"] == 1
    assert diagnostics["preselection_invalid_total_pair_count"] == 0
    assert diagnostics["preselection_dropped_total_candidate_count"] == 1


def test_non_complementary_spread_pair_is_rejected_before_ranking():
    df = pd.DataFrame([
        _candidate(
            "spread_home",
            win_prob=0.90,
            ev=0.40,
            edge=0.25,
            spread_line=-1.5,
            live_spread_line=-1.5,
        ),
        _candidate(
            "spread_away",
            win_prob=0.88,
            ev=0.35,
            edge=0.22,
            spread_line=2.5,
            live_spread_line=2.5,
        ),
        _candidate(
            "total_over",
            win_prob=0.57,
            ev=0.02,
            edge=0.02,
            total_line=8.5,
            live_total_line=8.5,
            uploaded_total_line=8.5,
            upload_total_line=8.5,
        ),
        _candidate(
            "total_under",
            win_prob=0.53,
            ev=0.00,
            edge=0.00,
            total_line=8.5,
            live_total_line=8.5,
            uploaded_total_line=8.5,
            upload_total_line=8.5,
        ),
    ])
    diagnostics = {}

    out = build_best_picks_df(df, diagnostics_out=diagnostics)
    audit_frame = diagnostics["candidate_audit_df"]

    assert len(out) == 1
    assert out.iloc[0]["market_type"].startswith("total")
    assert not audit_frame["market_type"].astype(str).str.startswith("spread").any()
    assert diagnostics["preselection_invalid_spread_pair_count"] == 2
    assert diagnostics["preselection_dropped_spread_candidate_count"] == 2
