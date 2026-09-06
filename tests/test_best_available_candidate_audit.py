"""Best-available selection and commercial product-boundary regressions."""
from __future__ import annotations

import pandas as pd

from core.smart_parlay_engine import generate_smart_parlays
from core.streamlit_pipeline import (
    _evidence_gated_wnba_under_finalist_penalty,
    build_best_picks_df,
    classify_best_available_picks,
)


def _candidate(
    market_type: str,
    probability: float,
    ev: float,
    *,
    league: str = "MLB",
) -> dict:
    is_total = market_type.startswith("total")
    spread_line = -1.5 if market_type == "spread_home" else 1.5 if market_type == "spread_away" else pd.NA
    total_line = 178.5 if league == "WNBA" else 8.5
    return {
        "game_id": "best-available-audit",
        "league": league,
        "home_team": "Chicago Cubs",
        "away_team": "Pittsburgh Pirates",
        "game_date": pd.Timestamp("2026-07-27", tz="UTC"),
        "market_type": market_type,
        "total_line": total_line if is_total else pd.NA,
        "spread_line": spread_line,
        "calibrated_probability": probability,
        "model_probability": probability,
        "ml_probability": probability,
        "expected_value": ev,
        "edge": ev,
        "market_probability": 0.50,
        "kalshi_probability": 0.50,
        "consensus_agreement": "Neutral",
        "odds_american": -110,
        "odds_source": "test",
        "line_source": "live",
        "market_line_source": "live",
        "live_total_line": total_line,
        "is_live_data": True,
        "used_stale_features": False,
    }


def test_exported_game_is_verified_rank_one_and_audits_every_candidate(monkeypatch):
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    analysis = pd.DataFrame([
        _candidate("total_over", probability=0.64, ev=0.10),
        _candidate("total_under", probability=0.56, ev=0.02),
    ])
    diagnostics: dict = {}
    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert len(best) == 1
    winner = best.iloc[0]
    assert winner["market_type"] == "total_over"
    assert int(winner["best_available_rank"]) == 1
    assert int(winner["best_available_family_rank"]) == 1
    assert bool(winner["best_available_selection_verified"])
    assert int(winner["best_available_candidate_count"]) == 2
    assert str(winner["matchup_id"]).startswith("2026-07-27|")
    assert winner["best_available_runner_up_market_type"] == "total_under"
    assert float(winner["best_available_score_gap"]) > 0
    assert bool(winner["selection_probability_pair_normalized"])

    audit = diagnostics["candidate_audit_df"]
    assert len(audit) == 2
    assert int(audit["best_available_selected"].sum()) == 1
    selected = audit[audit["best_available_selected"]].iloc[0]
    rejected = audit[~audit["best_available_selected"]].iloc[0]
    assert int(selected["best_available_rank"]) == 1
    assert selected["best_available_rejection_reason"] == "selected_best_available"
    assert rejected["best_available_rejection_reason"] == "lower_score_within_market_family"
    assert diagnostics["best_available_selection_verified"] is True
    assert diagnostics["best_available_selection_mismatch_count"] == 0
    assert audit["matchup_id"].astype(str).str.startswith("2026-07-27|").all()
    assert audit["selection_probability_pair_normalized"].all()
    assert bool(selected["selection_probability_pair_normalized"])
    assert not audit["wager_approved"].any()
    assert audit["export_role"].eq("RANKING CANDIDATE - BACKTEST ONLY").all()


def test_best_available_audit_compares_side_and_total_families(monkeypatch):
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    analysis = pd.DataFrame([
        _candidate("spread_home", probability=0.62, ev=0.08),
        _candidate("spread_away", probability=0.51, ev=-0.01),
        _candidate("total_over", probability=0.58, ev=0.04),
        _candidate("total_under", probability=0.50, ev=-0.02),
    ])
    diagnostics: dict = {}
    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert len(best) == 1
    audit = diagnostics["candidate_audit_df"]
    assert len(audit) == 4
    assert set(audit["market_family"]) == {"side", "total"}
    assert set(audit["market_type"]) == {
        "spread_home", "spread_away", "total_over", "total_under"
    }
    assert int(audit["best_available_selected"].sum()) == 1
    assert int(audit.loc[audit["best_available_selected"], "best_available_rank"].iloc[0]) == 1
    assert int(best.iloc[0]["best_available_candidate_count"]) == 4


def test_positive_ev_candidate_cannot_displace_probability_winner(monkeypatch):
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    analysis = pd.DataFrame([
        _candidate("spread_away", probability=0.58, ev=-0.08),
        _candidate("spread_home", probability=0.42, ev=0.03),
    ])
    diagnostics: dict = {}

    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert len(best) == 1
    winner = best.iloc[0]
    assert winner["market_type"] == "spread_away"
    assert not bool(winner["best_available_value_override_applied"])
    assert winner["best_available_value_override_from_pick"] == ""
    assert pd.isna(winner["best_available_value_override_ev_gain"])
    assert "win probability" in winner["best_available_selection_reason"].lower()
    assert diagnostics["best_available_value_override_count"] == 0

    audit = diagnostics["candidate_audit_df"]
    selected = audit[audit["best_available_selected"]].iloc[0]
    assert not bool(selected["best_available_value_override_applied"])
    assert int(selected["best_available_rank"]) == 1


def test_direction_evidence_penalty_still_blocks_opposed_candidate(monkeypatch):
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    analysis = pd.DataFrame([
        _candidate("spread_away", probability=0.58, ev=-0.08),
        _candidate("total_over", probability=0.42, ev=0.20),
    ])
    analysis.loc[analysis["market_type"].eq("total_over"), "kalshi_probability"] = 0.35
    diagnostics: dict = {}

    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert best.iloc[0]["market_type"] == "spread_away"
    assert not bool(best.iloc[0]["best_available_value_override_applied"])
    assert diagnostics["best_available_value_override_count"] == 0


def test_wnba_under_guard_requires_fresh_settled_direction_history():
    insufficient = {
        "buckets": {
            "WNBA:under:Agrees": {"n": 5, "win_rate": 0.0},
        }
    }
    recovered = {
        "buckets": {
            "WNBA:under:Agrees": {"n": 4, "win_rate": 0.50},
            "WNBA:under:Neutral": {"n": 4, "win_rate": 0.50},
        }
    }
    regressed = {
        "buckets": {
            "WNBA:under:Agrees": {"n": 5, "win_rate": 0.0},
            "WNBA:under:Neutral": {"n": 2, "win_rate": 0.50},
            "WNBA:under:Disagrees": {"n": 1, "win_rate": 0.0},
        }
    }

    penalty, diagnostics = _evidence_gated_wnba_under_finalist_penalty(insufficient)
    assert penalty == 0.0
    assert diagnostics["reason"] == "insufficient_direction_history"

    penalty, diagnostics = _evidence_gated_wnba_under_finalist_penalty(recovered)
    assert penalty == 0.0
    assert diagnostics["reason"] == "wnba_unders_not_materially_underperforming"

    penalty, diagnostics = _evidence_gated_wnba_under_finalist_penalty(regressed)
    assert penalty == 0.04
    assert diagnostics["applied"] is True
    assert diagnostics["under_sample"] == 8
    assert diagnostics["under_rate"] == 0.125


def test_fresh_wnba_under_regression_moves_close_finalist_to_side(monkeypatch):
    stats = {
        "overall": {"n": 200, "win_rate": 0.54},
        "buckets": {
            "WNBA:under:Agrees": {
                "n": 5,
                "wins": 0,
                "win_rate": 0.0,
            },
            "WNBA:under:Neutral": {
                "n": 2,
                "wins": 1,
                "win_rate": 0.50,
            },
            "WNBA:under:Disagrees": {
                "n": 1,
                "wins": 0,
                "win_rate": 0.0,
            },
        },
    }
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: stats)
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    analysis = pd.DataFrame([
        _candidate("spread_home", probability=0.51, ev=0.00, league="WNBA"),
        _candidate("spread_away", probability=0.49, ev=0.00, league="WNBA"),
        _candidate("total_under", probability=0.54, ev=0.02, league="WNBA"),
        _candidate("total_over", probability=0.46, ev=-0.02, league="WNBA"),
    ])
    diagnostics: dict = {}

    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert best.iloc[0]["market_type"] == "spread_home"
    assert diagnostics["wnba_under_finalist_penalty"]["applied"] is True
    audit = diagnostics["candidate_audit_df"]
    under = audit[audit["market_type"].eq("total_under")].iloc[0]
    assert bool(under["wnba_under_finalist_penalty_applied"])
    assert float(under["wnba_under_finalist_penalty_value"]) == 0.04
    assert under["wnba_under_finalist_penalty_reason"] == (
        "fresh_empirical_wnba_under_regression"
    )


def test_recent_bucket_regression_demotes_stale_confidence_from_finalist(monkeypatch):
    stats = {
        "overall": {"n": 300, "win_rate": 0.54},
        "buckets": {
            "MLB:side:Neutral": {
                "n": 100,
                "wins": 60,
                "win_rate": 0.60,
                "recent_n": 30,
                "recent_wins": 12,
                "recent_win_rate": 0.40,
            },
        },
    }
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: stats)
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)

    analysis = pd.DataFrame([
        _candidate("spread_home", probability=0.60, ev=-0.03),
        _candidate("spread_away", probability=0.40, ev=0.03),
        _candidate("total_over", probability=0.55, ev=0.01),
        _candidate("total_under", probability=0.45, ev=-0.01),
    ])
    diagnostics: dict = {}

    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert best.iloc[0]["market_type"] == "total_over"
    assert diagnostics["recent_regime_penalty_count"] == 2
    audit = diagnostics["candidate_audit_df"]
    side = audit[audit["market_type"].eq("spread_home")].iloc[0]
    assert bool(side["recent_regime_penalty_applied"])
    assert float(side["recent_regime_penalty_value"]) > 0.05
    assert side["recent_regime_penalty_reason"] == "fresh_recent_bucket_regression"


def test_recent_family_regression_demotes_sparse_over_buckets(monkeypatch):
    stats = {
        "overall": {"n": 500, "win_rate": 0.54},
        "buckets": {
            "NFL:over:Neutral": {
                "n": 8, "wins": 1, "win_rate": 0.125,
                "recent_n": 6, "recent_wins": 0,
            },
            "NFL:side:Neutral": {
                "n": 20, "wins": 11, "win_rate": 0.55,
                "recent_n": 12, "recent_wins": 7,
            },
        },
        "families": {
            "over": {
                "n": 120, "wins": 58, "win_rate": 0.48,
                "recent_n": 15, "recent_wins": 4,
            },
            "side": {
                "n": 300, "wins": 162, "win_rate": 0.54,
                "recent_n": 40, "recent_wins": 23,
            },
        },
    }
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: stats)
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)
    analysis = pd.DataFrame([
        _candidate("spread_home", probability=0.55, ev=0.00, league="NFL"),
        _candidate("spread_away", probability=0.45, ev=0.00, league="NFL"),
        _candidate("total_over", probability=0.59, ev=0.01, league="NFL"),
        _candidate("total_under", probability=0.41, ev=-0.01, league="NFL"),
    ])
    total_mask = analysis["market_type"].str.startswith("total")
    analysis.loc[total_mask, ["total_line", "live_total_line"]] = 36.5
    diagnostics: dict = {}

    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)

    assert best.iloc[0]["market_type"] == "spread_home"
    audit = diagnostics["candidate_audit_df"]
    over = audit[audit["market_type"].eq("total_over")].iloc[0]
    assert bool(over["recent_regime_penalty_applied"])
    assert over["recent_regime_penalty_reason"] == (
        "fresh_recent_family_regression"
    )


def test_commercial_tier_never_upgrades_an_unfunded_best_available_row():
    frame = pd.DataFrame([
        {
            "best_pick": "Over 8.5",
            "Pick_Status": "Actionable",
            "production_eligible": True,
            "production_bet_amount": 12.0,
            "production_expected_value": 0.05,
            "production_edge": 0.03,
            "market_line_source": "live",
            "line_consistency_flag": True,
            "line_event_identity_match_flag": True,
        },
        {
            "best_pick": "Under 8.5",
            "Pick_Status": "Below Threshold",
            "production_eligible": False,
            "production_bet_amount": 0.0,
            "production_expected_value": -0.03,
            "production_edge": -0.02,
            "market_line_source": "live",
            "line_consistency_flag": True,
            "line_event_identity_match_flag": True,
        },
    ])

    classified = classify_best_available_picks(frame)

    assert classified.loc[0, "commercial_tier"] == "Premium Pick"
    assert bool(classified.loc[0, "sellable_as_premium"])
    assert classified.loc[1, "commercial_tier"] == "Best Available / Pass"
    assert not bool(classified.loc[1, "sellable_as_premium"])
    assert bool(classified.loc[1, "best_available_only"])
    assert not bool(classified.loc[1, "qualified_pick"])
    assert classified.loc[1, "display_pick"] == "Under 8.5"
    assert bool(classified.loc[0, "wager_approved"])
    assert classified.loc[0, "export_role"] == "PRODUCTION WAGER"
    assert not bool(classified.loc[1, "wager_approved"])
    assert classified.loc[1, "export_role"] == "BEST AVAILABLE PICK - PASS / RESEARCH"
    assert classified.loc[1, "wager_instruction"].startswith("DO NOT BET")


def test_controlled_recovery_is_sellable_but_never_branded_as_premium():
    frame = pd.DataFrame([{
        "best_pick": "Cincinnati -1.5",
        "Pick_Status": "Actionable",
        "production_eligible": True,
        "production_bet_amount": 5.0,
        "production_expected_value": 0.04,
        "production_edge": 0.03,
        "market_line_source": "live",
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "controlled_card_recovery": True,
        "Pick_Quality": "D-Tier (Weak/Negative)",
        "tier_score": 5,
    }])

    classified = classify_best_available_picks(frame)
    row = classified.iloc[0]

    assert row["commercial_tier"] == "Controlled Value Pick"
    assert row["Pick_Quality"] == "Controlled Value (Price Edge)"
    assert row["tier_score"] == 5
    assert bool(row["sellable_as_value_card"])
    assert not bool(row["sellable_as_premium"])
    assert bool(row["wager_approved"])
    assert bool(row["qualified_pick"])
    assert row["export_role"] == "CONTROLLED VALUE WAGER"
    assert "not a Premium pick" in row["wager_instruction"]
    assert float(row["production_bet_amount"]) == 5.0


def test_public_pick_stays_visible_below_absolute_probability_or_value_gate():
    frame = pd.DataFrame([
        {
            "best_pick": "Under 8.5",
            "Pick_Status": "Below Threshold",
            "selection_probability_used": 0.54,
            "effective_expected_value": 0.05,
            "effective_edge": 0.03,
            "market_line_source": "live",
            "line_consistency_flag": True,
            "line_event_identity_match_flag": True,
        },
        {
            "best_pick": "Chicago +1.5",
            "Pick_Status": "Below Threshold",
            "selection_probability_used": 0.58,
            "effective_expected_value": 0.03,
            "effective_edge": 0.02,
            "market_line_source": "live",
            "line_consistency_flag": True,
            "line_event_identity_match_flag": True,
        },
        {
            "best_pick": "Seattle +1.5",
            "Pick_Status": "Below Threshold",
            "selection_probability_used": 0.60,
            "effective_expected_value": 0.03,
            "effective_edge": 0.02,
            "market_line_source": "live",
            "line_consistency_flag": True,
            "line_event_identity_match_flag": True,
        },
    ])

    classified = classify_best_available_picks(frame)

    assert classified["qualified_pick"].tolist() == [False, False, True]
    assert classified["display_pick"].tolist() == [
        "Under 8.5", "Chicago +1.5", "Seattle +1.5"
    ]
    assert "below 60%" in classified.loc[0, "qualification_reason"]
    assert "below 60%" in classified.loc[1, "qualification_reason"]
    assert classified.loc[2, "commercial_tier"] == "Qualified Lean / Pass"


def test_qualified_pick_uses_rowwise_fallback_when_production_metrics_are_sparse():
    frame = pd.DataFrame([{
        "best_pick": "Chicago +1.5",
        "Pick_Status": "Below Threshold",
        "selection_probability_used": pd.NA,
        "effective_win_probability": 0.61,
        "production_expected_value": pd.NA,
        "effective_expected_value": 0.03,
        "production_edge": pd.NA,
        "effective_edge": 0.02,
        "market_line_source": "live",
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
    }])

    classified = classify_best_available_picks(frame)

    assert bool(classified.loc[0, "qualified_pick"])
    assert classified.loc[0, "qualification_probability"] == 0.61
    assert classified.loc[0, "display_pick"] == "Chicago +1.5"


def test_final_empirical_edge_downgrades_preoverlay_qualified_lean_label():
    frame = pd.DataFrame([{
        "best_pick": "Washington +1.5",
        "Pick_Status": "Below Threshold",
        "selection_probability_used": 0.5772,
        "production_expected_value": 0.0024,
        "production_edge": 0.0226,
        "empirical_win_probability": 0.5400,
        "empirical_edge": -0.0450,
        "market_line_source": "live",
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "consensus_agreement": "Neutral",
    }])

    classified = classify_best_available_picks(frame)

    assert not bool(classified.loc[0, "qualified_pick"])
    assert classified.loc[0, "qualification_probability"] == 0.5400
    assert classified.loc[0, "commercial_tier"] == "Best Available / Pass"
    assert classified.loc[0, "export_role"] == "BEST AVAILABLE PICK - PASS / RESEARCH"
    assert classified.loc[0, "display_pick"] == "Washington +1.5"
    assert "below 60%" in classified.loc[0, "qualification_reason"]


def _parlay_leg(matchup_id: str, pick: str) -> dict:
    return {
        "matchup_id": matchup_id,
        "best_pick": pick,
        "league": "MLB",
        "Pick_Status": "Actionable",
        "production_eligible": True,
        "calibrated_probability": 0.65,
        "effective_win_probability": 0.65,
        "market_probability": 0.52,
        "edge": 0.05,
        "odds_american": -110,
        "decimal_odds": 1.91,
        "consensus_agreement": "Agrees",
        "Conviction_Score": 80.0,
    }


def test_strict_production_parlays_are_explicitly_premium():
    frame = pd.DataFrame([
        _parlay_leg("game-a", "Cubs moneyline"),
        _parlay_leg("game-b", "Pirates moneyline"),
    ])

    parlays = generate_smart_parlays(frame, num_rr_candidates=5, calibration=None)

    assert not parlays.empty
    assert parlays["production_safety_mode"].all()
    assert parlays["premium_eligible"].all()
    assert parlays["sellable_as_premium"].all()
    assert parlays["parlay_class"].eq("Premium").all()
    assert parlays["commercial_warning"].eq("").all()


def test_strict_production_parlays_fail_closed_without_explicit_identity_or_price():
    frame = pd.DataFrame([
        _parlay_leg("game-a", "Cubs moneyline"),
        _parlay_leg("game-b", "Pirates moneyline"),
    ])

    for required_column in ("matchup_id", "market_probability"):
        parlays = generate_smart_parlays(
            frame.drop(columns=[required_column]),
            num_rr_candidates=5,
            calibration=None,
        )
        assert parlays.empty

def test_coverage_rows_cannot_retain_any_stake_like_value():
    frame = pd.DataFrame([
        {
            "best_pick": "Under 8.5",
            "Pick_Status": "Below Threshold",
            "production_eligible": False,
            "production_bet_amount": 12.0,
            "Kelly_Bet_Size": 9.0,
            "Play_Stake": 7.0,
            "recommended_bet": 5.0,
            "Suggested_Stake": 3.0,
            "production_expected_value": 0.05,
            "production_edge": 0.03,
            "market_line_source": "live",
            "line_consistency_flag": True,
            "line_event_identity_match_flag": True,
        }
    ])

    classified = classify_best_available_picks(frame)
    row = classified.iloc[0]

    assert not bool(row["wager_approved"])
    assert row["export_role"] == "BEST AVAILABLE PICK - PASS / RESEARCH"
    for column in (
        "production_bet_amount",
        "Kelly_Bet_Size",
        "Play_Stake",
        "recommended_bet",
        "Suggested_Stake",
    ):
        assert float(row[column]) == 0.0

def test_resolved_model_direction_survives_overlays_and_both_ranking_stages(monkeypatch):
    monkeypatch.setattr('core.empirical_tiers.load_bucket_stats', lambda: {})
    monkeypatch.setattr('core.probability_calibration.load_calibration', lambda: None)
    rows = [_candidate('total_over', .49, .02), _candidate('total_under', .51, .02)]
    for row, p in zip(rows, [.64, .36]):
        row.update(ml_probability=p, ml_target=row['market_type'],
                   ml_probability_source='score-distribution-v1:mlb',
                   ml_feature_quality='resolved_team_scoring_stats')
    diagnostics = {}
    best = build_best_picks_df(pd.DataFrame(rows), diagnostics_out=diagnostics)
    assert best.iloc[0].market_type == 'total_over'
    audit = diagnostics['candidate_audit_df']
    winner = audit[audit.best_available_selected].iloc[0]
    assert winner.best_available_rank == 1
    assert winner.best_available_family_rank == 1
    assert bool(winner.best_available_selection_verified)
    assert audit.loc[audit.market_type.eq('total_under'), 'model_direction_guard_applied'].all()
    assert not audit.wager_approved.any()
