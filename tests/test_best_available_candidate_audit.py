"""Best-available selection and commercial product-boundary regressions."""
from __future__ import annotations

import pandas as pd

from core.smart_parlay_engine import generate_smart_parlays
from core.streamlit_pipeline import (
    build_best_picks_df,
    classify_best_available_picks,
)


def _candidate(market_type: str, probability: float, ev: float) -> dict:
    is_total = market_type.startswith("total")
    spread_line = -1.5 if market_type == "spread_home" else 1.5 if market_type == "spread_away" else pd.NA
    return {
        "game_id": "best-available-audit",
        "league": "MLB",
        "home_team": "Chicago Cubs",
        "away_team": "Pittsburgh Pirates",
        "game_date": pd.Timestamp("2026-07-27", tz="UTC"),
        "market_type": market_type,
        "total_line": 8.5 if is_total else pd.NA,
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
        "live_total_line": 8.5,
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


def test_commercial_tier_never_upgrades_an_unfunded_best_available_row():
    frame = pd.DataFrame([
        {
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
    assert bool(classified.loc[0, "wager_approved"])
    assert classified.loc[0, "export_role"] == "PRODUCTION WAGER"
    assert not bool(classified.loc[1, "wager_approved"])
    assert classified.loc[1, "export_role"] == "COVERAGE PICK - PASS"
    assert classified.loc[1, "wager_instruction"].startswith("DO NOT TREAT")


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

def test_coverage_rows_cannot_retain_any_stake_like_value():
    frame = pd.DataFrame([
        {
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
    assert row["export_role"] == "COVERAGE PICK - PASS"
    for column in (
        "production_bet_amount",
        "Kelly_Bet_Size",
        "Play_Stake",
        "recommended_bet",
        "Suggested_Stake",
    ):
        assert float(row[column]) == 0.0
