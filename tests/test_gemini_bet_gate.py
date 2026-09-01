from __future__ import annotations

import pandas as pd

from app_core.gemini_bet_gate import apply_gemini_bet_gate, classify_gemini_review
from core.streamlit_pipeline import optimize_portfolio_allocation
from integrations.gemini_client import run_gemini_analysis, run_gemini_prop_analysis


def _review_row(**overrides) -> dict:
    row = {
        "best_pick": "Home +1.5",
        "gemini_pick": "Home +1.5",
        "gemini_confidence": "HIGH",
        "gemini_flags": "",
        "gemini_explanation": "The calibrated probability clears the offered price.",
        "gemini_risk_notes": "Normal market movement risk.",
        "production_eligible": True,
        "Kelly_Bet_Size": 10.0,
    }
    row.update(overrides)
    return row


def _portfolio_row(**overrides) -> dict:
    row = {
        "league": "MLB",
        "home_team": "Home",
        "away_team": "Away",
        "best_pick": "Home +1.5",
        "Pick_Status": "Actionable",
        "market_type": "spread_home",
        "market_line_source": "live",
        "market_line_used": 1.5,
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "line_provenance_warning": "",
        "model_status": "ML Model",
        "stats_source": "live",
        "fallback_summary_by_league": "",
        "run_health_warning": "",
        "degraded_feature_subset_flag": False,
        "odds_american": -110,
        "decimal_odds": 1.0 + (100.0 / 110.0),
        "empirical_win_probability": 0.58,
        "effective_win_probability": 0.58,
        "effective_expected_value": 0.05,
        "expected_value": 0.05,
    }
    row.update(overrides)
    return row


def test_high_confidence_matching_review_approves_without_increasing_stake():
    result = apply_gemini_bet_gate(
        pd.DataFrame([_review_row()]),
        enabled=True,
        product="prop",
    ).iloc[0]

    assert result["gemini_review_status"] == "APPROVE"
    assert bool(result["gemini_approved"])
    assert result["gemini_stake_multiplier"] == 1.0
    assert result["Kelly_Bet_Size"] == 10.0


def test_medium_review_reduces_prop_stake_to_seventy_five_percent():
    result = apply_gemini_bet_gate(
        pd.DataFrame([_review_row(gemini_confidence="MEDIUM")]),
        enabled=True,
        product="prop",
    ).iloc[0]

    assert bool(result["gemini_approved"])
    assert result["gemini_stake_multiplier"] == 0.75
    assert result["Kelly_Bet_Size"] == 7.5


def test_disagreement_low_confidence_and_blocking_flags_hold_at_zero():
    rows = [
        _review_row(gemini_pick="Away -1.5"),
        _review_row(gemini_confidence="LOW"),
        _review_row(gemini_flags="no_value_at_price"),
        _review_row(
            gemini_pick="No Gemini pick",
            gemini_confidence="",
            gemini_explanation="Gemini analysis unavailable",
            gemini_risk_notes="Gemini analysis unavailable",
        ),
    ]
    result = apply_gemini_bet_gate(
        pd.DataFrame(rows), enabled=True, product="prop"
    )

    assert result["gemini_review_status"].tolist() == [
        "OPPOSE", "LOW_CONFIDENCE", "HOLD", "UNAVAILABLE"
    ]
    assert not result["gemini_approved"].any()
    assert result["Kelly_Bet_Size"].eq(0.0).all()
    assert not result["production_eligible"].any()


def test_gemini_never_promotes_an_ineligible_row():
    result = apply_gemini_bet_gate(
        pd.DataFrame([_review_row(production_eligible=False)]),
        enabled=True,
        product="best_pick",
    ).iloc[0]

    assert bool(result["gemini_approved"])
    assert not bool(result["production_eligible"])


def test_explicit_gemini_abstention_is_audited_separately_from_api_failure():
    status, reason, multiplier = classify_gemini_review(
        _review_row(gemini_pick="none", gemini_confidence="LOW")
    )

    assert status == "ABSTAIN"
    assert "abstained" in reason.lower()
    assert multiplier == 0.0


def test_disabled_gate_leaves_existing_eligibility_and_stake_untouched():
    result = apply_gemini_bet_gate(
        pd.DataFrame([_review_row()]),
        enabled=False,
        product="prop",
    ).iloc[0]

    assert result["gemini_review_status"] == "DISABLED"
    assert bool(result["production_eligible"])
    assert result["Kelly_Bet_Size"] == 10.0


def test_portfolio_holds_unapproved_review_and_scales_medium_review():
    base = _portfolio_row(
        gemini_gate_enabled=False,
        gemini_approved=False,
        gemini_stake_multiplier=1.0,
    )
    normal = optimize_portfolio_allocation(pd.DataFrame([base]), bankroll=1000.0)
    normal_amount = float(normal.iloc[0]["production_bet_amount"])
    assert normal_amount > 0.0

    held = optimize_portfolio_allocation(
        pd.DataFrame([{
            **base,
            "gemini_gate_enabled": True,
            "gemini_approved": False,
            "gemini_stake_multiplier": 0.0,
        }]),
        bankroll=1000.0,
    ).iloc[0]
    medium = optimize_portfolio_allocation(
        pd.DataFrame([{
            **base,
            "gemini_gate_enabled": True,
            "gemini_approved": True,
            "gemini_stake_multiplier": 0.75,
        }]),
        bankroll=1000.0,
    ).iloc[0]

    assert not bool(held["production_eligible"])
    assert float(held["production_bet_amount"]) == 0.0
    assert float(medium["production_bet_amount"]) == round(normal_amount * 0.75, 2)


def test_game_and_prop_wrappers_preserve_structured_review_fields(monkeypatch):
    def fake_batch(payload, session_state=None):
        return {
            str(item["game_id"]): {
                "recommended_bet": item["side_a"]["best_pick"],
                "confidence": "MEDIUM",
                "explanation": "Price and calibrated probability align.",
                "risk_notes": "Normal variance.",
                "flags": ["contrarian"],
            }
            for item in payload
        }

    monkeypatch.setattr(
        "app_core.llm_assistant.generate_batch_confidence_explanation",
        fake_batch,
    )
    game = pd.DataFrame([{
        "game_id": "g1",
        "matchup_id": "m1",
        "league": "MLB",
        "home_team": "Home",
        "away_team": "Away",
        "market_type": "total_under",
        "best_pick": "Under 8.5",
        "odds_american": -110,
        "market_probability": 0.52,
        "calibrated_probability": 0.57,
        "expected_value": 0.08,
        "edge": 0.05,
    }])
    game_result = run_gemini_analysis(game, analysis_df=game).iloc[0]

    prop = pd.DataFrame([{
        "league": "MLB",
        "matchup": "Away @ Home",
        "player": "Pitcher One",
        "participant_type": "pitcher",
        "market_type": "pitcher_strikeouts_over",
        "best_pick": "Pitcher One Over 5.5 Strikeouts",
        "line": 5.5,
        "expected_count": 6.4,
        "odds_american": -105,
        "MarketProbability": 0.51,
        "WinProbability": 0.58,
        "expected_value": 0.10,
        "edge": 0.07,
    }])
    prop_result = run_gemini_prop_analysis(prop).iloc[0]

    for result in (game_result, prop_result):
        assert result["gemini_confidence"] == "MEDIUM"
        assert result["gemini_flags"] == "contrarian"
        assert bool(result["gemini_reviewed"])
        assert classify_gemini_review(result)[0] == "APPROVE"
