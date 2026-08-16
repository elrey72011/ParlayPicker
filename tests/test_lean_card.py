"""All-games lean view: model's read on every game, tiered BET / LEAN / AVOID.

Pins the honest tiering so a near-efficient slate (mostly Below Threshold / No Play) still
produces a usable full-board read: the +EV-but-below-bar picks surface as LEAN, the proven
ones as BET, and the negative-EV / Kalshi-fading ones as AVOID.
"""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.lean_card import build_all_games_lean_card, classify_lean_tier


def test_classify_tiers():
    assert classify_lean_tier(
        "Actionable", 0.05, "Neutral", calibrated_win=0.58, break_even=0.524
    ) == "BET"
    assert classify_lean_tier(
        "Actionable", 0.05, "Neutral", calibrated_win=0.535, break_even=0.524
    ) == "LEAN"  # Actionable label alone cannot fund a thin priced edge
    assert classify_lean_tier("Below Threshold", 0.03, "Agrees") == "LEAN"   # +EV, not fading -> lean
    assert classify_lean_tier("Below Threshold", 0.03, "Disagrees") == "AVOID"  # fading Kalshi -> avoid
    assert classify_lean_tier("No Play", -0.02, "Neutral") == "AVOID"        # -EV -> avoid
    assert classify_lean_tier("Below Threshold", 0.0, "Neutral") == "AVOID"  # zero EV -> not a lean


def test_calibration_gate_demotes_overconfident_lean():
    # Positive raw EV, but calibrated win below break-even -> AVOID (the 6-23 pattern).
    assert classify_lean_tier("Below Threshold", 0.02, "Neutral",
                              calibrated_win=0.46, break_even=0.526) == "AVOID"
    # Calibrated win still beats break-even -> stays LEAN.
    assert classify_lean_tier("Below Threshold", 0.02, "Agrees",
                              calibrated_win=0.57, break_even=0.524) == "LEAN"
    # Actionable is independently price-gated; negative calibrated edge is a pass.
    assert classify_lean_tier("Actionable", 0.05, "Agrees",
                              calibrated_win=0.40, break_even=0.55) == "AVOID"


def _df(rows):
    return pd.DataFrame(rows)


def test_lean_card_orders_and_labels_full_slate():
    df = _df([
        {"league": "MLB", "home_team": "A", "away_team": "B", "best_pick": "Over 8.5",
         "Pick_Status": "No Play", "effective_expected_value": -0.02,
         "effective_win_probability": 0.49, "effective_edge": 0.01,
         "odds_american": -110, "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0},
        {"league": "MLB", "home_team": "C", "away_team": "D", "best_pick": "Under 8.5",
         "Pick_Status": "Below Threshold", "effective_expected_value": 0.03,
         "effective_win_probability": 0.53, "effective_edge": 0.035,
         "odds_american": -110, "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0},
        {"league": "MLB", "home_team": "E", "away_team": "F", "best_pick": "Under 7.5",
         "Pick_Status": "Actionable", "effective_expected_value": 0.05,
         "effective_win_probability": 0.57, "effective_edge": 0.06,
         "odds_american": -110, "consensus_agreement": "Agrees", "Kelly_Bet_Size": 8.0},
    ])
    card = build_all_games_lean_card(df, calibration=None)         # raw behavior, deterministic
    assert list(card["Tier"]) == ["BET", "LEAN", "AVOID"]          # ordered by tier
    assert card.iloc[0]["Matchup"] == "F @ E"
    assert float(card.iloc[0]["Suggested_Stake"]) == 8.0            # stake only on BET
    assert card.iloc[0]["Bet_Decision"] == "BET"
    assert float(card.iloc[1]["Suggested_Stake"]) == 0.0           # LEAN is a read, no stake
    assert card.iloc[1]["Bet_Decision"] == "BEST AVAILABLE - PASS"
    assert float(card.iloc[2]["Suggested_Stake"]) == 0.0


def test_ranks_by_empirical_edge_not_model_ev():
    # Two LEAN picks, same -110 odds. Pick A has HIGHER model EV (raw win .60) but sits in a
    # bad bucket (under:Disagrees); pick B has LOWER model EV (raw win .57) but a proven bucket
    # (under:Agrees). Ranked by model EV, A wins; ranked by empirical edge, B must come first.
    cal = [[0.50, 0.46], [0.57, 0.55], [0.60, 0.57], [0.65, 0.60]]
    bstats = {
        "overall": {"n": 342, "win_rate": 0.51},
        "buckets": {
            "MLB:under:Agrees": {"n": 65, "wins": 41, "win_rate": 0.631},
            "MLB:under:Disagrees": {"n": 33, "wins": 14, "win_rate": 0.424},
        },
    }
    df = _df([
        {"league": "MLB", "home_team": "A", "away_team": "B", "best_pick": "Under 8.5",
         "market_type": "total_under", "Pick_Status": "Below Threshold",
         "effective_expected_value": 0.15, "effective_win_probability": 0.60,
         "effective_edge": 0.08, "odds_american": -110, "consensus_agreement": "Disagrees",
         "Kelly_Bet_Size": 0.0},
        {"league": "MLB", "home_team": "C", "away_team": "D", "best_pick": "Under 7.5",
         "market_type": "total_under", "Pick_Status": "Below Threshold",
         "effective_expected_value": 0.06, "effective_win_probability": 0.57,
         "effective_edge": 0.04, "odds_american": -110, "consensus_agreement": "Agrees",
         "Kelly_Bet_Size": 0.0},
    ])
    card = build_all_games_lean_card(df, calibration=cal, bucket_stats=bstats)
    # B (under:Agrees, lower EV) ranks ABOVE A (under:Disagrees, higher EV) by empirical edge.
    assert card.iloc[0]["Pick"] == "Under 7.5"
    assert card.iloc[0]["Consensus"] == "Agrees"
    assert card.iloc[0]["Emp_Edge"] > card.iloc[1]["Emp_Edge"]
    assert card.iloc[0]["EV"] < card.iloc[1]["EV"]   # ...even though its model EV is lower


def test_lean_card_empty_when_no_games():
    assert build_all_games_lean_card(pd.DataFrame()).empty
    assert build_all_games_lean_card(None).empty


def test_lean_card_handles_export_column_names():
    # Post-export rename uses Home/Away/WinProbability instead of home_team/expected_value.
    df = _df([
        {"League": "MLB", "Home": "A", "Away": "B", "best_pick": "Under 8.5",
         "Pick_Status": "Below Threshold", "expected_value": 0.02,
         "WinProbability": 0.52, "edge": 0.03, "consensus_agreement": "Agrees",
         "Kelly_Bet_Size": 0.0},
    ])
    card = build_all_games_lean_card(df, calibration=None)
    assert len(card) == 1
    assert card.iloc[0]["Tier"] == "LEAN"
    assert card.iloc[0]["Matchup"] == "B @ A"


def test_lean_card_shows_best_available_pick_and_keeps_pass():
    df = _df([{
        "league": "MLB",
        "home_team": "A",
        "away_team": "B",
        "best_pick": "Under 8.5",
        "display_pick": "NO QUALIFIED PICK",
        "qualified_pick": False,
        "qualification_reason": "NO QUALIFIED PICK: win probability is below 55%.",
        "Pick_Status": "Below Threshold",
        "effective_expected_value": 0.03,
        "effective_win_probability": 0.53,
        "effective_edge": 0.02,
        "odds_american": -110,
        "consensus_agreement": "Neutral",
        "Kelly_Bet_Size": 0.0,
    }])

    card = build_all_games_lean_card(df, calibration=None, bucket_stats=None)

    assert card.loc[0, "Pick"] == "Under 8.5"
    assert card.loc[0, "Tier"] == "AVOID"
    assert card.loc[0, "Bet_Decision"] == "BEST AVAILABLE - PASS"
    assert card.loc[0, "Selection_Mode"] == "Best Available Pick / Pass"


def test_final_avoid_tier_downgrades_stale_qualified_lean_label():
    df = _df([{
        "league": "MLB",
        "home_team": "Philadelphia",
        "away_team": "Washington",
        "best_pick": "Washington +1.5",
        "qualified_pick": True,
        "qualification_reason": "Qualified research lean from pre-overlay metrics.",
        "Pick_Status": "Below Threshold",
        "effective_expected_value": 0.0024,
        "effective_win_probability": 0.5400,
        "effective_edge": 0.0226,
        "odds_american": -141,
        "consensus_agreement": "Neutral",
        "line_consistency_flag": True,
        "line_event_identity_match_flag": True,
        "Kelly_Bet_Size": 0.0,
    }])

    card = build_all_games_lean_card(df, calibration=None, bucket_stats=None)
    exported = attach_play_stakes(card, unit=1.0)

    assert card.loc[0, "Tier"] == "AVOID"
    assert not bool(card.loc[0, "Qualified_Pick"])
    assert card.loc[0, "Bet_Decision"] == "BEST AVAILABLE - PASS"
    assert card.loc[0, "Selection_Mode"] == "Best Available Pick / Pass"
    assert card.loc[0, "Qualification_Reason"] == (
        "PASS: final empirical tier is AVOID at the offered price."
    )
    assert exported.loc[0, "Export_Role"] == "BEST AVAILABLE PICK - PASS / RESEARCH"


def test_play_card_preserves_pipeline_and_run_stamps():
    df = _df([
        {
            "pipeline_build": "2026-07-30e-spread-price-pairing-guard",
            "export_run_id": "20260816T145118Z",
            "League": "WNBA",
            "Home": "Chicago",
            "Away": "Connecticut",
            "best_pick": "Chicago -4.5",
            "Pick_Status": "No Play",
            "expected_value": -0.02,
            "WinProbability": 0.51,
            "edge": 0.0,
            "odds_american": -111,
            "consensus_agreement": "Agrees",
            "Kelly_Bet_Size": 0.0,
        }
    ])

    card = build_all_games_lean_card(df, calibration=None, bucket_stats=None)

    assert card.columns[0] == "pipeline_build"
    assert card["pipeline_build"].eq(
        "2026-07-30e-spread-price-pairing-guard"
    ).all()
    assert card.columns[1] == "export_run_id"
    assert card["export_run_id"].eq("20260816T145118Z").all()


def test_build_applies_calibration_gate_end_to_end():
    # Raw win 0.53 @ -110 (b/e .524) looks like a LEAN, but a calibration that shrinks the
    # 0.50-0.55 band to ~.45 pushes it below break-even -> AVOID. A strong 0.62 pick survives.
    cal = [[0.45, 0.40], [0.55, 0.46], [0.62, 0.585], [0.70, 0.66]]
    df = _df([
        {"league": "MLB", "home_team": "A", "away_team": "B", "best_pick": "Under 8.5",
         "Pick_Status": "Below Threshold", "effective_expected_value": 0.02,
         "effective_win_probability": 0.53, "odds_american": -110,
         "consensus_agreement": "Neutral", "Kelly_Bet_Size": 0.0},
        {"league": "MLB", "home_team": "C", "away_team": "D", "best_pick": "Over 8.5",
         "Pick_Status": "Below Threshold", "effective_expected_value": 0.04,
         "effective_win_probability": 0.62, "odds_american": -110,
         "consensus_agreement": "Agrees", "Kelly_Bet_Size": 0.0},
    ])
    card = build_all_games_lean_card(df, calibration=cal)
    by_pick = {r["Pick"]: r["Tier"] for _, r in card.iterrows()}
    assert by_pick["Under 8.5"] == "AVOID"   # calibrated below break-even
    assert by_pick["Over 8.5"] == "LEAN"     # calibrated still beats break-even
    # Calib_Win% is surfaced for transparency.
    assert (card["Calib_Win%"] < card["Win%"]).all()


# -- Play stakes: every game playable at flat recreational units (owner, 4 Jul) --
from app_core.lean_card import (  # noqa: E402
    AVOID_NEAR_EDGE,
    PLAY_UNITS_AVOID_FAR,
    PLAY_UNITS_AVOID_NEAR,
    PLAY_UNITS_BET,
    PLAY_UNITS_CONTROLLED_VALUE,
    PLAY_UNITS_LEAN,
    attach_play_stakes,
)


def _play_card():
    return pd.DataFrame({
        "Matchup": ["A @ B", "C @ D", "E @ F", "G @ H"],
        "Tier": ["BET", "LEAN", "AVOID", "AVOID"],
        "Emp_Edge": [0.04, 0.01, -0.02, -0.12],
        "Suggested_Stake": [7.0, 0.0, 0.0, 0.0],
    })


def test_only_bet_rows_get_a_positive_play_stake():
    out = attach_play_stakes(_play_card(), unit=5.0)
    assert out.loc[out["Tier"] == "BET", "Play_Stake"].gt(0).all()
    assert out.loc[out["Tier"] != "BET", "Play_Stake"].eq(0).all()


def test_units_scale_down_with_confidence():
    out = attach_play_stakes(_play_card(), unit=5.0)
    by_matchup = out.set_index("Matchup")["Play_Units"]
    assert by_matchup["A @ B"] == PLAY_UNITS_BET
    assert by_matchup["C @ D"] == PLAY_UNITS_LEAN
    assert by_matchup["E @ F"] == PLAY_UNITS_AVOID_NEAR   # -0.02 >= -0.05: thin miss
    assert by_matchup["G @ H"] == PLAY_UNITS_AVOID_FAR    # -0.12: clearly losing price
    assert AVOID_NEAR_EDGE == -0.05


def test_bet_tier_keeps_larger_kelly_stake():
    card = _play_card()
    card.loc[0, "Suggested_Stake"] = 25.0  # Kelly above 2u at $5
    out = attach_play_stakes(card, unit=5.0)
    assert out.set_index("Matchup").loc["A @ B", "Play_Stake"] == 25.0


def test_controlled_value_bet_uses_small_units_and_distinct_public_label():
    card = pd.DataFrame({
        "Matchup": ["Athletics @ Cincinnati"],
        "Tier": ["BET"],
        "Emp_Edge": [0.042],
        "Suggested_Stake": [0.0],
        "Qualified_Pick": [True],
        "Controlled_Value_Card": [True],
    })

    out = attach_play_stakes(card, unit=5.0)
    row = out.iloc[0]

    assert row["Play_Units"] == PLAY_UNITS_CONTROLLED_VALUE
    assert row["Play_Stake"] == 2.5
    assert bool(row["Wager_Approved"])
    assert row["Export_Role"] == "CONTROLLED VALUE WAGER"
    assert "not a Premium pick" in row["Wager_Instruction"]


def test_empty_card_is_safe():
    assert attach_play_stakes(pd.DataFrame()).empty
    assert attach_play_stakes(None).empty


# -- score_best_picks_rows: index-aligned tiers for the MAIN card (owner, 4 Jul) --
from app_core.lean_card import score_best_picks_rows  # noqa: E402


def test_score_rows_is_index_aligned_and_unsorted():
    df = pd.DataFrame({
        "league": ["MLB", "MLB"],
        "home_team": ["Cincinnati", "Cleveland"],
        "away_team": ["Baltimore", "Chicago White Sox"],
        "Pick_Status": ["Below Threshold", "Actionable"],
        "best_pick": ["Baltimore +1.5", "Chicago White Sox +1.5"],
        "effective_expected_value": [0.03, 0.05],
        "consensus_agreement": ["Agrees", "Agrees"],
        "effective_win_probability": [0.58, 0.60],
        "effective_edge": [0.02, 0.03],
        "odds_american": [-110, -110],
        "Kelly_Bet_Size": [0.0, 12.0],
    }, index=[7, 3])
    out = score_best_picks_rows(df, calibration=None, bucket_stats=None)
    assert list(out.index) == [7, 3]          # aligned, original order kept
    assert out.loc[3, "Tier"] == "BET"
    assert out.loc[3, "Suggested_Stake"] == 12.0
    assert out.loc[7, "Tier"] in ("LEAN", "AVOID")


def test_score_rows_then_play_stakes_keeps_non_bets_at_zero():
    df = pd.DataFrame({
        "league": ["MLB"] * 3,
        "home_team": ["A", "B", "C"],
        "away_team": ["X", "Y", "Z"],
        "Pick_Status": ["No Play", "Below Threshold", "High Variance/Speculative"],
        "best_pick": ["Over 7.5", "Under 8.5", "X +1.5"],
        "effective_expected_value": [-0.03, 0.02, 0.10],
        "consensus_agreement": ["Agrees", "Agrees", "Neutral"],
        "effective_win_probability": [0.51, 0.56, 0.63],
        "effective_edge": [0.0, 0.02, 0.06],
        "odds_american": [-110, -110, -140],
        "Kelly_Bet_Size": [0.0, 0.0, 0.0],
    })
    staked = attach_play_stakes(score_best_picks_rows(df, calibration=None, bucket_stats=None), unit=5.0)
    assert (staked["Play_Stake"] == 0).all()
    assert list(staked.index) == list(df.index)


def test_actionable_row_below_absolute_margin_is_visible_but_unstaked():
    df = pd.DataFrame({
        "league": ["MLB"],
        "home_team": ["A"],
        "away_team": ["B"],
        "Pick_Status": ["Actionable"],
        "best_pick": ["A +1.5"],
        "effective_expected_value": [0.03],
        "consensus_agreement": ["Agrees"],
        "effective_win_probability": [0.535],
        "effective_edge": [0.02],
        "odds_american": [-110],
        "Kelly_Bet_Size": [10.0],
    })
    scored = score_best_picks_rows(df, calibration=None, bucket_stats=None)
    staked = attach_play_stakes(scored, unit=5.0)

    assert scored.iloc[0]["Tier"] == "LEAN"
    assert scored.iloc[0]["Bet_Decision"] == "BEST AVAILABLE - PASS"
    assert not bool(scored.iloc[0]["Production_Gate_Pass"])
    assert float(staked.iloc[0]["Play_Stake"]) == 0.0


def test_actionable_row_with_positive_ev_and_absolute_edge_is_funded():
    df = pd.DataFrame({
        "league": ["MLB"],
        "home_team": ["A"],
        "away_team": ["B"],
        "Pick_Status": ["Actionable"],
        "best_pick": ["A +1.5"],
        "effective_expected_value": [0.05],
        "consensus_agreement": ["Agrees"],
        "effective_win_probability": [0.58],
        "effective_edge": [0.05],
        "odds_american": [-110],
        "Kelly_Bet_Size": [10.0],
    })
    staked = attach_play_stakes(
        score_best_picks_rows(df, calibration=None, bucket_stats=None),
        unit=5.0,
    )

    assert staked.iloc[0]["Tier"] == "BET"
    assert staked.iloc[0]["Bet_Decision"] == "BET"
    assert bool(staked.iloc[0]["Production_Gate_Pass"])
    assert float(staked.iloc[0]["Play_Stake"]) == 10.0


def test_controlled_recovery_keeps_empirical_price_authority_in_public_card():
    # The recovered row cleared its exact-price gate on empirical probability.
    # Its legacy effective probability is deliberately below the absolute edge
    # margin and must not be used to reverse the final public wager decision.
    df = pd.DataFrame({
        "league": ["MLB"],
        "home_team": ["San Francisco"],
        "away_team": ["Texas"],
        "Pick_Status": ["Actionable"],
        "best_pick": ["San Francisco +1.5"],
        "qualified_pick": [True],
        "controlled_card_recovery": [True],
        "wager_approved": [True],
        "production_eligible": [True],
        "production_bet_amount": [5.0],
        "effective_expected_value": [0.04],
        "consensus_agreement": ["Agrees"],
        "effective_win_probability": [0.56023],
        "empirical_win_probability": [0.573626],
        "effective_edge": [0.03],
        "odds_american": [-120],
        "line_consistency_flag": [True],
        "line_event_identity_match_flag": [True],
        "Kelly_Bet_Size": [5.0],
    })

    scored = score_best_picks_rows(df, calibration=None, bucket_stats=None)
    staked = attach_play_stakes(scored, unit=5.0)

    assert scored.loc[0, "Calib_Win%"] == pytest.approx(0.573626)
    assert scored.loc[0, "Absolute_Edge"] == pytest.approx(
        0.573626 - (120 / 220)
    )
    assert scored.loc[0, "Tier"] == "BET"
    assert scored.loc[0, "Bet_Decision"] == "BET"
    assert bool(scored.loc[0, "Production_Gate_Pass"])
    assert bool(scored.loc[0, "Controlled_Value_Card"])
    assert float(staked.loc[0, "Play_Stake"]) == 5.0


def test_explicit_unfunded_authorization_downgrades_mathematical_bet_to_lean():
    df = pd.DataFrame({
        "league": ["MLB"],
        "home_team": ["Colorado"],
        "away_team": ["Tampa Bay"],
        "Pick_Status": ["Actionable"],
        "best_pick": ["Under 11.5"],
        "qualified_pick": [True],
        "wager_approved": [False],
        "production_eligible": [False],
        "production_bet_amount": [0.0],
        "effective_expected_value": [0.05],
        "consensus_agreement": ["Agrees"],
        "effective_win_probability": [0.58],
        "effective_edge": [0.05],
        "odds_american": [-110],
        "Kelly_Bet_Size": [0.0],
    })

    scored = score_best_picks_rows(df, calibration=None, bucket_stats=None)
    staked = attach_play_stakes(scored, unit=5.0)

    assert scored.iloc[0]["Tier"] == "LEAN"
    assert scored.iloc[0]["Bet_Decision"] == "QUALIFIED LEAN - PASS"
    assert not bool(scored.iloc[0]["Production_Gate_Pass"])
    assert "approval" in scored.iloc[0]["Production_Gate_Reason"]
    assert float(staked.iloc[0]["Play_Stake"]) == 0.0
    assert not bool(staked.iloc[0]["Wager_Approved"])


def test_build_card_matches_row_scores():
    df = pd.DataFrame({
        "league": ["MLB", "MLB"],
        "home_team": ["Cincinnati", "Cleveland"],
        "away_team": ["Baltimore", "Chicago White Sox"],
        "Pick_Status": ["Below Threshold", "No Play"],
        "best_pick": ["Baltimore +1.5", "Chicago White Sox +1.5"],
        "effective_expected_value": [0.03, -0.04],
        "consensus_agreement": ["Agrees", "Agrees"],
        "effective_win_probability": [0.58, 0.60],
        "effective_edge": [0.02, -0.01],
        "odds_american": [-110, -110],
        "Kelly_Bet_Size": [0.0, 0.0],
    })
    rows = score_best_picks_rows(df, calibration=None, bucket_stats=None)
    card = build_all_games_lean_card(df, calibration=None, bucket_stats=None)
    assert sorted(card["Tier"]) == sorted(rows["Tier"])
    assert set(card["Pick"]) == set(rows["Pick"])


def test_started_games_get_zero_play_stake():
    df = pd.DataFrame({
        "league": ["MLB", "MLB"],
        "home_team": ["New York Yankees", "Cincinnati"],
        "away_team": ["Minnesota", "Baltimore"],
        "Pick_Status": ["No Play", "No Play"],
        "best_pick": ["Over 10.5", "Baltimore +1.5"],
        "effective_expected_value": [0.0, -0.01],
        "consensus_agreement": ["Agrees", "Agrees"],
        "effective_win_probability": [0.57, 0.63],
        "effective_edge": [0.0, 0.01],
        "odds_american": [-130, -170],
        "Kelly_Bet_Size": [0.0, 0.0],
        "status_blocker_stage": ["game_already_started", "baseline_guardrail"],
    })
    out = attach_play_stakes(score_best_picks_rows(df, calibration=None, bucket_stats=None), unit=5.0)
    assert out.iloc[0]["Play_Stake"] == 0.0
    assert out.iloc[0]["Tier"] == "STARTED"
    assert out.iloc[1]["Play_Stake"] == 0.0


def test_hopeless_prices_get_zero_play_stake():
    # CWS +5.5 at -1718 (4 Jul): Emp_Edge -0.35 - no recreational stake at any size.
    card = pd.DataFrame({
        "Matchup": ["A @ B", "C @ D"],
        "Tier": ["AVOID", "AVOID"],
        "Emp_Edge": [-0.35, -0.10],
        "Suggested_Stake": [0.0, 0.0],
    })
    out = attach_play_stakes(card, unit=5.0)
    assert out.iloc[0]["Play_Stake"] == 0.0
    assert out.iloc[1]["Play_Stake"] == 0.0


def test_unresolved_line_is_unavailable_and_never_staked():
    df = pd.DataFrame({
        "league": ["MLB"],
        "home_team": ["A"],
        "away_team": ["B"],
        "Pick_Status": ["No Play"],
        "best_pick": ["Total line unresolved"],
        "effective_expected_value": [0.02],
        "consensus_agreement": ["Agrees"],
        "effective_win_probability": [0.60],
        "effective_edge": [0.03],
        "odds_american": [-110],
        "Kelly_Bet_Size": [0.0],
    })
    out = attach_play_stakes(
        score_best_picks_rows(df, calibration=None, bucket_stats=None),
        unit=1.0,
    )
    assert out.iloc[0]["Tier"] == "UNAVAILABLE"
    assert out.iloc[0]["Bet_Decision"] == "UNAVAILABLE"
    assert out.iloc[0]["Play_Stake"] == 0.0


def test_line_availability_is_distinct_from_wager_approval():
    source = pd.DataFrame([{
        "league": "MLB",
        "home_team": "Texas",
        "away_team": "Seattle",
        "best_pick": "Under 8.0",
        "Pick_Status": "No Play",
        "effective_expected_value": -0.01,
        "expected_value": -0.01,
        "edge": -0.01,
        "effective_win_probability": 0.49,
        "odds_american": -110,
        "consensus_agreement": "Neutral",
        "Kelly_Bet_Size": 0.0,
    }])

    out = attach_play_stakes(
        build_all_games_lean_card(source, calibration=None, bucket_stats=None),
        unit=1.0,
    )

    assert bool(out.iloc[0]["Line_Available"])
    assert "Playable" not in out.columns
    assert not bool(out.iloc[0]["Wager_Approved"])
    assert out.iloc[0]["Bet_Decision"] == "BEST AVAILABLE - PASS"


def test_exported_started_tier_stays_unplayable_on_all_games_card():
    source = pd.DataFrame([{
        "league": "MLB",
        "home_team": "Texas",
        "away_team": "Seattle",
        "best_pick": "Over 8.5",
        "Pick_Status": "No Play",
        "effective_expected_value": -0.02,
        "expected_value": -0.02,
        "edge": -0.01,
        "effective_win_probability": 0.49,
        "odds_american": -110,
        "consensus_agreement": "Neutral",
        "Play_Tier": "STARTED",
        "status_blocker_stage": "some_later_guard",
    }])

    out = attach_play_stakes(
        build_all_games_lean_card(source, calibration=None, bucket_stats=None),
        unit=1.0,
    )

    assert bool(out.iloc[0]["Started"])
    assert not bool(out.iloc[0]["Line_Available"])
    assert out.iloc[0]["Tier"] == "STARTED"
    assert out.iloc[0]["Bet_Decision"] == "STARTED"
    assert float(out.iloc[0]["Play_Units"]) == 0.0
    assert float(out.iloc[0]["Play_Stake"]) == 0.0
    assert not bool(out.iloc[0]["All_Row_Bet"])


def test_repaired_upload_fallback_is_unavailable_for_production_stake():
    source = pd.DataFrame([{
        "league": "MLB",
        "home_team": "Texas",
        "away_team": "Seattle",
        "best_pick": "Under 8.0",
        "Pick_Status": "Actionable",
        "effective_expected_value": 0.06,
        "expected_value": 0.06,
        "edge": 0.05,
        "effective_win_probability": 0.60,
        "odds_american": -108,
        "consensus_agreement": "Agrees",
        "line_consistency_flag": True,
        "line_event_identity_match_flag": False,
        "market_line_source_detail": "upload_total_fallback_after_rejected_live",
        "Kelly_Bet_Size": 10.0,
    }])

    out = attach_play_stakes(
        build_all_games_lean_card(source, calibration=None, bucket_stats=None),
        unit=1.0,
    )

    assert not bool(out.iloc[0]["Started"])
    assert not bool(out.iloc[0]["Line_Available"])
    assert out.iloc[0]["Tier"] == "UNAVAILABLE"
    assert out.iloc[0]["Bet_Decision"] == "UNAVAILABLE"
    assert float(out.iloc[0]["Play_Units"]) == 0.0
    assert float(out.iloc[0]["Play_Stake"]) == 0.0
    assert not bool(out.iloc[0]["All_Row_Bet"])


def test_play_card_labels_production_wagers_and_coverage_passes_explicitly():
    out = attach_play_stakes(_play_card(), unit=5.0)
    approved = out["Wager_Approved"]

    assert approved.sum() == 1
    assert out.loc[approved, "Export_Role"].eq("PRODUCTION WAGER").all()
    assert out.loc[approved, "Wager_Instruction"].str.startswith("APPROVED").all()
    assert out.loc[~approved, "Export_Role"].eq("COVERAGE PICK - PASS").all()
    assert out.loc[~approved, "Wager_Instruction"].str.startswith("DO NOT BET").all()
