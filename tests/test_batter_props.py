import pandas as pd

from app_core.batter_prop_pipeline import score_batter_prop
from app_core.mlb_batter_stats import batter_form_from_gamelog
from app_core.prop_odds_ingest import parse_pitcher_props
from app_core.prop_runner import (
    apply_batter_probation_exposure_cap,
    apply_novig_minimum_selection,
    apply_production_prop_gate,
    apply_extended_prop_stake_cap,
    apply_prop_game_guard,
    apply_prop_slate_guard,
    apply_prop_stake_status,
    build_prop_card,
)
from scripts.grade_props import grade_card


def _event():
    return {
        "home_team": "New York Mets",
        "away_team": "Boston Red Sox",
        "bookmakers": [{
            "key": "draftkings",
            "markets": [{
                "key": "batter_hits",
                "outcomes": [
                    {"name": "Over", "description": "Juan Soto", "point": 0.5, "price": -150},
                    {"name": "Under", "description": "Juan Soto", "point": 0.5, "price": 125},
                ],
            }],
        }],
    }


def test_batter_market_parses_participant_type():
    rows = parse_pitcher_props(_event(), "batter_hits")
    assert len(rows) == 1
    assert rows[0]["player"] == "Juan Soto"
    assert rows[0]["batter"] == "Juan Soto"
    assert rows[0]["pitcher"] is None
    assert rows[0]["participant_type"] == "batter"


def test_batter_form_blends_season_and_recent_without_target_date():
    splits = []
    for day in range(1, 31):
        splits.append({
            "date": f"2026-06-{day:02d}",
            "stat": {
                "hits": 2 if day > 20 else 1,
                "totalBases": 3 if day > 20 else 1,
                "plateAppearances": 4,
            },
        })
    form = batter_form_from_gamelog(splits, last_n=10, as_of_date="2026-07-01")
    assert form["n_games"] == 30
    assert form["avg_plate_appearances"] == 4
    assert 1.0 < form["hits_per_game"] < 2.0
    assert form["total_bases_per_game"] > form["hits_per_game"]


def test_batter_hits_scoring_requires_real_price_edge():
    row = parse_pitcher_props(_event(), "batter_hits")[0]
    scored = score_batter_prop(row, {
        "hits_per_game": 1.1,
        "total_bases_per_game": 1.7,
        "n_games": 80,
        "avg_plate_appearances": 4.2,
    })
    assert scored["recommendation"] == "over"
    assert scored["expected_stat"] == "hits"
    assert 0.55 <= scored["model_p_over"] <= 0.75
    assert 0.04 <= scored["best_edge"] <= 0.10


def test_batter_card_stays_research_only_while_on_probation():
    prop = parse_pitcher_props(_event(), "batter_hits")[0]
    card = build_prop_card(
        object(), "2026-07-10", 2026, 1000.0,
        kelly_per_pick_pct=0.01,
        kelly_total_pct=0.03,
        list_events=lambda c, sk, d: [{"id": "event-1"}],
        props_fetch=lambda c, sk, eid: [prop],
        schedule_fetch=lambda d: [],
        batter_form_fetch=lambda name, season, as_of_date=None: {
            "hits_per_game": 1.1,
            "total_bases_per_game": 1.7,
            "n_games": 80,
            "avg_plate_appearances": 4.2,
        },
    )
    assert len(card) == 1
    row = card.iloc[0]
    assert row["participant_type"] == "batter"
    assert row["market_type"] == "batter_hits_over"
    assert row["best_pick"] == "Juan Soto Over 0.5 Hits"
    assert bool(row["Market_Probation"])
    assert not bool(row["production_eligible"])
    assert row["Kelly_Bet_Size"] == 0.0
    assert row["Pick_Status"] == "Qualified / No Stake"
    assert row["Stake_Status"] == "Qualified / No Stake"
    assert "probation" in row["Status_Reason"].lower()


def test_batter_prop_grades_against_batting_result():
    card = pd.DataFrame([{
        "player": "Juan Soto",
        "batter": "Juan Soto",
        "participant_type": "batter",
        "market_type": "batter_total_bases_over",
        "best_pick": "Juan Soto Over 1.5 TB",
        "line": 1.5,
        "odds_american": -110,
        "Kelly_Bet_Size": 1.0,
        "Pick_Status": "Actionable",
        "expected_stat": "total_bases",
        "expected_count": 1.8,
    }])
    rows = grade_card(
        card,
        "2026-07-10",
        {"juan soto": 1},
        lambda player_id, participant_type: {"hits": 1, "total_bases": 2},
    )
    assert rows[0]["participant_type"] == "batter"
    assert rows[0]["stat"] == "total_bases"
    assert rows[0]["actual_value"] == 2
    assert rows[0]["result"] == "WIN"


def test_batter_probation_has_separate_aggregate_exposure_cap():
    card = pd.DataFrame({
        "participant_type": ["batter"] * 10 + ["pitcher"],
        "market_type": ["batter_hits_over"] * 10 + ["pitcher_strikeouts_under"],
        "Market_Probation": [True] * 10 + [False],
        "Kelly_Bet_Size": [0.51] * 10 + [5.0],
        "WinProbability": [0.70 - i * 0.01 for i in range(10)] + [0.65],
        "edge": [0.05] * 11,
    })
    out = apply_batter_probation_exposure_cap(card, bankroll=1000.0)
    batter_stakes = out.loc[out["participant_type"].eq("batter"), "Kelly_Bet_Size"]
    batter_total = batter_stakes.sum()
    assert batter_total == 7.0
    assert set(batter_stakes) == {0.0, 1.0}
    assert batter_stakes.iloc[:7].eq(1.0).all()
    assert batter_stakes.iloc[7:].eq(0.0).all()
    assert out.loc[out["participant_type"].eq("pitcher"), "Kelly_Bet_Size"].iloc[0] == 5.0
    assert bool(out["batter_probation_cap_applied"].iloc[0])
    assert out["batter_probation_exposure_before"].iloc[0] == 5.1
    assert out["batter_probation_exposure_cap"].iloc[0] == 7.5
    assert out["batter_probation_exposure_after"].iloc[0] == batter_total
    assert out["batter_probation_minimum_bet"].iloc[0] == 1.0
    assert out["batter_probation_selected_count"].iloc[0] == 7


def test_batter_probation_cap_is_noop_below_limit():
    card = pd.DataFrame({
        "participant_type": ["batter", "pitcher"],
        "market_type": ["batter_hits_over", "pitcher_strikeouts_under"],
        "Market_Probation": [True, False],
        "Kelly_Bet_Size": [1.0, 5.0],
    })
    out = apply_batter_probation_exposure_cap(card, bankroll=1000.0)
    assert out["Kelly_Bet_Size"].tolist() == [1.0, 5.0]
    assert not bool(out["batter_probation_cap_applied"].iloc[0])


def test_novig_minimum_applies_to_every_positive_prop_family():
    card = pd.DataFrame({
        "market_type": [
            "batter_hits_under", "pitcher_strikeouts_under",
            "pitcher_walks_under", "pitcher_walks_over",
        ],
        "Market_Probation": [True, False, True, True],
        "Kelly_Bet_Size": [1.0, 0.59, 0.59, 0.35],
        "WinProbability": [0.76, 0.65, 0.60, 0.58],
        "edge": [0.08, 0.11, 0.07, 0.05],
    })
    out = apply_novig_minimum_selection(
        card, bankroll=1000.0, total_cap_pct=0.03
    )
    positive = out.loc[out["Kelly_Bet_Size"].gt(0), "Kelly_Bet_Size"]
    assert positive.eq(1.0).all()
    assert out["Kelly_Bet_Size"].sum() == 4.0
    assert out["novig_minimum_applied"].iloc[0]
    assert out["novig_selected_count"].iloc[0] == 4
    assert out["novig_exposure_before"].iloc[0] == 2.53
    assert out["novig_exposure_after"].iloc[0] == 4.0


def test_novig_minimum_prioritizes_proven_markets_when_capacity_is_tight():
    card = pd.DataFrame({
        "market_type": [
            "batter_hits_under", "pitcher_strikeouts_under", "pitcher_walks_under",
        ],
        "Market_Probation": [True, False, True],
        "Kelly_Bet_Size": [0.5, 0.5, 0.5],
        "WinProbability": [0.80, 0.60, 0.70],
        "edge": [0.10, 0.05, 0.08],
    })
    out = apply_novig_minimum_selection(
        card, bankroll=100.0, total_cap_pct=0.02
    )
    assert out["Kelly_Bet_Size"].tolist() == [1.0, 1.0, 0.0]
    assert out["novig_exposure_after"].iloc[0] == 2.0


def test_production_gate_requires_proven_three_percent_ev_and_allowed_market():
    card = pd.DataFrame({
        "player": ["Zack Littell", "Jared Jones", "Ty France", "Weak TB"],
        "matchup": ["A @ B", "C @ D", "E @ F", "G @ H"],
        "market_type": [
            "pitcher_strikeouts_over",
            "pitcher_strikeouts_under",
            "batter_hits_under",
            "batter_total_bases_under",
        ],
        "best_pick": ["A", "B", "C", "D"],
        "line": [2.5, 5.5, 1.5, 1.5],
        "odds_american": [-169, -150, -264, -120],
        "WinProbability": [0.70, 0.70, 0.70, 0.70],
        "expected_count": [3.2, 4.8, 0.8, 0.8],
        "expected_value": [0.1238, 0.0038, 0.0624, 0.10],
        "Market_Probation": [False, False, True, False],
        "Pick_Status": ["Actionable"] * 4,
        "Kelly_Bet_Size": [0.55] * 4,
    })
    out = apply_production_prop_gate(card)
    assert out["production_eligible"].tolist() == [True, False, False, False]
    assert out["Kelly_Bet_Size"].tolist() == [0.55, 0.0, 0.0, 0.0]
    assert "below 3.0%" in out.loc[1, "production_gate_reason"]
    assert "probation" in out.loc[2, "production_gate_reason"]
    assert "production-disabled" in out.loc[3, "production_gate_reason"]

def test_production_gate_requires_probability_and_directional_projection_cushion():
    card = pd.DataFrame({
        "player": ["Strong Under", "Strong Over", "Extended", "Low Probability", "Thin Cushion"],
        "matchup": ["A @ B", "C @ D", "E @ F", "G @ H", "I @ J"],
        "market_type": ["pitcher_strikeouts_under", "pitcher_strikeouts_over",
                        "pitcher_strikeouts_under", "pitcher_strikeouts_under",
                        "pitcher_strikeouts_under"],
        "best_pick": ["Under 5.5", "Over 4.5", "Under 5.5", "Under 5.5", "Under 5.5"],
        "line": [5.5, 4.5, 5.5, 5.5, 5.5],
        "expected_count": [4.8, 5.2, 4.7, 4.7, 5.2],
        "odds_american": [-120, -115, -120, -120, -120],
        "WinProbability": [0.66, 0.65, 0.61, 0.59, 0.66],
        "expected_value": [0.08] * 5,
        "Market_Probation": [False] * 5,
        "Pick_Status": ["Actionable"] * 5,
        "Kelly_Bet_Size": [1.0] * 5,
    })
    out = apply_production_prop_gate(card)
    assert out["production_eligible"].tolist() == [True, True, True, False, False]
    assert out["Prop_Tier"].tolist() == ["Core", "Core", "Extended", "Research", "Research"]
    assert out["production_projection_cushion"].round(2).tolist() == [0.7, 0.7, 0.8, 0.8, 0.3]
    assert "below 60%" in out.loc[3, "production_gate_reason"]
    assert "below 0.50" in out.loc[4, "production_gate_reason"]


def test_game_guard_funds_only_highest_ranked_prop_per_matchup():
    card = pd.DataFrame({
        "matchup": ["A @ B", "A @ B", "A @ B", "C @ D"],
        "WinProbability": [0.68, 0.65, 0.63, 0.64],
        "production_projection_cushion": [0.6, 1.0, 1.2, 0.7],
        "expected_value": [0.08, 0.12, 0.13, 0.07],
        "edge": [0.06, 0.10, 0.11, 0.05],
        "Kelly_Bet_Size": [1.0] * 4,
        "production_eligible": [True] * 4,
        "production_gate_reason": ["Production qualified"] * 4,
        "Pick_Status": ["Actionable"] * 4,
    })
    out = apply_prop_game_guard(card)
    assert out["Kelly_Bet_Size"].tolist() == [1.0, 1.0, 0.0, 1.0]
    assert out["production_eligible"].tolist() == [True, True, False, True]
    assert "ranked higher" in out.loc[2, "production_gate_reason"]
    assert out.loc[0, "game_funded_before"] == 4
    assert out.loc[0, "game_funded_after"] == 3


def test_slate_guard_caps_funded_props_at_five_and_prioritizes_core():
    card = pd.DataFrame({
        "Prop_Tier": ["Extended"] * 5 + ["Core"],
        "WinProbability": [0.61] * 5 + [0.70],
        "production_projection_cushion": [0.7] * 6,
        "expected_value": [0.08] * 6,
        "edge": [0.06] * 6,
        "Kelly_Bet_Size": [1.0] * 6,
        "production_eligible": [True] * 6,
        "production_gate_reason": ["Qualified"] * 6,
    })
    out = apply_prop_slate_guard(card)
    assert int(out["production_eligible"].sum()) == 5
    assert bool(out.loc[5, "production_eligible"])
    assert out["slate_funded_before"].iloc[0] == 6
    assert out["slate_funded_after"].iloc[0] == 5


def test_extended_props_are_flat_one_dollar_after_novig_selection():
    card = pd.DataFrame({
        "Prop_Tier": ["Core", "Extended"],
        "Kelly_Bet_Size": [2.5, 3.0],
        "production_eligible": [True, True],
    })
    out = apply_extended_prop_stake_cap(card)
    assert out["Kelly_Bet_Size"].tolist() == [2.5, 1.0]
    assert bool(out["extended_stake_cap_applied"].iloc[0])



def test_production_gate_rejects_malformed_identity():
    card = pd.DataFrame({
        "player": ["Trevor Rogers"],
        "matchup": [None],
        "market_type": [None],
        "best_pick": [None],
        "line": [1.5],
        "odds_american": [-153],
        "expected_value": [0.10],
        "Market_Probation": [False],
        "Pick_Status": ["Actionable"],
        "Kelly_Bet_Size": [1.0],
    })
    out = apply_production_prop_gate(card)
    assert not bool(out.loc[0, "production_identity_valid"])
    assert not bool(out.loc[0, "production_eligible"])
    assert out.loc[0, "Kelly_Bet_Size"] == 0.0
    assert out.loc[0, "Pick_Status"] == "Rejected"


def test_novig_minimum_never_promotes_production_ineligible_row():
    card = pd.DataFrame({
        "market_type": ["pitcher_strikeouts_over", "pitcher_strikeouts_under"],
        "Market_Probation": [False, False],
        "production_eligible": [True, False],
        "Kelly_Bet_Size": [0.55, 0.75],
        "WinProbability": [0.70, 0.75],
        "edge": [0.08, 0.10],
    })
    out = apply_novig_minimum_selection(card, bankroll=1000.0, total_cap_pct=0.03)
    assert out["Kelly_Bet_Size"].tolist() == [1.0, 0.0]
    assert out["novig_selected_count"].iloc[0] == 1




def test_prop_stake_status_distinguishes_funded_from_unstaked_candidates():
    card = pd.DataFrame({
        "best_pick": ["Funded Over 0.5 Hits", "Capped Under 1.5 TB"],
        "Pick_Status": ["Actionable", "Actionable"],
        "Kelly_Bet_Size": [1.0, 0.0],
    })
    out = apply_prop_stake_status(card)
    assert out["Pick_Status"].tolist() == ["Actionable", "Qualified / No Stake"]
    assert out["Stake_Status"].tolist() == ["Funded", "Qualified / No Stake"]
    assert "not selected" in out.loc[1, "Status_Reason"]
