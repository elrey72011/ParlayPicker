import pandas as pd

from app_core.batter_prop_pipeline import score_batter_prop
from app_core.mlb_batter_stats import batter_form_from_gamelog
from app_core.prop_odds_ingest import parse_pitcher_props
from app_core.prop_runner import apply_batter_probation_exposure_cap, build_prop_card
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


def test_batter_card_is_actionable_but_starts_on_probation():
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
    assert row["Kelly_Bet_Size"] <= 1.0


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

