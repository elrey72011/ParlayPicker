import pandas as pd

from app_core.batter_prop_pipeline import score_batter_prop
from app_core.mlb_batter_stats import batter_form_from_gamelog
from app_core.prop_odds_ingest import parse_pitcher_props
from app_core.prop_runner import build_prop_card
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

