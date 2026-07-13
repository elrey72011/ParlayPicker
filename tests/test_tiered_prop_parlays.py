import pandas as pd

from app_core.best_duos import build_tiered_prop_parlays, duos_to_smart_parlays


def test_tiered_parlay_menu_includes_production_and_research_duos():
    props = pd.DataFrame({
        "league": ["MLB"] * 5,
        "matchup": ["A @ B", "C @ D", "E @ F", "G @ H", "I @ J"],
        "best_pick": [
            "P1 Over 5.5 Ks", "P2 Over 4.5 Ks",
            "B1 Under 1.5 Hits", "P3 Over 1.5 BBs", "B2 Under 1.5 Hits",
        ],
        "WinProbability": [0.67, 0.63, 0.78, 0.72, 0.71],
        "expected_value": [0.08, 0.08, 0.08, 0.08, 0.08],
        "edge": [0.07, 0.07, 0.07, 0.07, 0.07],
        "odds_american": [-110, -110, -110, -110, -110],
        "book": ["novig"] * 5,
        "Pick_Status": ["Actionable"] * 5,
        "Market_Probation": [False, False, True, True, True],
        "Kelly_Bet_Size": [1.0] * 5,
    })
    out = build_tiered_prop_parlays(None, props, bankroll=1000.0)
    assert len(out) == 2
    assert set(out["risk_tier"]) == {"Controlled", "Probation / Research"}
    assert out["group_id"].is_unique
    assert out["group_id"].tolist() == ["strict_duo_1", "strict_duo_2"]
    assert out["best_payout_book"].eq("Novig").all()
    assert out["recommended_bet"].ge(1.0).all()
    assert out.loc[out["risk_tier"].eq("Probation / Research"), "recommended_bet"].iloc[0] == 1.0



def _two_prop_card(*, books=("novig", "novig"), probabilities=(0.67, 0.66), odds=(-110, -110)):
    return pd.DataFrame({
        "league": ["MLB", "MLB"],
        "matchup": ["A @ B", "C @ D"],
        "best_pick": ["P1 Under 5.5 Ks", "P2 Over 1.5 BBs"],
        "WinProbability": list(probabilities),
        "expected_value": [0.08, 0.08],
        "edge": [0.07, 0.07],
        "odds_american": list(odds),
        "book": list(books),
        "Pick_Status": ["Actionable", "Actionable"],
        "Market_Probation": [False, False],
        "Kelly_Bet_Size": [1.0, 1.0],
    })


def test_tiered_parlays_reject_cross_book_prices():
    out = build_tiered_prop_parlays(
        None,
        _two_prop_card(books=("novig", "draftkings")),
        bankroll=1000.0,
    )
    assert out.empty


def test_strict_parlays_require_five_percent_post_haircut_ev():
    out = build_tiered_prop_parlays(
        None,
        _two_prop_card(probabilities=(0.60, 0.60), odds=(-125, -125)),
        bankroll=1000.0,
    )
    assert out.empty


def test_parlay_below_sportsbook_minimum_is_suppressed():
    duos = pd.DataFrame([{
        "leg1": "A",
        "leg2": "B",
        "leg1_prob": 0.71,
        "leg2_prob": 0.71,
        "combined_probability": 0.5006,
        "combined_decimal": 2.0,
        "parlay_ev": 0.0012,
        "common_book": "novig",
        "production_safety_mode": True,
        "probation_parlay_mode": False,
        "model_risk_haircut": 0.90,
    }])
    assert duos_to_smart_parlays(duos, bankroll=1000.0).empty


def test_strict_parlay_rejects_two_markets_for_same_player():
    props = pd.DataFrame({
        "league": ["MLB", "MLB"],
        "player": ["Shane Baz", "Shane Baz"],
        "matchup": ["A @ B", "C @ D"],
        "best_pick": ["Shane Baz Over 1.5 BBs", "Shane Baz Under 5.5 Ks"],
        "WinProbability": [0.70, 0.69],
        "expected_value": [0.10, 0.10],
        "edge": [0.08, 0.08],
        "odds_american": [-110, -110],
        "book": ["novig", "novig"],
        "Pick_Status": ["Actionable", "Actionable"],
        "Market_Probation": [False, False],
        "Kelly_Bet_Size": [1.0, 1.0],
    })
    assert build_tiered_prop_parlays(None, props, bankroll=1000.0).empty
