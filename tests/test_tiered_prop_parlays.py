import pandas as pd

from app_core.best_duos import build_tiered_prop_parlays


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
        "Pick_Status": ["Actionable"] * 5,
        "Market_Probation": [False, False, True, True, True],
        "Kelly_Bet_Size": [1.0] * 5,
    })
    out = build_tiered_prop_parlays(None, props, bankroll=1000.0)
    assert len(out) == 2
    assert set(out["risk_tier"]) == {"Controlled", "Probation / Research"}
    assert out["group_id"].is_unique
    assert out["group_id"].tolist() == ["strict_duo_1", "strict_duo_2"]
    assert out.loc[out["risk_tier"].eq("Probation / Research"), "recommended_bet"].iloc[0] == 1.0
