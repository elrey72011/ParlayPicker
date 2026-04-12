import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

def test_side_floor_and_consensus_overlays():
    # Helper to generate rows for pipeline processing
    def create_row(home_team, away_team, market_type, win_prob, ev, edge, consensus_agreement="Agrees"):
        is_total = "total" in market_type.lower()

        # In the pipeline, consensus_agreement is dynamically computed:
        # gap = calibrated_probability - kalshi_probability
        # Agrees: gap >= 0.03
        # Disagrees: gap <= -0.03
        # Neutral: otherwise

        if consensus_agreement == "Agrees":
            k_prob = win_prob - 0.04
        elif consensus_agreement == "Disagrees":
            k_prob = win_prob + 0.04
        else:
            k_prob = win_prob

        return {
            "league": "NBA",
            "home_team": home_team,
            "away_team": away_team,
            "game_date": "2024-01-01",
            "matchup_id": f"2024-01-01|{home_team}|{away_team}",
            "market_type": market_type,
            "expected_value": ev,
            "edge": edge,
            "calibrated_probability": win_prob,
            "ml_probability": win_prob,
            "odds_american": -110,
            "spread_line": -5.0 if not is_total else pd.NA,
            "total_line": 210.0 if is_total else pd.NA,
            "consensus_agreement": consensus_agreement,
            "is_live_data": True,
            "used_stale_features": False,
            "odds_source": "odds_api",
            "kalshi_probability": k_prob
        }

    df = pd.DataFrame([
        # 1. Side rows failing SIDE_MIN_WIN_PROB (0.52)
        create_row("A", "B", "spread_home", 0.51, 0.03, 0.03, "Agrees"),

        # 2. Side rows above SIDE_MIN_WIN_PROB (0.52)
        create_row("C", "D", "spread_home", 0.53, 0.03, 0.03, "Agrees"),

        # 3. Neutral row failing the overlay (prob 0.57 < 0.58, EV/Edge passing baseline)
        create_row("E", "F", "total_over", 0.57, 0.035, 0.045, "Neutral"),

        # 4. Neutral row passing the overlay (prob 0.58, EV 0.03, Edge 0.04)
        create_row("G", "H", "total_over", 0.58, 0.035, 0.045, "Neutral"),

        # 5. Disagrees row failing the overlay (prob 0.59 < 0.60)
        create_row("I", "J", "total_over", 0.59, 0.045, 0.055, "Disagrees"),

        # 6. Disagrees row passing the overlay (prob 0.61, EV 0.045, Edge 0.055)
        create_row("K", "L", "total_over", 0.61, 0.045, 0.055, "Disagrees"),

        # 7. Agrees row remaining Actionable (passes baseline totals)
        create_row("M", "N", "total_under", 0.57, 0.015, 0.025, "Agrees"),
    ])

    best = build_best_picks_df(df)

    # 1. Side failing floor -> Below Threshold
    assert best.loc[best["home_team"] == "A", "Pick_Status"].iloc[0] == "Below Threshold"

    # 2. Side passing floor -> Actionable
    assert best.loc[best["home_team"] == "C", "Pick_Status"].iloc[0] == "Actionable"

    # 3. Neutral failing overlay -> Below Threshold
    assert best.loc[best["home_team"] == "E", "Pick_Status"].iloc[0] == "Below Threshold"

    # 4. Neutral passing overlay -> Actionable
    assert best.loc[best["home_team"] == "G", "Pick_Status"].iloc[0] == "Actionable"

    # 5. Disagrees failing overlay -> High Variance/Speculative
    assert best.loc[best["home_team"] == "I", "Pick_Status"].iloc[0] == "High Variance/Speculative"

    # 6. Disagrees passing overlay -> Actionable
    assert best.loc[best["home_team"] == "K", "Pick_Status"].iloc[0] == "Actionable"

    # 7. Agrees passing baseline -> Actionable
    assert best.loc[best["home_team"] == "M", "Pick_Status"].iloc[0] == "Actionable"
