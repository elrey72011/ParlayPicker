import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

def test_side_floor_and_consensus_overlays(monkeypatch):
    monkeypatch.setattr("app_core.weights_config.EMPIRICAL_TIER_OVERLAY_ENABLED", False)
    # Helper to generate rows for pipeline processing
    def create_row(home_team, away_team, market_type, win_prob, ev, edge, consensus_agreement="Agrees"):
        is_total = "total" in market_type.lower()

        # In the pipeline, consensus_agreement is dynamically computed from the
        # DIRECTIONAL Kalshi read (16 Jun): does Kalshi back the SAME side as the
        # pick (kalshi_probability for our side > 0.50), regardless of which model
        # is more confident?
        #   Agrees:    kalshi_probability >= 0.52  (Kalshi backs our side)
        #   Disagrees: kalshi_probability <= 0.48  (Kalshi backs the other side)
        #   Neutral:   ~0.50

        if consensus_agreement == "Agrees":
            k_prob = 0.55
        elif consensus_agreement == "Disagrees":
            k_prob = 0.45
        else:
            k_prob = 0.50

        return {
            "league": "NBA",
            "home_team": home_team,
            "away_team": away_team,
            "game_date": "2026-08-01",
            "matchup_id": f"2026-08-01|{home_team}|{away_team}",
            "market_type": market_type,
            "expected_value": ev,
            "edge": edge,
            "calibrated_probability": win_prob,
            "ml_probability": win_prob,
            "odds_american": -110,
            "decimal_odds": 1.0 + (100.0 / 110.0),
            "market_probability": 110.0 / 210.0,
            "spread_line": -5.0 if not is_total else pd.NA,
            "total_line": 210.0 if is_total else pd.NA,
            "consensus_agreement": consensus_agreement,
            "is_live_data": True,
            "used_stale_features": False,
            "odds_source": "odds_api",
            "line_source": "live",
            "live_spread_line": -5.0 if not is_total else pd.NA,
            "live_total_line": 210.0 if is_total else pd.NA,
            "kalshi_probability": k_prob
        }

    df = pd.DataFrame([
        # 1. Side rows failing SIDE_MIN_WIN_PROB (0.52)
        create_row("A", "B", "spread_home", 0.51, 0.03, 0.03, "Agrees"),

        # 2. Side rows above SIDE_MIN_WIN_PROB (0.52)
        create_row("C", "D", "spread_home", 0.53, 0.03, 0.03, "Agrees"),

        # 3. Neutral row failing the overlay (prob 0.57 < 0.58, EV/Edge passing baseline)
        create_row("E", "F", "total_over", 0.57, 0.035, 0.045, "Neutral"),

        # 4. Neutral row now clears with the explicit NBA total-over bonus calibration
        create_row("G", "H", "total_over", 0.58, 0.035, 0.045, "Neutral"),

        # 5. Disagrees row now clears softened NBA total penalties in STANDARD profile
        create_row("I", "J", "total_over", 0.59, 0.045, 0.055, "Disagrees"),

        # 6. Disagrees row with decent stats should stay Actionable under the softer NBA pass
        create_row("K", "L", "total_over", 0.61, 0.045, 0.055, "Disagrees"),

        # 7. Agrees row remaining Actionable (passes baseline totals)
        create_row("M", "N", "total_under", 0.57, 0.015, 0.025, "Agrees"),
    ])

    best = build_best_picks_df(df)

    # 1. Side failing floor -> Below Threshold
    assert best.loc[best["home_team"] == "A", "Pick_Status"].iloc[0] == "Below Threshold"

    # 2. A side above the generic floor still cannot bypass the owner's stake floor.
    side = best.loc[best["home_team"] == "C"].iloc[0]
    assert side["Pick_Status"] == "High Variance/Speculative"
    assert "stake floor" in side["Status_Reason"]

    # 3. Neutral failing overlay -> Below Threshold
    assert best.loc[best["home_team"] == "E", "Pick_Status"].iloc[0] == "Below Threshold"

    # 4. The retired NBA Over bonus cannot bypass the current NBA probability floor.
    assert best.loc[best["home_team"] == "G", "Pick_Status"].iloc[0] == "Below Threshold"

    # 5. A disagreeing market does not relax the NBA total floor.
    assert best.loc[best["home_team"] == "I", "Pick_Status"].iloc[0] == "Below Threshold"

    # 6. Even the higher raw probability remains below the post-shrink NBA floor.
    assert best.loc[best["home_team"] == "K", "Pick_Status"].iloc[0] == "Below Threshold"

    # 7. Agrees does not bypass under-specific thresholds -> Below Threshold
    assert best.loc[best["home_team"] == "M", "Pick_Status"].iloc[0] == "Below Threshold"
