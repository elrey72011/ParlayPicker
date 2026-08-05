import pandas as pd

import core.streamlit_pipeline as sp


def _candidate(matchup, home, away, market_type, probability, ev):
    is_total = market_type.startswith("total")
    is_home = market_type == "spread_home"
    return {
        "matchup_id": matchup,
        "market_type": market_type,
        "expected_value": ev,
        "edge": ev,
        "league": "NBA",
        "home_team": home,
        "away_team": away,
        "game_date": "2026-07-30",
        "model_probability": probability,
        "ml_probability": probability,
        "calibrated_probability": probability,
        "market_probability": 0.50,
        "odds_american": -110,
        "line_source": "live",
        "live_total_line": 220.5 if is_total else pd.NA,
        "total_line": 220.5 if is_total else pd.NA,
        "live_spread_line": -3.5 if is_home else pd.NA,
        "spread_line": -3.5 if is_home else pd.NA,
        "is_live_data": True,
        "odds_source": "odds_api",
    }


def test_material_positive_ev_advantage_can_displace_probability_winner(monkeypatch):
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})

    frame = pd.DataFrame([
        _candidate("G1", "A", "B", "spread_home", 0.63, 0.04),
        _candidate("G1", "A", "B", "total_over", 0.57, 0.10),
        _candidate("G2", "C", "D", "spread_home", 0.54, 0.03),
        _candidate("G2", "C", "D", "total_over", 0.61, 0.14),
        # Unrelated games deliberately change each family's slate distribution.
        _candidate("G3", "E", "F", "spread_home", 0.80, 0.01),
        _candidate("G3", "E", "F", "total_over", 0.40, 0.30),
        _candidate("G4", "G", "H", "spread_home", 0.30, 0.25),
        _candidate("G4", "G", "H", "total_over", 0.75, 0.02),
    ])

    best = sp.build_best_picks_df(frame)

    g1 = best[(best["home_team"] == "A") & (best["away_team"] == "B")].iloc[0]
    g2 = best[(best["home_team"] == "C") & (best["away_team"] == "D")].iloc[0]
    assert g1["market_type"] == "total_over"
    assert bool(g1["best_available_value_override_applied"])
    assert g2["market_type"] == "total_over"


def test_unrelated_slate_rows_cannot_flip_a_games_value_dominance_winner(monkeypatch):
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})

    target = pd.DataFrame([
        _candidate("G1", "A", "B", "spread_home", 0.62, 0.03),
        _candidate("G1", "A", "B", "total_over", 0.56, 0.20),
    ])
    expanded = pd.concat(
        [
            target,
            pd.DataFrame([
                _candidate("G2", "C", "D", "spread_home", 0.90, 0.01),
                _candidate("G2", "C", "D", "total_over", 0.20, 0.40),
                _candidate("G3", "E", "F", "spread_home", 0.25, 0.40),
                _candidate("G3", "E", "F", "total_over", 0.85, 0.01),
            ]),
        ],
        ignore_index=True,
    )

    target_pick = sp.build_best_picks_df(target).iloc[0]["market_type"]
    expanded_pick = sp.build_best_picks_df(expanded)
    expanded_pick = expanded_pick[expanded_pick["home_team"] == "A"].iloc[0]["market_type"]

    assert target_pick == "total_over"
    assert expanded_pick == target_pick
