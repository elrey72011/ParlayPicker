import pandas as pd

from app_core.best_duos import build_best_duos


def _games():
    return pd.DataFrame({
        "league": ["MLB"],
        "home_team": ["Cincinnati"],
        "away_team": ["Philadelphia"],
        "best_pick": ["Cincinnati +1.5"],
        "Pick_Status": ["Actionable"],
        "production_eligible": [True],
        "odds_american": [-102],
        "empirical_win_probability": [0.60],
        "empirical_edge": [0.04],
        "effective_expected_value": [0.08],
        "game_already_started_flag": [False],
    })


def _props():
    return pd.DataFrame({
        "league": ["MLB", "MLB", "MLB"],
        "pitcher": ["Janson Junk", "Nathan Eovaldi", "Bryce Miller"],
        "matchup": [
            "Seattle Mariners @ Miami Marlins",
            "Los Angeles Angels @ Texas Rangers",
            "Seattle Mariners @ Miami Marlins",
        ],
        "best_pick": [
            "Janson Junk Under 1.5 BBs",
            "Nathan Eovaldi Over 6.5 Ks",
            "Bryce Miller Over 17.5 Outs",
        ],
        "WinProbability": [0.736, 0.6912, 0.5996],
        "expected_value": [0.0752, 0.1152, 0.1211],
        "edge": [0.0805, 0.1003, 0.0828],
        "odds_american": [-217, -163, -115],
        "Pick_Status": ["Actionable"] * 3,
        "Market_Probation": [False, False, True],
    })


def test_strict_duos_apply_haircut_and_exclude_probation():
    out = build_best_duos(None, _props(), strict=True, max_duos=5)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["boards"] == "prop+prop"
    assert row["production_safety_mode"] is True
    assert row["model_risk_haircut"] == 0.97
    assert row["combined_probability"] < 0.5087
    assert row["parlay_ev"] > 0
    assert "Bryce Miller" not in row["leg1"] + row["leg2"]


def test_strict_mixed_duo_requires_production_game():
    out = build_best_duos(_games(), _props(), strict=True, require_mixed=True, max_duos=5)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["boards"] == "game+prop"
    assert row["production_safety_mode"] is True


def test_strict_mode_rejects_nonproduction_game():
    games = _games().assign(production_eligible=False)
    out = build_best_duos(games, _props(), strict=True, require_mixed=True)
    assert out.empty
