Exit code: 0
Wall time: 1 seconds
Output:
import pandas as pd

from app_core.best_duos import build_best_duos, duos_to_smart_parlays


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
    assert bool(row["production_safety_mode"]) is True
    assert row["model_risk_haircut"] == 0.90
    assert row["combined_probability"] < 0.5087
    assert row["parlay_ev"] > 0
    assert "Bryce Miller" not in row["leg1"] + row["leg2"]


def test_strict_duos_require_sixty_percent_legs():
    props = _props().copy()
    props.loc[0, "WinProbability"] = 0.59
    out = build_best_duos(None, props, strict=True, max_duos=5)
    assert out.empty


def test_strict_mixed_duo_requires_production_game():
    out = build_best_duos(_games(), _props(), strict=True, require_mixed=True, max_duos=5)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["boards"] == "game+prop"
    assert bool(row["production_safety_mode"]) is True


def test_strict_mode_rejects_nonproduction_game():
    games = _games().assign(production_eligible=False)
    out = build_best_duos(games, _props(), strict=True, require_mixed=True)
    assert out.empty


def test_same_bet_direction_does_not_create_false_game_overlap():
    props = pd.DataFrame({
        "league": ["MLB", "MLB"],
        "matchup": [
            "Houston Astros @ Texas Rangers",
            "Colorado Rockies @ San Francisco Giants",
        ],
        "best_pick": [
            "Hunter Brown Under 6.5 Ks",
            "Robbie Ray Under 5.5 Ks",
        ],
        "WinProbability": [0.7139, 0.6500],
        "expected_value": [0.10, 0.10],
        "edge": [0.1024, 0.1063],
        "odds_american": [-170, -127],
        "Pick_Status": ["Actionable", "Actionable"],
        "Market_Probation": [False, False],
    })
    out = build_best_duos(None, props, strict=True)
    assert len(out) == 1
    assert "Hunter Brown" in out.iloc[0]["leg1"] + out.iloc[0]["leg2"]
    assert "Robbie Ray" in out.iloc[0]["leg1"] + out.iloc[0]["leg2"]


def test_strict_duos_map_to_smart_parlay_export_with_capped_stake():
    duos = build_best_duos(None, _props(), strict=True, max_duos=1)
    out = duos_to_smart_parlays(duos, bankroll=1000.0)
    assert len(out) == 1
    assert out.iloc[0]["legs"] == 2
    assert out.iloc[0]["production_safety_mode"]
    assert 0 < out.iloc[0]["recommended_bet"] <= 2.5
    assert " | " in out.iloc[0]["parlay_legs"]

