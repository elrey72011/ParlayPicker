"""Best Duos (owner, 8 Jul): likeliest 2-leg parlays, cross-board, no shared
games between legs, no leg reused across duos, ranked by joint probability."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core.best_duos import build_best_duos


def _games():
    return pd.DataFrame({
        "league": ["MLB", "MLB"],
        "home_team": ["San Francisco", "Cincinnati"],
        "away_team": ["Toronto", "Philadelphia"],
        "best_pick": ["San Francisco +1.5", "Under 8.5"],
        "Pick_Status": ["Actionable", "Near Miss"],
        "Kelly_Bet_Size": [15.0, 0.0],
        "odds_american": [-163, 120],
        "empirical_win_probability": [0.62, 0.58],
        "status_blocker_stage": ["", ""],
        "game_already_started_flag": [False, False],
    })


def _props():
    return pd.DataFrame({
        "league": ["MLB", "MLB", "MLB"],
        "pitcher": ["Zack Wheeler", "Shane Baz", "Kevin Gausman"],
        "matchup": [
            "Philadelphia Phillies @ Cincinnati Reds",   # overlaps game 2 AND Wheeler
            "Chicago Cubs @ Baltimore Orioles",
            "Toronto Blue Jays @ San Francisco Giants",  # overlaps game 1
        ],
        "best_pick": ["Zack Wheeler Over 6.5 Ks", "Shane Baz Under 2.5 BBs", "Kevin Gausman Under 5.5 Ks"],
        "WinProbability": [0.68, 0.677, 0.63],
        "odds_american": [-208, -182, -135],
        "Pick_Status": ["Actionable"] * 3,
        "Kelly_Bet_Size": [1.28, 3.1, 3.1],
    })


def test_top_duo_is_highest_joint_probability_without_overlap():
    duos = build_best_duos(_games(), _props(), max_duos=3)
    assert not duos.empty
    top = duos.iloc[0]
    # Wheeler (68%) + Baz (67.7%) are the two best legs and share no game.
    assert "Wheeler" in top["leg1"] and "Baz" in top["leg2"]
    assert abs(top["combined_probability"] - 0.68 * 0.677) < 1e-3


def test_same_game_legs_never_pair():
    duos = build_best_duos(_games(), _props(), max_duos=10)
    for _, d in duos.iterrows():
        joined = f"{d['leg1']} | {d['leg2']}".lower()
        # Wheeler prop and the Cincinnati/Philadelphia total must never co-occur
        assert not ("wheeler" in joined and "under 8.5 (philadelphia @ cincinnati)" in joined)
        # Gausman prop and the SF +1.5 run-line share the Toronto/SF game
        assert not ("gausman" in joined and "san francisco +1.5" in joined)


def test_legs_not_reused_across_duos():
    duos = build_best_duos(_games(), _props(), max_duos=5)
    legs = list(duos["leg1"]) + list(duos["leg2"])
    assert len(legs) == len(set(legs))


def test_floor_and_empty_safety():
    lowprob = _props().assign(WinProbability=[0.50, 0.51, 0.52])
    assert build_best_duos(None, lowprob).empty
    assert build_best_duos(None, None).empty
    # one qualifying leg is not a parlay
    one = _props().head(1)
    assert build_best_duos(None, one).empty


def test_require_mixed_pairs_one_game_with_one_prop():
    duos = build_best_duos(_games(), _props(), max_duos=3, require_mixed=True)
    assert not duos.empty
    for _, d in duos.iterrows():
        assert sorted(d["boards"].split("+")) == ["game", "prop"]
    # The best legal mixed pair: SF +1.5 (62%) can't pair with Gausman (same
    # game), so it takes the best non-overlapping prop.
    top = duos.iloc[0]
    joined = f"{top['leg1']} {top['leg2']}"
    assert "+1.5" in joined or "Under 8.5" in joined  # contains a game leg
