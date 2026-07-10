"""Pitcher-prop expansion (4 Jul): outs + walks join strikeouts on the SAME engine.

Owner directive: more high-win-probability bets, built where the edge is proven â€”
the prop framework (19-10, 65.5% on strikeouts). Outs and walks are count stats
projectable from the game logs we already fetch; hits/ERs are excluded
(defense/BABIP-dominated). Every guard applies unchanged: starter-sample floor,
projection-consistency gate, implausible-edge cap, min-edge/+EV, 55% win floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from app_core.mlb_pitcher_stats import pitcher_form_from_gamelog
from app_core.prop_odds_ingest import parse_pitcher_props
from app_core.prop_pipeline import PITCHER_PROP_SPECS, score_strikeout_prop
from app_core.prop_runner import build_prop_card
from grade_props import _stat_for_market, grade_card

_FORM = {"k_per_9": 9.5, "bb_per_9": 2.1, "avg_innings": 5.8, "n_games": 5}


def _event(markets):
    return {
        "home_team": "Chicago Cubs",
        "away_team": "St. Louis Cardinals",
        "bookmakers": [{"key": "draftkings", "markets": markets}],
    }


def _market(key, pitcher, line, over, under):
    return {
        "key": key,
        "outcomes": [
            {"description": pitcher, "name": "Over", "point": line, "price": over},
            {"description": pitcher, "name": "Under", "point": line, "price": under},
        ],
    }


def test_parse_pitcher_props_per_market_from_one_payload():
    ev = _event([
        _market("pitcher_strikeouts", "Shota Imanaga", 5.5, -110, -120),
        _market("pitcher_outs", "Shota Imanaga", 16.5, -115, -115),
        _market("pitcher_walks", "Shota Imanaga", 1.5, 105, -135),
    ])
    ks = parse_pitcher_props(ev, "pitcher_strikeouts")
    outs = parse_pitcher_props(ev, "pitcher_outs")
    walks = parse_pitcher_props(ev, "pitcher_walks")
    assert [r["market_key"] for r in ks + outs + walks] == [
        "pitcher_strikeouts", "pitcher_outs", "pitcher_walks",
    ]
    assert outs[0]["line"] == 16.5 and walks[0]["line"] == 1.5


def test_form_now_carries_bb_per_9():
    splits = [
        {"stat": {"inningsPitched": "6.0", "strikeOuts": 7, "baseOnBalls": 2}},
        {"stat": {"inningsPitched": "5.1", "strikeOuts": 5, "baseOnBalls": 1}},
    ]
    form = pitcher_form_from_gamelog(splits)
    assert form["bb_per_9"] > 0
    assert abs(form["bb_per_9"] - 9.0 * 3 / (6.0 + 5 + 1 / 3)) < 1e-9


def test_outs_market_scores_on_workload():
    row = {"pitcher": "P", "line": 16.5, "over_odds": -110, "under_odds": -110,
           "market_key": "pitcher_outs"}
    out = score_strikeout_prop(row, _FORM)
    assert out["expected_ks"] == 17.4  # 5.8 innings x 3 outs
    assert out["recommendation"] in ("over", "no_play")  # never crashes, projects over


def test_walks_market_requires_bb_rate():
    row = {"pitcher": "P", "line": 1.5, "over_odds": -110, "under_odds": -110,
           "market_key": "pitcher_walks"}
    no_bb = {k: v for k, v in _FORM.items() if k != "bb_per_9"}
    assert score_strikeout_prop(row, no_bb)["recommendation"] == "no_data"
    scored = score_strikeout_prop(row, _FORM)
    assert scored["expected_ks"] == 1.35  # 2.1/9 * 5.8
    assert scored["recommendation"] in ("over", "under", "no_play")


def test_strikeout_rows_without_market_key_unchanged():
    row = {"pitcher": "P", "line": 5.5, "over_odds": -110, "under_odds": -110}
    out = score_strikeout_prop(row, _FORM)
    assert out["expected_ks"] > 5.5  # 9.5/9 * 5.8 â‰ˆ 6.1, opponent-neutral
    assert out["recommendation"] in ("over", "under", "no_play")


def test_build_prop_card_labels_each_market():
    props = [
        {"pitcher": "Ace Guy", "line": 5.5, "over_odds": -115, "under_odds": -105,
         "book": "draftkings", "home_team": "H", "away_team": "A",
         "market_key": "pitcher_strikeouts"},
        {"pitcher": "Ace Guy", "line": 15.5, "over_odds": -120, "under_odds": -104,
         "book": "draftkings", "home_team": "H", "away_team": "A",
         "market_key": "pitcher_outs"},
    ]
    card = build_prop_card(
        object(), "2026-07-04", 2026, 1000.0,
        kelly_per_pick_pct=0.01, kelly_total_pct=0.03,
        min_edge=0.0, min_win_probability=0.50,
        enabled_markets=("pitcher_strikeouts", "pitcher_outs", "pitcher_walks"),
        max_plausible_edge=1.0,
        list_events=lambda c, s, d: [{"id": "e1"}],
        props_fetch=lambda c, s, e: props,
        schedule_fetch=lambda d: [{"home_pitcher": "Ace Guy", "home_pitcher_id": 1,
                                   "away_pitcher": None, "away_pitcher_id": None,
                                   "home_team_id": 10, "away_team_id": 11}],
        form_fetch=lambda pid, season: _FORM,
        team_k_fetch=lambda tid, season: 0.22,
    )
    assert not card.empty
    picks = " | ".join(card["best_pick"])
    types = set(card["market_type"])
    assert any(t.startswith("pitcher_outs_") for t in types)
    assert "Outs" in picks and ("Ks" in picks)


def test_outs_are_not_production_staked_by_default():
    props = [{
        "pitcher": "Ace Guy", "line": 15.5, "over_odds": -120, "under_odds": -104,
        "book": "draftkings", "home_team": "H", "away_team": "A",
        "market_key": "pitcher_outs",
    }]
    card = build_prop_card(
        object(), "2026-07-04", 2026, 1000.0,
        kelly_per_pick_pct=0.01, kelly_total_pct=0.03,
        min_edge=0.0, min_win_probability=0.50,
        max_plausible_edge=1.0,
        list_events=lambda c, s, d: [{"id": "e1"}],
        props_fetch=lambda c, s, e: props,
        schedule_fetch=lambda d: [{"home_pitcher": "Ace Guy", "home_pitcher_id": 1,
                                   "away_pitcher": None, "away_pitcher_id": None,
                                   "home_team_id": 10, "away_team_id": 11}],
        form_fetch=lambda pid, season: _FORM,
        team_k_fetch=lambda tid, season: 0.22,
    )
    assert card.empty


def test_grading_selects_stat_by_market():
    card = pd.DataFrame([
        {"pitcher": "P", "market_type": "pitcher_strikeouts_over", "best_pick": "P Over 4.5 Ks",
         "line": 4.5, "odds_american": -110, "Kelly_Bet_Size": 5.0, "Pick_Status": "Actionable"},
        {"pitcher": "P", "market_type": "pitcher_outs_under", "best_pick": "P Under 16.5 Outs",
         "line": 16.5, "odds_american": -110, "Kelly_Bet_Size": 5.0, "Pick_Status": "Actionable"},
        {"pitcher": "P", "market_type": "pitcher_walks_over", "best_pick": "P Over 1.5 BBs",
         "line": 1.5, "odds_american": -110, "Kelly_Bet_Size": 5.0, "Pick_Status": "Actionable"},
    ])
    actual = {"ks": 6, "outs": 15, "walks": 1}
    rows = grade_card(card, "2026-07-04", {"p": 1}, lambda pid: actual)
    results = {(_stat_for_market(r["pick"], "")): r["result"] for r in rows}
    by_pick = {r["pick"]: r["result"] for r in rows}
    assert by_pick["P Over 4.5 Ks"] == "WIN"       # 6 > 4.5
    assert by_pick["P Under 16.5 Outs"] == "WIN"   # 15 < 16.5
    assert by_pick["P Over 1.5 BBs"] == "LOSS"     # 1 < 1.5


def test_grading_int_actual_backwards_compatible():
    card = pd.DataFrame([
        {"pitcher": "P", "market_type": "pitcher_strikeouts_over", "best_pick": "P Over 4.5 Ks",
         "line": 4.5, "odds_american": -110, "Kelly_Bet_Size": 5.0, "Pick_Status": "Actionable"},
    ])
    rows = grade_card(card, "2026-07-03", {"p": 1}, lambda pid: 7)
    assert rows[0]["result"] == "WIN"


# â”€â”€ Market probation (6 Jul): stakes follow each market's graded record â”€â”€
from app_core.prop_runner import (  # noqa: E402
    PROBATION_STAKE,
    apply_market_probation,
    market_records_from_log,
)


def test_market_records_classify_picks_from_log():
    log = pd.DataFrame({
        "pick": ["A Over 4.5 Ks", "B Under 16.5 Outs", "C Over 1.5 BBs", "D Under 15.5 Outs"],
        "result": ["WIN", "LOSS", "WIN", "LOSS"],
    })
    rec = market_records_from_log(log)
    assert rec["pitcher_strikeouts"] == (1, 0)
    assert rec["pitcher_outs"] == (0, 2)
    assert rec["pitcher_walks"] == (1, 0)


def test_underwater_market_gets_probation_stake():
    card = pd.DataFrame({
        "market_type": ["pitcher_outs_under", "pitcher_strikeouts_over"],
        "best_pick": ["May Under 15.5 Outs", "Buehler Over 3.5 Ks"],
        "Kelly_Bet_Size": [7.5, 7.5],
    })
    records = {"pitcher_outs": (2, 5), "pitcher_strikeouts": (24, 12)}
    out = apply_market_probation(card, records)
    assert out.iloc[0]["Kelly_Bet_Size"] == PROBATION_STAKE   # outs capped
    assert bool(out.iloc[0]["Market_Probation"])
    assert out.iloc[1]["Kelly_Bet_Size"] == 7.5               # Ks untouched
    assert not bool(out.iloc[1]["Market_Probation"])


def test_small_samples_and_winning_markets_not_touched():
    card = pd.DataFrame({
        "market_type": ["pitcher_walks_over"],
        "best_pick": ["X Over 1.5 BBs"],
        "Kelly_Bet_Size": [5.0],
    })
    # walks 5-4 (55.6%) â€” above water, no probation
    out = apply_market_probation(card, {"pitcher_walks": (5, 4)})
    assert out.iloc[0]["Kelly_Bet_Size"] == 5.0
    # tiny sample (n<6) â€” never judged
    out2 = apply_market_probation(card, {"pitcher_walks": (1, 4)})
    assert out2.iloc[0]["Kelly_Bet_Size"] == 5.0


def test_stat_market_ignores_stat_words_in_pitcher_names():
    # "Walker" contains "walk"; market_type must decide (6 Jul misgrade).
    assert _stat_for_market("pitcher_outs_under", "Walker Buehler Under 15.5 Outs") == "outs"
    assert _stat_for_market("pitcher_strikeouts_over", "Walker Buehler Over 3.5 Ks") == "ks"
    # fallback path (no market_type) uses padded tokens, so names still can't collide
    assert _stat_for_market("", "Walker Buehler Under 15.5 Outs") == "outs"


def test_market_records_ignore_stat_words_in_names():
    log = pd.DataFrame({
        "pick": ["Walker Buehler Over 3.5 Ks", "Walker Buehler Under 15.5 Outs"],
        "result": ["WIN", "WIN"],
    })
    rec = market_records_from_log(log)
    assert rec.get("pitcher_strikeouts") == (1, 0)
    assert rec.get("pitcher_outs") == (1, 0)
    assert "pitcher_walks" not in rec

