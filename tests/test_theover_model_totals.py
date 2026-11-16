"""TheOver's model-backed totals pick moves the app's win estimate (owner 2026-11-16).

The owner approved this on 16 November 2026: it reverses, for this one signal, the 7 November
rule that TheOver's signals are kept per row and never used. It moves the selection probability
only, so these tests also pin that no stake or wager approval can come from it.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from app_core import theover_model_totals as tmt


def _rows(*rows):
    return pd.DataFrame([{"matchup_id": game, "market_type": kind, "best_available_probability": p,
                          "theover_pick_agrees": picked, "theover_model_hit_rate": hit}
                         for game, kind, p, picked, hit in rows])


def test_backed_is_a_total_theover_picked_with_a_hit_rate_listed():
    frame = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73),
                  ("g", "spread_home", 0.52, 1.0, 0.73),            # a side, not a total
                  ("h", "total_over", 0.50, 1.0, None),             # no hit rate listed
                  ("i", "total_over", 0.50, None, 0.80),            # no pick
                  ("j", "total_over", 0.50, 1.0, 1.0), ("k", "total_over", 0.50, 1.0, 0.0))   # not a rate
    assert list(tmt.backed(frame)) == [True, False, False, False, False, False, False]
    assert not tmt.backed(frame.drop(columns=["theover_model_hit_rate"])).any()


def test_the_picked_side_gains_what_the_other_side_loses():
    frame = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73),
                  ("g", "spread_home", 0.52, None, None), ("g", "spread_away", 0.48, None, None),
                  ("h", "total_over", 0.51, 0.0, None), ("h", "total_under", 0.49, 1.0, None))
    frame.index = [9, 4, 7, 1, 3, 8]
    values, moved = tmt.apply(frame)
    assert list(values.index) == [9, 4, 7, 1, 3, 8]
    assert values.loc[9] == pytest.approx(0.545) and values.loc[4] == pytest.approx(0.455)
    assert values.loc[9] + values.loc[4] == pytest.approx(1.0)              # the pair sums to what it did
    assert list(values.loc[[7, 1, 3, 8]]) == [0.52, 0.48, 0.51, 0.49]       # sides and unbacked totals never move
    assert list(moved) == pytest.approx([0.05, -0.05, 0.0, 0.0, 0.0, 0.0])
    assert tmt.SHIFT == 0.05


def test_the_shift_is_cut_back_at_the_limits_and_some_games_are_left_alone():
    near = _rows(("g", "total_over", 0.97, 1.0, 0.8), ("g", "total_under", 0.03, 0.0, 0.8))
    values, moved = tmt.apply(near)
    assert list(values) == pytest.approx([0.99, 0.01]) and list(moved) == pytest.approx([0.02, -0.02])
    full = _rows(("g", "total_over", 0.99, 1.0, 0.8), ("g", "total_under", 0.01, 0.0, 0.8))
    assert list(tmt.apply(full)[1]) == [0.0, 0.0]
    both = _rows(("g", "total_over", 0.5, 1.0, 0.8), ("g", "total_under", 0.5, 1.0, 0.8))      # the upload names both sides
    assert list(tmt.apply(both)[0]) == [0.5, 0.5]
    blank = _rows(("g", "total_over", None, 1.0, 0.8), ("g", "total_under", 0.5, 0.0, 0.8))    # nothing to move
    assert pd.isna(tmt.apply(blank)[0].iloc[0]) and tmt.apply(blank)[0].iloc[1] == 0.5
    alone = _rows(("g", "total_over", 0.5, 1.0, 0.8))                                          # the other side is not a candidate
    assert list(tmt.apply(alone)[0]) == pytest.approx([0.55])
    two = _rows(("g", "total_under", 0.495, 1.0, 0.73), ("g", "total_over", 0.505, 0.0, 0.73))
    assert list(tmt.apply(two, shift=0)[0]) == [0.495, 0.505]
    assert list(tmt.apply(two.drop(columns=["matchup_id"]))[0]) == [0.495, 0.505]
    assert tmt.apply(two.iloc[0:0])[0].empty


# ---- through the real pick pipeline ----

@pytest.fixture
def _pregame(monkeypatch):
    monkeypatch.setattr("app_core.candidate_chronology.now_utc", lambda: pd.Timestamp("2026-07-27T21:00:00Z"))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)


def _candidate(kind, probability, **extra):
    total = kind.startswith("total")
    line = 8.5 if total else (-1.5 if kind == "spread_home" else 1.5)
    quotes = [{"book": "novig", "market_type": k, "point": point, "price": -110, "recorded_at": "2026-07-27T20:55:00Z"}
              for k, point in (("total_over", 8.5), ("total_under", 8.5), ("spread_home", -1.5), ("spread_away", 1.5))]
    row = {"game_id": "theover-model", "league": "MLB", "home_team": "Chicago Cubs", "away_team": "Pittsburgh Pirates",
           "game_date": pd.Timestamp("2026-07-27", tz="UTC"), "game_start_utc": "2026-07-27T23:00:00Z",
           "odds_recorded_at": "2026-07-27T20:55:00Z", "market_type": kind, "total_line": line if total else pd.NA,
           "spread_line": pd.NA if total else line, "calibrated_probability": probability, "model_probability": probability,
           "ml_probability": probability, "expected_value": round(probability * (1 + 100 / 110) - 1, 4), "edge": 0.0,
           "market_probability": 0.50, "kalshi_probability": 0.50, "consensus_agreement": "Neutral", "odds_american": -110,
           "odds_source": "test", "line_source": "live", "market_line_source": "live", "live_total_line": 8.5,
           "is_live_data": True, "used_stale_features": False, "provider_quotes": json.dumps(quotes)}
    row.update(extra)
    return row


def _game(hit, side=0.52):
    return pd.DataFrame([
        _candidate("total_over", 0.505, theover_pick_agrees=0.0, theover_model_hit_rate=hit),
        _candidate("total_under", 0.495, theover_pick_agrees=1.0, theover_model_hit_rate=hit),
        _candidate("spread_home", side), _candidate("spread_away", 1 - side)])


def _run(frame):
    from core.streamlit_pipeline import build_best_picks_df
    diagnostics: dict = {}
    best = build_best_picks_df(frame, diagnostics_out=diagnostics)
    return best, diagnostics["candidate_audit_df"].set_index("market_type"), diagnostics


def test_the_model_backed_total_becomes_the_games_pick(_pregame):
    before, plain, _ = _run(_game(float("nan")))
    after, audit, diagnostics = _run(_game(0.73))
    # Without a listed hit rate the pick is what it always was; with one it is TheOver's side of the total.
    assert list(before["market_type"]) == ["spread_home"] and list(after["market_type"]) == ["total_under"]
    assert audit.loc["total_under", "best_available_probability"] == pytest.approx(0.545)
    assert audit.loc["total_over", "best_available_probability"] == pytest.approx(0.455)
    assert list(audit.loc[["total_under", "total_over", "spread_home", "spread_away"], "best_available_theover_model_shift"]) == \
        pytest.approx([0.05, -0.05, 0.0, 0.0])
    assert audit.loc["total_under", "best_available_probability_source"] == "calibrated_probability_pair_normalized+theover_model_total"
    assert audit.loc["spread_home", "best_available_probability_source"] == "calibrated_probability_pair_normalized"
    assert set(audit["best_available_selection_policy"]) == {"probability-first-v1"}     # same sort, a moved estimate
    assert diagnostics["theover_model_totals"] == {"status": "OK", "shift": 0.05, "games_moved": 1}
    assert not plain["best_available_theover_model_shift"].any()


def test_no_stake_or_approval_can_come_from_the_moved_estimate(_pregame):
    _, plain, _ = _run(_game(float("nan")))
    after, audit, _ = _run(_game(0.73))
    # What the wager checks read is identical on every candidate, moved or not.
    for column in ("calibrated_probability", "expected_value", "market_probability"):
        assert list(audit.loc[plain.index, column]) == pytest.approx(list(plain[column]))
    pick = after.iloc[0]
    assert pick["calibrated_probability"] == pytest.approx(0.495) and pick["expected_value"] < 0
    assert pick["best_available_probability"] == pytest.approx(0.545)
    assert pick["Kelly_Bet_Size"] == 0 and pick["production_bet_amount"] == 0
    assert not bool(pick["qualified_pick"]) and not bool(pick["production_eligible"]) and not bool(pick["wager_approved"])
    assert pick["Pick_Status"] != "Actionable"
    assert not audit["wager_approved"].any()


def test_a_clearly_stronger_side_is_still_the_pick(_pregame):
    best, audit, _ = _run(_game(0.73, side=0.60))
    assert list(best["market_type"]) == ["spread_home"]
    assert audit.loc["total_under", "best_available_probability"] == pytest.approx(0.545)   # moved, and still second
    assert best["second_best_pick"].iloc[0] == audit.loc["total_under", "best_pick"]


def test_the_board_says_the_estimate_was_moved():
    from app_core.per_game_boards import per_game_board
    final = {"league": "NCAAB", "Home": "Rice", "Away": "Tulsa", "best_pick": "Under 145.5", "market_type": "total_under",
             "matchup_id": "g1", "odds_american": -110, "total_line": 145.5, "Play_Stake": 0, "production_eligible": False,
             "best_available_selection_policy": "probability-first-v1", "best_available_probability": 0.545,
             "best_available_probability_source": "calibrated_probability_pair_normalized+theover_model_total"}
    moved = per_game_board(pd.DataFrame([final]), None, "overall").iloc[0]
    assert moved["probability_basis"] == "Candidate win estimate, moved by TheOver model total (unvalidated)"
    assert moved["win_probability"] == pytest.approx(0.545) and not bool(moved["Bettable"])
    plain = dict(final, best_available_probability=0.495, best_available_probability_source="calibrated_probability_pair_normalized")
    assert per_game_board(pd.DataFrame([plain]), None, "overall").iloc[0]["probability_basis"] == "Candidate win estimate (pair-normalized)"


def test_the_shift_reaches_the_audit_and_the_export_and_stays_out_of_the_hashed_predictions():
    from core.streamlit_pipeline import BEST_PICK_COLUMNS, REQUIRED_BEST_PICK_EXPORT_COLUMNS
    assert tmt.COLUMN in BEST_PICK_COLUMNS and tmt.COLUMN in REQUIRED_BEST_PICK_EXPORT_COLUMNS
    from app_core import pinnacle_model as pm
    assert not {"apply", "backed"} & set(pm.PREDICTION_FUNCTIONS)
