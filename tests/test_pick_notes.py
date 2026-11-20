"""Best price at a candidate's line and the tie flag (app_core.pick_notes; owner 2026-11-15)."""
from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from app_core import pick_notes


def _quote(book, market_type, point, price, recorded_at="2026-11-15T23:40:00Z"):
    return {"book": book, "market_type": market_type, "point": point, "price": price, "recorded_at": recorded_at}


def _row(market_type="spread_home", line=-3.5, quotes=()):
    total = market_type.startswith("total")
    return {"market_type": market_type, "spread_line": None if total else line, "total_line": line if total else None,
            "provider_quotes": json.dumps(list(quotes))}


@pytest.mark.parametrize("american, expected", [(150, 2.5), (-200, 1.5), (100, 2.0), (-100, 2.0), ("-110", 1 + 100 / 110)])
def test_decimal_converts_a_real_american_price(american, expected):
    assert pick_notes.decimal(american) == pytest.approx(expected)


@pytest.mark.parametrize("american", [99, -99, 0, None, float("nan"), "abc", True, 10001, -10001, float("inf")])
def test_decimal_refuses_what_is_not_a_price(american):
    assert pick_notes.decimal(american) is None


def test_best_price_is_the_highest_payout_for_the_same_side_at_the_same_line():
    found = pick_notes.best_price(_row(quotes=[
        _quote("novig", "spread_home", -3.5, -112),
        _quote("fanduel", "spread_home", -3.5, -108),
        _quote("draftkings", "spread_home", -3.5, -110),
        _quote("betmgm", "spread_home", -4.5, 105),        # another line: not the same bet
        _quote("fanduel", "spread_away", 3.5, 120),        # the other side
        _quote("pinnacle", "spread_home", -3.5, 101),      # a reference, never an entry
    ]))
    assert found == {"best_line_price": -108.0, "best_line_price_book": "FanDuel", "best_line_price_books": 3}


def test_a_plus_price_beats_any_minus_price():
    found = pick_notes.best_price(_row(quotes=[_quote("novig", "spread_home", -3.5, -102),
                                               _quote("betmgm", "spread_home", -3.5, 100)]))
    assert (found["best_line_price"], found["best_line_price_book"]) == (100.0, "BetMGM")


def test_every_book_at_the_best_price_is_named_in_stake_book_order():
    found = pick_notes.best_price(_row("total_over", 145.5, [
        _quote("betmgm", "total_over", 145.5, -105),
        _quote("novig_us", "total_over", 145.5, -105),
        _quote("draftkings", "total_over", 145.5, -110),
    ]))
    assert found == {"best_line_price": -105.0, "best_line_price_book": "Novig, BetMGM", "best_line_price_books": 3}


def test_a_quote_much_older_than_the_newest_one_is_left_out():
    quotes = [_quote("novig", "spread_home", -3.5, -112, "2026-11-15T23:40:00Z"),
              _quote("fanduel", "spread_home", -3.5, 105, "2026-11-15T23:19:59Z"),     # 20 minutes and a second older
              _quote("draftkings", "spread_home", -3.5, -109, "2026-11-15T23:20:00Z")]  # exactly 20 minutes: kept
    found = pick_notes.best_price(_row(quotes=quotes))
    assert found == {"best_line_price": -109.0, "best_line_price_book": "DraftKings", "best_line_price_books": 2}


def test_quotes_without_an_update_time_or_a_real_price_are_left_out():
    quotes = [_quote("novig", "spread_home", -3.5, -112),
              _quote("fanduel", "spread_home", -3.5, 150, recorded_at=None),
              _quote("draftkings", "spread_home", -3.5, 150, recorded_at="2026-11-15T23:40:00"),   # no time zone
              _quote("betmgm", "spread_home", -3.5, 50)]
    assert pick_notes.best_price(_row(quotes=quotes)) == {
        "best_line_price": -112.0, "best_line_price_book": "Novig", "best_line_price_books": 1}


def test_a_book_quoting_the_line_twice_counts_once_at_its_newest_quote():
    quotes = [_quote("fanduel", "spread_home", -3.5, 120, "2026-11-15T23:30:00Z"),
              _quote("fanduel", "spread_home", -3.5, -115, "2026-11-15T23:40:00Z")]
    assert pick_notes.best_price(_row(quotes=quotes)) == {
        "best_line_price": -115.0, "best_line_price_book": "FanDuel", "best_line_price_books": 1}


@pytest.mark.parametrize("row", [
    _row(quotes=[]),
    _row(quotes=[_quote("pinnacle", "spread_home", -3.5, -105)]),
    dict(_row(), provider_quotes="not json"),
    dict(_row(), provider_quotes=None),
    dict(_row(), spread_line=None),
    dict(_row(), market_type=""),
])
def test_no_usable_quote_gives_no_price(row):
    found = pick_notes.best_price(row)
    assert math.isnan(found["best_line_price"])
    assert (found["best_line_price_book"], found["best_line_price_books"]) == ("", 0)


def test_prices_keeps_the_index_and_handles_an_empty_or_bare_frame():
    quotes = [_quote("novig", "total_under", 145.5, -110), _quote("fanduel", "total_under", 145.5, -104)]
    frame = pd.DataFrame([_row("total_under", 145.5, quotes), _row("total_over", 145.5, quotes)], index=[7, 3])
    out = pick_notes.prices(frame)
    assert list(out.index) == [7, 3] and list(out.columns) == list(pick_notes.PRICE_FIELDS)
    assert out.loc[7].to_dict() == {"best_line_price": -104.0, "best_line_price_book": "FanDuel", "best_line_price_books": 2}
    assert math.isnan(out.loc[3, "best_line_price"]) and out.loc[3, "best_line_price_books"] == 0
    assert pick_notes.prices(frame.iloc[0:0]).empty
    bare = pick_notes.prices(pd.DataFrame({"market_type": ["spread_home"]}))
    assert math.isnan(bare.loc[0, "best_line_price"]) and bare.loc[0, "best_line_price_book"] == ""


def _candidates(*rows):
    return pd.DataFrame([{"matchup_id": game, "best_available_rank": rank, "best_available_probability": probability,
                          "market_type": market_type, "best_pick": pick}
                         for game, rank, probability, market_type, pick in rows])


def test_four_level_candidates_are_one_tie_and_no_lean():
    out = pick_notes.ties(_candidates(
        ("g1", 1, 0.5, "spread_away", "Tulsa +3.5"), ("g1", 2, 0.5, "spread_home", "Rice -3.5"),
        ("g1", 3, 0.5, "total_over", "Over 145.5"), ("g1", 4, 0.5, "total_under", "Under 145.5")))
    assert list(out["best_available_tie_count"]) == [4, 4, 4, 4]
    assert out["best_available_no_lean"].all()
    assert out.loc[0, "best_available_tied_with"] == "Rice -3.5 | Over 145.5 | Under 145.5"
    assert pick_notes.tie_note(*out.loc[0]) == "No lean (tie)"


def test_a_tie_with_the_other_market_is_level_but_still_a_lean_in_its_own_market():
    out = pick_notes.ties(_candidates(
        ("g1", 1, 0.5238, "spread_home", "Rice -3.5"), ("g1", 2, 0.5238, "total_over", "Over 145.5"),
        ("g1", 3, 0.4762, "spread_away", "Tulsa +3.5"), ("g1", 4, 0.4762, "total_under", "Under 145.5")))
    assert list(out["best_available_tie_count"]) == [2, 2, 2, 2]
    assert not out["best_available_no_lean"].any()
    assert pick_notes.tie_note(*out.loc[0]) == "Level with Over 145.5"
    assert pick_notes.tie_note(*out.loc[0], other_markets=False) == ""


def test_a_clear_first_place_is_no_tie():
    out = pick_notes.ties(_candidates(
        ("g1", 1, 0.55, "total_over", "Over 145.5"), ("g1", 2, 0.51, "spread_home", "Rice -3.5"),
        ("g1", 3, 0.49, "spread_away", "Tulsa +3.5"), ("g1", 4, 0.45, "total_under", "Under 145.5")))
    assert list(out["best_available_tie_count"]) == [1, 1, 1, 1]
    assert list(out["best_available_tied_with"]) == ["", "", "", ""]
    assert not out["best_available_no_lean"].any()
    assert pick_notes.tie_note(*out.loc[0]) == ""


def test_a_tie_is_within_a_billionth_and_never_across_games():
    out = pick_notes.ties(_candidates(
        ("g1", 1, 0.5 + 1e-12, "spread_away", "Tulsa +3.5"), ("g1", 2, 0.5, "spread_home", "Rice -3.5"),
        ("g2", 1, 0.5 + 1e-6, "spread_away", "Navy +1.5"), ("g2", 2, 0.5, "spread_home", "Army -1.5"),
        ("g3", 1, 0.5, "spread_away", "Iowa +2.5")))
    assert list(out["best_available_tie_count"]) == [2, 2, 1, 1, 1]
    assert list(out["best_available_no_lean"]) == [True, True, False, False, False]


def test_unranked_rows_have_no_count_and_missing_probabilities_tie_with_each_other():
    frame = _candidates(
        ("g1", 1, None, "spread_away", "Tulsa +3.5"), ("g1", 2, None, "spread_home", "Rice -3.5"),
        ("g1", None, 0.9, "total_over", "Over 145.5"))
    frame.index = [11, 5, 8]
    out = pick_notes.ties(frame)
    assert list(out.index) == [11, 5, 8]
    assert out.loc[11, "best_available_tie_count"] == 2 and bool(out.loc[11, "best_available_no_lean"])
    assert pd.isna(out.loc[8, "best_available_tie_count"]) and not out.loc[8, "best_available_no_lean"]
    assert out.loc[5, "best_available_tied_with"] == "Tulsa +3.5"


def test_ties_on_an_empty_or_bare_frame_is_all_defaults():
    assert pick_notes.ties(pd.DataFrame(columns=["matchup_id"])).empty
    out = pick_notes.ties(pd.DataFrame({"matchup_id": ["g1"]}))
    assert list(out.columns) == list(pick_notes.TIE_FIELDS)
    assert pd.isna(out.loc[0, "best_available_tie_count"]) and out.loc[0, "best_available_tied_with"] == ""


@pytest.mark.parametrize("count, others, no_lean, expected", [
    (4, "a | b | c", True, "No lean (tie)"),
    (2, "Over 145.5", "True", "No lean (tie)"),
    (3, "Over 145.5 | Over 146", False, "Level with Over 145.5, Over 146"),
    (1, "", False, ""),
    (pd.NA, float("nan"), pd.NA, ""),
    (None, None, None, ""),
])
def test_tie_note_wording(count, others, no_lean, expected):
    assert pick_notes.tie_note(count, others, no_lean) == expected


# ---- the pipeline carries both notes and its pick is unchanged ----

@pytest.fixture
def _fixed_pregame_clock(monkeypatch):
    monkeypatch.setattr("app_core.candidate_chronology.now_utc", lambda: pd.Timestamp("2026-07-27T21:00:00Z"))
    monkeypatch.setattr("core.empirical_tiers.load_bucket_stats", lambda: {})
    monkeypatch.setattr("core.probability_calibration.load_calibration", lambda: None)


def _pool_row(market_type, probability, quotes):
    total = market_type.startswith("total")
    line = 8.5 if total else (-1.5 if market_type == "spread_home" else 1.5)
    return {
        "game_id": "pick-notes", "league": "MLB", "home_team": "Chicago Cubs", "away_team": "Pittsburgh Pirates",
        "game_date": pd.Timestamp("2026-07-27", tz="UTC"), "game_start_utc": "2026-07-27T23:00:00Z",
        "odds_recorded_at": "2026-07-27T20:55:00Z", "market_type": market_type,
        "total_line": line if total else pd.NA, "spread_line": pd.NA if total else line,
        "calibrated_probability": probability, "model_probability": probability, "ml_probability": probability,
        "expected_value": 0.0, "edge": 0.0, "market_probability": 0.50, "kalshi_probability": 0.50,
        "consensus_agreement": "Neutral", "odds_american": -110, "odds_source": "test", "line_source": "live",
        "market_line_source": "live", "live_total_line": 8.5, "is_live_data": True, "used_stale_features": False,
        "provider_quotes": json.dumps(quotes),
    }


def _mlb_quotes():
    at = "2026-07-27T20:55:00Z"
    return [_quote("novig", "total_over", 8.5, -110, at), _quote("fanduel", "total_over", 8.5, -104, at),
            _quote("draftkings", "total_over", 8.5, -104, at), _quote("pinnacle", "total_over", 8.5, 102, at),
            _quote("novig", "total_under", 8.5, -110, at), _quote("novig", "spread_home", -1.5, -110, at),
            _quote("novig", "spread_away", 1.5, -110, at)]


def test_pipeline_audit_carries_the_best_price_and_leaves_the_pick_alone(_fixed_pregame_clock):
    from core.streamlit_pipeline import build_best_picks_df
    quotes = _mlb_quotes()
    analysis = pd.DataFrame([_pool_row("total_over", 0.56, quotes), _pool_row("total_under", 0.44, quotes),
                             _pool_row("spread_home", 0.52, quotes), _pool_row("spread_away", 0.48, quotes)])
    diagnostics: dict = {}
    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"].set_index("market_type")

    assert list(best["market_type"]) == ["total_over"]
    assert audit.loc["total_over", "best_available_selected"]
    assert audit.loc["total_over", "best_line_price"] == -104
    assert audit.loc["total_over", "best_line_price_book"] == "FanDuel, DraftKings"
    assert audit.loc["total_over", "best_line_price_books"] == 3
    probability = audit.loc["total_over", "best_available_probability"]
    assert audit.loc["total_over", "best_line_expected_value"] == pytest.approx(probability * (1 + 100 / 104) - 1, abs=1e-4)
    assert audit.loc["total_over", "best_line_expected_value"] > audit.loc["total_over", "best_available_expected_value"]
    assert audit.loc["spread_home", "best_line_price"] == -110 and audit.loc["spread_home", "best_line_price_book"] == "Novig"
    assert list(audit["best_available_tie_count"]) == [1, 1, 1, 1]
    assert not audit["best_available_no_lean"].any()
    assert diagnostics["pick_notes"] == {"status": "OK", "selected_with_better_price": 1, "selected_no_lean": 0}
    for column in pick_notes.PRICE_FIELDS + pick_notes.TIE_FIELDS + ("best_line_expected_value",):
        assert column in best.columns


def test_pipeline_marks_a_four_way_tie_and_still_selects_by_the_same_rule(_fixed_pregame_clock):
    from core.streamlit_pipeline import build_best_picks_df
    quotes = _mlb_quotes()
    analysis = pd.DataFrame([_pool_row(kind, 0.5, quotes)
                             for kind in ("total_over", "total_under", "spread_home", "spread_away")])
    diagnostics: dict = {}
    best = build_best_picks_df(analysis, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]

    # The tie-break is unchanged: market name decides, so the away spread is still the pick.
    assert list(best["market_type"]) == ["spread_away"]
    assert list(audit["best_available_tie_count"]) == [4, 4, 4, 4]
    assert audit["best_available_no_lean"].all()
    assert bool(best["best_available_no_lean"].iloc[0]) and best["best_available_tie_count"].iloc[0] == 4
    assert diagnostics["pick_notes"]["selected_no_lean"] == 1


def test_both_notes_reach_the_export_column_lists():
    from core.streamlit_pipeline import BEST_PICK_COLUMNS, REQUIRED_BEST_PICK_EXPORT_COLUMNS
    for column in pick_notes.PRICE_FIELDS + pick_notes.TIE_FIELDS + ("best_line_expected_value",):
        assert column in BEST_PICK_COLUMNS and column in REQUIRED_BEST_PICK_EXPORT_COLUMNS


def test_nothing_that_selects_or_stakes_reads_the_notes_back():
    """Display and audit only: the fields are written once and named by the grids and export lists.

    One reader stands outside them, added with the owner's approval on 2026-11-19: the read-only tally of
    TheOver's plan (``core/theover_plan.py``), which counts a bet's units at the best price, as the plan's
    rules say. ``tests/test_theover_plan_report.py`` pins that only the morning report loads the tally and
    that it saves nothing, so the price still cannot reach a pick or a stake.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    allowed = {"app_core/pick_notes.py", "core/streamlit_pipeline.py", "app/ui/daily_dashboard.py", "streamlit_app.py",
               "core/theover_plan.py"}
    fields = pick_notes.PRICE_FIELDS + pick_notes.TIE_FIELDS + ("best_line_expected_value",)
    users = set()
    for folder in ("app", "app_core", "core", "services", "scripts"):
        for path in (root / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="replace")
            if "pick_notes" in text or any(field in text for field in fields):
                users.add(path.relative_to(root).as_posix())
    users |= {"streamlit_app.py"} if "pick_notes" in (root / "streamlit_app.py").read_text(encoding="utf-8") else set()
    assert users == allowed


# ---- the homepage board ----

def _board_and_candidates():
    result = pd.DataFrame({"candidate_id": ["", ""], "matchup_id": ["g1", "g2"],
                           "market_type": ["spread_away", "total_over"], "pick": ["Tulsa +3.5", "Over 145.5"]})
    candidates = pd.DataFrame({
        "candidate_id": ["c1", "c2", "c3"], "matchup_id": ["g1", "g2", "g2"],
        "market_type": ["spread_away", "total_over", "spread_home"],
        "best_pick": ["Tulsa +3.5", "Over 145.5", "Rice -3.5"],
        "best_line_price": [-105.0, float("nan"), -110.0], "best_line_price_book": ["FanDuel", "", "Novig"],
        "best_available_tie_count": pd.array([4, 2, 2], dtype="Int64"),
        "best_available_tied_with": ["Rice -3.5 | Over 140 | Under 140", "Rice -3.5", "Over 145.5"],
        "best_available_no_lean": [True, False, False]})
    return result, candidates


def test_board_shows_the_best_price_and_its_book_per_row():
    from app.ui.daily_dashboard import best_price_shadow
    result, candidates = _board_and_candidates()
    price, book = best_price_shadow(result, candidates)
    assert price.iloc[0] == -105 and pd.isna(price.iloc[1])
    assert list(book) == ["FanDuel", ""]
    assert best_price_shadow(result, candidates.drop(columns=["best_line_price"])) is None
    assert best_price_shadow(result, None) is None
    assert best_price_shadow(result, candidates.assign(best_line_price=float("nan"))) is None


def test_board_marks_ties_and_a_family_board_only_marks_its_own_coin_flip():
    from app.ui.daily_dashboard import tie_notes
    result, candidates = _board_and_candidates()
    assert list(tie_notes(result, candidates)) == ["No lean (tie)", "Level with Rice -3.5"]
    assert list(tie_notes(result, candidates, "totals")) == ["No lean (tie)", ""]
    quiet = candidates.assign(best_available_tie_count=pd.array([1, 1, 1], dtype="Int64"),
                              best_available_tied_with="", best_available_no_lean=False)
    assert tie_notes(result, quiet) is None
    assert tie_notes(result, candidates.drop(columns=["best_available_no_lean"])) is None


def _board_app():
    import pandas as pd
    import streamlit as st
    from app.ui.daily_dashboard import render_daily_dashboard
    today, details = st.tabs(["Today", "Details"])
    board = pd.DataFrame([{"league": "NCAAB", "Home": "Rice", "Away": "Tulsa", "best_pick": "Tulsa +3.5",
                           "market_type": "spread_away", "matchup_id": "g1", "production_win_probability": 0.5,
                           "production_edge": 0.0, "production_expected_value": -0.045, "odds_american": -110,
                           "Play_Stake": 0, "production_eligible": False}])
    candidates = pd.DataFrame([{"matchup_id": "g1", "market_type": "spread_away", "best_pick": "Tulsa +3.5",
                                "best_line_price": -105.0, "best_line_price_book": "FanDuel",
                                "best_available_tie_count": 4,
                                "best_available_tied_with": "Rice -3.5 | Over 140.5 | Under 140.5",
                                "best_available_no_lean": True}])
    render_daily_dashboard(today.empty(), details.empty(), board, candidates)


def test_the_homepage_board_renders_both_notes_beside_the_pick():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(_board_app).run()
    assert not app.exception
    frame = next(d.value for d in app.dataframe if "Wager status" in d.value.columns)
    assert list(frame.columns) == ["Game", "Best pick", "Tie", "Odds", "Best price", "Best price book",
                                   "Win estimate", "EV estimate", "Wager status"]
    row = frame.iloc[0]
    assert (row["Best pick"], row["Tie"], row["Odds"], row["Best price"], row["Best price book"]) == (
        "Tulsa +3.5", "No lean (tie)", -110, -105, "FanDuel")
    assert row["Wager status"] == "PASS"
    assert any("Best price is the best price for the same side at the same line" in c.value for c in app.caption)
