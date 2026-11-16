"""TheOver's model-backed totals picks listed on the homepage (owner 2026-11-16)."""
from __future__ import annotations

import pandas as pd

from app.ui.daily_dashboard import THEOVER_MODEL_CAPTION, THEOVER_MODEL_COLUMNS, theover_model_board


def _candidate(game, market_type, pick, *, picked=1.0, hit=0.73, context="CURRENT_PREGAME", start="2026-11-16T23:00:00Z",
               selected=False, odds=-110.0, best=-104.0, book="FanDuel", line=145.5, **extra):
    row = {"matchup_id": game, "league": "NCAAB", "away_team": "Tulsa", "home_team": game, "market_type": market_type,
           "best_pick": pick, "theover_pick_agrees": picked, "theover_model_hit_rate": hit, "theover_pick_line": line,
           "candidate_context": context, "pregame_quote_valid": True, "odds_american": odds, "best_line_price": best,
           "best_line_price_book": book, "best_available_selected": selected, "game_start_utc": start}
    row.update(extra)
    return row


def test_one_row_per_game_for_the_side_theover_picked_with_a_listed_hit_rate():
    candidates = pd.DataFrame([
        _candidate("Rice", "total_under", "Under 145.5", selected=True),
        _candidate("Rice", "total_over", "Over 145.5", picked=0.0),                 # the side it did not pick
        _candidate("Rice", "spread_home", "Rice -3.5"),                             # not a total
        _candidate("Navy", "total_over", "Over 139.5", hit=float("nan")),           # no hit rate listed
        _candidate("Army", "total_over", "Over 131.5", picked=float("nan")),        # no TheOver pick
        _candidate("Iowa", "total_over", "Over 150.5", context="POST_START_DIAGNOSTIC"),
        _candidate("Utah", "total_over", "Over 151.5", pregame_quote_valid=False),
        _candidate("Yale", "total_over", "Over 141.5", hit=0.61, start="2026-11-16T21:00:00Z", best=float("nan"), book=""),
    ])
    board = theover_model_board(candidates)
    assert list(board.columns) == THEOVER_MODEL_COLUMNS
    # In start order: Yale tips first.
    assert list(board["Game"]) == ["Tulsa at Yale", "Tulsa at Rice"]
    rice = board.iloc[1]
    assert (rice["Sport"], rice["Pick"], rice["Odds"], rice["Best price"], rice["Best price book"]) == (
        "NCAAB", "Under 145.5", -110, -104, "FanDuel")
    assert (rice["TheOver's line"], rice["Model hit rate"], rice["Also the app's pick"]) == (145.5, 0.73, "Yes")
    yale = board.iloc[0]
    assert pd.isna(yale["Best price"]) and yale["Best price book"] == "" and yale["Also the app's pick"] == ""


def test_a_game_listed_twice_appears_once_and_older_audits_without_a_context_still_work():
    candidates = pd.DataFrame([_candidate("Rice", "total_under", "Under 145.5"),
                               _candidate("Rice", "total_under", "Under 145.5", odds=-112.0)])
    assert len(theover_model_board(candidates)) == 1
    bare = candidates.drop(columns=["candidate_context", "pregame_quote_valid", "best_line_price", "best_line_price_book",
                                    "theover_pick_line", "game_start_utc"])
    board = theover_model_board(bare)
    assert len(board) == 1 and pd.isna(board.loc[0, "Best price"]) and board.loc[0, "Best price book"] == ""


def test_nothing_to_list_gives_an_empty_board_with_its_columns():
    for candidates in (None, pd.DataFrame(), pd.DataFrame([{"market_type": "total_over"}]),
                       pd.DataFrame([_candidate("Rice", "spread_home", "Rice -3.5")]),
                       pd.DataFrame([_candidate("Rice", "total_over", "Over 145.5", hit=1.0)])):
        board = theover_model_board(candidates)
        assert board.empty and list(board.columns) == THEOVER_MODEL_COLUMNS


def test_the_caption_says_what_it_does_and_what_it_cannot_do():
    assert "adds 5 points to the app's win estimate" in THEOVER_MODEL_CAPTION
    assert "147-92" in THEOVER_MODEL_CAPTION and "113-119" in THEOVER_MODEL_CAPTION
    assert "Nine nights is not proof" in THEOVER_MODEL_CAPTION
    assert "No stake or wager approval comes from it." in THEOVER_MODEL_CAPTION
    from app_core.theover_model_totals import SHIFT
    assert SHIFT == 0.05                                    # the caption's "5 points"


def test_the_list_changes_nothing_it_reads():
    candidates = pd.DataFrame([_candidate("Rice", "total_under", "Under 145.5"),
                               _candidate("Rice", "total_over", "Over 145.5", picked=0.0)])
    before = candidates.copy(deep=True)
    theover_model_board(candidates)
    pd.testing.assert_frame_equal(candidates, before)


def _app():
    import pandas as pd
    import streamlit as st
    from app.ui.daily_dashboard import render_daily_dashboard
    today, details = st.tabs(["Today", "Details"])
    board = pd.DataFrame([{"league": "NCAAB", "Home": "Rice", "Away": "Tulsa", "best_pick": "Tulsa +3.5",
                           "market_type": "spread_away", "matchup_id": "g1", "production_win_probability": 0.52,
                           "production_edge": 0.0, "production_expected_value": -0.02, "odds_american": -110,
                           "Play_Stake": 0, "production_eligible": False}])
    candidates = pd.DataFrame([
        {"matchup_id": "g1", "league": "NCAAB", "away_team": "Tulsa", "home_team": "Rice", "market_type": "total_under",
         "best_pick": "Under 145.5", "theover_pick_agrees": 1.0, "theover_model_hit_rate": 0.73, "theover_pick_line": 146.5,
         "candidate_context": "CURRENT_PREGAME", "pregame_quote_valid": True, "odds_american": -110.0,
         "best_line_price": -104.0, "best_line_price_book": "FanDuel", "best_available_selected": False,
         "game_start_utc": "2026-11-16T23:00:00Z"}])
    render_daily_dashboard(today.empty(), details.empty(), board, candidates)


def test_the_homepage_shows_the_list_under_the_boards():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(_app).run()
    assert not app.exception
    assert any(e.label == "TheOver model totals · 1 pick(s)" for e in app.expander)
    frame = next(d.value for d in app.dataframe if "Model hit rate" in d.value.columns)
    assert list(frame.columns) == THEOVER_MODEL_COLUMNS
    row = frame.iloc[0]
    assert (row["Game"], row["Pick"], row["TheOver's line"], row["Odds"], row["Best price"], row["Best price book"],
            row["Model hit rate"], row["Also the app's pick"]) == (
        "Tulsa at Rice", "Under 145.5", 146.5, -110, -104, "FanDuel", "73%", "")
    assert any(c.value == THEOVER_MODEL_CAPTION for c in app.caption)
    # The list only reads: the saved pick above it is shown as saved, and nothing is approved.
    picks = next(d.value for d in app.dataframe if "Wager status" in d.value.columns)
    assert picks.iloc[0]["Best pick"] == "Tulsa +3.5" and picks.iloc[0]["Wager status"] == "PASS"
