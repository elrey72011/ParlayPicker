"""A total survives when the regular books sit a point apart and none quotes their midpoint (owner-approved 2026-11-20).

On 2026-11-19 Novig showed placeholder prices for UMBC at Ohio State (179.5 at -9900 and -100000), FanDuel
quoted 156.5 and DraftKings 155.5. The reference for a total is the number the standard books agree on or,
when two of them disagree, their median: 156.0, which no book quotes. The game lost both totals rows, and
TheOver's Over 156.5 with them. Now, when no book quotes the reference at all, the first standard book with
both prices within half a point of it is read: its own line with its own two prices.

Quote values are the export's. The date is in the future so nothing here depends on a clock.
"""
import pandas as pd
import pytest
from pregame_selection_fixture import build_pregame_best_picks_df

from app_core.per_game_boards import per_game_board
from core.streamlit_pipeline import (
    SPLIT_TOTAL_TOL, _consensus_total_line, _consistent_total_book, _expand_live_odds_to_bet_rows,
    _priced_total_book, _trusted_live_line_source_mask,
)

PLACEHOLDER = (179.5, -9900, -100000)
FANDUEL, DRAFTKINGS = (156.5, -110, -110), (155.5, -112, -108)


def _game(**totals):
    """The game with each named book's total as ``(line, over price, under price)``; ``None`` leaves a price out."""
    row = {"league": "NCAAB", "home_team": "Ohio State", "away_team": "Umbc", "game_date": "2099-11-19",
           "matchup_id": "ohio-state-umbc", "commence_time_raw": "2099-11-20T00:00:00Z",
           "game_time_est": "2099-11-19 7:00 PM ET"}
    for book, (line, over, under) in totals.items():
        row[f"{book}_over_point"] = row[f"{book}_under_point"] = line
        if over is not None:
            row[f"{book}_over_price"] = over
        if under is not None:
            row[f"{book}_under_price"] = under
    return row


def _totals(row):
    expanded, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([row]), None)
    return expanded[expanded.market_type.str.startswith("total")].set_index("market_type")


def _unresolved(totals):
    return (len(totals) == 2 and totals.odds_american.isna().all() and totals.opposing_odds_american.isna().all()
            and set(totals.odds_source) == {"rejected_live_total_price"})


# ---- the game of 19 November ----

def test_the_game_keeps_its_total_at_the_first_regular_book_with_both_prices():
    row = _game(novig=PLACEHOLDER, fanduel=FANDUEL, draftkings=DRAFTKINGS)
    assert _consensus_total_line(row) == 156.0                       # the reference itself is not changed
    assert _priced_total_book(row, "over") == _priced_total_book(row, "under") == ("fanduel", "split")
    assert _consistent_total_book(row, "over") == _consistent_total_book(row, "under") == "fanduel"
    totals = _totals(row)
    assert set(totals.index) == {"total_over", "total_under"} and set(totals.total_line) == {156.5}
    assert set(totals.odds_american) == {-110} and set(totals.opposing_odds_american) == {-110}
    assert set(totals.odds_source) == {"odds_api"} and set(totals.opposing_odds_source) == {"fanduel"}
    assert set(totals.line_source) == {"fanduel_priced_total_split"}


def test_with_no_novig_total_at_all_the_split_books_are_read_the_same_way():
    totals = _totals(_game(fanduel=FANDUEL, draftkings=DRAFTKINGS))
    assert set(totals.total_line) == {156.5} and set(totals.line_source) == {"fanduel_priced_total_split"}


def test_a_book_without_both_prices_is_passed_over_and_the_next_gives_its_own_line_and_prices():
    row = _game(novig=PLACEHOLDER, fanduel=(156.5, -110, None), draftkings=DRAFTKINGS)
    assert _priced_total_book(row, "over") == _priced_total_book(row, "under") == ("draftkings", "split")
    totals = _totals(row)
    assert set(totals.total_line) == {155.5} and set(totals.line_source) == {"draftkings_priced_total_split"}
    assert totals.loc["total_over", "odds_american"] == -112 and totals.loc["total_over", "opposing_odds_american"] == -108
    assert totals.loc["total_under", "odds_american"] == -108 and totals.loc["total_under", "opposing_odds_american"] == -112
    assert set(totals.opposing_odds_source) == {"draftkings"}                     # never FanDuel's line with another book's price


# ---- what does not change ----

def test_a_priced_novig_total_within_a_point_still_sets_the_line():
    totals = _totals(_game(novig=(156.5, -104, -106), fanduel=FANDUEL, draftkings=DRAFTKINGS))
    assert set(totals.total_line) == {156.5} and totals.loc["total_over", "odds_american"] == -104
    assert not totals.line_source.str.contains("_priced_total_").any()            # Novig's own quote, as before


def test_books_agreeing_on_a_number_are_still_read_as_the_consensus():
    row = _game(novig=PLACEHOLDER, fanduel=FANDUEL, draftkings=DRAFTKINGS, betmgm=(155.5, -105, -115))
    assert _consensus_total_line(row) == 155.5
    assert _priced_total_book(row, "over") == ("draftkings", "consensus")
    assert set(_totals(row).line_source) == {"draftkings_priced_total_consensus"}


@pytest.mark.parametrize("other", [157.5, 157.0])
def test_books_more_than_a_point_apart_leave_the_total_unresolved_as_before(other):
    row = _game(novig=PLACEHOLDER, fanduel=(other, -110, -110), draftkings=DRAFTKINGS)
    assert SPLIT_TOTAL_TOL == 0.5
    assert _priced_total_book(row, "over") == _priced_total_book(row, "under") == (None, None)
    assert _unresolved(_totals(row))


def test_a_consensus_line_that_books_quote_without_prices_is_never_replaced_by_a_neighbour():
    row = _game(novig=PLACEHOLDER, fanduel=(156.5, None, None), draftkings=(156.5, None, None), betmgm=(157.0, -110, -110))
    assert _consensus_total_line(row) == 156.5
    assert _priced_total_book(row, "over") == _priced_total_book(row, "under") == (None, None)
    assert _unresolved(_totals(row))


def test_one_side_of_a_mismatched_pair_on_the_reference_gives_over_and_under_the_same_answer():
    row = _game(novig=PLACEHOLDER, fanduel=(156.0, -110, -110), draftkings=DRAFTKINGS, betmgm=(156.5, -105, -115))
    row["fanduel_under_point"] = 157.0                # FanDuel's Over sits on the reference, its Under does not
    assert _consensus_total_line(row) == 156.0
    assert _priced_total_book(row, "over") == _priced_total_book(row, "under") == (None, None)


def test_with_no_priced_pair_anywhere_no_price_is_made_up():
    row = _game(novig=PLACEHOLDER, fanduel=(156.5, -110, None), draftkings=(155.5, None, -108))
    assert _priced_total_book(row, "over") == (None, None) and _unresolved(_totals(row))


# ---- the label survives the line guards, the ranking and the board ----

def test_the_split_label_is_a_live_line_for_the_three_regular_books_only():
    sources = pd.Series(["fanduel_priced_total_split", "draftkings_priced_total_split", "betmgm_priced_total_split",
                         "novig_priced_total_split", "unknown_priced_total_split", "rejected_live_total_price"])
    assert _trusted_live_line_source_mask(sources).tolist() == [True, True, True, False, False, False]


def test_the_split_total_survives_ranking_and_reaches_the_board():
    row = _game(novig=PLACEHOLDER, fanduel=FANDUEL, draftkings=DRAFTKINGS)
    row.update({"novig_home_point": -54.5, "novig_away_point": 54.5, "novig_home_price": -9900, "novig_away_price": -7592,
                "fanduel_home_point": -28.5, "fanduel_away_point": 28.5, "fanduel_home_price": -118, "fanduel_away_price": -104,
                "draftkings_home_point": -28.5, "draftkings_away_point": 28.5, "draftkings_home_price": -110,
                "draftkings_away_price": -110})
    expanded, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([row]), None)
    # Synthetic probabilities exercise the ranking, not a forecast of the real game.
    expanded["calibrated_probability"] = expanded.market_type.map(
        {"spread_home": .51, "spread_away": .49, "total_over": .53, "total_under": .47})
    expanded["model_probability"] = expanded.calibrated_probability
    expanded["expected_value"] = -.02
    expanded["edge"] = -.01
    expanded["market_probability"] = .5
    diagnostics = {}
    final = build_pregame_best_picks_df(expanded, diagnostics_out=diagnostics)
    audit = diagnostics["candidate_audit_df"]
    ranked = audit[audit.market_type.str.startswith("total")]
    assert len(ranked) == 2 and set(ranked.total_line) == {156.5}
    assert diagnostics["preselection_dropped_total_candidate_count"] == 0
    assert set(ranked.line_source) == {"fanduel_priced_total_split"}
    final["Bettable"] = False
    final["Play_Stake"] = 0
    board = per_game_board(final, audit, "totals").iloc[0]
    assert board["pick"] in {"Over 156.5", "Under 156.5"} and board.odds == -110
    assert board.status == "PASS" and board.Play_Stake == 0
