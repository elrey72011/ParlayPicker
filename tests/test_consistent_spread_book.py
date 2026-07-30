"""Source the run line from a book whose spread agrees with its own moneyline (19 Jun).

The 10:49 raw_book_odds_diag proved novig + fanduel published HOU/CLE with the spread
flipped vs their own moneyline, while draftkings + betmgm had it right. We now take the
line AND price from the first internally-consistent book.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.streamlit_pipeline import (
    _build_moneyline_orientation_rows,
    _consistent_spread_book,
    _expand_live_odds_to_bet_rows,
    _filter_preselection_line_integrity,
    _novig_spread_is_consensus_outlier,
    _novig_spread_quote_for_favorite,
    _trusted_live_line_source_mask,
)


def _cleveland_row():
    # HOU home / CLE away. novig+fanduel flipped, draftkings+betmgm correct.
    return {
        "league": "MLB", "home_team": "Houston", "away_team": "Cleveland",
        "game_date": "2026-06-19", "matchup_id": "m", "commence_time_raw": "2026-06-19T00:11:00Z",
        "novig_home_point": 1.5, "novig_away_point": -1.5, "novig_home_price": -150, "novig_away_price": 174,
        "novig_h2h_home_price": -115, "novig_h2h_away_price": 113,
        "fanduel_home_point": 1.5, "fanduel_away_point": -1.5, "fanduel_home_price": -150, "fanduel_away_price": 160,
        "fanduel_h2h_home_price": -120, "fanduel_h2h_away_price": 102,
        "draftkings_home_point": -1.5, "draftkings_away_point": 1.5, "draftkings_home_price": 150, "draftkings_away_price": -190,
        "draftkings_h2h_home_price": -125, "draftkings_h2h_away_price": 104,
        "betmgm_home_point": -1.5, "betmgm_away_point": 1.5, "betmgm_home_price": 148, "betmgm_away_price": -185,
        "betmgm_h2h_home_price": -130, "betmgm_h2h_away_price": 105,
    }


def test_picks_first_consistent_book_skipping_flipped_ones():
    assert _consistent_spread_book(_cleveland_row()) == "draftkings"


def test_picks_novig_when_novig_is_consistent():
    # Colorado home (+1.5 underdog, ml +127) / Pittsburgh away (-1.5 fav, ml -130):
    # novig already agrees with its moneyline, so it's used unchanged.
    row = {
        "novig_home_point": 1.5, "novig_away_point": -1.5,
        "novig_h2h_home_price": 127, "novig_h2h_away_price": -130,
    }
    assert _consistent_spread_book(row) == "novig"


def test_none_when_no_book_consistent():
    row = {
        "novig_home_point": 1.5, "novig_away_point": -1.5,
        "novig_h2h_home_price": -115, "novig_h2h_away_price": 113,  # spread says away fav, ml says home fav
    }
    assert _consistent_spread_book(row) is None


def test_expand_orients_cleveland_to_plus_1_5_from_consistent_book():
    out, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([_cleveland_row()]), None)
    away = out[out.market_type == "spread_away"].iloc[0]
    home = out[out.market_type == "spread_home"].iloc[0]
    assert float(away["spread_line"]) == 1.5      # Cleveland +1.5 (was -1.5)
    assert float(away["odds_american"]) == -190.0  # draftkings' away price, not novig's
    assert float(home["spread_line"]) == -1.5     # Houston -1.5


def test_moneyline_export_preserves_favorite_orientation_without_creating_a_bet():
    rows = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Miami",
                "away_team": "Philadelphia",
                "homekalshi": "MIA",
                "awaykalshi": "PHI",
                "pick": "MIA",
                "line": -120,
            },
            {
                "league": "MLB",
                "home_team": "Los Angeles Angels",
                "away_team": "Houston",
                "homekalshi": "LAA",
                "awaykalshi": "HOU",
                "pick": "LAA",
                "line": -115,
            },
            {
                "league": "MLB",
                "home_team": "New York Mets",
                "away_team": "Atlanta",
                "homekalshi": "NYM",
                "awaykalshi": "ATL",
                "pick": "NYM",
                "line": 130,
            },
        ]
    )

    hints = _build_moneyline_orientation_rows(rows)[0]

    assert hints["market_type"].eq("orientation_hint").all()
    assert hints["orientation_favorite_side"].tolist() == ["home", "home", "away"]
    assert hints["odds_american"].isna().all()


def test_novig_moneyline_line_sources_are_trusted_live():
    mask = _trusted_live_line_source_mask(
        pd.Series(
            [
                "novig_moneyline_verified",
                "novig_moneyline_reoriented",
                "fanduel_standard_spread_consensus",
            ]
        )
    )

    assert mask.tolist() == [True, True, True]


def test_novig_reorientation_swaps_line_and_price_together():
    row = {
        "novig_home_point": 1.5,
        "novig_home_price": -208,
        "novig_away_point": -1.5,
        "novig_away_price": 194,
    }

    home_point, home_price, home_remapped = _novig_spread_quote_for_favorite(
        row, "home", "home"
    )
    away_point, away_price, away_remapped = _novig_spread_quote_for_favorite(
        row, "away", "home"
    )

    assert (home_point, home_price, home_remapped) == (-1.5, 194.0, True)
    assert (away_point, away_price, away_remapped) == (1.5, -208.0, True)


def test_expand_uses_novig_moneyline_for_miami_novig_run_line():
    live = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Miami",
                "away_team": "Philadelphia",
                "game_date": "2026-07-28",
                "matchup_id": "m",
                "commence_time_raw": "2026-07-28T22:40:00Z",
                "novig_home_point": 1.5,
                "novig_home_price": -208,
                "novig_away_point": -1.5,
                "novig_away_price": 194,
                "novig_h2h_home_price": -106,
                "novig_h2h_away_price": 104,
            }
        ]
    )
    hint = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Miami",
                "away_team": "Philadelphia",
                "game_date": "2026-07-28",
                "matchup_id": "m",
                "market_type": "orientation_hint",
                "orientation_favorite_side": "home",
            },
            {
                "league": "MLB",
                "home_team": "Miami",
                "away_team": "Philadelphia",
                "game_date": "2026-07-28",
                "matchup_id": "m",
                "market_type": "spread_home",
                "orientation_favorite_side": pd.NA,
            },
            {
                "league": "MLB",
                "home_team": "Miami",
                "away_team": "Philadelphia",
                "game_date": "2026-07-28",
                "matchup_id": "m",
                "market_type": "spread_away",
                "orientation_favorite_side": pd.NA,
            },
        ]
    )

    out, _ = _expand_live_odds_to_bet_rows(live, hint)
    home = out[out["market_type"] == "spread_home"].iloc[0]
    away = out[out["market_type"] == "spread_away"].iloc[0]

    assert float(home["spread_line"]) == -1.5
    assert float(home["odds_american"]) == 194.0
    assert home["line_source"] == "novig_moneyline_reoriented"
    assert home["orientation_source"].endswith("|novig_moneyline_favorite")
    assert float(away["spread_line"]) == 1.5
    assert float(away["odds_american"]) == -208.0

def test_novig_moneyline_beats_stale_theover_hint_for_toronto_run_line():
    # 29 Jul production regression: Novig showed TOR -1.5/+141 and
    # WAS +1.5/-147, but a conflicting TheOver hint reassigned the outcomes and
    # exported Toronto +1.5/-150. A complete Novig market must bind to itself.
    live = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Washington",
                "away_team": "Toronto",
                "game_date": "2026-07-29",
                "matchup_id": "tor-was",
                "commence_time_raw": "2026-07-29T17:05:00Z",
                "novig_home_point": 1.5,
                "novig_home_price": -147,
                "novig_away_point": -1.5,
                "novig_away_price": 141,
                "novig_h2h_home_price": 106,
                "novig_h2h_away_price": -108,
            }
        ]
    )
    stale_hint = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Washington",
                "away_team": "Toronto",
                "game_date": "2026-07-29",
                "matchup_id": "tor-was",
                "market_type": "orientation_hint",
                "orientation_favorite_side": "home",
            },
            {
                "league": "MLB",
                "home_team": "Washington",
                "away_team": "Toronto",
                "game_date": "2026-07-29",
                "matchup_id": "tor-was",
                "market_type": "spread_home",
                "orientation_favorite_side": pd.NA,
            },
            {
                "league": "MLB",
                "home_team": "Washington",
                "away_team": "Toronto",
                "game_date": "2026-07-29",
                "matchup_id": "tor-was",
                "market_type": "spread_away",
                "orientation_favorite_side": pd.NA,
            },
        ]
    )

    out, _ = _expand_live_odds_to_bet_rows(live, stale_hint)
    home = out[out["market_type"] == "spread_home"].iloc[0]
    away = out[out["market_type"] == "spread_away"].iloc[0]

    assert float(home["spread_line"]) == 1.5
    assert float(home["odds_american"]) == -147.0
    assert float(away["spread_line"]) == -1.5
    assert float(away["odds_american"]) == 141.0
    assert home["line_source"] == "novig_moneyline_verified"
    assert away["line_source"] == "novig_moneyline_verified"
    assert home["orientation_source"].endswith("|novig_moneyline_favorite")
    assert away["orientation_source"].endswith("|novig_moneyline_favorite")


def test_partial_novig_side_uses_oriented_standard_consensus_and_survives_preselection():
    # 30 Jul production regression: Tampa Bay/Texas exposed both Novig points,
    # but the Texas +1.5 outcome lacked a usable Novig price. The expansion kept
    # Tampa Bay -1.5 and silently lost Texas +1.5 during preselection even though
    # every standard book carried a real-priced +1.5.
    live_row = {
        "league": "MLB",
        "home_team": "Tampa Bay",
        "away_team": "Texas",
        "game_date": "2026-07-30",
        "matchup_id": "tb-tex",
        "commence_time_raw": "2026-07-30T16:11:00Z",
        "novig_home_point": -1.5,
        "novig_home_price": 132,
        "novig_away_point": 1.5,
        "novig_away_price": pd.NA,
        "novig_h2h_home_price": -217,
        "novig_h2h_away_price": 153,
        "novig_over_point": 6.5,
        "novig_over_price": -105,
        "novig_under_point": 6.5,
        "novig_under_price": -115,
        "fanduel_home_point": -1.5,
        "fanduel_home_price": 125,
        "fanduel_away_point": 1.5,
        "fanduel_away_price": -145,
        "fanduel_h2h_home_price": -210,
        "fanduel_h2h_away_price": 175,
        "draftkings_home_point": -1.5,
        "draftkings_home_price": 130,
        "draftkings_away_point": 1.5,
        "draftkings_away_price": -150,
        "draftkings_h2h_home_price": -215,
        "draftkings_h2h_away_price": 170,
        "betmgm_home_point": -1.5,
        "betmgm_home_price": 128,
        "betmgm_away_point": 1.5,
        "betmgm_away_price": -148,
        "betmgm_h2h_home_price": -212,
        "betmgm_h2h_away_price": 168,
    }
    hint = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Tampa Bay",
                "away_team": "Texas",
                "game_date": "2026-07-30",
                "matchup_id": "tb-tex",
                "market_type": market_type,
                "orientation_favorite_side": (
                    "home" if market_type == "orientation_hint" else pd.NA
                ),
            }
            for market_type in ("orientation_hint", "spread_home", "spread_away")
        ]
    )

    expanded, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([live_row]), hint)
    away = expanded[expanded["market_type"].eq("spread_away")].iloc[0]

    assert float(away["spread_line"]) == 1.5
    assert float(away["odds_american"]) == -145.0
    assert away["odds_source"] == "odds_api"
    assert away["line_source"] == "fanduel_standard_spread_consensus"
    assert away["orientation_source"].endswith(
        "|novig_moneyline_favorite|standard_spread_consensus"
    )

    retained = _filter_preselection_line_integrity(expanded)
    assert set(retained["market_type"]) == {
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    }


def test_rejects_washington_plus_5_5_alt_line_against_standard_consensus():
    # 29 Jul production regression: Novig exposed the alternate WAS +5.5 at
    # -1150 while every standard book carried the normal MLB run line at 1.5.
    # The alt outcome is real, but it is not a comparable Best Available spread.
    live_row = {
        "league": "MLB",
        "home_team": "Washington",
        "away_team": "Toronto",
        "game_date": "2026-07-29",
        "matchup_id": "tor-was-alt",
        "commence_time_raw": "2026-07-29T17:05:00Z",
        "novig_home_point": 5.5,
        "novig_home_price": -1150,
        "novig_away_point": -5.5,
        "novig_away_price": 1100,
        "novig_h2h_home_price": 108,
        "novig_h2h_away_price": -113,
        "fanduel_home_point": 1.5,
        "fanduel_home_price": -165,
        "fanduel_away_point": -1.5,
        "fanduel_away_price": 145,
        "fanduel_h2h_home_price": 110,
        "fanduel_h2h_away_price": -130,
        "draftkings_home_point": 1.5,
        "draftkings_home_price": -160,
        "draftkings_away_point": -1.5,
        "draftkings_away_price": 140,
        "draftkings_h2h_home_price": 108,
        "draftkings_h2h_away_price": -128,
        "betmgm_home_point": 1.5,
        "betmgm_home_price": -158,
        "betmgm_away_point": -1.5,
        "betmgm_away_price": 138,
        "betmgm_h2h_home_price": 105,
        "betmgm_h2h_away_price": -125,
    }
    hint = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Washington",
                "away_team": "Toronto",
                "game_date": "2026-07-29",
                "matchup_id": "tor-was-alt",
                "market_type": market_type,
                "orientation_favorite_side": (
                    "home" if market_type == "orientation_hint" else pd.NA
                ),
            }
            for market_type in ("orientation_hint", "spread_home", "spread_away")
        ]
    )

    assert _novig_spread_is_consensus_outlier(live_row)

    out, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([live_row]), hint)
    spreads = out[out["market_type"].isin(["spread_home", "spread_away"])]

    assert len(spreads) == 2
    assert spreads["spread_line"].isna().all()
    assert spreads["odds_american"].isna().all()
    assert spreads["line_source"].eq("rejected_live_spread_outlier").all()
    assert spreads["odds_source"].eq("rejected_live_spread_outlier").all()
    assert spreads["orientation_source"].str.endswith(
        "|novig_spread_consensus_outlier"
    ).all()

