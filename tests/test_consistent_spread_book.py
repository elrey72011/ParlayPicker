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
    _consistent_standard_spread_pair,
    _expand_live_odds_to_bet_rows,
    _filter_preselection_line_integrity,
    _novig_spread_is_consensus_outlier,
    _novig_spread_quote_for_favorite,
    _oriented_standard_spread_pair,
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
                "novig_team_bound_quote",
                "novig_moneyline_verified",
                "novig_moneyline_reoriented",
                "fanduel_standard_spread_consensus",
            ]
        )
    )

    assert mask.tolist() == [True, True, True, True]


def test_novig_team_binding_never_swaps_quotes_between_teams():
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

    assert (home_point, home_price, home_remapped) == (1.5, -208.0, False)
    assert (away_point, away_price, away_remapped) == (-1.5, 194.0, False)


def test_expand_preserves_novig_team_bound_miami_run_line():
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

    assert float(home["spread_line"]) == 1.5
    assert float(home["odds_american"]) == -208.0
    assert home["line_source"] == "novig_team_bound_quote"
    assert home["orientation_source"].endswith("|odds_api_team_binding")
    assert float(away["spread_line"]) == -1.5
    assert float(away["odds_american"]) == 194.0

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
    assert home["line_source"] == "novig_team_bound_quote"
    assert away["line_source"] == "novig_team_bound_quote"
    assert home["orientation_source"].endswith("|odds_api_team_binding")
    assert away["orientation_source"].endswith("|odds_api_team_binding")


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



def test_conflicting_spread_signs_use_the_corroborated_team_bound_pair():
    # DraftKings and BetMGM agree on the complete signed team pair while FanDuel is
    # the lone dissenting book. Preserve the majority's team binding and keep each
    # price attached to its quoted point; moneyline favoritism must not swap outcomes.
    live_row = {
        "league": "MLB",
        "home_team": "Tampa Bay",
        "away_team": "Texas",
        "game_date": "2026-07-30",
        "matchup_id": "tb-tex-price-pair",
        "commence_time_raw": "2026-07-30T16:11:00Z",
        "novig_home_point": 1.5,
        "novig_home_price": 189,
        "novig_away_point": -1.5,
        "novig_away_price": -107,
        "novig_h2h_home_price": -178,
        "novig_h2h_away_price": 167,
        "novig_over_point": 5.5,
        "novig_over_price": -105,
        "novig_under_point": 5.5,
        "novig_under_price": -115,
        "fanduel_home_point": -1.5,
        "fanduel_home_price": 125,
        "fanduel_away_point": 1.5,
        "fanduel_away_price": -145,
        "fanduel_h2h_home_price": -180,
        "fanduel_h2h_away_price": 140,
        "draftkings_home_point": 1.5,
        "draftkings_home_price": -150,
        "draftkings_away_point": -1.5,
        "draftkings_away_price": 130,
        "draftkings_h2h_home_price": -203,
        "draftkings_h2h_away_price": 153,
        "betmgm_home_point": 1.5,
        "betmgm_home_price": -148,
        "betmgm_away_point": -1.5,
        "betmgm_away_price": 128,
        "betmgm_h2h_home_price": -190,
        "betmgm_h2h_away_price": 150,
    }
    hint = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Tampa Bay",
                "away_team": "Texas",
                "game_date": "2026-07-30",
                "matchup_id": "tb-tex-price-pair",
                "market_type": market_type,
                "orientation_favorite_side": (
                    "home" if market_type == "orientation_hint" else pd.NA
                ),
            }
            for market_type in ("orientation_hint", "spread_home", "spread_away")
        ]
    )

    home_pair = _consistent_standard_spread_pair(live_row, "home")
    away_pair = _consistent_standard_spread_pair(live_row, "away")
    assert home_pair == (1.5, -150.0, 130.0, "draftkings")
    assert away_pair == (-1.5, 130.0, -150.0, "draftkings")

    expanded, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([live_row]), hint)
    spreads = expanded[
        expanded["market_type"].isin(["spread_home", "spread_away"])
    ].set_index("market_type")

    assert float(spreads.loc["spread_home", "spread_line"]) == 1.5
    assert float(spreads.loc["spread_home", "odds_american"]) == -150.0
    assert float(spreads.loc["spread_home", "opposing_odds_american"]) == 130.0
    assert float(spreads.loc["spread_away", "spread_line"]) == -1.5
    assert float(spreads.loc["spread_away", "odds_american"]) == 130.0
    assert float(spreads.loc["spread_away", "opposing_odds_american"]) == -150.0
    assert spreads["line_source"].eq(
        "draftkings_standard_spread_consensus"
    ).all()
    assert spreads["opposing_odds_source"].eq("draftkings").all()

def test_replaces_washington_plus_5_5_alt_line_with_standard_consensus():
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
    spreads = spreads.set_index("market_type")
    assert float(spreads.loc["spread_home", "spread_line"]) == 1.5
    assert float(spreads.loc["spread_home", "odds_american"]) == -165.0
    assert float(spreads.loc["spread_away", "spread_line"]) == -1.5
    assert float(spreads.loc["spread_away", "odds_american"]) == 145.0
    assert spreads["line_source"].eq(
        "fanduel_standard_spread_consensus"
    ).all()
    assert spreads["orientation_source"].str.contains(
        "novig_moneyline_favorite.*standard_spread_consensus",
        regex=True,
    ).all()


def test_totals_only_upload_preserves_standard_book_team_bindings():
    # 31 Jul production regression: a totals-only upload and pathological Novig
    # +/-5.5 alternate caused standard-book outcomes to be reassigned according to
    # the moneyline favorite. Outcome names already bind point and price to the team,
    # so the corroborated FanDuel/DraftKings pair must remain unchanged.
    live_row = {
        "league": "MLB",
        "home_team": "Atlanta",
        "away_team": "Washington",
        "game_date": "2026-07-31",
        "matchup_id": "wsh-atl",
        "commence_time_raw": "2026-07-31T23:15:00Z",
        "novig_home_point": 5.5,
        "novig_home_price": -99900,
        "novig_away_point": -5.5,
        "novig_away_price": 1000,
        "novig_h2h_home_price": -116,
        "novig_h2h_away_price": 106,
        "novig_over_point": 8.5,
        "novig_over_price": -110,
        "novig_under_point": 8.5,
        "novig_under_price": -110,
        "fanduel_home_point": 1.5,
        "fanduel_home_price": -196,
        "fanduel_away_point": -1.5,
        "fanduel_away_price": 162,
        "fanduel_h2h_home_price": -116,
        "fanduel_h2h_away_price": 106,
        "draftkings_home_point": 1.5,
        "draftkings_home_price": -190,
        "draftkings_away_point": -1.5,
        "draftkings_away_price": 156,
        "draftkings_h2h_home_price": -118,
        "draftkings_h2h_away_price": -102,
    }
    totals_only = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Atlanta",
                "away_team": "Washington",
                "game_date": "2026-07-31",
                "matchup_id": "wsh-atl",
                "market_type": market_type,
                "total_line": 8.5,
            }
            for market_type in ("total_over", "total_under")
        ]
    )

    home_pair = _oriented_standard_spread_pair(live_row, "home", "home")
    away_pair = _oriented_standard_spread_pair(live_row, "away", "home")
    assert home_pair == (1.5, -196.0, 162.0, "fanduel", False)
    assert away_pair == (-1.5, 162.0, -196.0, "fanduel", False)

    expanded, _ = _expand_live_odds_to_bet_rows(
        pd.DataFrame([live_row]), totals_only
    )
    retained = _filter_preselection_line_integrity(expanded)
    assert set(retained["market_type"]) == {
        "spread_home",
        "spread_away",
        "total_over",
        "total_under",
    }

    spreads = retained[
        retained["market_type"].isin(["spread_home", "spread_away"])
    ].set_index("market_type")
    assert spreads["candidate_source"].eq("live_market_only").all()
    assert float(spreads.loc["spread_home", "spread_line"]) == 1.5
    assert float(spreads.loc["spread_home", "odds_american"]) == -196.0
    assert float(spreads.loc["spread_home", "opposing_odds_american"]) == 162.0
    assert float(spreads.loc["spread_away", "spread_line"]) == -1.5
    assert float(spreads.loc["spread_away", "odds_american"]) == 162.0
    assert float(spreads.loc["spread_away", "opposing_odds_american"]) == -196.0
    assert spreads["line_source"].eq(
        "fanduel_standard_spread_consensus"
    ).all()
    assert ~spreads["orientation_source"].str.contains(
        "standard_spread_sign_rebound", regex=False
    ).any()


def test_aug1_yankees_quote_never_inherits_cubs_point_and_price():
    # Export regression: every standard book bound Cubs +1.5 to the negative
    # price and Yankees -1.5 to the positive price. The old moneyline-orientation
    # rebound fabricated Yankees +1.5 at the Cubs price, a quote no book offered.
    live_row = {
        "league": "MLB",
        "home_team": "Chicago Cubs",
        "away_team": "New York Yankees",
        "game_date": "2026-08-01",
        "matchup_id": "nyy-chc-aug1",
        "commence_time_raw": "2026-08-01T23:16:00Z",
        "novig_home_point": -3.5,
        "novig_home_price": -100000,
        "novig_away_point": 3.5,
        "novig_away_price": -733,
        "novig_h2h_home_price": -108,
        "novig_h2h_away_price": 104,
        "fanduel_home_point": 1.5,
        "fanduel_home_price": -205,
        "fanduel_away_point": -1.5,
        "fanduel_away_price": 168,
        "fanduel_h2h_home_price": -106,
        "fanduel_h2h_away_price": -102,
        "draftkings_home_point": 1.5,
        "draftkings_home_price": -204,
        "draftkings_away_point": -1.5,
        "draftkings_away_price": 167,
        "draftkings_h2h_home_price": -117,
        "draftkings_h2h_away_price": -103,
        "betmgm_home_point": 1.5,
        "betmgm_home_price": -210,
        "betmgm_away_point": -1.5,
        "betmgm_away_price": 170,
        "betmgm_h2h_home_price": -118,
        "betmgm_h2h_away_price": -102,
    }

    assert _consistent_standard_spread_pair(live_row, "home") == (
        1.5,
        -205.0,
        168.0,
        "fanduel",
    )
    assert _consistent_standard_spread_pair(live_row, "away") == (
        -1.5,
        168.0,
        -205.0,
        "fanduel",
    )

    expanded, _ = _expand_live_odds_to_bet_rows(pd.DataFrame([live_row]), None)
    spreads = expanded[expanded["market_type"].isin(["spread_home", "spread_away"])].set_index(
        "market_type"
    )
    assert float(spreads.loc["spread_home", "spread_line"]) == 1.5
    assert float(spreads.loc["spread_home", "odds_american"]) == -205.0
    assert float(spreads.loc["spread_away", "spread_line"]) == -1.5
    assert float(spreads.loc["spread_away", "odds_american"]) == 168.0

    royals_row = {
        "novig_home_point": 1.5,
        "novig_home_price": -167,
        "novig_away_point": -1.5,
        "novig_away_price": 156,
        "fanduel_home_point": 1.5,
        "fanduel_home_price": -184,
        "fanduel_away_point": -1.5,
        "fanduel_away_price": 155,
        "draftkings_home_point": 1.5,
        "draftkings_home_price": -180,
        "draftkings_away_point": -1.5,
        "draftkings_away_price": 148,
        "betmgm_home_point": 1.5,
        "betmgm_home_price": -185,
        "betmgm_away_point": -1.5,
        "betmgm_away_price": 150,
    }
    assert _consistent_standard_spread_pair(royals_row, "home") == (
        1.5,
        -184.0,
        155.0,
        "fanduel",
    )
    assert _consistent_standard_spread_pair(royals_row, "away") == (
        -1.5,
        155.0,
        -184.0,
        "fanduel",
    )



def test_rejects_pathological_novig_spread_pair_without_synthetic_minus_110():
    # 30 Jul production regression: only DraftKings carried a standard run line,
    # so Novig's alternate +/-3.5 market could not be identified as a consensus
    # outlier. The suspended Texas quote (-100000) was later sanitized to -110 and
    # became a fake +29.5% EV rank-1 pick. Reject both sides of the price pair.
    live_row = {
        "league": "MLB",
        "home_team": "Tampa Bay",
        "away_team": "Texas",
        "game_date": "2026-07-30",
        "matchup_id": "tb-tex-pathological-price",
        "commence_time_raw": "2026-07-30T16:11:00Z",
        "novig_home_point": -3.5,
        "novig_home_price": 111,
        "novig_away_point": 3.5,
        "novig_away_price": -100000,
        "novig_h2h_home_price": -2500,
        "novig_h2h_away_price": 900,
        "draftkings_home_point": 1.5,
        "draftkings_home_price": -10000,
        "draftkings_away_point": -1.5,
        "draftkings_away_price": 1440,
        "draftkings_h2h_home_price": -2300,
        "draftkings_h2h_away_price": 830,
    }
    hint = pd.DataFrame(
        [
            {
                "league": "MLB",
                "home_team": "Tampa Bay",
                "away_team": "Texas",
                "game_date": "2026-07-30",
                "matchup_id": "tb-tex-pathological-price",
                "market_type": market_type,
                "orientation_favorite_side": (
                    "home" if market_type == "orientation_hint" else pd.NA
                ),
            }
            for market_type in ("orientation_hint", "spread_home", "spread_away")
        ]
    )

    assert not _novig_spread_is_consensus_outlier(live_row)

    expanded, _ = _expand_live_odds_to_bet_rows(
        pd.DataFrame([live_row]), hint
    )
    spreads = expanded[
        expanded["market_type"].isin(["spread_home", "spread_away"])
    ]

    assert len(spreads) == 2
    assert spreads["spread_line"].isna().all()
    assert spreads["odds_american"].isna().all()
    assert spreads["opposing_odds_american"].isna().all()
    assert spreads["odds_source"].eq("rejected_live_spread_price").all()
    assert spreads["line_source"].eq("rejected_live_spread_price").all()
    assert spreads["orientation_source"].str.endswith(
        "|spread_price_pair_rejected"
    ).all()
    assert not (
        spreads["odds_source"].eq("fallback_novig")
        & spreads["odds_american"].eq(-110.0)
    ).any()


