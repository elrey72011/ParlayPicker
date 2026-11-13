"""Pinnacle movement columns on the exported candidate audit (owner 2026-11-12). SYNTHETIC rows; no Drive."""
from pathlib import Path

import pandas as pd

from app_core import audit_movement as am

ROOT = Path(__file__).resolve().parents[1]


def _move(**changes):
    row = dict(sport="NHL", market_family="PUCK_LINE", away_team="Utah Mammoth", home_team="Vegas Golden Knights",
               scheduled_start="2026-11-12T02:40:00+00:00", pinnacle_first_line=-1.5, pinnacle_line=-1.5, line_move=0.0,
               pinnacle_first_probability=0.33, pinnacle_probability=0.36, probability_shift=0.03,
               moved_toward="Vegas Golden Knights", captures=14, first_capture_at="2026-11-11T18:00:00+00:00",
               last_capture_at="2026-11-12T02:20:00+00:00", stake_books_at_line=["FanDuel", "Novig"],
               stake_books_off_line=[])
    return dict(row, **changes)


def _audit():
    rows = [("NHL", "Vegas", "Utah", "spread_home"), ("NHL", "Vegas", "Utah", "spread_away"),
            ("NHL", "Vegas", "Utah", "total_over"), ("NHL", "Vegas", "Utah", "total_under"),
            ("NBA", "Milwaukee", "Los Angeles", "spread_home"), ("NCAAF", "Toledo", "Umass", "spread_home"),
            ("NCAAB", "St. John's", "Colgate", "total_under")]
    return pd.DataFrame([dict(league=league, home_team=home, away_team=away, market_type=kind,
                              game_start_utc="2026-11-12T02:40:00Z", best_pick="x") for league, home, away, kind in rows])


def test_movement_is_joined_by_league_start_time_and_leading_team_words():
    moves = [_move(),
             _move(market_family="TOTAL", pinnacle_first_line=6.5, pinnacle_line=6.0, line_move=-0.5, moved_toward="Under",
                   stake_books_at_line=["Novig"], stake_books_off_line=["DraftKings"]),
             # Two Los Angeles teams at the same hour in the same league: the short name cannot tell them apart.
             _move(sport="NBA", market_family="SPREAD", home_team="Milwaukee Bucks", away_team="Los Angeles Lakers"),
             _move(sport="NBA", market_family="SPREAD", home_team="Milwaukee Bucks", away_team="Los Angeles Clippers"),
             _move(sport="NCAAB", market_family="TOTAL", home_team="St. John's Red Storm", away_team="Colgate Raiders",
                   scheduled_start="2026-11-12T02:45:00+00:00", moved_toward=None, probability_shift=0.0)]
    audit = _audit()
    before = audit.copy()
    out = am.with_movement(audit, moves)
    pd.testing.assert_frame_equal(audit, before)                      # the audit itself is never changed
    assert list(out.columns) == list(audit.columns) + list(am.COLUMNS)
    home, away, over, under, ambiguous, football, held = out.to_dict("records")
    assert home["pinnacle_current_probability"] == 0.36 and home["pinnacle_probability_shift"] == 0.03
    assert home["pinnacle_move_with_pick"] is True and away["pinnacle_move_with_pick"] is False
    assert home["pinnacle_stake_books_at_line"] == "FanDuel, Novig" and home["pinnacle_stake_books_off_line"] == ""
    assert (over["pinnacle_first_line"], over["pinnacle_current_line"], over["pinnacle_line_move"]) == (6.5, 6.0, -0.5)
    assert over["pinnacle_move_with_pick"] is False and under["pinnacle_move_with_pick"] is True
    assert under["pinnacle_stake_books_off_line"] == "DraftKings" and under["pinnacle_captures"] == 14
    assert under["pinnacle_movement_at"] == "2026-11-12T02:20:00+00:00"
    # Blank means not measured: an ambiguous game, and a sport the price watch does not capture.
    assert all(pd.isna(ambiguous[column]) for column in am.COLUMNS)
    assert all(pd.isna(football[column]) for column in am.COLUMNS)
    # Start times five minutes apart still match; no move gives no direction.
    assert held["pinnacle_captures"] == 14 and pd.isna(held["pinnacle_move_with_pick"])


def test_names_and_times_must_really_match():
    assert am._leads("New York", "New York Knicks") and am._leads("St. John's", "St. John's Red Storm")
    assert am._leads("Umass Lowell", "UMass Lowell River Hawks")
    assert not am._leads("York", "New York Knicks") and not am._leads("", "New York Knicks")
    assert not am._leads("North Carolina Wilmington", "UNC Wilmington Seahawks")     # left blank, never guessed
    audit = _audit().head(1)
    assert pd.isna(am.with_movement(audit, [_move(scheduled_start="2026-11-12T03:30:00+00:00")])
                   .loc[0, "pinnacle_captures"])                                      # 50 minutes apart
    assert pd.isna(am.with_movement(audit, [_move(home_team="Utah Mammoth", away_team="Vegas Golden Knights")])
                   .loc[0, "pinnacle_captures"])                                      # sides swapped: no match
    assert pd.isna(am.with_movement(audit, [_move(market_family="TOTAL")]).loc[0, "pinnacle_captures"])


def test_nothing_to_join_returns_the_audit_itself():
    audit = _audit()
    assert am.with_movement(audit, []) is audit and am.with_movement(audit, None) is audit
    assert am.with_movement(None, [_move()]) is None
    empty = audit.head(0)
    assert am.with_movement(empty, [_move()]) is empty
    bare = audit.drop(columns=["home_team"])
    assert am.with_movement(bare, [_move()]) is bare                 # an audit without the columns is left alone


def test_the_export_adds_the_columns_and_the_reader_never_raises(monkeypatch):
    source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert "with_movement(candidate_audit_df, movement_for_export()).to_csv(index=False, encoding=\"utf-8-sig\")" in source
    import app.ui.price_gap_candidates as panel
    monkeypatch.setattr(panel.st, "session_state", {"price_gap_candidates": {"movement": [_move()]}})
    assert panel.movement_for_export() == [_move()]                  # the list loaded in the panel is used first

    def broken():
        raise RuntimeError("drive unreachable")
    monkeypatch.setattr(panel.st, "session_state", {})
    monkeypatch.setattr(panel, "_movement_from_drive", broken)
    assert panel.movement_for_export() == []


def test_names_the_provider_shortens_or_the_audit_squashes_still_match():
    """Found 2026-11-13 on two real exports: 24 of 195 names did not match the provider's."""
    pairs = [("Jackson State", "Jackson St Tigers"), ("Sam Houston State", "Sam Houston St Bearkats"),
             ("Central Connecticut State", "Central Connecticut St Blue Devils"),
             ("Arkansaspine Bluff", "Arkansas-Pine Bluff Golden Lions"), ("Bethunecookman", "Bethune-Cookman Wildcats"),
             ("Marylandeastern Shore", "Maryland-Eastern Shore Hawks"), ("Saint Louis", "St Louis Blues"),
             ("Montreal", "Montréal Canadiens"), ("California Baptist", "Cal Baptist Lancers"),
             ("St. John's", "St. John's Red Storm"), ("New York", "New York Knicks")]
    assert [short for short, full in pairs if not am._leads(short, full)] == []
    # Still exact about which team.
    assert not am._leads("York", "New York Knicks")
    assert not am._leads("Michigan State", "Michigan Wolverines")
    assert not am._leads("North Carolina Asheville", "UNC Asheville Bulldogs")   # no alias: left blank, never guessed
    # A short name can lead two providers' names ("Michigan"): the start time and the other side decide.
    assert am._leads("Michigan", "Michigan St Spartans") and am._leads("Michigan", "Michigan Wolverines")
    assert not am._leads("Kent State", "Kentucky Wildcats") and not am._leads("", "Kent St Golden Flashes")
