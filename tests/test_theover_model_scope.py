"""Which leagues' TheOver model pick may feed the blend (owner 2026-11-17).

Over ten graded nights (7 to 16 November 2026) TheOver's model-backed totals pick went 81-45 in
college basketball and 10-15 in the NBA. The NBA's picks are now kept for the record only: the upload
reader marks each pick's row, the production pipeline marks it again once the league is settled, and
``theover_model_totals.derived`` leaves a marked-off game out of the blend. A frame with no mark is
treated as before (``tests/test_theover_model_totals.py`` covers that and is unchanged).
"""
from __future__ import annotations

import pandas as pd
import pytest

from app_core import rating_basis as rb
from app_core import theover_model_totals as tmt
from app_core.weights_config import FALLBACK_MARKET_WEIGHT, FALLBACK_THEOVER_WEIGHT

MARK = tmt.IN_BLEND_COLUMN


def _rows(*rows):
    return pd.DataFrame([{"matchup_id": game, "league": league, "market_type": kind, "market_probability": market,
                          "theover_pick_agrees": picked, "theover_model_hit_rate": hit}
                         for game, league, kind, market, picked, hit in rows])


SLATE = (("n", "NBA", "total_under", 0.488, 1.0, 0.875), ("n", "NBA", "total_over", 0.512, 0.0, 0.875),
         ("c", "NCAAB", "total_under", 0.495, 1.0, 0.73), ("c", "NCAAB", "total_over", 0.505, 0.0, 0.73),
         ("h", "NHL", "total_over", 0.50, 1.0, 0.62), ("h", "NHL", "total_under", 0.50, 0.0, 0.62),
         ("c", "NCAAB", "spread_home", 0.52, None, None),                      # a side: never marked
         ("x", "NBA", "total_over", 0.50, 1.0, None))                          # a pick with no hit rate listed


def test_the_nba_is_the_one_league_kept_for_the_record():
    assert tmt.RESEARCH_ONLY_LEAGUES == frozenset({"NBA"}) and MARK == "theover_model_in_blend"
    leagues = pd.Series(["NBA", " nba ", "NCAAB", "NHL", "NFL", "NCAAF", "MLB", "WNBA", None, ""])
    assert list(tmt.in_blend_mark(leagues)) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    assert list(tmt.in_blend_mark(None, index=[4, 9])) == [1.0, 1.0]            # no league column: nothing is held back


def test_the_mark_goes_on_both_sides_of_a_model_backed_total_and_nowhere_else():
    frame = _rows(*SLATE)
    marks = tmt.mark(frame)
    assert list(marks.iloc[:6]) == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert marks.iloc[6:].isna().all()
    assert tmt.mark(frame.drop(columns=["theover_model_hit_rate"])).isna().all()
    assert tmt.mark(frame.drop(columns=["league"])).iloc[:6].eq(1.0).all()      # no league: marked usable
    assert tmt.mark(frame.iloc[0:0]).empty


def test_a_marked_off_game_is_left_out_of_the_blend_and_the_rest_is_untouched():
    frame = _rows(*SLATE)
    unmarked, _ = tmt.blend_input(frame)                                         # as an analysis saved before the change
    assert list(unmarked.iloc[:6]) == pytest.approx([0.70, 0.30, 0.70, 0.30, 0.62, 0.38])
    frame[MARK] = tmt.mark(frame)
    values, source = tmt.blend_input(frame)
    assert values.iloc[:2].isna().all() and list(source.iloc[:2]) == ["", ""]    # the NBA game: nothing from TheOver
    assert list(values.iloc[2:6]) == pytest.approx([0.70, 0.30, 0.62, 0.38])
    assert list(source.iloc[2:6]) == [tmt.SOURCE] * 4
    assert list(tmt.in_blend(frame).iloc[:6]) == [False, False, True, True, True, True]
    assert list(tmt.backed(frame).iloc[:2]) == [True, False]                     # still TheOver's pick: it stays on the list


def test_an_uploaded_probability_for_an_nba_total_is_still_kept():
    frame = _rows(*SLATE[:2])
    frame[MARK] = tmt.mark(frame)
    uploaded = pd.Series([0.62, 0.38], index=frame.index)
    values, source = tmt.blend_input(frame, uploaded)                            # the mark is about the model pick only
    assert list(values) == pytest.approx([0.62, 0.38]) and list(source) == [tmt.UPLOAD] * 2


def test_with_the_mark_an_nba_total_is_the_price_alone_and_says_so():
    from core.streamlit_pipeline import compute_blended_probability
    frame = _rows(*SLATE[:4])
    frame[MARK] = tmt.mark(frame)
    theover, source = tmt.blend_input(frame)
    empty = pd.Series(float("nan"), index=frame.index)
    blended = compute_blended_probability(p_market=frame["market_probability"], p_kalshi=empty, p_ml=empty, p_theover=theover,
                                          p_sentiment=pd.Series(0.5, index=frame.index), league=frame["league"],
                                          market_type=frame["market_type"], theover_source=source)
    assert list(blended.iloc[:2]) == pytest.approx([0.488, 0.512])               # the NBA total: the book's price
    weight = FALLBACK_MARKET_WEIGHT + FALLBACK_THEOVER_WEIGHT
    assert blended.iloc[2] == pytest.approx((FALLBACK_MARKET_WEIGHT * 0.495 + FALLBACK_THEOVER_WEIGHT * 0.70) / weight)
    recorded = frame.assign(blend_in_market=frame["market_probability"], blend_in_kalshi=float("nan"),
                            blend_in_ml=float("nan"), blend_in_theover=theover, **{tmt.SOURCE_COLUMN: source})
    assert list(rb.basis(recorded)) == ["Price only", "Price only", "Price + TheOver model pick", "Price + TheOver model pick"]


def test_emptying_the_set_blends_the_nba_again(monkeypatch):
    monkeypatch.setattr(tmt, "RESEARCH_ONLY_LEAGUES", frozenset())
    frame = _rows(*SLATE[:2])
    frame[MARK] = tmt.mark(frame)
    assert list(frame[MARK]) == [1.0, 1.0]
    assert list(tmt.blend_input(frame)[0]) == pytest.approx([0.70, 0.30])


# ---- where the mark is stamped, and that it travels ----

TOTALS = pd.DataFrame({
    "League": ["NBA", "NBA", "NCAAB", "NCAAB"], "GameTime": ["7:00 PM"] * 4,
    "HomeTeam": ["Charlotte", "Atlanta", "Kennesaw State", "Rice"], "AwayTeam": ["Boston", "Washington", "Austin Peay", "Tulsa"],
    "HomeKalshi": ["CHARLOTTE", "ATLANTA", "KENNESAW STATE", "RICE"], "AwayKalshi": ["BOSTON", "WASHINGTON", "AUSTIN PEAY", "TULSA"],
    "Pick": ["OVER", "UNDER", "OVER", "UNDER"], "PickCode": ["OU_OVER", "OU_UNDER", "OU_OVER", "OU_UNDER"],
    "Line": [207.5, 228.5, 157.5, 145.5], "WinProbability": [None] * 4, "PublicBettingPct": [0.87, None, 0.66, None],
    "BettingDirection": ["OVER", None, "UNDER", None], "ModelHitRate": [None, 0.714, None, 0.73],
    "ModelText": [None, "Backed by Pulsar", None, "Backed by Nova"], "WinProbSource": ["not_provided"] * 4,
    "Market": ["Total"] * 4,
    "Matchup": ["Boston @ Charlotte", "Washington @ Atlanta", "Austin Peay @ Kennesaw State", "Tulsa @ Rice"]})


def test_the_upload_reader_marks_each_model_backed_pick_by_its_league():
    from core.streamlit_pipeline import build_theover_bet_rows
    rows = build_theover_bet_rows(None, TOTALS, ["NBA", "NCAAB"])
    by_game = rows.groupby("home_team")[MARK].apply(list).to_dict()
    assert by_game["Atlanta"] == [0.0, 0.0]                                      # NBA, model hit rate listed: record only
    assert by_game["Rice"] == [1.0, 1.0]                                         # college: may feed the blend
    assert pd.isna(by_game["Charlotte"]).all() and pd.isna(by_game["Kennesaw State"]).all()   # no hit rate: no mark
    atlanta = rows[rows.home_team.eq("Atlanta") & rows.market_type.eq("total_under")].iloc[0]
    assert atlanta.theover_pick_agrees == 1.0 and abs(atlanta.theover_model_hit_rate - 0.714) < 1e-9   # kept for the list


def test_the_mark_travels_with_the_hit_rate_through_every_column_list():
    from core.streamlit_pipeline import (BEST_PICK_COLUMNS, CANONICAL_BET_COLUMNS, REQUIRED_BEST_PICK_EXPORT_COLUMNS,
                                         THEOVER_RESEARCH_COLUMNS)
    for columns in (THEOVER_RESEARCH_COLUMNS, CANONICAL_BET_COLUMNS, BEST_PICK_COLUMNS, REQUIRED_BEST_PICK_EXPORT_COLUMNS):
        assert MARK in columns and "theover_model_hit_rate" in columns
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "core" / "streamlit_pipeline.py").read_text(encoding="utf-8")
    # The candidate audit lists the mark beside the hit rate.
    assert '"theover_model_hit_rate",\n        "theover_model_in_blend",' in text


def test_the_production_pipeline_marks_again_just_before_its_blend():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "core" / "streamlit_pipeline.py").read_text(encoding="utf-8")
    stamp = "merged[_theover_model_totals.IN_BLEND_COLUMN] = _theover_model_totals.mark(merged)"
    assert text.count(stamp) == 1
    assert text.index(stamp) < text.index("_theover_model_totals.blend_input(")  # the league is settled, then the blend reads it


def _game(league, marked):
    frame = pd.DataFrame([dict(matchup_id="g", league=league, home_team="Boston", away_team="Orlando", market_type=kind,
                               market_probability=0.5, odds_american=-110, opposing_odds_american=-110, total_line=212.5,
                               spread_line=None, theover_pick_agrees=picked, theover_model_hit_rate=0.875,
                               theover_probability=None, ml_probability=None, kalshi_probability=None,
                               win_prob_source="not_provided")
                          for kind, picked in (("total_under", 1.0), ("total_over", 0.0))])
    if marked:
        frame[MARK] = tmt.mark(frame)
    return frame


def test_the_re_blend_and_the_analysis_blend_leave_a_marked_nba_total_at_the_price():
    import streamlit_app as app
    from core.streamlit_pipeline import _apply_analysis_calculations
    blended = (FALLBACK_MARKET_WEIGHT * 0.5 + FALLBACK_THEOVER_WEIGHT * 0.70) / (FALLBACK_MARKET_WEIGHT + FALLBACK_THEOVER_WEIGHT)

    again = app._recompute_consensus_from_kalshi(_game("NBA", marked=True)).set_index("market_type")
    assert again.loc["total_under", "calibrated_probability"] == pytest.approx(0.5)
    assert again.loc["total_over", "calibrated_probability"] == pytest.approx(0.5)
    assert pd.isna(again.loc["total_under", "blend_in_theover"]) and again.loc["total_under", "blend_in_theover_source"] == ""
    assert again.loc["total_under", MARK] == 0.0                                 # the mark is still on the row afterwards

    nba = _apply_analysis_calculations(_game("NBA", marked=True)).set_index("market_type")
    assert nba.loc["total_under", "calibrated_probability"] == pytest.approx(0.5)

    # A college game with the mark is blended exactly as before.
    college = app._recompute_consensus_from_kalshi(_game("NCAAB", marked=True)).set_index("market_type")
    assert college.loc["total_under", "calibrated_probability"] == pytest.approx(blended)
    assert college.loc["total_under", "blend_in_theover_source"] == tmt.SOURCE and college.loc["total_under", MARK] == 1.0
    shown = _apply_analysis_calculations(_game("NCAAB", marked=True)).set_index("market_type")
    assert shown.loc["total_under", "calibrated_probability"] == pytest.approx(blended)


# ---- on the homepage list ----

def test_the_list_keeps_nba_picks_and_says_they_are_for_the_record():
    from app.ui.daily_dashboard import RECORD_ONLY, THEOVER_MODEL_CAPTION, THEOVER_MODEL_COLUMNS, theover_model_board
    base = {"theover_pick_agrees": 1.0, "theover_pick_line": 212.5, "odds_american": -110, "candidate_context": "CURRENT_PREGAME",
            "pregame_quote_valid": True, "best_available_selected": False, "game_start_utc": "2026-11-18T00:00:00Z"}
    candidates = pd.DataFrame([
        dict(base, league="NBA", away_team="Orlando", home_team="Boston", matchup_id="n", market_type="total_under",
             best_pick="Under 212.5", theover_model_hit_rate=0.875, theover_model_in_blend=0.0),
        dict(base, league="NCAAB", away_team="Tulsa", home_team="Rice", matchup_id="c", market_type="total_under",
             best_pick="Under 145.5", theover_model_hit_rate=0.73, theover_model_in_blend=1.0, best_available_selected=True),
        dict(base, league="NHL", away_team="Ottawa", home_team="Boston", matchup_id="h", market_type="total_over",
             best_pick="Over 5.5", theover_model_hit_rate=0.62, theover_model_in_blend=float("nan"))])
    board = theover_model_board(candidates).set_index("Sport")
    assert list(board.columns) == [c for c in THEOVER_MODEL_COLUMNS if c != "Sport"]
    assert "Used in the estimate" in THEOVER_MODEL_COLUMNS and RECORD_ONLY == "No: record only"
    assert board.loc["NBA", "Used in the estimate"] == RECORD_ONLY and board.loc["NBA", "Pick"] == "Under 212.5"
    assert board.loc["NCAAB", "Used in the estimate"] == "Yes" and board.loc["NCAAB", "Also the app's pick"] == "Yes"
    assert board.loc["NHL", "Used in the estimate"] == "Yes"                     # no mark: it was used
    older = theover_model_board(candidates.drop(columns=["theover_model_in_blend"]))
    assert list(older["Used in the estimate"]) == ["Yes", "Yes", "Yes"]         # an analysis saved before the change
    assert "its NBA picks are listed for the record only" in THEOVER_MODEL_CAPTION
    assert "10-15" in THEOVER_MODEL_CAPTION and "81-45" in THEOVER_MODEL_CAPTION


# ---- end to end, through the production pipeline ----

def test_through_the_whole_pipeline_the_nba_total_stays_at_the_price_and_the_college_total_is_blended(monkeypatch):
    import core.streamlit_pipeline as sp
    games = {"league": ["NBA", "NCAAB"], "home_team": ["Atlanta Hawks", "Rice"], "away_team": ["Washington Wizards", "Tulsa"]}
    base = pd.DataFrame({**games, "game_date": ["2026-11-18T00:00:00Z"] * 2, "odds_american": [-110, -110]})
    totals = pd.DataFrame({**games, "line": [228.5, 145.5], "pick": ["UNDER", "UNDER"], "pickcode": ["OU_UNDER", "OU_UNDER"],
                           "winprobability": [None, None], "winprobsource": ["not_provided"] * 2,
                           "modelhitrate": [0.875, 0.73], "odds_american": [-110, -110]})
    monkeypatch.setattr(sp, "load_base_data", lambda: base)
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe", lambda _sports: pd.DataFrame())   # the upload becomes the slate
    analysis, _best, _diagnostics = sp.run_analysis_pipeline(sports=["NBA", "NCAAB"], max_rows=40, use_ml=False,
                                                             spreads_df=None, totals_df=totals)
    rows = analysis[analysis["market_type"].astype(str).str.startswith("total")].set_index(["league", "market_type"])
    assert len(rows) == 4

    nba_under, nba_over = rows.loc[("NBA", "total_under")], rows.loc[("NBA", "total_over")]
    assert nba_under[MARK] == 0.0 and nba_over[MARK] == 0.0
    assert pd.isna(nba_under["blend_in_theover"]) and pd.isna(nba_over["blend_in_theover"])
    assert nba_under[tmt.SOURCE_COLUMN] == "" and nba_over[tmt.SOURCE_COLUMN] == ""
    assert nba_under["calibrated_probability"] == pytest.approx(nba_under["market_probability"])      # the price, nothing else
    assert nba_under["theover_pick_agrees"] == 1.0 and nba_under["theover_model_hit_rate"] == pytest.approx(0.875)   # still recorded

    college_under, college_over = rows.loc[("NCAAB", "total_under")], rows.loc[("NCAAB", "total_over")]
    assert college_under[MARK] == 1.0 and college_under[tmt.SOURCE_COLUMN] == tmt.SOURCE
    assert college_under["blend_in_theover"] == pytest.approx(0.70) and college_over["blend_in_theover"] == pytest.approx(0.30)
    weight = FALLBACK_MARKET_WEIGHT + FALLBACK_THEOVER_WEIGHT
    market = college_under["market_probability"]
    assert college_under["calibrated_probability"] == pytest.approx(
        (FALLBACK_MARKET_WEIGHT * market + FALLBACK_THEOVER_WEIGHT * 0.70) / weight)
