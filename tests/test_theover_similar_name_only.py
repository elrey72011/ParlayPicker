"""TheOver's model pick on a game matched by name similarity alone: listed for the record, never blended (owner 2026-11-18). SYNTHETIC."""
import numpy as np
import pandas as pd
import pytest

from app_core import rating_basis as rb
from app_core import theover_model_totals as tm


def _game(orientation):
    """One game's two totals rows, the Under picked by TheOver's model at 70%."""
    data = {"league": ["NCAAB", "NCAAB"], "market_type": ["total_over", "total_under"], "matchup_id": ["g", "g"],
            "theover_pick_agrees": [0.0, 1.0], "theover_model_hit_rate": [0.7, 0.7]}
    if orientation is not None:
        data["orientation_source"] = [orientation, orientation]
    return pd.DataFrame(data)


def test_a_pick_on_a_game_the_upload_did_not_match_is_marked_for_the_record_only():
    assert (tm.MATCH_COLUMN, tm.UNMATCHED) == ("orientation_source", "default")
    for orientation in ("default", "default|novig_moneyline_favorite|standard_spread_consensus", " Default "):
        frame = _game(orientation)
        assert tm.by_similarity_only(frame).tolist() == [True, True], orientation
        assert tm.mark(frame).tolist() == [0.0, 0.0], orientation
        frame[tm.IN_BLEND_COLUMN] = tm.mark(frame)
        assert tm.backed(frame).tolist() == [False, True] and not tm.in_blend(frame).any(), orientation
        assert tm.derived(frame).isna().all(), orientation
        value, source = tm.blend_input(frame)
        assert value.isna().all() and source.tolist() == ["", ""], orientation


def test_a_game_matched_any_other_way_is_marked_as_before():
    # Exactly, through the name key, by its teams under another id, or by the candidate step's own similarity
    # match (whose line the app rejects elsewhere); and a frame that does not say how its games were matched.
    for orientation in ("exact_match", "exact_match|college_name_key", "canonical_match", "fuzzy_match", "", None):
        frame = _game(orientation)
        assert not tm.by_similarity_only(frame).any(), orientation
        assert tm.mark(frame).tolist() == [1.0, 1.0], orientation
        frame[tm.IN_BLEND_COLUMN] = tm.mark(frame)
        assert tm.derived(frame).tolist() == pytest.approx([0.3, 0.7]), orientation
        value, source = tm.blend_input(frame)
        assert value.tolist() == pytest.approx([0.3, 0.7]) and source.tolist() == [tm.SOURCE, tm.SOURCE], orientation
    unsaid = _game("exact_match")
    unsaid["orientation_source"] = [np.nan, pd.NA]
    assert tm.mark(unsaid).tolist() == [1.0, 1.0]
    # A league kept for the record stays 0 however its game was matched; a row with no model pick stays unmarked.
    assert tm.mark(_game("exact_match").assign(league="NBA")).tolist() == [0.0, 0.0]
    assert tm.mark(_game("default").assign(league="NBA")).tolist() == [0.0, 0.0]
    assert tm.mark(_game("default").assign(theover_model_hit_rate=np.nan)).isna().all()
    assert tm.mark(_game("default").assign(market_type=["spread_home", "spread_away"])).isna().all()
    assert not tm.by_similarity_only(None).any() and not tm.by_similarity_only(pd.DataFrame({"league": ["NCAAB"]})).any()
    # Only the unmatched game of a slate is taken out.
    slate = pd.concat([_game("default"), _game("exact_match").assign(matchup_id="h")], ignore_index=True)
    assert tm.mark(slate).tolist() == [0.0, 0.0, 1.0, 1.0]
    slate[tm.IN_BLEND_COLUMN] = tm.mark(slate)
    assert tm.derived(slate).tolist() == pytest.approx([np.nan, np.nan, 0.3, 0.7], nan_ok=True)


def test_the_list_says_why_a_pick_is_not_used():
    from app.ui.daily_dashboard import RECORD_ONLY, SIMILAR_NAME_ONLY, THEOVER_MODEL_CAPTION, theover_model_board
    row = {"market_type": "total_under", "theover_pick_agrees": 1.0, "theover_model_hit_rate": 0.7, "theover_pick_line": 141.5,
           "candidate_context": "CURRENT_PREGAME", "away_team": "Away", "best_pick": "Under 141.5", "odds_american": -108.0}
    candidates = pd.DataFrame([
        dict(row, league="NCAAB", home_team="Unmatched", matchup_id="a", theover_model_in_blend=0.0, orientation_source="default"),
        dict(row, league="NBA", home_team="Pro", matchup_id="b", theover_model_in_blend=0.0, orientation_source="exact_match"),
        dict(row, league="NCAAB", home_team="Matched", matchup_id="c", theover_model_in_blend=1.0,
             orientation_source="exact_match|college_name_key"),
        dict(row, league="NCAAB", home_team="Unsaid", matchup_id="d", theover_model_in_blend=0.0)])
    used = theover_model_board(candidates).set_index("Game")["Used in the estimate"].to_dict()
    assert used == {"Away at Unmatched": SIMILAR_NAME_ONLY, "Away at Pro": RECORD_ONLY, "Away at Matched": "Yes",
                    "Away at Unsaid": RECORD_ONLY}
    assert SIMILAR_NAME_ONLY == "No: matched by a similar name only" and RECORD_ONLY == "No: record only"
    older = theover_model_board(candidates.drop(columns=["orientation_source"])).set_index("Game")["Used in the estimate"].to_dict()
    assert older["Away at Unmatched"] == RECORD_ONLY and older["Away at Matched"] == "Yes"
    assert "matched only by a similar team name" in THEOVER_MODEL_CAPTION and "not certainly the one TheOver picked" in THEOVER_MODEL_CAPTION


# ---- through the production pipeline ----

def _feed_row(home, away, total=141.5):
    row = {"league": "NCAAB", "home_team": home, "away_team": away, "commence_time_raw": "2031-01-15T00:00:00Z",
           "game_date": "2031-01-14"}
    for book in ("novig", "fanduel", "draftkings", "betmgm"):
        row.update({f"{book}_home_price": -110, f"{book}_home_point": -11.5, f"{book}_away_price": -110,
                    f"{book}_away_point": 11.5, f"{book}_over_price": -112, f"{book}_over_point": total,
                    f"{book}_under_price": -108, f"{book}_under_point": total,
                    f"{book}_h2h_home_price": -650, f"{book}_h2h_away_price": 475})
    return row


def _totals(home, away):
    return pd.DataFrame({"league": ["NCAAB"], "home_team": [home], "away_team": [away], "game_date": ["2031-01-14"],
                         "line": [141.5], "pick": ["UNDER"], "pickcode": ["OU_UNDER"], "winprobability": [None],
                         "winprobsource": ["not_provided"], "modelhitrate": [0.632], "odds_american": [-108]})


def _run(monkeypatch, feed_home, uploaded_home, similarity):
    import core.streamlit_pipeline as sp
    # Two spellings of a made-up school that no key settles, scoring ``similarity`` in 100 whatever library reads them.
    monkeypatch.setattr(sp, "_team_similarity_score",
                        lambda left, right: 100 if str(left).strip().lower() == str(right).strip().lower() else similarity)
    monkeypatch.setattr(sp, "load_base_data", lambda: pd.DataFrame())
    monkeypatch.setattr(sp, "fetch_live_odds_dataframe",
                        lambda *args, **kwargs: pd.DataFrame([_feed_row(feed_home, "Mist County")]))
    analysis, _best, diagnostics = sp.run_analysis_pipeline(sports=["NCAAB"], max_rows=40, use_ml=False, spreads_df=None,
                                                             totals_df=_totals(uploaded_home, "Mist County"))
    return analysis, diagnostics["candidate_generation_diagnostics"]


def test_through_the_pipeline_a_pick_attached_by_similarity_does_not_move_the_rating(monkeypatch):
    from app.ui.daily_dashboard import SIMILAR_NAME_ONLY, theover_model_board
    analysis, counts = _run(monkeypatch, "Wobegon Lake", "Lake Wobegon", similarity=70)
    rows = analysis.set_index("market_type")
    assert len(rows) == 4 and rows["orientation_source"].str.startswith("default").all()
    assert (counts["games_matched_exact"], counts["games_matched_fuzzy"]) == (0, 0)
    under, over = rows.loc["total_under"], rows.loc["total_over"]
    # TheOver's columns are on the game, for the record ...
    assert under["theover_pick_agrees"] == 1.0 and under["theover_model_hit_rate"] == pytest.approx(0.632)
    assert over["theover_pick_agrees"] == 0.0
    # ... and the pick moves nothing: no blend input, the rating is the price, and the label says so.
    assert under["theover_model_in_blend"] == 0.0 and over["theover_model_in_blend"] == 0.0
    assert pd.isna(under["blend_in_theover"]) and pd.isna(over["blend_in_theover"])
    assert under["calibrated_probability"] == pytest.approx(under["blend_in_market"])
    assert over["calibrated_probability"] == pytest.approx(over["blend_in_market"])
    assert under["calibrated_probability"] + over["calibrated_probability"] == pytest.approx(1.0)
    assert set(rb.basis(analysis)) == {rb.PRICE_ONLY}
    board = theover_model_board(analysis)
    assert len(board) == 1 and board.loc[0, "Pick"] == "Under 141.5" and board.loc[0, "Model hit rate"] == pytest.approx(0.632)
    assert board.loc[0, "Used in the estimate"] == SIMILAR_NAME_ONLY


def test_through_the_pipeline_a_game_with_no_similar_upload_carries_nothing_from_theover(monkeypatch):
    analysis, counts = _run(monkeypatch, "Wobegon Lake", "Lake Wobegon", similarity=40)
    rows = analysis.set_index("market_type")
    assert rows["orientation_source"].str.startswith("default").all()
    assert rows["theover_model_hit_rate"].isna().all() and rows["theover_model_in_blend"].isna().all()
    assert rows.loc["total_under", "calibrated_probability"] == pytest.approx(rows.loc["total_under", "blend_in_market"])
    assert set(rb.basis(analysis)) == {rb.PRICE_ONLY}
