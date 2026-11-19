"""A pick whose estimate is the sportsbook's price alone is shown as "No lean (price only)" (owner 2026-11-19).

SYNTHETIC rows. The rule lives in ``app_core.rating_basis.withhold``; the homepage board, Pick Details,
the Full Pick Board's research card and the compact workbook apply it. It hides a pick, it does not
change one: the selection, the candidate audit and the saved evidence are untouched, and a row with an
approved stake is never withheld.
"""
from pathlib import Path

import pandas as pd

from app_core import rating_basis as rb

ROOT = Path(__file__).resolve().parents[1]


def test_withhold_replaces_the_pick_and_empties_what_describes_it():
    frame = pd.DataFrame({"pick": ["Under 145.5", "Brown +2.5", "Duke -7.5"], "odds": [-110.0, -108.0, -112.0],
                          "why": ["a", "b", "c"], "game": ["g1", "g2", "g3"]}, index=[7, 7, 9])   # an index can repeat
    labels = ["Price + TheOver model pick", "Price only", ""]
    out = rb.withhold(frame, labels, "pick", ("odds", "why", "not a column"))
    assert list(out["pick"]) == ["Under 145.5", rb.NO_LEAN, "Duke -7.5"] and rb.NO_LEAN == "No lean (price only)"
    assert out["odds"].isna().tolist() == [False, True, False] and list(out["why"]) == ["a", "", "c"]
    assert list(out["game"]) == ["g1", "g2", "g3"]                    # what names the game stays
    assert list(frame["pick"]) == ["Under 145.5", "Brown +2.5", "Duke -7.5"]      # the caller's frame is not changed
    pd.testing.assert_frame_equal(rb.withhold(frame, ["Price + Kalshi"] * 3, "pick", ("odds",)), frame)


def test_the_rule_is_display_only():
    from app_core import pinnacle_model as pm
    assert "withhold" not in pm.PREDICTION_FUNCTIONS
    pipeline = (ROOT / "core" / "streamlit_pipeline.py").read_text(encoding="utf-8")
    assert "withhold" not in pipeline and "NO_LEAN" not in pipeline       # the selection never reads it


# ---- the compact workbook ----

def test_the_compact_workbook_shows_no_pick_and_no_price_on_the_price_alone():
    import streamlit_app as app
    base = {"league": "NCAAB", "Home": "Rice", "Away": "Tulsa", "odds_american": -110, "odds_source": "odds_api",
            "Wager_Instruction": "DO NOT BET - $0 PASS / RESEARCH", "Bettable": False, "Play_Stake": 0.0, "Kelly_Bet_Size": 0.0}
    compact = app._build_compact_export_frame(pd.DataFrame([
        dict(base, best_pick="Under 145.5", rating_basis="Price + TheOver model pick"),
        dict(base, best_pick="Tulsa +3.5", rating_basis="Price only"),
        dict(base, best_pick="Rice -3.5", rating_basis="Price only", Bettable=True, Play_Stake=25.0,
             Wager_Instruction="BET - APPROVED")]))
    assert list(compact["best_pick"]) == ["Under 145.5", rb.NO_LEAN, "Rice -3.5"]     # an approved stake keeps its pick
    assert compact["Pick Price"].isna().tolist() == [False, True, False]
    assert list(compact["rating_basis"]) == ["Price + TheOver model pick", "Price only", "Price only"]
    assert list(compact["Play_Stake"]) == [0.0, 0.0, 25.0]


def test_the_full_pick_board_card_applies_the_same_rule():
    text = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
    assert 'no_edge_view = _withhold(' in text and 'no_edge_view["Built From"], "Best Pick",' in text
    assert 'or \\"No lean (price only)\\" where' in text


# ---- the homepage board and Pick Details ----

def _board_app():
    import pandas as pd
    import streamlit as st
    from app.ui.daily_dashboard import render_daily_dashboard
    today, details = st.tabs(["Today", "Details"])
    base = {"league": "NCAAB", "production_edge": 0.0, "Play_Stake": 0, "production_eligible": False, "odds_american": -110}
    board = pd.DataFrame([
        dict(base, Home="Rice", Away="Tulsa", best_pick="Under 145.5", market_type="total_under", matchup_id="g1",
             production_win_probability=0.57, production_expected_value=0.088, rating_basis="Price + TheOver model pick"),
        dict(base, Home="Yale", Away="Brown", best_pick="Brown +2.5", market_type="spread_away", matchup_id="g2",
             production_win_probability=0.51, production_expected_value=-0.026, rating_basis="Price only")])
    candidates = pd.DataFrame([
        {"matchup_id": "g1", "market_type": "total_under", "best_pick": "Under 145.5", "rating_basis": "Price + TheOver model pick"},
        {"matchup_id": "g2", "market_type": "spread_away", "best_pick": "Brown +2.5", "rating_basis": "Price only"}])
    render_daily_dashboard(today.empty(), details.empty(), board, candidates)


def test_the_homepage_board_shows_no_lean_and_nothing_about_the_pick_it_withholds():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(_board_app).run()
    assert not app.exception
    frame = next(d.value for d in app.dataframe if "Wager status" in d.value.columns)
    assert list(frame["Best pick"]) == ["Under 145.5", rb.NO_LEAN]
    assert list(frame["Built from"]) == ["Price + TheOver model pick", "Price only"]
    assert list(frame["Wager status"]) == ["PASS", "PASS"] and list(frame["Game"]) == ["Tulsa at Rice", "Brown at Yale"]
    assert frame["Odds"].isna().tolist() == [False, True]
    assert frame.loc[1, "Win estimate"] == "" and frame.loc[1, "EV estimate"] == "" and frame.loc[0, "Win estimate"] != ""
    assert any(rb.NO_LEAN_CAPTION in c.value for c in app.caption)
    details = next(d.value for d in app.dataframe if "approval_reason" in d.value.columns)
    assert list(details["pick"]) == ["Under 145.5", rb.NO_LEAN]
    assert "Brown +2.5" not in " ".join(str(d.value.to_dict()) for d in app.dataframe)      # on no table of the page


def test_pick_details_names_no_pick_for_such_a_game():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(_board_app).run()
    choice = next(s for s in app.selectbox if s.label == "Game selection")
    assert choice.options == ["NCAAB · Tulsa at Rice · Under 145.5", "NCAAB · Brown at Yale · " + rb.NO_LEAN]
    assert not any(rb.NO_LEAN in i.value for i in app.info) and len(app.table) == 1      # the first game shows its selection
    app = choice.set_value(1).run()
    assert not app.exception
    assert any(i.value.startswith(rb.NO_LEAN + ".") and rb.NO_LEAN_CAPTION in i.value for i in app.info)
    assert len(app.table) == 0 and "Brown +2.5" not in " ".join(str(m.value) for m in list(app.markdown) + list(app.caption))


def _approved_board_app():
    import pandas as pd
    import app_core.per_game_boards as boards
    result = pd.DataFrame([
        {"league": "NCAAB", "matchup": "Brown at Yale", "start": "", "pick": "Brown +2.5", "selection_label": "Best Overall",
         "status": "APPROVED", "approval_reason": "", "odds": -108.0, "win_probability": 0.51, "edge": 0.0, "ev": 0.0,
         "selection_score": 0.5, "probability_basis": "", "matchup_id": "g2", "market_type": "spread_away"}])
    boards.per_game_board = lambda board, candidates, family: result
    from app.ui.daily_dashboard import _render_game_board
    candidates = pd.DataFrame([{"matchup_id": "g2", "market_type": "spread_away", "best_pick": "Brown +2.5", "rating_basis": "Price only"}])
    _render_game_board(pd.DataFrame([{"x": 1}]), candidates, "overall")


def test_an_approved_wager_is_never_withheld_on_the_board():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(_approved_board_app).run()
    assert not app.exception
    frame = next(d.value for d in app.dataframe if "Wager status" in d.value.columns)
    assert list(frame["Best pick"]) == ["Brown +2.5"] and list(frame["Built from"]) == ["Price only"]
    assert list(frame["Odds"]) == [-108.0]
