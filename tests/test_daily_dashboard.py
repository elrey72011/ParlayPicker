import pandas as pd
from streamlit.testing.v1 import AppTest
from app.ui.daily_dashboard import daily_board


def test_daily_board_reconciles_stale_stakes_without_mutating_source():
    source = pd.DataFrame([
        {"best_pick": "Research", "Play_Stake": 12, "production_eligible": False, "Bet_Decision": "BET"},
        {"best_pick": "Approved", "Play_Stake": 5, "production_eligible": True, "Bet_Decision": "BET"},
        {"best_pick": "Unfunded", "Play_Stake": 0, "production_eligible": True, "Bet_Decision": "BET"},
    ])
    board = daily_board(source)
    assert board.Bettable.tolist() == [False, True, False]
    assert board.Play_Stake.tolist() == [0, 5, 0]
    assert source.Play_Stake.tolist() == [12, 5, 0]


def _app():
    import pandas as pd
    import streamlit as st
    from app.ui.daily_dashboard import render_daily_dashboard
    today, details = st.tabs(["Today", "Pick Details"])
    render_daily_dashboard(today.empty(), details.empty(), pd.DataFrame([
        {"league": "MLB", "Home": "Home", "Away": "Away", "best_pick": "Over 8", "Play_Stake": 0, "production_eligible": False},
    ]))


def test_pass_only_run_has_no_approved_download_and_explains_decision():
    app = AppTest.from_function(_app).run()
    assert not app.exception
    assert [m.value for m in app.metric] == ["1", "0", "1"]
    assert all(b.proto.label != "Download approved game wagers" for b in app.get("download_button"))
    assert any("PASS" in m.value for m in app.markdown)
    app.selectbox[0].set_value(0).run()
    assert not app.exception
