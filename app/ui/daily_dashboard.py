"""Daily workflow rendered only from the reconciled game export."""
from __future__ import annotations

import pandas as pd
import streamlit as st
from app_core.export_scope import label_wager_export


def _text(row, *columns, default="Not available"):
    for column in columns:
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value)
    return default


def daily_board(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconcile authorization before presenting any daily decision."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    return label_wager_export(frame).reset_index(drop=True)


def render_daily_dashboard(today, details, frame: pd.DataFrame) -> None:
    board = daily_board(frame)
    with today.container():
        st.subheader("Your daily game card")
        st.caption("Saved analysis · Game markets only. Player props and parlays remain in Workspace → Full Pick Board and Parlays.")
        if board.empty:
            st.info("Start with your sport and bankroll, then select Run Master Analysis. Add optional files under Settings & research first.")
        else:
            approved = board.loc[board["Bettable"]]
            a, b, c = st.columns(3)
            a.metric("Games reviewed", len(board))
            b.metric("Approved game wagers", len(approved))
            c.metric("Passes", len(board) - len(approved))
            if approved.empty:
                st.info("No game wagers cleared the final checks in this run. Open Pick Details to see each selection and its pass reason.")
            else:
                st.caption("Approval reflects this saved run. Check the current line and price before acting.")
                for _, row in approved.iterrows():
                    with st.container(border=True):
                        st.caption(_text(row, "league") + " · " + _text(row, "Commence (Local)", "Local Date"))
                        st.markdown("**" + _text(row, "Away") + " at " + _text(row, "Home") + "**")
                        st.write(_text(row, "display_pick", "best_pick"))
                        st.write("Odds: " + _text(row, "odds_american") + " · Approved stake: $" + _text(row, "Play_Stake", default="0"))
                st.download_button("Download approved game wagers", approved.to_csv(index=False),
                                   "approved-game-wagers.csv", "text/csv", key="daily_approved_export")
            st.caption("Use Results after games finish. Readiness for grading does not establish a profitable edge or an achieved win rate.")
    with details.container():
        st.subheader("Understand a selection")
        if board.empty:
            st.info("Run an analysis to inspect game selections and their final decisions.")
            return
        labels = [f"{_text(row, 'league')} · {_text(row, 'Away')} at {_text(row, 'Home')} · {_text(row, 'display_pick', 'best_pick')}" for _, row in board.iterrows()]
        if st.session_state.get("daily_game_selection", 0) not in range(len(board)):
            st.session_state["daily_game_selection"] = 0
        selected = st.selectbox("Game selection", range(len(board)), format_func=lambda i: labels[i], key="daily_game_selection")
        row = board.iloc[selected]
        with st.container(border=True):
            st.markdown("**" + ("Approved wager" if row["Bettable"] else "PASS · No approved wager") + "**")
            st.write(_text(row, "Wager_Instruction"))
            st.write(_text(row, "Production_Gate_Reason", "Status_Reason", "qualification_reason"))
            st.caption("Scheduled: " + _text(row, "Commence (Local)", "Local Date"))
            st.caption("Quote captured: " + _text(row, "odds_recorded_at"))
        field_labels = {"best_pick": "Selection", "odds_american": "American odds", "odds_source": "Quote source",
                  "Play_Stake": "Approved stake ($)", "production_win_probability": "Production win estimate",
                  "expected_value": "Expected return per dollar", "edge": "Estimated edge"}
        fields = [c for c in field_labels if c in board]
        st.table(pd.DataFrame({"Detail": [field_labels[c] for c in fields], "Value": [_text(row, c) for c in fields]}).set_index("Detail"))
        st.caption("Probabilities are model estimates, not observed win rates. Full candidate audits and research exports are in Workspace.")
