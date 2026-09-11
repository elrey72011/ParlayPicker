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


def _render_game_board(board, candidates, family):
    from app_core.per_game_boards import per_game_board
    result = per_game_board(board, candidates, family)
    if result.empty:
        st.info("Run an analysis to populate the game board.")
        return
    display = result.rename(columns={"league":"Sport", "matchup":"Game", "start":"Start", "pick":"Best pick", "selection_label":"Selection", "status":"Wager status", "approval_reason":"Wager explanation", "odds":"Odds", "win_probability":"Win estimate", "edge":"Edge", "ev":"EV estimate", "selection_score":"Selection score", "probability_basis":"Win estimate source"})
    columns = ["Game", "Best pick", "Odds", "Win estimate", "EV estimate", "Wager status"]
    display = display[columns].copy()
    for column in ("Win estimate", "EV estimate"):
        display[column] = display[column].map(lambda v: f"{v:.1%}" if pd.notna(v) else "Unavailable")
    st.dataframe(display, hide_index=True, width="stretch", height=min(760, 38 * (len(display) + 1)), column_config={"Selection score": st.column_config.NumberColumn(format="%.3f"), "Odds": st.column_config.NumberColumn(format="%+.0f")})
    with st.expander("Selection details and wager explanations", expanded=False):
        st.dataframe(result[["matchup", "pick", "approval_reason", "edge", "selection_score", "probability_basis", "start"]], hide_index=True, width="stretch")
        st.caption("Best Overall, Best Side, and Best Total identify the selected pick for each game. Wager status is separate: PASS means no approved wager. Ranking uses the composite selection score first, followed by probability, tier, EV, and edge as tie-breakers; the highest EV does not necessarily rank first. Selection score is not a win probability. Edge compares the displayed estimate with price break-even. Alternate candidates use their calibrated estimates; only exact final tickets carry final production metrics and approval.")
    st.download_button("Download this game board", result.to_csv(index=False), family+"-per-game.csv", "text/csv", key="per_game_export_"+family)


def render_daily_dashboard(today, details, frame: pd.DataFrame, candidates: pd.DataFrame | None = None) -> None:
    board = daily_board(frame)
    with today.container():
        st.subheader("Best picks")
        st.caption("Saved analysis · One selection per game in each view. Game markets only. Player props and parlays remain in Workspace → Full Pick Board and Parlays.")
        if board.empty:
            st.info("Start with your sport and bankroll, then select Run Game Analysis. Add optional files under Settings & research first.")
        else:
            approved = board.loc[board["Bettable"]]
            a, b, c = st.columns(3)
            a.metric("Games reviewed", len(board))
            b.metric("Approved game wagers", len(approved))
            c.metric("Games without an approved wager", len(board) - len(approved))
            overall, sides, totals = st.tabs(["Overall Best Pick", "Sides", "Totals"], key="daily_market_navigation", on_change="rerun")
            with overall:
                _render_game_board(board, candidates, "overall")
            with sides:
                st.caption("Best moneyline or spread for each game, selected from the full ranked candidate audit.")
                _render_game_board(board, candidates, "sides")
            with totals:
                st.caption("Best over or under for each game, selected from the full ranked candidate audit.")
                _render_game_board(board, candidates, "totals")
            if not approved.empty:
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
                  "production_expected_value": "Production expected return per dollar", "production_edge": "Production edge"}
        fields = [c for c in field_labels if c in board]
        st.table(pd.DataFrame({"Detail": [field_labels[c] for c in fields], "Value": [_text(row, c) for c in fields]}).set_index("Detail"))
        with st.expander("Gemini review", expanded=False):
            st.write("Agreement: " + _text(row, "gemini_agreement", default="Not reviewed"))
            st.write(_text(row, "gemini_explanation", default="No saved review for this selection."))
            st.write("Missing information: " + _text(row, "gemini_missing_information"))
            st.caption("Reviewed: " + _text(row, "gemini_reviewed_at") + " · Model: " + _text(row, "gemini_review_model"))
            st.caption("Gemini agreement is qualitative review, not a win probability or wager approval.")
        st.caption("Probabilities are model estimates, not observed win rates. Full candidate audits and research exports are in Workspace.")
