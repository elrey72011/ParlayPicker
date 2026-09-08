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


def _metric_value(row, column, *, points=False):
    import math
    value = pd.to_numeric(row.get(column), errors="coerce")
    if pd.isna(value) or not math.isfinite(value):
        return "Not available"
    return f"{value * 100:+.1f} pp" if points else f"{value:.1%}"


def _render_featured_card(board, family):
    from app_core.featured_picks import featured_picks
    ranked = featured_picks(board, family)
    if ranked.empty:
        st.info("No selection with a final win estimate and usable price is available in this category.")
        return
    row = ranked.iloc[0]
    approved = bool(row["_featured_approved"])
    with st.container(border=True):
        st.caption(_text(row, "league") + " · " + _text(row, "Commence (Local)", "Local Date"))
        st.markdown("### " + _text(row, "display_pick", "best_pick"))
        st.write(_text(row, "Away") + " at " + _text(row, "Home"))
        st.markdown("**" + ("APPROVED WAGER" if approved else "PASS · Best available research selection") + "**")
        odds = float(row["odds_american"])
        st.write(f"Price: {odds:+.0f} · " + _text(row, "odds_source"))
        a, b, c = st.columns(3)
        a.metric("Estimated win %", _metric_value(row, "production_win_probability"))
        b.metric("Edge vs. break-even", _metric_value(row, "production_edge", points=True))
        c.metric("Estimated EV", _metric_value(row, "production_expected_value"))
        st.caption("Win % leads this ranking. Edge is the probability advantage over the price's break-even point; EV is the estimated return per dollar.")
        if approved:
            st.write("Why this pick: highest final estimated win probability among approved selections in this view; edge and EV break ties.")
            st.caption("Approved stake: $" + _text(row, "Play_Stake", default="0"))
        else:
            st.write("Why it leads this view: highest final estimated win probability among the available research selections. It has not cleared the wager checks.")
            st.caption(_text(row, "Production_Gate_Reason", "Status_Reason", "qualification_reason"))
        st.caption("Quote captured: " + _text(row, "odds_recorded_at"))
    st.caption("Estimated metrics explain the selection; they are not proof of a win or an achieved hit rate. Approval and price reflect the saved run.")


def render_daily_dashboard(today, details, frame: pd.DataFrame) -> None:
    board = daily_board(frame)
    with today.container():
        st.subheader("Best picks")
        st.caption("Saved analysis · Ranked from the finalized game card. Game markets only. Player props and parlays remain in Workspace → Full Pick Board and Parlays.")
        if board.empty:
            st.info("Start with your sport and bankroll, then select Run Master Analysis. Add optional files under Settings & research first.")
        else:
            approved = board.loc[board["Bettable"]]
            a, b, c = st.columns(3)
            a.metric("Games reviewed", len(board))
            b.metric("Approved game wagers", len(approved))
            c.metric("Passes", len(board) - len(approved))
            overall, sides, totals = st.tabs(["Overall Best Pick", "Sides", "Totals"], key="daily_market_navigation", on_change="rerun")
            with overall:
                _render_featured_card(board, "overall")
            with sides:
                st.caption("Moneylines and spreads from the finalized game card.")
                _render_featured_card(board, "sides")
            with totals:
                st.caption("Overs and unders from the finalized game card.")
                _render_featured_card(board, "totals")
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
                  "expected_value": "Expected return per dollar", "edge": "Estimated edge"}
        fields = [c for c in field_labels if c in board]
        st.table(pd.DataFrame({"Detail": [field_labels[c] for c in fields], "Value": [_text(row, c) for c in fields]}).set_index("Detail"))
        st.caption("Probabilities are model estimates, not observed win rates. Full candidate audits and research exports are in Workspace.")
