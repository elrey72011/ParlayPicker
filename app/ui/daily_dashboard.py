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


STALE_QUOTE_MINUTES = 30   # the market gate's own limit (core.sport_market_gate)


def daily_board(frame: pd.DataFrame, now=None) -> pd.DataFrame:
    """Reconcile authorization before presenting any daily decision.

    The market gate checks quote freshness only when picks are refreshed, so an approved stake
    stayed on screen after its quote aged out (audit 2026-11-09). An approved row whose quote is
    older than ``STALE_QUOTE_MINUTES`` now shows as a PASS with no stake until picks are refreshed.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()
    board = label_wager_export(frame).reset_index(drop=True)
    if "Bettable" not in board:
        return board
    from datetime import datetime, timedelta, timezone
    now = now or datetime.now(timezone.utc)
    column = next((c for c in ("odds_recorded_at", "quote_timestamp", "quote_time") if c in board), None)
    if column is None:
        return board
    stamps = pd.to_datetime(board[column], utc=True, errors="coerce")
    stale = board["Bettable"].astype(bool) & stamps.notna() & (stamps < now - timedelta(minutes=STALE_QUOTE_MINUTES))
    if stale.any():
        board.loc[stale, "Bettable"] = False
        if "Play_Stake" in board:
            board.loc[stale, "Play_Stake"] = 0
        board.loc[stale, "Production_Gate_Reason"] = (
            f"Quote older than {STALE_QUOTE_MINUTES} minutes: refresh picks before placing this wager")
    return board


def wager_rejection_summary(board):
    """Count one primary saved blocker per unapproved game."""
    from app_core.recommendation_quality import quality_reason
    reasons = []
    for _, row in board.iterrows():
        if row.get('Bettable') is True or str(row.get('Bettable')).lower() == 'true':
            continue
        reason = quality_reason(row) or _text(
            row, 'Production_Gate_Reason', 'status_blocker_reason', 'qualification_reason',
            default='No funded wager remains after final checks')
        reasons.append({'League': _text(row, 'league', 'League'), 'Reason': reason})
    if not reasons:
        return pd.DataFrame(columns=['League', 'Reason', 'Games'])
    return pd.DataFrame(reasons).groupby(['League', 'Reason'], sort=False).size().reset_index(name='Games')


def _candidate_values(result, candidates, columns):
    """Per board row, the first of ``columns`` (in order) from its saved candidate, or None.

    Matched by ``candidate_id``; board rows built from the final export carry none, so it
    falls back to game + market + pick label when that names exactly one candidate.
    """
    if candidates is None or getattr(candidates, "empty", True):
        return None
    values = None
    for column in columns:
        if column in candidates:
            numbers = pd.to_numeric(candidates[column], errors="coerce")
            values = numbers if values is None else values.fillna(numbers)
    if values is None:
        return None
    probs = pd.Series(float("nan"), index=result.index)
    if "candidate_id" in result and "candidate_id" in candidates:
        lookup = dict(zip(candidates["candidate_id"].astype(str), values))
        probs = result["candidate_id"].astype(str).map(lookup).astype("float64")
    # Board rows built from the final export carry no candidate_id (2026-11-06: every
    # Overall row read "Unavailable"), so fall back to the game, market and pick label,
    # used only when it names exactly one candidate.
    keys = ("matchup_id", "market_type")
    if all(k in result for k in keys + ("pick",)) and all(k in candidates for k in keys + ("best_pick",)):
        def key(frame, pick):
            return (frame["matchup_id"].astype(str) + "|" + frame["market_type"].astype(str) + "|"
                    + frame[pick].astype(str))
        candidate_keys = key(candidates, "best_pick")
        unique = ~candidate_keys.duplicated(keep=False)
        by_key = dict(zip(candidate_keys[unique], values[unique]))
        probs = probs.fillna(key(result, "pick").map(by_key).astype("float64"))
    return probs.where((probs > 0) & (probs < 1))


def _ev(result, probs):
    odds = pd.to_numeric(result.get("odds"), errors="coerce")
    decimal = odds.map(lambda o: (1 + o / 100 if o > 0 else 1 + 100 / abs(o)) if pd.notna(o) and abs(o) >= 100 else float("nan"))
    return probs * decimal - 1


def gemini_research_labels(result, board):
    """'Agrees · HIGH' / 'Disagrees (Under 152.5) · LOW' per board row from the Gemini research
    review saved on the final picks (``app_core.gemini_research_review``), else None."""
    if board is None or getattr(board, "empty", True) or "gemini_research_agrees" not in board:
        return None
    keys = ("matchup_id", "market_type")
    if not all(k in result for k in keys + ("pick",)) or not all(k in board for k in keys + ("best_pick",)):
        return None

    def label(row):
        agrees = pd.to_numeric(row.get("gemini_research_agrees"), errors="coerce")
        if pd.isna(agrees):
            return None
        confidence = str(row.get("gemini_research_confidence") or "").strip()
        text = "Agrees" if agrees == 1 else f"Disagrees ({row.get('gemini_research_pick')})"
        return text + (f" · {confidence}" if confidence else "")
    lookup = {f"{r['matchup_id']}|{r['market_type']}|{r['best_pick']}": label(r) for _, r in board.iterrows()}
    labels = (result["matchup_id"].astype(str) + "|" + result["market_type"].astype(str) + "|"
              + result["pick"].astype(str)).map(lookup)
    return labels if labels.notna().any() else None


def market_shadow(result, candidates):
    """(market no-vig probability, market EV) per board row from its saved candidate, or None.

    The candidate carries ``fair_market_probability`` (else ``market_probability``) for the
    picked side. EV uses the row's American odds.
    """
    probs = _candidate_values(result, candidates, ("fair_market_probability", "market_probability"))
    return None if probs is None else (probs, _ev(result, probs))


def pinnacle_shadow(result, candidates):
    """(Pinnacle no-vig probability at the pick's line, EV against it) per board row, or None."""
    probs = _candidate_values(result, candidates, ("pinnacle_fair_probability",))
    if probs is None or not probs.notna().any():
        return None
    return probs, _ev(result, probs)


def research_model_shadow(result, candidates):
    """(research model win probability, its EV) per board row from the saved candidate, or None.

    ``app_core.research_model_column``: the research loop's score model (NBA, NCAAB, NHL), at
    the pick's own side and line; EV counts a push as stake returned.
    """
    probs = _candidate_values(result, candidates, ("research_model_probability",))
    if probs is None or not probs.notna().any():
        return None
    # The lookup keeps values in (0, 1), so a zero push reads as missing: 0.
    push = _candidate_values(result, candidates, ("research_model_push_probability",))
    push = push.fillna(0.0) if push is not None else pd.Series(0.0, index=probs.index)
    decimal = pd.to_numeric(result.get("odds"), errors="coerce").map(_decimal).astype("float64")
    return probs, probs * (decimal - 1) - (1 - probs - push)


PINNACLE_MIN_EV = 0.02


def _decimal(odds):
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if pd.isna(odds) or abs(odds) < 100:
        return None
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


BOOK_UNKNOWN = "Not recorded"


def _entry_books(row):
    """The book(s) offering this candidate's exact side, line and price; None when only Pinnacle does.

    The feed's name (``the_odds_api``) is not a book (owner 2026-11-13). The saved bookmaker is used
    when there is one; otherwise the books in the row's own provider quotes that match it exactly.
    """
    saved = _text(row, "quote_bookmaker", default="")
    if saved:
        return None if "pinnacle" in saved.lower() else saved
    try:
        from app_core.prediction_evidence import matching_quotes
        books = sorted({str(quote.get("book") or "").strip() for quote in matching_quotes(row)} - {""})
    except Exception:
        books = []
    if not books:
        return BOOK_UNKNOWN
    stake = [name for name in books if "pinnacle" not in name.lower()]
    return ", ".join(stake) if stake else None


def pinnacle_value_board(candidates, min_ev=PINNACLE_MIN_EV):
    """Candidates whose price beats Pinnacle's no-vig probability at the same line (research only).

    Uses each candidate's saved ``pinnacle_fair_probability`` (``core.pinnacle_anchor``:
    Pinnacle posted both sides at exactly that line, fresh at refresh) and its own
    American odds: EV = fair x decimal - 1. Keeps EV >= ``min_ev``, pregame candidates
    only, never a Pinnacle price as the entry, and the best EV per game and market.
    """
    columns = ["Sport", "Game", "Pick", "Odds", "Book", "Pinnacle fair", "EV vs Pinnacle"]
    if candidates is None or getattr(candidates, "empty", True) or "pinnacle_fair_probability" not in candidates:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in candidates.to_dict("records"):
        if str(row.get("candidate_context") or "").upper() == "POST_START_DIAGNOSTIC":
            continue
        try:
            fair = float(row.get("pinnacle_fair_probability"))
        except (TypeError, ValueError):
            continue
        decimal = _decimal(row.get("odds_american"))
        if decimal is None or pd.isna(fair) or not 0 < fair < 1:
            continue
        if "pinnacle" in _text(row, "quote_bookmaker", "odds_feed_source", "odds_source", default="").lower():
            continue
        book = _entry_books(row)
        if book is None:
            continue
        ev = fair * decimal - 1
        if ev < min_ev:
            continue
        family = "Totals" if str(row.get("market_type") or "").lower().startswith("total") else "Sides"
        game = f"{_text(row, 'away_team', default='?')} at {_text(row, 'home_team', default='?')}"
        rows.append({"key": (_text(row, "matchup_id", default=game), family), "Sport": _text(row, "league", default=""),
                     "Game": game, "Pick": _text(row, "best_pick", default=""), "Odds": float(row["odds_american"]),
                     "Book": book, "Pinnacle fair": fair, "EV vs Pinnacle": ev})
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).sort_values("EV vs Pinnacle", ascending=False)
    return frame.drop_duplicates("key")[columns].reset_index(drop=True)


def _render_pinnacle_value(candidates):
    board = pinnacle_value_board(candidates)
    with st.expander(f"Pinnacle value (research) · {len(board)} price(s) beat Pinnacle", expanded=False):
        if board.empty:
            st.info("No current price beats Pinnacle's no-vig line by 2% or more. Pinnacle must post both "
                    "sides at the same line, updated within 20 minutes of Refresh.")
        else:
            shown = board.copy()
            for column in ("Pinnacle fair", "EV vs Pinnacle"):
                shown[column] = shown[column].map(lambda v: f"{v:.1%}")
            st.dataframe(shown, hide_index=True, width="stretch",
                         column_config={"Odds": st.column_config.NumberColumn(format="%+.0f")})
        st.caption("Research only: no stake, approval or change to Best picks. Each row is the best price per game and "
                   "market whose expected value against Pinnacle's no-vig probability at the same line is at least 2%. "
                   "Pinnacle is the sharp reference, never the entry price. Its edge is being measured in the "
                   "forward shadow before it can move any pick.")


def _render_game_board(board, candidates, family):
    from app_core.per_game_boards import per_game_board
    from app_core.coverage_presentation import CoverageBindingConflict
    try:
        result = per_game_board(board, candidates, family)
    except CoverageBindingConflict as exc:
        # Keep the schedule visible and the owner Publish panel reachable.
        # Field-level diagnostics stay behind its token gate; no rejected metrics.
        st.error('Game board could not be built: '+str(exc))
        decisions = board.attrs.get('slate_coverage', {}).get('decisions', [])
        st.dataframe(pd.DataFrame([dict(Event=d['canonical_event_id'],
            Game=d['away_team']+' at '+d['home_team'], Start=d['original_start'],
            **{'Coverage decision': d['coverage_decision_state'], 'Wager status': 'PASS',
               'Preview blocker': str(exc)}) for d in decisions]), hide_index=True, width='stretch')
        return
    if result.empty:
        st.info("Run an analysis to populate the game board.")
        return
    display = result.rename(columns={"league":"Sport", "matchup":"Game", "start":"Start", "pick":"Best pick", "selection_label":"Selection", "status":"Wager status", "approval_reason":"Wager explanation", "odds":"Odds", "win_probability":"Win estimate", "edge":"Edge", "ev":"EV estimate", "selection_score":"Selection score", "probability_basis":"Win estimate source"})
    columns = ["Game", "Best pick", "Odds", "Win estimate", "EV estimate", "Wager status"]
    market = market_shadow(result, candidates)
    if market is not None:
        # Shadow only (owner request 2026-11-06): the market's no-vig price next to the
        # app's estimate. It changes no pick, stake or approval.
        display["Market estimate"], display["Market EV"] = market
        columns[4:4] = ["Market estimate", "Market EV"]
    pinnacle = pinnacle_shadow(result, candidates)
    if pinnacle is not None:
        # Research reference (owner 2026-11-07): Pinnacle's no-vig price at the pick's exact
        # line, where Pinnacle posted it; changes no pick, stake or approval.
        display["Pinnacle fair"], display["EV vs Pinnacle"] = pinnacle
        at = columns.index("Wager status")
        columns[at:at] = ["Pinnacle fair", "EV vs Pinnacle"]
    model = research_model_shadow(result, candidates)
    if model is not None:
        # Research only (owner 2026-11-07): the unvalidated research model's view of the pick;
        # changes no pick, stake or approval.
        display["Research model"], display["Research model EV"] = model
        at = columns.index("Wager status")
        columns[at:at] = ["Research model", "Research model EV"]
    gemini = gemini_research_labels(result, board)
    if gemini is not None:
        # Research only (owner 2026-11-07): Gemini's review of the top picks; never a stake.
        display["Gemini (research)"] = gemini.fillna("Not reviewed")
        at = columns.index("Wager status")
        columns[at:at] = ["Gemini (research)"]
    if 'coverage_decision_state' in result:
        display['Coverage decision'] = result['coverage_decision_state']
        display['Coverage explanation'] = result['coverage_explanation']
        columns += ['Coverage decision', 'Coverage explanation']
    display = display[columns].copy()
    for column in [c for c in ("Win estimate", "EV estimate", "Market estimate", "Market EV", "Pinnacle fair",
                               "EV vs Pinnacle", "Research model", "Research model EV") if c in display]:
        display[column] = display[column].map(lambda v: f"{v:.1%}" if pd.notna(v) else "Unavailable")
    st.dataframe(display, hide_index=True, width="stretch", height=min(760, 38 * (len(display) + 1)), column_config={"Selection score": st.column_config.NumberColumn(format="%.3f"), "Odds": st.column_config.NumberColumn(format="%+.0f")})
    with st.expander("Selection details and wager explanations", expanded=False):
        st.dataframe(result[["matchup", "pick", "approval_reason", "edge", "selection_score", "probability_basis", "start"]], hide_index=True, width="stretch")
        st.caption("Best Overall, Best Side, and Best Total identify the selected pick for each game. Wager status is separate: PASS means no approved wager. Ranking uses the composite selection score first, followed by probability, tier, EV, and edge as tie-breakers; the highest EV does not necessarily rank first. Selection score is not a win probability. Edge compares the displayed estimate with price break-even. Estimate source is shown for each selection; NFL context-backed estimates retain the prior result and injury-report status in the export. Only exact final tickets carry final production metrics and approval.")
    st.download_button("Download this game board", result.to_csv(index=False), family+"-per-game.csv", "text/csv", key="per_game_export_"+family)


def render_daily_dashboard(today, details, frame: pd.DataFrame, candidates: pd.DataFrame | None = None) -> None:
    board = daily_board(frame)
    with today.container():
        from app.ui.slate_coverage import render_coverage
        render_coverage(frame.attrs.get('slate_coverage'), key='today_slate_coverage')
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
            with st.expander("Why picks are not eligible for wagers", expanded=approved.empty):
                st.dataframe(wager_rejection_summary(board), hide_index=True, width="stretch")
                st.caption("One primary blocker per game from the saved analysis. Research locks preserve selections; they do not approve wagers. Refresh picks after an input fix to evaluate prices and eligibility again.")
            overall, sides, totals = st.tabs(["Overall Best Pick", "Sides", "Totals"], key="daily_market_navigation", on_change="rerun")
            with overall:
                _render_game_board(board, candidates, "overall")
            with sides:
                st.caption("Best moneyline or spread for each game, selected from the full ranked candidate audit.")
                _render_game_board(board, candidates, "sides")
            with totals:
                st.caption("Best over or under for each game, selected from the full ranked candidate audit.")
                _render_game_board(board, candidates, "totals")
            _render_pinnacle_value(candidates)
            if not approved.empty:
                st.download_button("Download approved game wagers", approved.to_csv(index=False),
                                   "approved-game-wagers.csv", "text/csv", key="daily_approved_export")
            st.caption("Use Results after games finish. Readiness for grading does not establish a profitable edge or an achieved win rate.")
    with details.container():
        render_coverage(frame.attrs.get('slate_coverage'), key='details_slate_coverage')
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
