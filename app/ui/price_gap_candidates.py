"""Price-gap candidates panel: live gaps against Pinnacle, research only and marked unproven."""
import pandas as pd
import streamlit as st

from app_core import price_gap_candidates as gaps

COLUMNS = ("sport", "market_family", "away_team", "home_team", "scheduled_start", "sportsbook", "pick",
           "american_odds", "fair_american_odds", "fair_probability", "edge", "captured_at")
MOVEMENT_COLUMNS = ("sport", "market_family", "away_team", "home_team", "scheduled_start", "pinnacle_first_line",
                    "pinnacle_line", "line_move", "pinnacle_first_probability", "pinnacle_probability",
                    "probability_shift", "moved_toward", "stake_books_at_line", "stake_books_off_line", "captures")


def _movement_frame(rows):
    frame = pd.DataFrame(rows).reindex(columns=MOVEMENT_COLUMNS)
    for column in ("stake_books_at_line", "stake_books_off_line"):
        frame[column] = frame[column].map(lambda value: ", ".join(value) if isinstance(value, list) else "")
    return frame


def _from_drive():
    from app_core.evidence_drive import DriveStore
    from app_core.evidence_remote import settings
    folder = settings()[0]
    if not folder:
        raise RuntimeError("MISSING_DRIVE_FOLDER")
    return gaps.latest(DriveStore(folder), folder)


@st.cache_data(ttl=300, show_spinner=False)
def _movement_from_drive():
    """The newest published movement rows from the last 30 minutes; [] when there are none or Drive is unreachable."""
    try:
        from app_core.evidence_drive import DriveStore
        from app_core.evidence_remote import settings
        folder = settings()[0]
        if not folder:
            return []
        found = gaps.latest(DriveStore(folder), folder, slots=6)
        return list((found or {}).get("movement") or [])
    except Exception:
        return []


def movement_for_export():
    """Movement rows for the audit export: the list loaded in the panel, else the newest one on Drive
    (read at most once every five minutes). Never raises: the export must not depend on it."""
    try:
        loaded = st.session_state.get("price_gap_candidates") or {}
        return list(loaded.get("movement") or []) or _movement_from_drive()
    except Exception:
        return []


AUTO_RELOAD_SECONDS = 300          # the price watch publishes about every 15 minutes; a small file, no odds credits


def _load():
    try:
        st.session_state["price_gap_candidates"] = _from_drive() or {"none": True}
        st.session_state["price_gap_candidates_checked"] = pd.Timestamp.now(tz="America/New_York").strftime("%-I:%M %p ET")
    except Exception as exc:
        st.session_state.pop("price_gap_candidates", None)
        st.error(f"The candidates could not be read from Drive ({type(exc).__name__}).")


def _body(auto):
    """The list and the movement table. Run as a fragment, so a reload redraws only this panel."""
    if st.button("Load latest candidates", key="price_gap_candidates_load"):
        with st.spinner("Reading the latest candidates from Drive…"):
            _load()
    elif auto:
        _load()
    found = st.session_state.get("price_gap_candidates")
    if not found:
        return
    checked = st.session_state.get("price_gap_candidates_checked")
    if checked:
        st.caption(f"Last checked {checked}." + (f" Reloading every {AUTO_RELOAD_SECONDS // 60} minutes while this "
                                                 "page is open." if auto else ""))
    if found.get("none"):
        st.info("No candidate list was published in the last hour. The price watch may not be running.")
        return
    st.write(f"{len(found['candidates'])} candidate(s) at {found['min_edge']:.0%} or more, captured "
             f"{found['generated_at']} UTC ({found['age_minutes']:.0f} min ago).")
    if found.get("stale"):
        st.error("This list is more than 15 minutes old. The prices have almost certainly moved.")
    if found["candidates"]:
        st.dataframe(pd.DataFrame(found["candidates"]).reindex(columns=COLUMNS), hide_index=True,
                     column_config={"edge": st.column_config.NumberColumn("edge", format="percent"),
                                    "fair_probability": st.column_config.NumberColumn("fair probability",
                                                                                      format="percent")})
    else:
        st.caption("No stake book was priced above Pinnacle's fair price in that capture.")
    moves = found.get("movement")
    if moves:
        st.markdown("**Pinnacle line movement, per game**")
        st.caption("Where Pinnacle's main line and fair price were when the watch first saw the game, and where "
                   "they are now. The line is the home line (spreads) or the total; the probability is the home "
                   "side's or the Over's. Movement is since the first stored capture, not since the market opened. "
                   "A book listed as off the line has not followed Pinnacle yet. Research only: a move is not a "
                   "reason to bet.")
        st.dataframe(_movement_frame(moves), hide_index=True, column_config={
            column: st.column_config.NumberColumn(column.replace("_", " "), format="percent")
            for column in ("pinnacle_first_probability", "pinnacle_probability", "probability_shift")})


def render_price_gap_candidates():
    with st.expander("Price-gap candidates (research only)", expanded=False):
        st.warning(gaps.NOTICE)
        st.caption("A stake book's price that is better than Pinnacle's fair price for the same side and line, in the "
                   "price watch's newest capture. The list is published while the price watch runs (evenings). "
                   "Gaps close within minutes: check the price at the book before reading anything into a row.")
        # Owner 2026-11-13: the list should move without a manual reload. Off by default; it reads one small
        # file from Drive and spends no odds credits. It only runs while the page is open.
        auto = st.toggle(f"Reload automatically every {AUTO_RELOAD_SECONDS // 60} minutes",
                         key="price_gap_candidates_auto", value=False)
        st.fragment(_body, run_every=AUTO_RELOAD_SECONDS if auto else None)(auto)
