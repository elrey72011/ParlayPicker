from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


GAME_KEY_COLUMNS = ["league", "away_team", "home_team", "game_date"]
DIAGNOSTIC_COLUMNS = [
    "league",
    "away_team",
    "home_team",
    "game_date",
    "market_type",
    "best_pick",
    "kalshi_match_status",
    "kalshi_match_reason",
    "kalshi_series_ticker",
    "kalshi_series_fetch_error",
    "kalshi_candidate_event_count",
    "kalshi_dated_candidate_event_count",
    "kalshi_best_event_score",
    "kalshi_event_ticker",
    "kalshi_market_ticker",
    "kalshi_market_title",
    "kalshi_probability",
    "kalshi_line",
    "kalshi_line_diff",
]


def _market_family(values: pd.Series) -> pd.Series:
    text = values.astype("string").fillna("").str.lower()
    family = pd.Series("other", index=values.index, dtype="string")
    family.loc[text.str.contains("spread", na=False)] = "spread"
    family.loc[text.str.contains("total|over|under", regex=True, na=False)] = "total"
    family.loc[text.str.contains("moneyline|h2h", regex=True, na=False)] = "moneyline"
    return family


def build_kalshi_diagnostic_frames(df: pd.DataFrame) -> dict[str, Any]:
    """Build complete, exportable Kalshi coverage diagnostics without UI truncation."""
    if df is None or df.empty or "kalshi_match_status" not in df.columns:
        return {
            "overall": {
                "attempted_rows": 0,
                "matched_rows": 0,
                "row_coverage": 0.0,
                "attempted_games": 0,
                "matched_games": 0,
                "game_coverage": 0.0,
            },
            "matched": pd.DataFrame(),
            "misses": pd.DataFrame(),
            "reason_summary": pd.DataFrame(columns=["reason", "rows"]),
            "league_summary": pd.DataFrame(),
            "market_summary": pd.DataFrame(),
        }

    work = df.copy()
    work["_kalshi_matched"] = (
        work["kalshi_match_status"].astype("string").fillna("").str.lower().eq("matched")
    )
    if "market_type" in work.columns:
        work["market_family"] = _market_family(work["market_type"])
    else:
        work["market_family"] = "unknown"

    visible_columns = [column for column in DIAGNOSTIC_COLUMNS if column in work.columns]
    if "market_family" not in visible_columns:
        market_insert = visible_columns.index("market_type") + 1 if "market_type" in visible_columns else 4
        visible_columns.insert(market_insert, "market_family")

    matched = work.loc[work["_kalshi_matched"], visible_columns].copy()
    misses = work.loc[~work["_kalshi_matched"], visible_columns].copy()
    reason_summary = (
        misses.get("kalshi_match_reason", pd.Series("unknown", index=misses.index))
        .astype("string")
        .fillna("unknown")
        .replace("", "unknown")
        .value_counts()
        .rename_axis("reason")
        .reset_index(name="rows")
    )

    row_summary = (
        work.groupby(["league"], dropna=False)["_kalshi_matched"]
        .agg(attempted_rows="size", matched_rows="sum")
        .reset_index()
        if "league" in work.columns
        else pd.DataFrame()
    )
    if not row_summary.empty:
        row_summary["missed_rows"] = row_summary["attempted_rows"] - row_summary["matched_rows"]
        row_summary["row_coverage"] = row_summary["matched_rows"] / row_summary["attempted_rows"].clip(lower=1)

    game_key_columns = [column for column in GAME_KEY_COLUMNS if column in work.columns]
    game_summary = pd.DataFrame()
    attempted_games = matched_games = 0
    if len(game_key_columns) == len(GAME_KEY_COLUMNS):
        games = (
            work.groupby(game_key_columns, dropna=False)["_kalshi_matched"]
            .any()
            .rename("game_matched")
            .reset_index()
        )
        attempted_games = int(len(games))
        matched_games = int(games["game_matched"].sum())
        game_summary = (
            games.groupby("league", dropna=False)["game_matched"]
            .agg(attempted_games="size", matched_games="sum")
            .reset_index()
        )
        game_summary["missed_games"] = game_summary["attempted_games"] - game_summary["matched_games"]
        game_summary["game_coverage"] = game_summary["matched_games"] / game_summary["attempted_games"].clip(lower=1)

    if row_summary.empty:
        league_summary = game_summary
    elif game_summary.empty:
        league_summary = row_summary
    else:
        league_summary = row_summary.merge(game_summary, on="league", how="outer")

    market_summary = (
        work.groupby("market_family", dropna=False)["_kalshi_matched"]
        .agg(attempted_rows="size", matched_rows="sum")
        .reset_index()
    )
    market_summary["missed_rows"] = market_summary["attempted_rows"] - market_summary["matched_rows"]
    market_summary["row_coverage"] = market_summary["matched_rows"] / market_summary["attempted_rows"].clip(lower=1)

    attempted_rows = int(len(work))
    matched_rows = int(work["_kalshi_matched"].sum())
    return {
        "overall": {
            "attempted_rows": attempted_rows,
            "matched_rows": matched_rows,
            "row_coverage": matched_rows / max(attempted_rows, 1),
            "attempted_games": attempted_games,
            "matched_games": matched_games,
            "game_coverage": matched_games / max(attempted_games, 1),
        },
        "matched": matched,
        "misses": misses,
        "reason_summary": reason_summary,
        "league_summary": league_summary,
        "market_summary": market_summary,
    }


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def render_kalshi_diagnostics(df: pd.DataFrame) -> None:
    st.subheader("Kalshi Diagnostics")
    frames = build_kalshi_diagnostic_frames(df)
    overall = frames["overall"]
    if overall["attempted_rows"] == 0:
        st.info("No Kalshi diagnostic rows found in analysis output.")
        return

    row_col, game_col = st.columns(2)
    row_col.metric(
        "Candidate-row coverage",
        f"{overall['matched_rows']}/{overall['attempted_rows']} ({overall['row_coverage']:.0%})",
    )
    game_col.metric(
        "Game-event coverage",
        f"{overall['matched_games']}/{overall['attempted_games']} ({overall['game_coverage']:.0%})",
    )
    st.caption(
        "Candidate-row coverage counts every spread/total direction attempted. "
        "Game-event coverage counts a game as covered when at least one candidate row matched."
    )

    st.markdown("#### Coverage by league")
    st.dataframe(frames["league_summary"], width="stretch", hide_index=True)
    st.markdown("#### Coverage by market family")
    st.dataframe(frames["market_summary"], width="stretch", hide_index=True)
    st.markdown("#### Miss reasons")
    st.dataframe(frames["reason_summary"], width="stretch", hide_index=True)

    matched = frames["matched"]
    misses = frames["misses"]
    matched_col, missed_col = st.columns(2)
    matched_col.download_button(
        "Download all matched Kalshi rows",
        data=_csv_bytes(matched),
        file_name="kalshi_matched_rows.csv",
        mime="text/csv",
        key="download_all_kalshi_matched_rows",
    )
    missed_col.download_button(
        "Download all missed Kalshi rows",
        data=_csv_bytes(misses),
        file_name="kalshi_missed_rows.csv",
        mime="text/csv",
        key="download_all_kalshi_missed_rows",
    )

    st.markdown(f"#### Matched details ({len(matched)})")
    st.dataframe(matched, width="stretch", hide_index=True)
    st.markdown(f"#### Miss details ({len(misses)})")
    st.dataframe(misses, width="stretch", hide_index=True)
