"""
ParlayDesk v9.2 (Vertex-first) — stabilized for Streamlit Cloud.
This rewrite focuses on safe initialization, unique widget keys, and
robust fallbacks for missing credentials.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

from parlaydesk_init import (
    init_app_state,
    init_vertex_from_secrets,
    get_kalshi_integrator,
    make_key,
)
from parlaydesk_ui import render_debug_panel

# VertexMasterAnalyzer contract note:
# - __init__(odds_api_client=None, sportsdata_clients=None, apisports_clients=None,
#            sentiment_analyzer=None, local_ml_predictor=None, theover_data=None,
#            kalshi_integrator=None, use_kalshi=True)
# - Main method: analyze_all_games(games: List[dict], league: str = "NBA") -> pandas.DataFrame
# - Expected output columns include: ["League", "Home", "Away", "Commence (UTC)",
#   "Market", "Pick", "AI_Prob", "AI_Edge"] with optional Kalshi metadata.
from vertex_master_analyzer import VertexMasterAnalyzer

st.set_page_config(page_title="ParlayDesk", layout="wide")

SPORT_OPTIONS = {
    "NBA": "basketball_nba",
    "NCAAB": "basketball_ncaab",
    "NFL": "americanfootball_nfl",
    "NCAAF": "americanfootball_ncaaf",
    "NHL": "icehockey_nhl",
    "MLB": "baseball_mlb",
}


def fetch_odds_games(odds_client: Any, sport_key: str) -> List[Dict[str, Any]]:
    if odds_client is None:
        return []
    try:
        raw = odds_client.get_odds(sport_key)
    except Exception as exc:
        st.warning(f"Failed to fetch odds from TheOddsAPI: {exc}")
        return []

    games: List[Dict[str, Any]] = []
    for g in raw or []:
        commence = g.get("commence_time")
        try:
            commence_dt = datetime.fromisoformat(str(commence).replace("Z", "+00:00"))
        except Exception:
            commence_dt = None
        home = g.get("home_team")
        away = None
        teams = g.get("teams") or []
        if teams:
            away = teams[0] if teams[0] != home else (teams[1] if len(teams) > 1 else None)
        games.append(
            {
                "sport_key": sport_key,
                "home_team": home,
                "away_team": away,
                "commence_time": commence_dt,
                "bookmakers": g.get("bookmakers") or [],
            }
        )
    return games


def load_theover_data() -> Dict[str, pd.DataFrame]:
    """Load supplemental theover.ai CSVs if present."""
    data: Dict[str, pd.DataFrame] = {}
    base = Path("data")
    for name in ["theover_spreads.csv", "theover_totals.csv", "theover_moneyline.csv"]:
        path = base / name
        if path.exists():
            try:
                data[name] = pd.read_csv(path)
            except Exception:
                continue
    return data


def build_all_games(odds_client: Any, sport_key: str) -> List[Dict[str, Any]]:
    base_games = fetch_odds_games(odds_client, sport_key)
    if base_games:
        return base_games
    # Odds API missing or returned nothing; fall back to CSV-only mode.
    csv_path = Path("data/theover_spreads.csv")
    if not csv_path.exists():
        st.info("No live odds available and no local CSV data found.")
        return []
    try:
        df = pd.read_csv(csv_path)
        games: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            games.append(
                {
                    "sport_key": sport_key,
                    "home_team": row.get("home_team") or row.get("home"),
                    "away_team": row.get("away_team") or row.get("away"),
                    "commence_time": None,
                    "bookmakers": [],
                }
            )
        st.warning("Operating in CSV-only mode; odds may be stale.")
        return games
    except Exception as exc:
        st.warning(f"Failed to load backup CSV games: {exc}")
        return []


def run_vertex_master(
    all_games: List[Dict[str, Any]],
    league_label: str,
    odds_client: Any,
    sportsdata_clients: Dict[str, Any],
    apisports_clients: Dict[str, Any],
    sentiment_analyzer: Any,
    theover_data: Dict[str, pd.DataFrame],
    kalshi_integrator: Any,
) -> pd.DataFrame:
    analyzer = VertexMasterAnalyzer(
        odds_api_client=odds_client,
        sportsdata_clients=sportsdata_clients,
        apisports_clients=apisports_clients,
        sentiment_analyzer=sentiment_analyzer,
        local_ml_predictor=None,
        theover_data=theover_data,
        kalshi_integrator=kalshi_integrator,
        use_kalshi=bool(kalshi_integrator),
    )
    return analyzer.analyze_all_games(all_games, league=league_label)


def render_game_tab(config: Dict[str, Any]):
    col1, col2 = st.columns([2, 1])
    with col1:
        league = st.selectbox(
            "League",
            list(SPORT_OPTIONS.keys()),
            key=make_key("league-select"),
        )
    with col2:
        st.write("\n")
        refresh = st.button("Refresh games", key=make_key("refresh-games"))

    if refresh or "all_games" not in st.session_state:
        st.session_state["all_games"] = build_all_games(
            config["odds_client"], SPORT_OPTIONS[league]
        )
        st.session_state["selected_league"] = league

    games = st.session_state.get("all_games", [])
    st.caption(f"Loaded {len(games)} games for {st.session_state.get('selected_league', league)}")
    if games:
        df = pd.DataFrame(
            [
                {
                    "League": st.session_state.get("selected_league", league),
                    "Home": g.get("home_team"),
                    "Away": g.get("away_team"),
                    "Commence": g.get("commence_time"),
                }
                for g in games
            ]
        )
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No games available.")


def render_vertex_tab(config: Dict[str, Any]):
    league = st.session_state.get("selected_league", list(SPORT_OPTIONS.keys())[0])
    games = st.session_state.get("all_games", [])
    st.write(f"Using {len(games)} games from {league}.")

    vertex_status = init_vertex_from_secrets(config["logger"])
    st.write(vertex_status["message"])

    theover_data = load_theover_data()
    kalshi = get_kalshi_integrator(config["logger"])
    if not kalshi:
        st.info("Kalshi not configured; proceeding without Kalshi matching.")

    run_button = st.button("Run Vertex AI Master Analysis", key=make_key("run-vertex", league))
    if not run_button:
        return

    if not games:
        st.warning("Load games first from the Games tab.")
        return

    try:
        results = run_vertex_master(
            games,
            league,
            config["odds_client"],
            config["sportsdata_clients"],
            config["apisports_clients"],
            config["sentiment_analyzer"],
            theover_data,
            kalshi,
        )
        if results is None or results.empty:
            st.warning("Master analysis returned no rows.")
            return

        display_cols = [
            c
            for c in [
                "League",
                "Home",
                "Away",
                "Commence (UTC)",
                "Market",
                "Pick",
                "AI_Prob",
                "AI_Edge",
                "kalshi_available",
                "kalshi_label",
            ]
            if c in results.columns
        ]
        st.dataframe(results[display_cols], use_container_width=True)
    except Exception:
        tb = traceback.format_exc()
        st.session_state["last_exception"] = tb
        st.error("Vertex master analysis failed. See debug panel for details.")
        st.code(tb)


def render_settings_tab(config: Dict[str, Any]):
    st.subheader("Configuration overview")
    st.write("API configuration is derived from Streamlit secrets or environment variables.")
    st.json(st.session_state.get("api_config", {}))


def main():
    config = init_app_state()
    st.title("ParlayDesk — Vertex-first")

    tabs = st.tabs(["Games & Odds", "Vertex Master Analysis", "Settings"])
    tab_renderers = [render_game_tab, render_vertex_tab, render_settings_tab]

    for tab, renderer in zip(tabs, tab_renderers):
        with tab:
            try:
                renderer(config)
            except Exception:
                tb = traceback.format_exc()
                st.session_state["last_exception"] = tb
                st.error("Section failed. Check debug panel for details.")
                st.code(tb)

    render_debug_panel(
        st.session_state.get("api_config", {}),
        analyzer_path=str(Path(__file__).name),
        last_exception=st.session_state.get("last_exception"),
    )


if __name__ == "__main__":
    main()
