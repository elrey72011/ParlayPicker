import json
import os
import re
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st
from app_core.kalshi_integrator import KalshiIntegrator, LEAGUE_SERIES_MAP
from app_core.sentiment_pipeline import build_team_sentiment_map
from vertex_master_analyzer import blended_win_prob

# Must be the first Streamlit call
st.set_page_config(page_title="ParlayDesk", layout="wide")

# ... [Keep your helper utilities: read_secret, american_to_implied_prob, safe_iso, get_local_tz, parse_commence_to_utc, normalize_commence_times, nba_abbrev, fmt_local_time, extract_best_market] ...

# -----------------
# MASTER ANALYSIS TAB
# -----------------

with tab_master:
    st.header("Master Analysis")
    games = st.session_state.get("games", [])
    if st.button("Run Master Analysis"):
        sentiment_map, _ = build_team_sentiment_map(news_api_key, games, league) if news_api_key else ({}, {})
        kalshi_markets = st.session_state["kalshi_integrator"].get_sports_markets(league) if st.session_state["kalshi_integrator"] else []
        
        rows_out = []
        master_stats = {"market_rows_out": 0}

        # --- CORRECTED MASTER ANALYSIS LOOP ---
        for idx, g in enumerate(games):
            warnings: List[str] = list(g.get("warnings") or [])
            league_name = g.get("league")
            home = g.get("home_team")
            away = g.get("away_team")
        
            # 1. INITIALIZE DATA IMMEDIATELY (Fixes NameErrors)
            h_code = nba_abbrev(home)
            a_code = nba_abbrev(away)
            
            # Standardize display strings
            commence_iso = g.get("commence_time_iso_utc") or safe_iso(g.get("commence_time_iso"))
            commence_local = fmt_local_time(g.get("commence_time_local"))
            
            # CALCULATE external dependencies BEFORE generating rows
            vertex_prob_home = get_vertex_prob(g)
            home_sent = sentiment_map.get(home, 0.0)
            away_sent = sentiment_map.get(away, 0.0)
            sentiment_diff = home_sent - away_sent
        
            # Kalshi Matching
            filtered_markets = filter_kalshi_game_markets(
                kalshi_markets, g.get("commence_time_utc"), league_name,
                home, away, h_code, a_code
            )
            deduped = {m.get("event_ticker") or m.get("ticker"): m for m in filtered_markets}
            winner_reason_override = "winner_not_in_fetched_markets" if not filtered_markets else None
            kalshi_matches, _ = match_kalshi_market(g, list(deduped.values()), winner_reason_override)
            
            kalshi_winner = kalshi_matches.get("winner", {})

            # 2. GENERATE ROWS (Now safely uses pre-calculated variables)
            if not (g.get("home_ml_price") or g.get("home_spread_point") or g.get("total_point")):
                rows_out.append({
                    "League": league_name, "Home": home, "Away": away,
                    "Commence (UTC)": commence_iso,
                    "Market": "None",
                    "AI_Prob": vertex_prob_home, # Now safely defined
                    "kalshi_matched": kalshi_winner.get("kalshi_matched"),
                    "Sentiment_Diff": sentiment_diff,
                })
                master_stats["market_rows_out"] += 1
                continue

            # Moneyline Row
            if g.get("home_ml_price") is not None:
                pick = home if (implied_h or 0) >= (implied_a or 0) else away
                rows_out.append({
                    "League": league_name, "Home": home, "Away": away,
                    "Market": "Moneyline", "Pick": pick,
                    "AI_Prob": blended_for_selection(pick, market_home_prob),
                    "Kalshi_Prob": k_winner.get("kalshi_prob"),
                    "Sentiment_Diff": sentiment_diff
                })
                master_stats["market_rows_out"] += 1

        st.dataframe(pd.DataFrame(rows_out))
