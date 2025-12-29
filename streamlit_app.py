import streamlit as st
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple, Union, cast
from zoneinfo import ZoneInfo

# Import Core Components
from app_core.apisports import APISportsBasketballClient, APISportsFootballClient, APISportsHockeyClient
from app_core.sportsdata import SportsDataNBAClient, SportsDataNCAABClient, SportsDataNFLClient, SportsDataNCAAFClient, SportsDataNHLClient
from app_core.feature_processing import run_roi_pipeline_validation, TeamNameMatcher
from app_core.sentiment_pipeline import SentimentPipeline

# Helper imports from app_core
try:
    from app_core.kalshi_integrator import league_game_prefix, team_code_for_league
except ImportError:
    def league_game_prefix(league: str) -> str:
        return f"KX{league.upper()}GAME"
    def team_code_for_league(league: str, team: str) -> str:
        return team[:3].upper()

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_local_tz() -> str:
    """Return the local timezone string."""
    return "America/New_York"

def parse_commence_to_utc(date_str: str) -> datetime:
    """Parse a commence time string to UTC datetime."""
    try:
        if not date_str:
            return datetime.now(timezone.utc)
        dt = datetime.fromisoformat(str(date_str).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)

def kalshi_date_token_from_local(dt: Optional[datetime]) -> str:
    """Generate Kalshi date token from local time."""
    if not dt:
        return "UNKNOWN"
    return dt.strftime("%y%b%d").upper()

def team_code_candidates(league: str, team: Any) -> List[str]:
    """Get candidate team codes for a team."""
    if not team:
        return []
    code = team_code_for_league(league, str(team))
    return [code] if code else []

def enrich_game_context(games_data: List[Dict], api_clients: Dict[str, Any]) -> List[Dict]:
    """
    Enrich game data with stats using fuzzy matching as the only way to match teams.
    """
    if not games_data:
        return []

    # 1. Fetch all available stats into a lookup dataframe
    from app_core.feature_processing import fetch_and_process_standings

    stats_df = fetch_and_process_standings(api_clients)

    # 2. Create a lookup dictionary: normalized_name -> stats_row
    stats_lookup = {}
    if not stats_df.empty:
        for _, row in stats_df.iterrows():
            norm_name = row.get('team_norm')
            if norm_name:
                stats_lookup[norm_name] = row.to_dict()

    enriched_games = []

    for game in games_data:
        g = game.copy()

        # Use fuzzy matcher to normalize names
        home_team = str(g.get('home_team', ''))
        away_team = str(g.get('away_team', ''))

        home_norm = TeamNameMatcher.normalize(home_team)
        away_norm = TeamNameMatcher.normalize(away_team)

        home_stats = stats_lookup.get(home_norm, {})
        away_stats = stats_lookup.get(away_norm, {})

        # Inject stats with defaults if missing
        g['home_win_pct'] = home_stats.get('win_pct', 0.5)
        g['home_ppg'] = home_stats.get('ppg', 50.0)
        g['home_oppg'] = home_stats.get('oppg', 50.0)
        g['home_streak'] = home_stats.get('streak', 0.0)

        g['away_win_pct'] = away_stats.get('win_pct', 0.5)
        g['away_ppg'] = away_stats.get('ppg', 50.0)
        g['away_oppg'] = away_stats.get('oppg', 50.0)
        g['away_streak'] = away_stats.get('streak', 0.0)

        enriched_games.append(g)

    return enriched_games

def filter_kalshi_game_markets(
    markets: List[Dict[str, Any]],
    game_time_utc: Optional[datetime],
    league: str,
    home_team: Any = None,
    away_team: Any = None,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Flexible filtering for game markets with title-based fallback."""
    try:
        tz_name = get_local_tz()
        local_tz = None
        try:
            local_tz = ZoneInfo(tz_name)
        except Exception:
            local_tz = None

        game_dt = game_time_utc
        if isinstance(game_dt, str):
            game_dt = parse_commence_to_utc(game_dt)
        if isinstance(game_dt, datetime) and game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=timezone.utc)
        game_local = game_dt.astimezone(local_tz) if (game_dt and local_tz) else game_dt
        date_token = game_local.strftime("%y%b%d").upper() if game_local else kalshi_date_token_from_local(game_time_utc)
        date_token = date_token or "UNKNOWN"

        league_upper = (league or "").upper()
        winner_prefix = league_game_prefix(league_upper)
        prefix_overrides = (st.session_state.get("kalshi_game_prefix_map") or {}).get(
            league_upper
        )
        allowed_prefixes = [p for p in [winner_prefix, prefix_overrides] if p]

        home_codes = []
        away_codes = []
        if home_code:
            home_codes.append(str(home_code).upper())
        home_codes.extend(team_code_candidates(league, home_team))
        if away_code:
            away_codes.append(str(away_code).upper())
        away_codes.extend(team_code_candidates(league, away_team))

        def ticker_upper(market: Dict[str, Any]) -> str:
            return str(market.get("event_ticker") or market.get("ticker") or "").upper()

        allowed_date_tokens: List[str] = []
        if game_local:
            base_date = game_local.date()
            for delta in (-1, 0, 1):
                allowed_date_tokens.append(
                    (base_date + timedelta(days=delta)).strftime("%y%b%d").upper()
                )
        
        home_codes_set = set(c.upper() for c in home_codes if c)
        away_codes_set = set(c.upper() for c in away_codes if c)

        filtered = []
        for m in markets:
            ticker = ticker_upper(m)
            
            if league == "NCAAB" and "GAME" in ticker and "NCAA" in ticker:
                 has_any_team = any(c in ticker for c in home_codes_set.union(away_codes_set))
                 if has_any_team:
                      filtered.append(m)
                      continue

            if not any(p in ticker for p in allowed_prefixes):
                continue

            date_match = False
            for dt_tok in allowed_date_tokens:
                if dt_tok in ticker:
                    date_match = True
                    break
            
            if not date_match:
                 continue

            has_home = any(c in ticker for c in home_codes_set)
            has_away = any(c in ticker for c in away_codes_set)
            
            if has_home and has_away:
                filtered.append(m)
        
        return filtered

    except Exception:
        return []

if __name__ == "__main__":
    st.title("ParlayDesk (System Restored)")
    st.success("Core modules and headers restored. Application ready for logic injection.")
