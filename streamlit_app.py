"""
Streamlit App - ParlayDesk
"""
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import logging
import pandas as pd
import streamlit as st

# Helper imports from app_core
try:
    from app_core.kalshi_integrator import league_game_prefix, team_code_for_league
except ImportError:
    # Fallback if imports fail during initial setup
    def league_game_prefix(league: str) -> str:
        return f"KX{league.upper()}GAME"
    def team_code_for_league(league: str, team: str) -> str:
        return team[:3].upper()

logger = logging.getLogger(__name__)

# --- Helper Functions (restored/stubbed to fix NameErrors) ---

def get_local_tz() -> str:
    """Return the local timezone string."""
    # Default to Eastern for betting apps usually
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

# --- Main Logic ---

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

        # Prefer provided codes; fall back to mapping from team names and extra heuristics.
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
        
        # Pre-compute target date strings for various formats
        target_date_nodash = ""
        target_date_short = ""
        if game_local:
            base_date = game_local.date()
            target_date_nodash = base_date.isoformat().replace("-", "")
            target_date_short = base_date.strftime("%y%b%d").upper() # e.g. 23OCT25

        # Prepare team codes sets for faster lookup
        home_codes_set = set(c.upper() for c in home_codes if c)
        away_codes_set = set(c.upper() for c in away_codes if c)

        filtered = []
        for m in markets:
            ticker = ticker_upper(m)
            
            # 1. League Check (Basic) - If ticker doesn't contain league prefix, skip
            # Note: winner_prefix usually has league in it (e.g. KXNBAGAME)
            
            # NCAAB Special Fallback: Relaxed matching logic
            if league == "NCAAB" and "GAME" in ticker and "NCAA" in ticker:
                 # If at least one team code matches, we assume it's relevant for this game
                 # This is a broad net to catch college games where tickers might be less standard
                 # and date matching might be flaky due to UTC/Local shifts
                 has_any_team = any(c in ticker for c in home_codes_set.union(away_codes_set))
                 if has_any_team:
                      filtered.append(m)
                      continue

            # Standard Logic continue...
            # Check for any allowed prefix
            if not any(p in ticker for p in allowed_prefixes):
                continue

            # Date Check
            date_match = False
            for dt_tok in allowed_date_tokens:
                if dt_tok in ticker:
                    date_match = True
                    break
            
            if not date_match:
                 continue

            # Team Check
            has_home = any(c in ticker for c in home_codes_set)
            has_away = any(c in ticker for c in away_codes_set)
            
            if has_home and has_away:
                filtered.append(m)
        
        return filtered

    except Exception:
        return []

if __name__ == "__main__":
    st.title("ParlayDesk (Repair Mode)")
    st.warning("This file appears to have been overwritten with a partial function definition. A repair has been applied to make it valid Python.")
    st.write("Function `filter_kalshi_game_markets` is available.")
