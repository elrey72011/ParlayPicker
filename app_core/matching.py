
import re
import datetime
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process
import streamlit as st

# Import dependency from existing app_core module
from app_core.kalshi_integrator import (
    league_game_prefix,
    team_code_for_league,
)

def get_local_tz() -> str:
    """Return configured or system local timezone string."""
    try:
        if "general" in st.secrets:
            return st.secrets["general"].get("timezone", "America/New_York")
        return st.secrets.get("timezone", "America/New_York")
    except Exception:
        return "America/New_York"

def parse_commence_to_utc(raw: Any) -> Optional[datetime]:
    if not raw:
        return None
    try:
        # If already datetime
        if isinstance(raw, datetime):
            if raw.tzinfo is None:
                return raw.replace(tzinfo=timezone.utc)
            return raw.astimezone(timezone.utc)
        # Isoformat string
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def kalshi_date_token_from_local(date_val: Any) -> Optional[str]:
    """Return YYMONDD token (e.g., 25DEC16) for local YYYY-MM-DD date strings."""
    try:
        if not date_val:
            return None
        parsed = datetime.fromisoformat(str(date_val))
        return parsed.strftime("%y%b%d").upper()
    except Exception:
        return None

def team_code_candidates(league: str, team_name: Any) -> List[str]:
    primary = (team_code_for_league(league, team_name) or "").upper()
    cleaned = re.sub(r"[^A-Z0-9 ]", " ", str(team_name or "").upper()).strip()
    tokens = [t for t in cleaned.split() if t]

    candidates: List[str] = []
    if primary:
        candidates.append(primary)
    for tok in tokens:
        if tok:
            candidates.extend([tok, tok[:3], tok[:2]])
    if tokens:
        initials = "".join(t[0] for t in tokens if t)
        if len(initials) >= 2:
            candidates.append(initials)
            candidates.append(initials[:2])
        first_two_initials = "".join(t[0] for t in tokens[:2] if t)
        if len(first_two_initials) >= 2:
            candidates.append(first_two_initials)

        # Common college-style abbreviations (e.g., ARST, MOSU)
        if len(tokens) >= 2 and tokens[1] in {"STATE", "ST"}:
            first = tokens[0]
            first2 = first[:2]
            first3 = first[:3]
            candidates.extend([f"{first2}ST", f"{first3}ST", f"{first2}SU", f"{first3}SU"])
        if len(tokens) >= 2 and tokens[1] in {"UNIVERSITY", "UNIV", "U"}:
            first = tokens[0]
            first2 = first[:2]
            first3 = first[:3]
            candidates.extend([f"{first2}U", f"{first3}U"])
    deduped = [c for c in dict.fromkeys(candidates) if c]
    return deduped

def normalize_market_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    # Remove common noise words in sports names, and 'vs'/'at' for clean tokenization
    text = re.sub(r'\b(state|st\.|univ\.|university|bc|unlv|la|vs|at)\b', '', text)
    # Remove apostrophes specifically to handle "John's" -> "johns"
    text = text.replace("'", "")
    # Keep alphanumeric AND spaces
    return re.sub(r'[^a-z0-9 ]', '', text).strip()

def filter_kalshi_game_markets(
    markets: List[Dict[str, Any]],
    game_time_utc: Optional[datetime],
    league: str,
    home_team: Any = None,
    away_team: Any = None,
    home_code: Optional[str] = None,
    away_code: Optional[str] = None,
    prefix_overrides: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Robust 3-step filtering engine:
    1. Date/Time Alignment (+/- 24 hours to be safe)
    2. Normalization
    3. Fuzzy Matching (RapidFuzz)
    """
    try:
        # --- Step 0: Parse Game Time ---
        game_dt = game_time_utc
        if isinstance(game_dt, str):
            game_dt = parse_commence_to_utc(game_dt)
        if isinstance(game_dt, datetime) and game_dt.tzinfo is None:
            game_dt = game_dt.replace(tzinfo=timezone.utc)

        # NOTE: We proceed even if game_dt is missing, relying on name matching,
        # but logically we should have a date.
        # If we have a date, we use it to filter aggressively.

        matched: List[Dict[str, Any]] = []

        # Prepare search terms
        home_norm = normalize_market_text(str(home_team or ""))
        away_norm = normalize_market_text(str(away_team or ""))

        # Create a space-less version for partial_ratio containment checks (legacy style)
        home_simple = home_norm.replace(" ", "")
        away_simple = away_norm.replace(" ", "")

        for m in markets or []:
            # --- Step 1: Date/Time Alignment ---
            # Try to respect the date if possible
            ticker = str(m.get("event_ticker") or m.get("ticker") or "").upper()

            # Check if it's a game market
            if "GAME" not in ticker and not m.get("title", "").upper().count(" VS "):
                # If it doesn't look like a game, skip?
                # Kalshi usually has GAME in ticker for games.
                # But let's be lenient if title matches strongly.
                pass

            # Date Filter
            if game_dt:
                # Try to extract date from Open Time or Ticker
                m_open = m.get("open_time")
                if m_open:
                    m_dt = parse_commence_to_utc(m_open)
                    if m_dt:
                        diff = abs((m_dt - game_dt).total_seconds())
                        # 24 hours window (86400s) to catch weird open times or timezones
                        if diff > 86400:
                            continue

            # --- Step 2 & 3: Fuzzy Matching ---
            market_title = normalize_market_text(m.get("title", ""))
            market_ticker_norm = normalize_market_text(ticker)

            # 1. Direct Containment (using space-preserved norms)
            # "army" in "army vs navy" -> True
            # "black knights" in "army black knights" -> True
            home_in = home_norm in market_title or home_norm in market_ticker_norm
            away_in = away_norm in market_title or away_norm in market_ticker_norm

            is_match = False
            if home_in and away_in:
                is_match = True
            else:
                # 2. Token Set Ratio (Handles "Army Navy" vs "Navy Army")
                # This is powerful because we preserved spaces!
                combo_query = f"{home_norm} {away_norm}"

                # token_sort_ratio handles reordering
                score_sort = fuzz.token_sort_ratio(combo_query, market_title)

                # partial_ratio handles substrings (legacy requirement)
                # For partial ratio, space-less might be safer if we are unsure of separators
                # but with spaces, partial_ratio still works if substring exists.
                score_partial = fuzz.partial_ratio(home_simple + away_simple, market_title.replace(" ", ""))

                if score_sort > 80 or score_partial > 85:
                    is_match = True

            if is_match:
                matched.append(m)

        return matched
    except Exception:
        return []
