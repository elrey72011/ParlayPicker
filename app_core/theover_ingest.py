"""
TheOver.ai Public Betting Ingestion Module

Parses raw text from TheOver.ai public betting picks and normalizes it for integration.
Supports parsing from Excel uploads (Totals/Sides) and text paste fallbacks.
Now uses Canonical Keys for reliable matching.
"""
import pandas as pd
import re
import logging
from typing import Tuple, Optional, Dict, List, Any
from datetime import datetime
from app_core.feature_processing import robust_normalize_team

logger = logging.getLogger(__name__)

# User-specified Alias Map for TheOver input specifically
THEOVER_LEAGUE_ALIASES = {
    "NFL": {
        "carolina": "carolina panthers",
        "chicago": "chicago bears",
        "green bay": "green bay packers",
        "l.a. rams": "los angeles rams",
        "la rams": "los angeles rams",
        "l.a. chargers": "los angeles chargers",
        "la chargers": "los angeles chargers",
        "new york giants": "new york giants",
        "new york jets": "new york jets",
        "ny giants": "new york giants",
        "ny jets": "new york jets",
        "washington": "washington commanders",
        "kc": "kansas city chiefs",
        "sf": "san francisco 49ers",
        "ne": "new england patriots",
        "tb": "tampa bay buccaneers",
        "lv": "las vegas raiders",
        "no": "new orleans saints",
    },
    "NHL": {
        "carolina": "carolina hurricanes",
        "vegas": "vegas golden knights",
        "ny rangers": "new york rangers",
        "n.y. rangers": "new york rangers",
        "st. louis": "st louis blues",
        "st louis": "st louis blues",
        "montreal": "montreal canadiens",
        "florida": "florida panthers",
        "tampa bay": "tampa bay lightning",
        "colorado": "colorado avalanche",
    },
    "NBA": {
        "ny knicks": "new york knicks",
        "la lakers": "los angeles lakers",
        "la clippers": "los angeles clippers",
        "gs warriors": "golden state warriors",
    }
}

def normalize_theover_team_for_ingest(team_raw: str, league: str) -> str:
    """
    Wrapper around robust_normalize_team that applies TheOver-specific aliases first.
    """
    if not team_raw:
        return ""

    clean_raw = str(team_raw).strip()
    lower_raw = clean_raw.lower()

    # 1. Apply TheOver Specific Aliases (City -> Full Name)
    if league in THEOVER_LEAGUE_ALIASES:
        aliases = THEOVER_LEAGUE_ALIASES[league]
        if lower_raw in aliases:
            clean_raw = aliases[lower_raw]
        # Also check for exact keys in aliases (sometimes casing matters less, but we lowered it)

    # 2. Use the system standard normalizer
    # robust_normalize_team handles lowercase, strip, mascot stripping (if college), etc.
    return robust_normalize_team(clean_raw, league=league)

def generate_canonical_key(league: str, date_str: str, away_norm: str, home_norm: str) -> str:
    """
    {league}|{local_date}|{away_norm}|{home_norm}
    """
    return f"{league}|{date_str}|{away_norm}|{home_norm}"

def parse_theover_excel(file_buffer, pick_type_hint: str = "UNKNOWN") -> pd.DataFrame:
    """
    Parse an Excel file for TheOver.ai data.

    Args:
        file_buffer: The file-like object from st.file_uploader.
        pick_type_hint: "TOTAL" or "SIDE" to guide default pick type if not detected.

    Returns:
        pd.DataFrame: Normalized DataFrame ready for merging.
    """
    try:
        df = pd.read_excel(file_buffer)
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        return pd.DataFrame()

    # Clean column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Map flexible columns
    league_cols = [c for c in df.columns if c in ("league", "sport")]
    col_league = league_cols[0] if league_cols else None

    # Date
    date_candidates = ["date", "game_date", "time"]
    col_date = next((c for c in date_candidates if c in df.columns), None)

    # Teams
    away_candidates = ["away", "away_team", "visitor", "road"]
    home_candidates = ["home", "home_team"]
    col_away = next((c for c in away_candidates if c in df.columns), None)
    col_home = next((c for c in home_candidates if c in df.columns), None)

    col_matchup = None
    if not (col_away and col_home):
        matchup_candidates = ["matchup", "game", "teams", "match"]
        col_matchup = next((c for c in matchup_candidates if c in df.columns), None)

    # Pick Info
    pick_candidates = ["pick", "selection", "play", "team", "side", "total_pick"]
    col_pick = next((c for c in pick_candidates if c in df.columns), None)

    line_candidates = ["line", "total", "points", "number", "odds", "spread"]
    col_line = next((c for c in line_candidates if c in df.columns), None)

    # Metadata
    model_candidates = ["source_model", "model", "source", "capper", "backed_by"]
    col_model = next((c for c in model_candidates if c in df.columns), None)

    hit_rate_candidates = ["hit_rate", "win_pct", "rate", "accuracy", "history", "confidence"]
    col_hit_rate = next((c for c in hit_rate_candidates if c in df.columns), None)

    records = []

    # Default Date if missing = Today
    today_str = datetime.now().strftime("%Y-%m-%d")

    for _, row in df.iterrows():
        # League
        raw_league = str(row[col_league]) if col_league else "UNKNOWN"
        league = raw_league.upper().strip()
        if "NBA" in league: league = "NBA"
        elif "NFL" in league: league = "NFL"
        elif "NHL" in league: league = "NHL"
        elif "MLB" in league: league = "MLB"
        elif any(x in league for x in ["NCAAB", "CBB", "COLLEGE BASKETBALL"]): league = "NCAAB"
        elif any(x in league for x in ["NCAAF", "CFB", "COLLEGE FOOTBALL"]): league = "NCAAF"

        # Date
        date_val = today_str
        if col_date:
            try:
                # Try to parse date
                d = pd.to_datetime(row[col_date])
                date_val = d.strftime("%Y-%m-%d")
            except:
                pass

        # Teams
        raw_away = ""
        raw_home = ""

        if col_away and col_home:
            raw_away = str(row[col_away])
            raw_home = str(row[col_home])
        elif col_matchup:
            val = str(row[col_matchup])
            if " @ " in val:
                parts = val.split(" @ ")
                raw_away, raw_home = parts[0], parts[1]
            elif " at " in val:
                parts = val.split(" at ")
                raw_away, raw_home = parts[0], parts[1]
            elif " vs " in val:
                parts = val.split(" vs ")
                if len(parts) == 2:
                    raw_away, raw_home = parts[0], parts[1]

        # Normalization
        away_norm = normalize_theover_team_for_ingest(raw_away, league)
        home_norm = normalize_theover_team_for_ingest(raw_home, league)

        # Pick
        raw_pick = str(row[col_pick]) if col_pick else ""
        pick_type = pick_type_hint

        # Heuristic for pick type if UNKNOWN
        if pick_type == "UNKNOWN":
            if "OVER" in raw_pick.upper() or "UNDER" in raw_pick.upper():
                pick_type = "TOTAL"
            else:
                pick_type = "SIDE"

        line_val = None
        if col_line:
            try:
                line_val = float(row[col_line])
            except:
                pass

        # If TOTAL, try to extract line from pick if line_val is missing
        if pick_type == "TOTAL":
            match = re.search(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', raw_pick, re.IGNORECASE)
            if match:
                if line_val is None:
                    line_val = float(match.group(2))
                raw_pick = match.group(1).upper() # OVER or UNDER

        # Model & Hit Rate
        source_model = str(row[col_model]) if col_model else "TheOver"
        hit_rate = 0.0
        if col_hit_rate:
            try:
                raw_hr = str(row[col_hit_rate]).replace("%", "")
                hit_rate = float(raw_hr)
                if hit_rate > 1.0: hit_rate /= 100.0
            except:
                pass

        canon_key = generate_canonical_key(league, date_val, away_norm, home_norm)

        rec = {
            "theover_key": canon_key,
            "league": league,
            "date_local": date_val,
            "away_norm": away_norm,
            "home_norm": home_norm,
            "theover_pick": raw_pick,
            "theover_market_type": pick_type,
            "theover_line": line_val,
            "theover_model": source_model,
            "theover_hit_rate": hit_rate,
            "raw_text": str(row.to_dict())
        }
        records.append(rec)

    return pd.DataFrame(records)

def parse_theover_public_betting_text(raw_text: str, pick_type_hint: str = "UNKNOWN") -> pd.DataFrame:
    """
    Parse raw text paste from TheOver.ai.
    Lines often look like:
    "NBA • Today • 7:00 PM" (League/Time line)
    "Team A @ Team B" (Matchup)
    "Over 220.5" or "Team A -5" (Pick)
    "Backed by Atom (68%)" (Model info)
    """
    rows = []

    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

    current_league = "UNKNOWN"
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_away = ""
    current_home = ""

    # State tracking
    i = 0
    while i < len(lines):
        line = lines[i]

        # 1. League/Date detection
        # e.g. "NBA • Today • 7:00 PM" or just "NBA"
        # Heuristics:
        if any(x in line.upper() for x in ["NBA", "NFL", "NHL", "NCAAB", "NCAAF", "CBB", "CFB"]):
            upper = line.upper()
            if "NBA" in upper: current_league = "NBA"
            elif "NFL" in upper: current_league = "NFL"
            elif "NHL" in upper: current_league = "NHL"
            elif "NCAAB" in upper or "CBB" in upper: current_league = "NCAAB"
            elif "NCAAF" in upper or "CFB" in upper: current_league = "NCAAF"

            # TODO: Parse date from "Today", "Tomorrow", or specific date if present?
            # For now, assume Today matches the Master slate date usually.
            i += 1
            continue

        # 2. Matchup detection
        # "Team A @ Team B" or "Team A vs Team B"
        if " @ " in line or " vs " in line:
            splitter = " @ " if " @ " in line else " vs "
            parts = line.split(splitter)
            if len(parts) >= 2:
                current_away = parts[0].strip()
                current_home = parts[1].strip()
                # Clean up if there are odds attached (e.g. "Team A (+100) @ Team B (-120)")
                # Usually TheOver headers are just names
                i += 1
                continue

        # 3. Pick detection
        # "Over 46.5 (-110)" or "Chiefs -3 (-110)"
        # Regex for total
        total_match = re.search(r'^(Over|Under)\s+([0-9]+\.?[0-9]*)', line, re.IGNORECASE)
        pick_type = "UNKNOWN"
        pick_val = None
        pick_line = None

        if total_match:
            pick_type = "TOTAL"
            pick_val = total_match.group(1).upper()
            pick_line = float(total_match.group(2))
        else:
            # Maybe Side?
            # Check for spread/odds patterns?
            # Or assume if it's not metadata, it's a pick.
            # But we need to distinguish from "Backed by..."
            if not line.startswith("Backed by") and not line.startswith("Analyze"):
                pick_type = "SIDE"
                pick_val = line # e.g. "Chiefs -3 (-110)"
                # Clean it
                pick_val = pick_val.split("(")[0].strip()

        if pick_type != "UNKNOWN":
            # Look ahead for "Backed by..."
            model_name = "TheOver"
            hit_rate = 0.0

            # Check next line
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if next_line.startswith("Backed by"):
                    # "Backed by Atom (68%)"
                    m = re.search(r'Backed by (.*?) \((\d+)%\)', next_line)
                    if m:
                        model_name = m.group(1).strip()
                        hit_rate = float(m.group(2)) / 100.0
                    else:
                        # Maybe "Backed by Atom"
                        model_name = next_line.replace("Backed by ", "").strip()
                    i += 1 # Consume this line

            # Determine implied market type if hint provided
            if pick_type_hint != "UNKNOWN":
                market_type = pick_type_hint
            else:
                market_type = pick_type

            # Normalization
            away_norm = normalize_theover_team_for_ingest(current_away, current_league)
            home_norm = normalize_theover_team_for_ingest(current_home, current_league)

            canon_key = generate_canonical_key(current_league, current_date, away_norm, home_norm)

            rows.append({
                "theover_key": canon_key,
                "league": current_league,
                "date_local": current_date,
                "away_norm": away_norm,
                "home_norm": home_norm,
                "theover_pick": pick_val,
                "theover_market_type": market_type,
                "theover_line": pick_line,
                "theover_model": model_name,
                "theover_hit_rate": hit_rate,
                "raw_text": line
            })

        i += 1

    return pd.DataFrame(rows)

def process_theover_inputs(
    totals_file=None,
    sides_file=None,
    totals_paste=None,
    sides_paste=None
) -> pd.DataFrame:
    """
    Main ingestion entry point.
    Merges all inputs into a single dataframe indexed by 'theover_key'.
    Handles multiple picks per game (e.g. Total AND Side).
    """
    dfs = []

    # 1. Totals Ingestion
    if totals_file:
        dfs.append(parse_theover_excel(totals_file, pick_type_hint="TOTAL"))
    elif totals_paste and totals_paste.strip():
        dfs.append(parse_theover_public_betting_text(totals_paste, pick_type_hint="TOTAL"))

    # 2. Sides Ingestion
    if sides_file:
        dfs.append(parse_theover_excel(sides_file, pick_type_hint="SIDE"))
    elif sides_paste and sides_paste.strip():
        dfs.append(parse_theover_public_betting_text(sides_paste, pick_type_hint="SIDE"))

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)

    if combined.empty:
        return combined

    # Deduplication strategy:
    # We might have multiple picks for the same game (Total and Side).
    # We want to keep them distinct.
    # But if we have duplicates for the SAME market type (e.g. 2 totals for same game), pick best hit rate.

    combined = combined.sort_values(by="theover_hit_rate", ascending=False)

    # We need to return a structure that can be easily queried by Master Analysis.
    # Master Analysis iterates by Game.
    # So we want to find all TheOver picks for a given Key.

    # Let's drop exact duplicates of (key, market_type)
    deduped = combined.drop_duplicates(subset=["theover_key", "theover_market_type"], keep="first")

    return deduped
