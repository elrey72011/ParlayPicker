"""
TheOver.ai Public Betting Ingestion Module

Parses raw text from TheOver.ai public betting picks and normalizes it for integration.
Supports parsing from Excel uploads (Totals/Sides) and text paste fallbacks.
Now uses Canonical Keys for reliable matching.
"""
import pandas as pd
import re
import logging
from typing import Tuple, Optional, Dict, List, Any, Union
from pathlib import Path
from datetime import datetime
from app_core.feature_processing import robust_normalize_team

logger = logging.getLogger("app_core.theover_ingest")

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
    Generates a canonical key for matching against the master schedule.
    Format: {league}|{local_date}|{away_norm}|{home_norm}

    This must match the key generation logic in streamlit_app.py 'Run Master Analysis' loop.
    Ensure 'away_norm' and 'home_norm' have been normalized via robust_normalize_team.
    """
    return f"{league}|{date_str}|{away_norm}|{home_norm}"

def _read_excel_safe(path: Union[str, Any], sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except ImportError as exc:
        logger.error("Failed to read Excel file: %s", exc)
        return pd.DataFrame()
    except Exception as exc:
        logger.error("Failed to read Excel file %s sheet %s: %s", str(path)[:50], sheet_name, exc)
        return pd.DataFrame()

def load_theover_sides(path: Union[str, Any]) -> pd.DataFrame:
    df = _read_excel_safe(path, sheet_name="Table1")
    logger.info("TheOver Sides Table1 raw shape: %s | columns: %s",
                getattr(df, "shape", None),
                list(df.columns) if hasattr(df, "columns") else None)

    if df.empty:
        return df

    # Relaxed validation: Just check for critical columns for keys
    required_critical = ["League", "HomeTeam", "AwayTeam"]
    missing = [c for c in required_critical if c not in df.columns]
    if missing:
        logger.error("Table1 missing critical columns: %s", missing)
        return pd.DataFrame()

    df = df.copy()
    # Ensure types for critical columns
    if "League" in df.columns:
        df["League"] = df["League"].astype(str).str.upper()
    if "HomeTeam" in df.columns:
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    if "AwayTeam" in df.columns:
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    return df

def debug_load_theover_totals(totals_path: Union[str, Any]) -> pd.DataFrame:
    try:
        # Direct read using openpyxl, bypassing other checks
        df = pd.read_excel(totals_path, sheet_name="TotalsRaw", engine="openpyxl")
        logger.info("DEBUG TheOver TotalsRaw shape: %s | columns: %s",
                    df.shape, list(df.columns))
        return df
    except Exception as e:
        logger.error(f"DEBUG TheOver load failed: {e}")
        return pd.DataFrame()

def load_theover_totals(path: Union[str, Any]) -> pd.DataFrame:
    df = _read_excel_safe(path, sheet_name="TotalsRaw")
    logger.info("TheOver TotalsRaw raw shape: %s | columns: %s",
                getattr(df, "shape", None),
                list(df.columns) if hasattr(df, "columns") else None)

    if df.empty:
        return df

    # Relaxed validation
    required_critical = ["League", "HomeTeam", "AwayTeam"]
    missing = [c for c in required_critical if c not in df.columns]
    if missing:
        logger.error("TotalsRaw missing critical columns: %s", missing)
        return pd.DataFrame()

    df = df.copy()
    if "League" in df.columns:
        df["League"] = df["League"].astype(str).str.upper()
    if "HomeTeam" in df.columns:
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    if "AwayTeam" in df.columns:
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
    return df

def _transform_theover_df(df: pd.DataFrame, pick_type_default: str) -> pd.DataFrame:
    """
    Transforms the structured TheOver dataframes (from load_theover_*)
    into the standardized records format expected by the application.
    """
    if df.empty:
        return pd.DataFrame()

    records = []
    # Use the current slate date from the system, or today's date if not passed.
    # The application code generally assumes "today" for the slate being analyzed.
    slate_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Transforming TheOver DataFrame ({pick_type_default}) with {len(df)} rows.")

    for _, row in df.iterrows():
        # League normalization
        raw_league = str(row.get("League", "UNKNOWN")).strip().upper()
        if "NBA" in raw_league: league = "NBA"
        elif "NFL" in raw_league: league = "NFL"
        elif "NHL" in raw_league: league = "NHL"
        elif "MLB" in raw_league: league = "MLB"
        elif any(x in raw_league for x in ["NCAAB", "CBB", "COLLEGE BASKETBALL"]): league = "NCAAB"
        elif any(x in raw_league for x in ["NCAAF", "CFB", "COLLEGE FOOTBALL"]): league = "NCAAF"
        else: league = raw_league

        # Date - usually not in Excel, default to today
        date_val = slate_date

        # Teams
        raw_home = str(row.get("HomeTeam", "")).strip()
        raw_away = str(row.get("AwayTeam", "")).strip()

        if not raw_home or not raw_away:
            continue

        home_norm = normalize_theover_team_for_ingest(raw_home, league)
        away_norm = normalize_theover_team_for_ingest(raw_away, league)

        # Pick & Line
        raw_pick = str(row.get("Pick", "")).strip()

        line_val = None
        try:
            l_raw = row.get("Line")
            if pd.notnull(l_raw):
                line_val = float(l_raw)
        except (ValueError, TypeError):
            pass

        # Hit Rate / Win Probability
        hit_rate = 0.0
        try:
            wp = row.get("WinProbability")
            if wp is not None and pd.notnull(wp):
                s_wp = str(wp).replace("%", "").strip()
                if s_wp:
                    hit_rate = float(s_wp)
                    if hit_rate > 1.0: hit_rate /= 100.0
        except (ValueError, TypeError):
            pass

        # Market Type
        market = str(row.get("Market", pick_type_default)).upper()
        if "TOTAL" in market:
            pick_type = "TOTAL"
            # If line is missing, try to extract from pick
            if line_val is None:
                match = re.search(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', raw_pick, re.IGNORECASE)
                if match:
                    line_val = float(match.group(2))
                    raw_pick = match.group(1).upper()
        elif "SPREAD" in market or "SIDE" in market or "MONEYLINE" in market:
            pick_type = "SIDE"
        else:
            pick_type = pick_type_default

        # Canonical Key
        canon_key = generate_canonical_key(league, date_val, away_norm, home_norm)

        records.append({
            "theover_key": canon_key,
            "league": league,
            "date_local": date_val,
            "away_norm": away_norm,
            "home_norm": home_norm,
            "theover_pick": raw_pick,
            "theover_market_type": pick_type,
            "theover_line": line_val,
            "theover_model": "TheOver",
            "theover_hit_rate": hit_rate,
            "raw_text": str(row.to_dict())
        })

    logger.info(f"TheOver totals rows retained for matching: {len(records)}")
    if records:
        logger.info(f"TheOver totals sample keys: {[r['theover_key'] for r in records[:3]]}")
    else:
        logger.warning(f"TheOver {pick_type_default} produced 0 records after transformation.")

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
        if any(x in line.upper() for x in ["NBA", "NFL", "NHL", "NCAAB", "NCAAF", "CBB", "CFB"]):
            upper = line.upper()
            if "NBA" in upper: current_league = "NBA"
            elif "NFL" in upper: current_league = "NFL"
            elif "NHL" in upper: current_league = "NHL"
            elif "NCAAB" in upper or "CBB" in upper: current_league = "NCAAB"
            elif "NCAAF" in upper or "CFB" in upper: current_league = "NCAAF"
            i += 1
            continue

        # 2. Matchup detection
        if " @ " in line or " vs " in line:
            splitter = " @ " if " @ " in line else " vs "
            parts = line.split(splitter)
            if len(parts) >= 2:
                current_away = parts[0].strip()
                current_home = parts[1].strip()
                i += 1
                continue

        # 3. Pick detection
        total_match = re.search(r'^(Over|Under)\s+([0-9]+\.?[0-9]*)', line, re.IGNORECASE)
        pick_type = "UNKNOWN"
        pick_val = None
        pick_line = None

        if total_match:
            pick_type = "TOTAL"
            pick_val = total_match.group(1).upper()
            pick_line = float(total_match.group(2))
        else:
            if not line.startswith("Backed by") and not line.startswith("Analyze"):
                pick_type = "SIDE"
                pick_val = line
                pick_val = pick_val.split("(")[0].strip()

        if pick_type != "UNKNOWN":
            model_name = "TheOver"
            hit_rate = 0.0

            # Check next line for metadata
            if i + 1 < len(lines):
                next_line = lines[i+1]
                if next_line.startswith("Backed by"):
                    m = re.search(r'Backed by (.*?) \((\d+)%\)', next_line)
                    if m:
                        model_name = m.group(1).strip()
                        hit_rate = float(m.group(2)) / 100.0
                    else:
                        model_name = next_line.replace("Backed by ", "").strip()
                    i += 1

            if pick_type_hint != "UNKNOWN":
                market_type = pick_type_hint
            else:
                market_type = pick_type

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
    sides_paste=None,
    games=None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main ingestion entry point.
    Merges all inputs into a single dataframe indexed by 'theover_key'.
    Handles multiple picks per game (e.g. Total AND Side).
    Returns (DataFrame, stats_dict).
    """
    dfs = []
    stats = {
        "raw_total_rows": 0,
        "raw_sides_rows": 0,
        "raw_totals_rows": 0,
        "files_processed": []
    }

    # Debugging / Loading Totals
    totals_df = pd.DataFrame()
    if totals_file:
        logger.info("DEBUG TheOver totals input provided: %s", totals_file)
        if isinstance(totals_file, (str, Path)):
             logger.info("DEBUG TheOver totals exists: %s", Path(totals_file).exists())
        else:
             logger.info("DEBUG TheOver totals is a file-like object")

        try:
            totals_df = pd.read_excel(totals_file, sheet_name="TotalsRaw", engine="openpyxl")
            logger.info("DEBUG TotalsRaw shape: %s | columns: %s", totals_df.shape, list(totals_df.columns))
            stats["raw_totals_rows"] = len(totals_df)
            stats["files_processed"].append("totals_file")
        except Exception as exc:
            logger.error("DEBUG Failed to read TotalsRaw: %s", exc)
            # Fallback or empty
            totals_df = pd.DataFrame()

    # Debugging / Loading Sides
    sides_df = pd.DataFrame()
    if sides_file:
        logger.info("DEBUG TheOver sides input provided: %s", sides_file)
        if isinstance(sides_file, (str, Path)):
             logger.info("DEBUG TheOver sides exists: %s", Path(sides_file).exists())
        else:
             logger.info("DEBUG TheOver sides is a file-like object")

        try:
            sides_df = pd.read_excel(sides_file, sheet_name="Table1", engine="openpyxl")
            logger.info("DEBUG Table1 shape: %s | columns: %s", sides_df.shape, list(sides_df.columns))
            stats["raw_sides_rows"] = len(sides_df)
            stats["files_processed"].append("sides_file")
        except Exception as exc:
            logger.error("DEBUG Failed to read Table1: %s", exc)
            sides_df = pd.DataFrame()

    total_rows_parsed = len(totals_df)
    # Matching logic using games is "Pending" but we log the counts
    matched_count = 0
    unmatched_count = 0

    logger.info(
        "TheOver Matching Debug\nTotal Rows Parsed: %d\nMatched Games: %d\nUnmatched Games: %d",
        total_rows_parsed,
        matched_count,
        unmatched_count,
    )

    # 1. Excel Ingestion (Strict Sheets)
    # Using the loaded DFs
    if not totals_df.empty:
        try:
            # We assume column names are correct or compatible with _transform_theover_df
            # load_theover_totals previously cleaned them.
            # We can replicate basic cleanup here if needed.
            # _transform_theover_df handles "League", "HomeTeam", "AwayTeam"

            # Ensure critical column types if possible
            if "League" in totals_df.columns:
                totals_df["League"] = totals_df["League"].astype(str).str.upper()
            if "HomeTeam" in totals_df.columns:
                totals_df["HomeTeam"] = totals_df["HomeTeam"].astype(str).str.strip()
            if "AwayTeam" in totals_df.columns:
                totals_df["AwayTeam"] = totals_df["AwayTeam"].astype(str).str.strip()

            processed = _transform_theover_df(totals_df, pick_type_default="TOTAL")
            if not processed.empty:
                dfs.append(processed)
        except Exception as e:
            logger.error(f"Error processing Totals Excel: {e}")

    if not sides_df.empty:
        try:
            if "League" in sides_df.columns:
                sides_df["League"] = sides_df["League"].astype(str).str.upper()
            if "HomeTeam" in sides_df.columns:
                sides_df["HomeTeam"] = sides_df["HomeTeam"].astype(str).str.strip()
            if "AwayTeam" in sides_df.columns:
                sides_df["AwayTeam"] = sides_df["AwayTeam"].astype(str).str.strip()

            processed = _transform_theover_df(sides_df, pick_type_default="SIDE")
            if not processed.empty:
                dfs.append(processed)
        except Exception as e:
            logger.error(f"Error processing Sides Excel: {e}")

    stats["raw_total_rows"] = stats["raw_totals_rows"] + stats["raw_sides_rows"]

    # 2. Text Paste Fallback
    if totals_paste and totals_paste.strip():
        # Note: text paste rows are not counted in "raw_total_rows" for Excel debug parity, or could be added separately
        dfs.append(parse_theover_public_betting_text(totals_paste, pick_type_hint="TOTAL"))

    if sides_paste and sides_paste.strip():
        dfs.append(parse_theover_public_betting_text(sides_paste, pick_type_hint="SIDE"))

    if not dfs:
        return pd.DataFrame(), stats

    combined = pd.concat(dfs, ignore_index=True)

    if combined.empty:
        return combined, stats

    # Deduplication strategy:
    # Prioritize picks with higher hit rate if duplicates exist for same game + market type
    combined = combined.sort_values(by="theover_hit_rate", ascending=False)
    deduped = combined.drop_duplicates(subset=["theover_key", "theover_market_type"], keep="first")

    return deduped, stats
