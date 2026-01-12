"""
TheOver.ai Public Betting Ingestion Module

Parses raw text from TheOver.ai public betting picks and normalizes it for integration.
Supports parsing from Excel uploads (Totals/Sides) and text paste fallbacks.
Now uses Canonical Keys for reliable matching via Advanced Fuzzy Matching.
"""
import pandas as pd
import re
import logging
from typing import Tuple, Optional, Dict, List, Any, Union
from datetime import datetime
from app_core.feature_processing import robust_normalize_team
from app_core.kalshi_integrator import team_code_for_league
from app_core.team_name_matcher import TeamNameMatcher

try:
    import rapidfuzz
    from rapidfuzz import process, fuzz
except ImportError:
    rapidfuzz = None
    process = None
    fuzz = None

logger = logging.getLogger("app_core.theover_ingest")

def generate_canonical_key(league: str, date_str: str, away_code: str, home_code: str) -> str:
    """
    Generates a canonical key for matching against the master schedule.
    Format: {league}|{away_code}|{home_code}|{local_date}
    """
    return f"{league}|{away_code}|{home_code}|{date_str}"

def load_theover_file(uploaded_file):
    """
    Robust file loader that attempts Excel first, then CSV.
    Never attempts UTF-8 decode on XLSX to prevent binary errors.
    """
    if uploaded_file is None:
        return pd.DataFrame()

    # Reset pointer if possible
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    # 1. Try Excel (openpyxl)
    try:
        return pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception:
        # 2. Try CSV
        try:
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
        except Exception:
            return pd.DataFrame()

def _parse_vertical_layout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses "vertical chunk" style spreadsheets where games are listed in blocks.
    Identifies games via keywords like '@' or 'vs' and associated 'Over'/'Under' keywords.
    """
    records = []

    # Iterate through all rows and scan for game patterns
    # A block typically looks like:
    # TeamA
    # TeamB (or @ TeamB)
    # Pick info

    # Convert entire dataframe to string for easier searching
    # Or iterate row by row. Since structure varies, row by row state machine is safer.

    current_game = None

    # Heuristic: convert df to list of lists
    rows = df.astype(str).values.tolist()

    for row in rows:
        row_text = " ".join([r for r in row if r and r.lower() != 'nan' and r.lower() != 'none']).strip()
        if not row_text:
            continue

        # Check for matchup
        if " @ " in row_text or " vs " in row_text:
            # Likely a game line
            parts = re.split(r" @ | vs ", row_text, flags=re.IGNORECASE)
            if len(parts) >= 2:
                # Potential game found
                away_raw = parts[0].strip()
                # Remove common garbage from end of home team if present
                home_raw = parts[1].split('(')[0].strip()

                # Store potential game context
                current_game = {
                    "away": away_raw,
                    "home": home_raw,
                }
                continue

        # Check for Pick info if we have a current game
        if current_game:
            # Look for Over/Under or Spread clues
            # Regex for Over/Under
            ou_match = re.search(r'(Over|Under)\s*(\d+\.?\d*)', row_text, re.IGNORECASE)
            if ou_match:
                records.append({
                    "HomeTeam": current_game["home"],
                    "AwayTeam": current_game["away"],
                    "Pick": f"{ou_match.group(1)} {ou_match.group(2)}",
                    "Market": "TOTAL",
                    "League": "UNKNOWN" # Will need external context or inference
                })
                # Don't clear current_game yet, might have multiple picks?
                # Usually vertical format is one pick per block.
                # Let's keep it until new game found or explicit clear?
                # Safer to keep.
                continue

            # Look for spread/ML clues (Team name + line)
            # This is harder without known team names.
            # But if the row starts with one of the teams?
            if current_game["home"] in row_text or current_game["away"] in row_text:
                 records.append({
                    "HomeTeam": current_game["home"],
                    "AwayTeam": current_game["away"],
                    "Pick": row_text, # Heuristic
                    "Market": "SIDE",
                    "League": "UNKNOWN"
                })

    return pd.DataFrame(records)

def parse_theover_csv(uploaded_file) -> pd.DataFrame:
    """
    Unified parser that handles:
    1. Standard Table (TotalsRaw.csv, Table1.csv)
    2. Vertical/Semi-structured layouts
    """
    df = load_theover_file(uploaded_file)
    if df.empty:
        return df

    # Normalize Headers: Uppercase, Strip
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Check for Standard Format
    required_std = {"HOMETEAM", "AWAYTEAM", "PICK"}
    if required_std.intersection(set(df.columns)):
        # Ensure 'LEAGUE' column exists
        if "LEAGUE" not in df.columns:
            df["LEAGUE"] = "UNKNOWN"
        return df

    # Fallback: Vertical Chunk Parser
    # If we don't have standard headers, try to parse the content
    parsed_vertical = _parse_vertical_layout(df)
    if not parsed_vertical.empty:
        # Uppercase the new headers
        parsed_vertical.columns = [str(c).strip().upper() for c in parsed_vertical.columns]
        return parsed_vertical

    return pd.DataFrame()

def _transform_theover_df(df: pd.DataFrame, pick_type_default: str, games: List[Dict[str, Any]], stats_collector: List[Dict]) -> pd.DataFrame:
    """
    Transforms the parsed TheOver dataframe into standardized records.
    Uses 'games' (Master Schedule) for Advanced Fuzzy Matching.
    """
    if df.empty:
        return pd.DataFrame()

    records = []
    # Use the current slate date from the system, or today's date if not passed.
    slate_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Transforming TheOver DataFrame ({pick_type_default}) with {len(df)} rows.")

    # Pre-process master schedule for matching
    # Create list of (Home, Away) tuples from master schedule
    master_schedule_tuples = []
    game_lookup = {} # (HomeNorm, AwayNorm) -> GameDict

    if games:
        for g in games:
            h = g.get("home_team", "")
            a = g.get("away_team", "")
            if h and a:
                # Store original names in tuple
                master_schedule_tuples.append((h, a))
                # Store normalized key for lookup
                key = (TeamNameMatcher.normalize(h), TeamNameMatcher.normalize(a))
                game_lookup[key] = g

    # Normalize column names one last time to be safe
    df.columns = [str(c).upper().strip() for c in df.columns]

    for _, row in df.iterrows():
        # League normalization
        raw_league = str(row.get("LEAGUE", "UNKNOWN")).strip().upper()

        # If league is unknown, try to infer? Or just pass "UNKNOWN" and let matcher handle
        league = raw_league # Default
        if "NBA" in raw_league: league = "NBA"
        elif "NFL" in raw_league: league = "NFL"
        elif "NHL" in raw_league: league = "NHL"
        elif "MLB" in raw_league: league = "MLB"
        elif any(x in raw_league for x in ["NCAAB", "CBB", "COLLEGE BASKETBALL"]): league = "NCAAB"
        elif any(x in raw_league for x in ["NCAAF", "CFB", "COLLEGE FOOTBALL"]): league = "NCAAF"

        # Extract Team Names
        csv_home = str(row.get("HOMETEAM", "")).strip()
        csv_away = str(row.get("AWAYTEAM", "")).strip()

        # If teams are missing, skip
        if not csv_home or not csv_away or csv_home == "nan" or csv_away == "nan":
            continue

        # --- ADVANCED FUZZY MATCHING ---
        matched_game_obj = None
        match_confidence = 0.0
        match_status = "FAIL"

        if master_schedule_tuples:
            # Use TeamNameMatcher to find best game
            # Threshold set to 0.70 to catch "Central Florida" -> "UCF"
            matched_tuple = TeamNameMatcher.match_game(csv_home, csv_away, master_schedule_tuples, threshold=0.70)

            if matched_tuple:
                # Retrieve the full game object
                h_matched, a_matched = matched_tuple
                key = (TeamNameMatcher.normalize(h_matched), TeamNameMatcher.normalize(a_matched))
                # Try finding it; if fuzzy matching swapped home/away, check both orientations
                matched_game_obj = game_lookup.get(key)
                if not matched_game_obj:
                     # Try swap
                     key_swap = (TeamNameMatcher.normalize(a_matched), TeamNameMatcher.normalize(h_matched))
                     matched_game_obj = game_lookup.get(key_swap)

                if matched_game_obj:
                    match_status = "MATCH"
                    # Calculate rough confidence score for logging (average of home/away fuzzy ratios)
                    s1 = TeamNameMatcher.similarity_score(TeamNameMatcher.normalize(csv_home), TeamNameMatcher.normalize(h_matched))
                    s2 = TeamNameMatcher.similarity_score(TeamNameMatcher.normalize(csv_away), TeamNameMatcher.normalize(a_matched))
                    match_confidence = (s1 + s2) / 2.0

        # Logging
        stats_collector.append({
            "csv_home": csv_home,
            "csv_away": csv_away,
            "matched_home": matched_game_obj.get("home_team") if matched_game_obj else None,
            "matched_away": matched_game_obj.get("away_team") if matched_game_obj else None,
            "confidence": f"{match_confidence:.2f}",
            "status": match_status
        })

        # --- KEY GENERATION ---
        if matched_game_obj:
            # Use Canonical Data from App
            # Use App's league (should be consistent, but trust App)
            league = matched_game_obj.get("league", league)
            # Use date from matched game (local date string)
            # The app games usually have 'commence_time_iso_local' or similar, but key uses YYYY-MM-DD
            # 'commence_date_local' is standard in this app
            date_val = matched_game_obj.get("commence_date_local") or slate_date

            # Use App's Team Codes logic (using the canonical names)
            home_code = team_code_for_league(league, matched_game_obj.get("home_team"))
            away_code = team_code_for_league(league, matched_game_obj.get("away_team"))

        else:
            # Fallback: Use CSV data (Legacy/Unmatched behavior)
            # Use current slate date as fallback
            date_val = slate_date
            home_code = team_code_for_league(league, csv_home)
            away_code = team_code_for_league(league, csv_away)

        canon_key = generate_canonical_key(league, date_val, away_code, home_code)

        # Pick & Line
        raw_pick = str(row.get("PICK", "")).strip()

        line_val = None
        try:
            l_raw = row.get("LINE")
            if pd.notnull(l_raw):
                line_val = float(l_raw)
        except (ValueError, TypeError):
            pass

        # Hit Rate / Win Probability
        hit_rate = 0.0
        try:
            wp = row.get("WINPROBABILITY")
            if wp is not None and pd.notnull(wp):
                s_wp = str(wp).replace("%", "").strip()
                if s_wp:
                    hit_rate = float(s_wp)
                    if hit_rate > 1.0: hit_rate /= 100.0
        except (ValueError, TypeError):
            pass

        # Market Type Determination
        market_raw = str(row.get("MARKET", pick_type_default)).upper()

        final_pick_type = pick_type_default
        final_pick_val = raw_pick

        # Basic parsing logic similar to before
        if "TOTAL" in market_raw:
             final_pick_type = "TOTAL"
        elif "SPREAD" in market_raw or "SIDE" in market_raw:
             final_pick_type = "SIDE"

        # If line is missing, try to extract from pick string
        if line_val is None:
            match = re.search(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', raw_pick, re.IGNORECASE)
            if match:
                line_val = float(match.group(2))
                if final_pick_type == "TOTAL":
                    final_pick_val = match.group(1).upper()
            else:
                # Spread extraction? e.g. "Team -5.5"
                match_spread = re.search(r'(-?\d+\.?\d*)$', raw_pick)
                if match_spread:
                     line_val = float(match_spread.group(1))

        records.append({
            "theover_key": canon_key,
            "league": league,
            "date_local": date_val,
            "away_code": away_code,
            "home_code": home_code,
            "theover_pick": final_pick_val,
            "theover_market_type": final_pick_type,
            "theover_line": line_val,
            "theover_model": "TheOver",
            "theover_hit_rate": hit_rate,
            "raw_text": str(row.to_dict()),
            "away_team_raw": csv_away,
            "home_team_raw": csv_home
        })

    return pd.DataFrame(records)

def parse_theover_public_betting_text(raw_text: str, pick_type_hint: str = "UNKNOWN") -> pd.DataFrame:
    """
    Parse raw text paste from TheOver.ai.
    """
    rows = []
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]

    current_league = "UNKNOWN"
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_away = ""
    current_home = ""

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
                pick_val = line.split("(")[0].strip()

        if pick_type != "UNKNOWN":
            model_name = "TheOver"
            hit_rate = 0.0

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

            # Use team_code_for_league for consistency
            away_code = team_code_for_league(current_league, current_away)
            home_code = team_code_for_league(current_league, current_home)

            canon_key = generate_canonical_key(current_league, current_date, away_code, home_code)

            rows.append({
                "theover_key": canon_key,
                "league": current_league,
                "date_local": current_date,
                "away_code": away_code,
                "home_code": home_code,
                "theover_pick": pick_val,
                "theover_market_type": market_type,
                "theover_line": pick_line,
                "theover_model": model_name,
                "theover_hit_rate": hit_rate,
                "raw_text": line,
                "away_team_raw": current_away,
                "home_team_raw": current_home
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
    Returns (DataFrame, stats_dict).
    """
    dfs = []
    stats = {
        "raw_total_rows": 0,
        "raw_sides_rows": 0,
        "raw_totals_rows": 0,
        "files_processed": [],
        "unmatched_examples": [], # For UI
        "matched_rows": 0,
        "unmatched_rows": 0
    }

    # Validation
    if games is None:
        logger.warning("No Master Schedule (games) provided to process_theover_inputs. Fuzzy matching will be skipped.")

    matching_logs = []

    # Process Files
    # We use our unified parser for both "Totals" and "Sides" slots, as file formats are erratic

    if totals_file:
        try:
            df = parse_theover_csv(totals_file)
            stats["raw_totals_rows"] = len(df)
            if not df.empty:
                processed = _transform_theover_df(df, "TOTAL", games, matching_logs)
                if not processed.empty:
                    dfs.append(processed)
            stats["files_processed"].append("totals_file")
        except Exception as e:
            logger.error(f"Error processing Totals file: {e}", exc_info=True)

    if sides_file:
        try:
            df = parse_theover_csv(sides_file)
            stats["raw_sides_rows"] = len(df)
            if not df.empty:
                processed = _transform_theover_df(df, "SIDE", games, matching_logs)
                if not processed.empty:
                    dfs.append(processed)
            stats["files_processed"].append("sides_file")
        except Exception as e:
            logger.error(f"Error processing Sides file: {e}", exc_info=True)

    # Process Text Pastes
    if totals_paste and totals_paste.strip():
        # Text paste parser handles its own key generation (simple) -
        # Refactoring text paste parser to use fuzzy matching is out of scope for "CSV Fix",
        # but we should at least acknowledge it exists.
        # The prompt specifically mentioned CSV ingestion fixes.
        dfs.append(parse_theover_public_betting_text(totals_paste, pick_type_hint="TOTAL"))

    if sides_paste and sides_paste.strip():
        dfs.append(parse_theover_public_betting_text(sides_paste, pick_type_hint="SIDE"))

    # Compute Stats
    stats["total_rows"] = stats["raw_totals_rows"] + stats["raw_sides_rows"]

    # Filter matching logs for failures
    failed_logs = [l for l in matching_logs if l["status"] == "FAIL"]
    stats["unmatched_examples"] = failed_logs[:10] # Top 10 failures
    stats["unmatched_rows"] = len(failed_logs)
    stats["matched_rows"] = len([l for l in matching_logs if l["status"] == "MATCH"])

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
