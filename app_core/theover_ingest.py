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

# Assuming rapidfuzz is available as per requirements
try:
    import rapidfuzz
    from rapidfuzz import process, fuzz
except ImportError:
    rapidfuzz = None
    process = None
    fuzz = None
    logging.getLogger("app_core.theover_ingest").warning("rapidfuzz not installed. Fuzzy matching will be disabled.")

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
            # Ensure messy files are handled with skip_blank_lines and on_bad_lines
            # Enforce utf-8-sig to handle BOM if present
            return pd.read_csv(uploaded_file, on_bad_lines='skip', skip_blank_lines=True, encoding='utf-8-sig')
        except Exception:
            return pd.DataFrame()

def _parse_block_layout(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses "block" style spreadsheets where games are listed in blocks.
    Identifies games by looking for the @ symbol in text lines rather than just row headers.
    Implements a scan window logic (8-10 lines) to find disconnected lines/picks.
    """
    records = []

    # Flatten to list of strings
    rows = df.astype(str).values.tolist()
    # Flatten structure: list of text lines (joining columns)
    text_lines = []
    for r in rows:
        # Join non-empty cells
        line = " ".join([x for x in r if x and x.lower() != 'nan' and x.lower() != 'none']).strip()
        if line:
            text_lines.append(line)

    i = 0
    while i < len(text_lines):
        line = text_lines[i]

        # 1. Find Game Line (contains "@")
        if "@" in line:
            parts = line.split("@")
            if len(parts) >= 2:
                away_raw = parts[0].strip()
                # Clean home team (remove parens etc)
                home_raw = parts[1].split('(')[0].split('vs')[0].strip()

                # We found a game. Now scan next 10 lines for Pick info.
                found_pick = False

                # Look ahead window
                for j in range(1, 11):
                    if i + j >= len(text_lines):
                        break

                    next_line = text_lines[i + j]

                    # Regex for Over/Under
                    # Matches "Over 45.5" or just "45.5" then "Over" logic would require state,
                    # but typically block format has "Over 55.5" or "55.5 Over" or similar.
                    ou_match = re.search(r'(Over|Under)\s*(\d+\.?\d*)', next_line, re.IGNORECASE)
                    if ou_match:
                        records.append({
                            "HOMETEAM": home_raw,
                            "AWAYTEAM": away_raw,
                            "PICK": f"{ou_match.group(1)} {ou_match.group(2)}",
                            "MARKET": "TOTAL",
                            "LEAGUE": "UNKNOWN"
                        })
                        found_pick = True
                        break # Found the total for this game

        i += 1

    return pd.DataFrame(records)

def parse_theover_csv(uploaded_file) -> pd.DataFrame:
    """
    Unified parser that handles:
    1. Standard Table (TotalsRaw.csv, Table1.csv)
    2. Vertical/Semi-structured layouts (Sheet1.csv)
    3. Header-agnostic Column Mapping
    """
    df = load_theover_file(uploaded_file)
    if df.empty:
        return df

    # Normalize Headers: Uppercase, Strip
    df.columns = [str(c).strip().upper() for c in df.columns]

    # Priorities for mapping to ensure correct assignment
    # (Target, List of Keywords)
    mappings = [
        ("LEAGUE", ["LEAGUE"]),
        ("HOMETEAM", ["HOMETEAM", "HOME", "TEAM1", "TEAM 1"]),
        ("AWAYTEAM", ["AWAYTEAM", "AWAY", "TEAM2", "TEAM 2"]),
        ("WINPROBABILITY", ["WINPROBABILITY", "PROB", "HITRATE", "SCORE"]),
        ("PICK", ["PICK"])
    ]

    # Robust Coalescing Logic:
    # Identify all columns that match keywords for a target, then coalesce them.
    # This handles mixed-schema files or files with alternative headers.

    for target, keywords in mappings:
        matching_cols = []
        for col in df.columns:
            # Check for keyword match
            if any(k in col for k in keywords):
                matching_cols.append(col)

        if matching_cols:
            # Combine first found column with others as fallback
            combined = df[matching_cols[0]]
            for other_col in matching_cols[1:]:
                combined = combined.combine_first(df[other_col])

            # Assign to standardized target column
            df[target] = combined

    # Check for Standard Format (post-mapping)
    required_std = {"HOMETEAM", "AWAYTEAM", "PICK"}

    # Validation Logic: Only return if at least 2 key columns are present.
    # If standard columns are missing, we MUST fallback to block parser.
    if len(required_std.intersection(set(df.columns))) >= 2:
        # Ensure 'LEAGUE' column exists
        if "LEAGUE" not in df.columns:
            df["LEAGUE"] = "UNKNOWN"
        return df

    # Fallback: Block Parser for Sheet1 style (Triggered if <2 standard cols found)
    parsed_block = _parse_block_layout(df)
    if not parsed_block.empty:
        return parsed_block

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
    master_teams_norm_map = {}
    master_team_names = []

    if games:
        for g in games:
            h = g.get("home_team", "")
            a = g.get("away_team", "")
            if h:
                # Normalization: Convert all incoming team names to UPPERCASE and strip extra spaces
                # This ensures the canonical name is used for matching.
                h_norm = TeamNameMatcher.normalize(h).upper().strip()
                if h_norm not in master_teams_norm_map:
                    master_teams_norm_map[h_norm] = []
                    master_team_names.append(h_norm)
                master_teams_norm_map[h_norm].append(g)
            if a:
                a_norm = TeamNameMatcher.normalize(a).upper().strip()
                if a_norm not in master_teams_norm_map:
                    master_teams_norm_map[a_norm] = []
                    master_team_names.append(a_norm)
                master_teams_norm_map[a_norm].append(g)

    # Normalize column names one last time to be safe
    df.columns = [str(c).upper().strip() for c in df.columns]

    for _, row in df.iterrows():
        # League normalization
        raw_league = str(row.get("LEAGUE", "UNKNOWN")).strip().upper()

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
        if not csv_home or not csv_away or csv_home.lower() == "nan" or csv_away.lower() == "nan":
            continue

        # --- ADVANCED FUZZY MATCHING ---
        matched_game_obj = None
        match_confidence = 0.0
        match_status = "FAIL"
        closest_matches = []

        # Helper to match a single team name
        def match_single_team(name_raw, candidates):
            if not name_raw or not candidates:
                return None, 0.0, []

            # Normalization: Convert to UPPERCASE as requested
            norm = TeamNameMatcher.normalize(name_raw).upper().strip()

            # 1. Exact Match
            if norm in candidates:
                return norm, 100.0, []

            # 2. Fuzzy Match (ExtractOne) with token_set_ratio
            if process:
                # Use score_cutoff=75.0 to enforce strict matching as requested
                # Using token_set_ratio which handles "Utah" -> "Utah Jazz" well
                res = process.extractOne(norm, candidates, scorer=fuzz.token_set_ratio, score_cutoff=75.0)
                if res:
                    match_str, score, _ = res
                    # Collect debug top 3 for transparency
                    top3 = process.extract(norm, candidates, scorer=fuzz.token_set_ratio, limit=3)
                    top3_fmt = [f"{m} ({s:.1f})" for m, s, _ in top3]
                    return match_str, score, top3_fmt

            return None, 0.0, []

        if master_team_names:
            # Match Home Team
            h_match, h_score, h_top3 = match_single_team(csv_home, master_team_names)

            # Match Away Team
            a_match, a_score, a_top3 = match_single_team(csv_away, master_team_names)

            # Check Thresholds (75%)
            if h_score >= 75.0 and a_score >= 75.0:
                # Find the intersection of games
                h_games = master_teams_norm_map.get(h_match, [])
                a_games = master_teams_norm_map.get(a_match, [])

                # Use ID or date matching if possible, but intersection is robust for daily slates
                common_games = [g for g in h_games if g in a_games]

                if common_games:
                    matched_game_obj = common_games[0]
                    match_confidence = (h_score + a_score) / 2.0
                    match_status = "MATCH"
                else:
                     match_status = "MISMATCH_PAIR"

            closest_matches = list(set(h_top3 + a_top3))[:3]

        # Logging
        stats_collector.append({
            "csv_home": csv_home,
            "csv_away": csv_away,
            "matched_home": matched_game_obj.get("home_team") if matched_game_obj else None,
            "matched_away": matched_game_obj.get("away_team") if matched_game_obj else None,
            "confidence": f"{match_confidence:.2f}",
            "status": match_status,
            "closest_matches": "; ".join(closest_matches)
        })

        # --- KEY GENERATION ---
        # Critical: Use matched canonical names if available
        if matched_game_obj:
            league = matched_game_obj.get("league", league)
            date_val = matched_game_obj.get("commence_date_local") or slate_date
            home_code = team_code_for_league(league, matched_game_obj.get("home_team"))
            away_code = team_code_for_league(league, matched_game_obj.get("away_team"))
        else:
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

        # Hit Rate
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

        # Market Type
        market_raw = str(row.get("MARKET", pick_type_default)).upper()
        final_pick_type = pick_type_default
        final_pick_val = raw_pick

        if "TOTAL" in market_raw:
             final_pick_type = "TOTAL"
        elif "SPREAD" in market_raw or "SIDE" in market_raw:
             final_pick_type = "SIDE"

        # Line Extraction
        if line_val is None:
            match = re.search(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', raw_pick, re.IGNORECASE)
            if match:
                line_val = float(match.group(2))
                if final_pick_type == "TOTAL":
                    final_pick_val = match.group(1).upper()
            else:
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

        if any(x in line.upper() for x in ["NBA", "NFL", "NHL", "NCAAB", "NCAAF", "CBB", "CFB"]):
            upper = line.upper()
            if "NBA" in upper: current_league = "NBA"
            elif "NFL" in upper: current_league = "NFL"
            elif "NHL" in upper: current_league = "NHL"
            elif "NCAAB" in upper or "CBB" in upper: current_league = "NCAAB"
            elif "NCAAF" in upper or "CFB" in upper: current_league = "NCAAF"
            i += 1
            continue

        if " @ " in line or " vs " in line:
            splitter = " @ " if " @ " in line else " vs "
            parts = line.split(splitter)
            if len(parts) >= 2:
                current_away = parts[0].strip()
                current_home = parts[1].strip()
                i += 1
                continue

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
        "unmatched_examples": [],
        "matched_rows": 0,
        "unmatched_rows": 0
    }

    if games is None:
        logger.warning("No Master Schedule (games) provided to process_theover_inputs. Fuzzy matching will be skipped.")

    matching_logs = []

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

    if totals_paste and totals_paste.strip():
        dfs.append(parse_theover_public_betting_text(totals_paste, pick_type_hint="TOTAL"))

    if sides_paste and sides_paste.strip():
        dfs.append(parse_theover_public_betting_text(sides_paste, pick_type_hint="SIDE"))

    stats["total_rows"] = stats["raw_totals_rows"] + stats["raw_sides_rows"]

    failed_logs = [l for l in matching_logs if l["status"] == "FAIL" or l["status"] == "MISMATCH_PAIR"]
    stats["unmatched_examples"] = failed_logs[:10]
    stats["unmatched_rows"] = len(failed_logs)
    stats["matched_rows"] = len([l for l in matching_logs if l["status"] == "MATCH"])

    if not dfs:
        return pd.DataFrame(), stats

    combined = pd.concat(dfs, ignore_index=True)

    if combined.empty:
        return combined, stats

    combined = combined.sort_values(by="theover_hit_rate", ascending=False)
    deduped = combined.drop_duplicates(subset=["theover_key", "theover_market_type"], keep="first")

    return deduped, stats
