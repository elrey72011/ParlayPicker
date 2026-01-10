"""
TheOver.ai Public Betting Ingestion Module

Parses raw text from TheOver.ai public betting picks and normalizes it for integration.
Supports parsing from Excel uploads (Totals/Sides) and text paste fallbacks.
"""
import pandas as pd
import re
import logging
from typing import Tuple, Optional, Dict, List, Any
from datetime import datetime
from app_core.team_name_matcher import TeamNameMatcher

logger = logging.getLogger(__name__)

def normalize_theover_team_name(team: str, league: str = "") -> str:
    """
    Normalize team name using TeamNameMatcher.

    Args:
        team: Raw team name
        league: League identifier (optional, for context if needed later)

    Returns:
        Normalized team string
    """
    # Use robust normalization from TeamNameMatcher
    # We strip mascots for cleaner matching against our internal DB
    # However, if stripping results in empty string (e.g. "Lakers" -> ""), fallback to original
    normalized = TeamNameMatcher.normalize(team, strip_mascots=True)
    if not normalized:
        normalized = TeamNameMatcher.normalize(team, strip_mascots=False)
    return normalized

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
        # Read the first sheet
        df = pd.read_excel(file_buffer)
    except Exception as e:
        logger.error(f"Failed to read Excel file: {e}")
        return pd.DataFrame()

    # Clean column names: lower, strip, spaces to underscores
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Map flexible columns to canonical names
    # Canonical schema: league, away_team, home_team, pick, line, status, source_model, hit_rate

    # 1. League
    league_cols = [c for c in df.columns if c in ("league", "sport")]
    col_league = league_cols[0] if league_cols else None

    # 2. Teams (Matchup or Home/Away)
    # Check for split columns first
    away_candidates = ["away", "away_team", "visitor", "road"]
    home_candidates = ["home", "home_team"]

    col_away = next((c for c in away_candidates if c in df.columns), None)
    col_home = next((c for c in home_candidates if c in df.columns), None)

    col_matchup = None
    if not (col_away and col_home):
        # Look for matchup column
        matchup_candidates = ["matchup", "game", "teams", "match"]
        col_matchup = next((c for c in matchup_candidates if c in df.columns), None)

    # 3. Pick
    pick_candidates = ["pick", "selection", "play", "team", "side", "total_pick"]
    col_pick = next((c for c in pick_candidates if c in df.columns), None)

    # 4. Line/Total
    line_candidates = ["line", "total", "points", "number", "odds", "spread"]
    col_line = next((c for c in line_candidates if c in df.columns), None)

    # 5. Metadata
    status_candidates = ["status"]
    col_status = next((c for c in status_candidates if c in df.columns), None)

    model_candidates = ["source_model", "model", "source", "capper"]
    col_model = next((c for c in model_candidates if c in df.columns), None)

    hit_rate_candidates = ["hit_rate", "win_pct", "rate", "accuracy", "history"]
    col_hit_rate = next((c for c in hit_rate_candidates if c in df.columns), None)

    star_candidates = ["star", "featured", "best_bet"]
    col_star = next((c for c in star_candidates if c in df.columns), None)

    # Processing Rows
    records = []

    for _, row in df.iterrows():
        # League Normalization
        raw_league = str(row[col_league]) if col_league else "UNKNOWN"
        league = raw_league.upper().strip()
        # Map common variants
        if "NBA" in league: league = "NBA"
        elif "NFL" in league: league = "NFL"
        elif "NHL" in league: league = "NHL"
        elif "MLB" in league: league = "MLB"
        elif any(x in league for x in ["NCAAB", "CBB", "COLLEGE BASKETBALL"]): league = "NCAAB"
        elif any(x in league for x in ["NCAAF", "CFB", "COLLEGE FOOTBALL"]): league = "NCAAF"

        # Teams
        raw_away = ""
        raw_home = ""

        if col_away and col_home:
            raw_away = str(row[col_away])
            raw_home = str(row[col_home])
        elif col_matchup:
            val = str(row[col_matchup])
            # Split logic
            if " @ " in val:
                parts = val.split(" @ ")
                raw_away, raw_home = parts[0], parts[1]
            elif " at " in val:
                parts = val.split(" at ")
                raw_away, raw_home = parts[0], parts[1]
            elif " vs " in val:
                # "Home vs Away" is ambiguous but usually means that order?
                # Or "Away vs Home"?
                # Convention: usually neutral or specific league style.
                # We'll assume Away vs Home for consistency unless we know better,
                # or rely on simple split.
                parts = val.split(" vs ")
                if len(parts) == 2:
                    raw_away, raw_home = parts[0], parts[1] # Guess

        # Pick Type Detection
        # If we passed a hint, start with that.
        pick_type = pick_type_hint

        # Refine based on content?
        # If pick column says "Over..." or "Under...", force TOTAL
        raw_pick = str(row[col_pick]) if col_pick else ""
        if "OVER" in raw_pick.upper() or "UNDER" in raw_pick.upper():
            pick_type = "TOTAL"

        # Line Parsing
        line_val = None
        if col_line:
            try:
                line_val = float(row[col_line])
            except:
                pass

        # Always try to clean raw_pick if it contains Over/Under info, even if line_val exists
        if pick_type == "TOTAL":
            match = re.search(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', raw_pick, re.IGNORECASE)
            if match:
                if line_val is None:
                    line_val = float(match.group(2))
                # Normalize pick to just OVER/UNDER
                raw_pick = match.group(1).upper()
            elif "OVER" in raw_pick.upper():
                raw_pick = "OVER"
            elif "UNDER" in raw_pick.upper():
                raw_pick = "UNDER"

        # Metadata extraction
        source_model = str(row[col_model]) if col_model else "TheOver"
        status = str(row[col_status]) if col_status else None

        hit_rate = 0.0
        if col_hit_rate:
            try:
                raw_hr = str(row[col_hit_rate]).replace("%", "")
                hit_rate = float(raw_hr)
                if hit_rate > 1.0: hit_rate /= 100.0
            except:
                pass

        is_star = False
        if col_star:
            val = str(row[col_star]).lower()
            if val in ["true", "1", "yes", "star"]:
                is_star = True

        # Construct Record
        rec = {
            "league": league,
            "date_local": datetime.now().strftime("%Y-%m-%d"),
            "raw_text": str(row.to_dict()),
            "source_model": source_model,
            "source_hit_rate": hit_rate,
            "star_rating": is_star,
            "status": status,
            "pick_type": pick_type,
            "away_team_raw": raw_away,
            "home_team_raw": raw_home,
            "away_team_norm": normalize_theover_team_name(raw_away, league),
            "home_team_norm": normalize_theover_team_name(raw_home, league),
        }

        if pick_type == "TOTAL":
            rec["pick_side"] = raw_pick.upper() # Should be OVER or UNDER
            rec["total_points"] = line_val
            rec["pick_team"] = None
        else:
            rec["pick_team"] = raw_pick # Team Name or indicator
            rec["pick_line"] = line_val
            rec["pick_side"] = None

        # Calculate Sort Score (Quality)
        # Priority: Star (1000) + Hit Rate (0-1) + 1 if recent?
        score = 0.0
        if is_star: score += 1000.0
        if hit_rate: score += (hit_rate * 100.0)
        rec["quality_score"] = score

        records.append(rec)

    return pd.DataFrame(records)

def parse_theover_public_betting_text(raw_text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse raw text paste from TheOver.ai Public Betting section.

    ... (Existing Implementation Kept for Fallback) ...
    """
    totals_rows = []
    sides_rows = []

    lines = raw_text.strip().split('\n')

    # Regex patterns
    pct_pattern = re.compile(r'(\d+(?:\.\d+)?)%')
    total_pattern = re.compile(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', re.IGNORECASE)
    star_pattern = re.compile(r'⭐')

    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue

        # 1. Extract Hit Rate
        hit_rate_match = pct_pattern.search(line)
        hit_rate = None
        if hit_rate_match:
            try:
                hit_rate = float(hit_rate_match.group(1)) / 100.0
            except ValueError:
                pass

        # 2. Extract Star
        has_star = bool(star_pattern.search(line))

        # 3. Detect Pick Type (Total vs Side)
        total_match = total_pattern.search(line)

        # 4. Extract League (Basic heuristic, optional)
        league = "UNKNOWN"
        line_upper = line.upper()
        if "NBA" in line_upper: league = "NBA"
        elif "NFL" in line_upper: league = "NFL"
        elif "NCAAB" in line_upper or "CBB" in line_upper: league = "NCAAB"
        elif "NCAAF" in line_upper or "CFB" in line_upper: league = "NCAAF"
        elif "NHL" in line_upper: league = "NHL"

        # 5. Extract Model Name
        source_model = "TheOver"
        if "(" in line and ")" in line:
            parens_content = line[line.find("(")+1:line.rfind(")")]
            if "-" in parens_content:
                parts = parens_content.split("-")
                for p in parts:
                    if "%" not in p:
                        source_model = p.strip()

        # 6. Parse Teams & Pick
        clean_line = line.replace("\t", " ").replace("|", " ")
        pick_type = "UNKNOWN"
        pick_val = None
        pick_side = None

        if total_match:
            pick_type = "TOTAL"
            pick_side = total_match.group(1).upper()
            try:
                pick_val = float(total_match.group(2))
            except ValueError:
                pick_val = 0.0
            teams_part = clean_line.replace(total_match.group(0), "")
        else:
            pick_type = "SIDE"
            teams_part = clean_line

        separator = None
        if " @ " in teams_part: separator = " @ "
        elif " vs " in teams_part: separator = " vs "
        elif " v " in teams_part: separator = " v "

        home_team = "Unknown"
        away_team = "Unknown"
        pick_team = "Unknown"

        if separator:
            parts = teams_part.split(separator)
            if len(parts) >= 2:
                t1 = parts[0].strip()
                t2 = parts[1].strip()

                if ":" in t1: t1 = t1.split(":")[-1].strip()

                t2_clean = []
                for word in t2.split():
                    if any(c.isdigit() for c in word) and word not in ['76ers', '49ers']:
                        break
                    if word.startswith("(") or word.startswith("-") or word.startswith("+"):
                        break
                    t2_clean.append(word)

                away_team = t1
                home_team = " ".join(t2_clean)

                if pick_type == "SIDE":
                    matchup_str = f"{t1}{separator}{t2}"
                    remainder = clean_line.replace(matchup_str, "")
                    t1_norm = TeamNameMatcher.normalize(t1, strip_mascots=True)
                    t2_norm = TeamNameMatcher.normalize(t2, strip_mascots=True)

                    if t1_norm and t1_norm in TeamNameMatcher.normalize(remainder, strip_mascots=True):
                        pick_team = t1
                    elif t2_norm and t2_norm in TeamNameMatcher.normalize(remainder, strip_mascots=True):
                        pick_team = t2
                    else:
                        words = []
                        for w in clean_line.split():
                            if any(c.isdigit() for c in w) and w not in ['76ers', '49ers']: break
                            words.append(w)
                        pick_team = " ".join(words)
        else:
            if pick_type == "SIDE":
                words = []
                for w in clean_line.split():
                    if any(c.isdigit() for c in w) and w not in ['76ers', '49ers']: break
                    words.append(w)
                pick_team = " ".join(words)

        row = {
            "league": league,
            "date_local": datetime.now().strftime("%Y-%m-%d"),
            "raw_text": line,
            "source_model": source_model,
            "source_hit_rate": hit_rate,
            "star_rating": has_star,
            "away_team_raw": away_team,
            "home_team_raw": home_team,
            "away_team_norm": normalize_theover_team_name(away_team),
            "home_team_norm": normalize_theover_team_name(home_team),
            "quality_score": (1000.0 if has_star else 0.0) + ((hit_rate or 0.0) * 100.0)
        }

        if pick_type == "TOTAL":
            row["pick_type"] = "TOTAL"
            row["pick_side"] = pick_side
            row["total_points"] = pick_val
            totals_rows.append(row)
        else:
            row["pick_type"] = "SIDE"
            row["pick_team"] = pick_team
            sides_rows.append(row)

    df_totals = pd.DataFrame(totals_rows)
    df_sides = pd.DataFrame(sides_rows)

    return df_totals, df_sides

def process_theover_inputs(
    totals_file: Optional[Any] = None,
    sides_file: Optional[Any] = None,
    totals_paste: Optional[str] = None,
    sides_paste: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main ingestion entry point. Prioritizes Excel files, falls back to text.
    Deduplicates and returns normalized DataFrames for Totals and Sides.

    Returns:
        (df_totals, df_sides)
    """
    df_totals = pd.DataFrame()
    df_sides = pd.DataFrame()

    # 1. Totals Ingestion
    if totals_file:
        df_totals = parse_theover_excel(totals_file, pick_type_hint="TOTAL")
    elif totals_paste and totals_paste.strip():
        df_totals, _ = parse_theover_public_betting_text(totals_paste)
        # discard unused sides return from text parser for this slot if strictly totals
        # but text parser returns both. We should probably merge if mixed?
        # For now, assume user pastes Totals into Totals box.

    # 2. Sides Ingestion
    if sides_file:
        df_sides = parse_theover_excel(sides_file, pick_type_hint="SIDE")
    elif sides_paste and sides_paste.strip():
        _, df_s = parse_theover_public_betting_text(sides_paste)
        df_sides = df_s

    # 3. Deduplication and Ranking
    # Rule: Keep BEST record per Game (League + Home + Away)
    def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        # Sort by Quality Score Descending
        df = df.sort_values(by="quality_score", ascending=False)
        # Drop duplicates on Key (League, AwayNorm, HomeNorm)
        # Note: If Home/Away are unknown, we might drop useful rows?
        # But we need to match them to Master Analysis anyway.
        df["match_key"] = df["league"] + "_" + df["away_team_norm"] + "_" + df["home_team_norm"]
        # Filter out empty keys (unparsed teams)
        valid_df = df[df["match_key"].str.contains("Unknown") == False]
        # Dedupe
        deduped = valid_df.drop_duplicates(subset=["match_key"], keep="first")
        return deduped.drop(columns=["match_key"])

    final_totals = _dedupe(df_totals)
    final_sides = _dedupe(df_sides)

    return final_totals, final_sides
