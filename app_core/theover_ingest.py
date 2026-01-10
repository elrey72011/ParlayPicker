"""
TheOver.ai Public Betting Ingestion Module

Parses raw text from TheOver.ai public betting picks and normalizes it for integration.
"""
import pandas as pd
import re
import logging
from typing import Tuple, Optional, Dict, List
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
    return TeamNameMatcher.normalize(team, strip_mascots=True)

def parse_theover_public_betting_text(raw_text: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse raw text paste from TheOver.ai Public Betting section.

    Expected format examples (flexible):
    - "NBA | LAL vs GSW | LAL -5.5 | 60% | Grok"
    - "NFL: KC @ BUF - Over 48.5 (65% - Nova)"
    - Tab separated table copies

    Returns:
        Tuple[df_totals, df_sides]
    """
    totals_rows = []
    sides_rows = []

    lines = raw_text.strip().split('\n')

    # Regex patterns
    # Pattern for explicit percentage: 65% or 65.5%
    pct_pattern = re.compile(r'(\d+(?:\.\d+)?)%')

    # Pattern for Total: Over/Under followed by number
    total_pattern = re.compile(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', re.IGNORECASE)

    # Pattern for Side: Team name usually implies side if not Over/Under
    # We'll rely on exclusion of Total pattern to identify Sides

    # Star pattern
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

        # 5. Extract Model Name (heuristic: looking for words like Grok, Nova, etc if known, or just text near %)
        # For now, we store raw line or try to find text inside parens if common format
        source_model = "TheOver"
        if "(" in line and ")" in line:
            # check content inside parens
            parens_content = line[line.find("(")+1:line.rfind(")")]
            if "-" in parens_content:
                parts = parens_content.split("-")
                # assume one part is model if not the %
                for p in parts:
                    if "%" not in p:
                        source_model = p.strip()

        # 6. Parse Teams & Pick
        # This is the hardest part without structured input.
        # We assume the line contains team names.
        # Strategy:
        # If Total: "Over 48.5" -> The teams are the rest of the text.
        # If Side: The pick is one of the teams.

        # Normalize line for easier parsing
        clean_line = line.replace("\t", " ").replace("|", " ")

        # Extract pick side/value
        pick_type = "UNKNOWN"
        pick_val = None
        pick_side = None # Over/Under/TeamName

        if total_match:
            pick_type = "TOTAL"
            pick_side = total_match.group(1).upper() # OVER or UNDER
            try:
                pick_val = float(total_match.group(2))
            except ValueError:
                pick_val = 0.0

            # Teams are everything else? This is messy.
            # Ideally we rely on the match logic in main app to associate by fuzzy matching ANY text in the line to a game.
            # But here we need structured columns.

            # Let's try to find "vs" or "@" to split teams
            teams_part = clean_line
            # Remove the total part
            teams_part = teams_part.replace(total_match.group(0), "")
        else:
            pick_type = "SIDE"
            # For sides, usually the team name IS the pick.
            # We need to extract the line if present (e.g. -5.5)
            # Find numbers with +/-
            spread_pattern = re.compile(r'([+-]?\d+\.?\d*)')
            # Warning: this matches everything.
            # Let's look for explicitly signed numbers or numbers at end?

            teams_part = clean_line

        # Generic Team Extraction
        # Split by "vs", "vs.", "@", " v "
        separator = None
        if " @ " in teams_part: separator = " @ "
        elif " vs " in teams_part: separator = " vs "
        elif " v " in teams_part: separator = " v "

        home_team = "Unknown"
        away_team = "Unknown"
        pick_team = "Unknown" # For sides

        if separator:
            parts = teams_part.split(separator)
            if len(parts) >= 2:
                # Away @ Home usually
                # But "Lakers vs Warriors" (Home vs Away? or just Matchup?)
                # Convention: Away @ Home. "Vs" is ambiguous, often Home vs Away.
                t1 = parts[0].strip()
                t2 = parts[1].strip()

                # Cleanup potential noise in team names (like dates, league names)
                # This is a best effort.

                # If Side, we need to know WHICH team is the pick.
                # Usually the line starts with the pick? Or the pick is listed explicitly?
                # "LAL -5.5" -> Pick is LAL.

                if pick_type == "SIDE":
                    # Heuristic: The pick team is often mentioned first or with the spread
                    # We will store the RAW text and let the matcher figure it out?
                    # No, goal 1 says: away_team, home_team, pick_team.

                    # If we can't parse perfectly, we might need the user to format it?
                    # Or we extract all potential team entities.

                    # Let's look for the team name in the line that matches the 'pick' logic.
                    # Actually, if we identify two teams, we can store them.
                    # Pick identification:
                    # If the line says "Lakers -5", Pick is Lakers.
                    pass

                # Assign to columns (best guess)
                # We will normalize these later during merge? No, goal 2 says merge using normalized names.
                # So we must extract them here.

                # Cleaning
                # Remove common prefixes/suffixes from the split parts if they contain metadata
                # e.g. "NBA: Lakers" -> "Lakers"
                if ":" in t1: t1 = t1.split(":")[-1].strip()

                # Remove everything after the team name in t2 (like odds, hit rate)
                # t2 might be "Warriors -5.5 60%..."
                # Heuristic: Stop at first digit? No, 76ers.
                # Stop at first " -" or " +" or "("?

                # We'll just store extracted strings and rely on fuzzy match to VALID teams later.
                # But for the DATAFRAME we need columns.

                away_team = t1
                home_team = t2.split(" ")[0] if " " in t2 else t2 # Very naive

                # Improved T2 extraction: take words until a number or symbol
                t2_clean = []
                for word in t2.split():
                    if any(c.isdigit() for c in word) and word not in ['76ers', '49ers']:
                        break
                    if word.startswith("(") or word.startswith("-") or word.startswith("+"):
                        break
                    t2_clean.append(word)
                home_team = " ".join(t2_clean)

        else:
            # If no separator, maybe it's just "Lakers -5" (Pick only?)
            # Or "Lakers vs Warriors" without spaces?
            # We'll put the whole text in "raw_text" and maybe just the first word as team?
            pass

        # Construct Row Data
        row = {
            "league": league,
            "date_local": datetime.now().strftime("%Y-%m-%d"), # Assume today's picks
            "raw_text": line,
            "source_model": source_model,
            "source_hit_rate": hit_rate,
            "star_rating": has_star,
            "away_team_raw": away_team,
            "home_team_raw": home_team,
            "away_team_norm": normalize_theover_team_name(away_team),
            "home_team_norm": normalize_theover_team_name(home_team)
        }

        if pick_type == "TOTAL":
            row["pick_type"] = "TOTAL"
            row["pick_side"] = pick_side
            row["total_points"] = pick_val
            totals_rows.append(row)
        else:
            # Side
            row["pick_type"] = "SIDE"
            # Attempt to extract pick team from the line
            # If "Lakers -5", pick is Lakers.
            # We'll assume the text starting the line is the pick team if it's not a matchup.
            # If it IS a matchup "Lakers @ Warriors", we need to know who the pick is.
            # Usually format is "Pick Team [Line] ..."
            # If separator exists, it's a matchup line.

            # Let's assume the 'pick_team' is the one that appears first or is explicitly stated?
            # Actually, "Lakers -5" -> Lakers is pick.
            # "Lakers @ Warriors" -> Who is pick? Context needed.
            # If the user pastes "Lakers -5", we know.
            # If the user pastes "Lakers @ Warriors - Lakers -5", we know.

            # Simple heuristic: The team name mentioned near the spread/odds.
            # Or simpler: store the raw teams and in the matching step,
            # if we match the game, we try to see which team in the game matches the start of the line?

            # Pick Parsing Logic
            pick_team = "Unknown"

            if not separator:
                # "Lakers -5.5 ..." -> Pick is Lakers
                words = []
                for w in clean_line.split():
                    if any(c.isdigit() for c in w) and w not in ['76ers', '49ers']: break
                    words.append(w)
                pick_team = " ".join(words)
            else:
                # Matchup line. "Lakers @ Warriors: Lakers -5"
                # Strategy: Look for the team name mentioned AFTER the matchup part, or near the spread.
                # Usually format: "Away @ Home | Pick Team Spread ..."
                # We can try to match the parsed 'away_team' or 'home_team' against the REST of the string.

                # 'teams_part' variable above only holds the matchup part if we split by separator.
                # Let's look at the FULL 'clean_line' again.
                # Remove the matchup part (t1 separator t2) from clean_line to find the pick part?

                matchup_str = f"{t1}{separator}{t2}"
                remainder = clean_line.replace(matchup_str, "")

                # Check if remainder contains Home or Away team name
                t1_norm = TeamNameMatcher.normalize(t1, strip_mascots=True)
                t2_norm = TeamNameMatcher.normalize(t2, strip_mascots=True)

                # Check if remainder contains these tokens
                # Simple check:
                if t1_norm and t1_norm in TeamNameMatcher.normalize(remainder, strip_mascots=True):
                    pick_team = t1
                elif t2_norm and t2_norm in TeamNameMatcher.normalize(remainder, strip_mascots=True):
                    pick_team = t2
                else:
                    # Fallback: Maybe the pick is the first word of the line?
                    # "Lakers -5 (LAL @ GSW)"
                    # Use the first word logic as fallback
                    words = []
                    for w in clean_line.split():
                        if any(c.isdigit() for c in w) and w not in ['76ers', '49ers']: break
                        words.append(w)
                    pick_team = " ".join(words)

            row["pick_team"] = pick_team

            sides_rows.append(row)

    df_totals = pd.DataFrame(totals_rows)
    df_sides = pd.DataFrame(sides_rows)

    return df_totals, df_sides
