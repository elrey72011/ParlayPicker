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
from datetime import datetime
from app_core.feature_processing import robust_normalize_team
from app_core.kalshi_integrator import team_code_for_league

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

def load_theover_sides(path: Union[str, Any]) -> pd.DataFrame:
    # Use specific sheet for Sides if possible, but load_theover_file is generic.
    # We will try to load the specific sheet inside a wrapper or modify load_theover_file to take args.
    # The user provided a simple load_theover_file. Let's stick to that pattern but allow sheet specification if it's excel.

    # Actually, the user instruction was specifically to implement load_theover_file.
    # But I need to respect the "Table1" / "TotalsRaw" sheet names if I can.
    # Let's use the logic but adapted to try specific sheets if it IS an excel file.

    df = pd.DataFrame()
    if hasattr(path, "seek"):
        path.seek(0)

    try:
        df = pd.read_excel(path, sheet_name="Table1", engine="openpyxl")
    except Exception:
        try:
            if hasattr(path, "seek"):
                path.seek(0)
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        except Exception:
            try:
                if hasattr(path, "seek"):
                    path.seek(0)
                df = pd.read_csv(path)
            except Exception:
                pass

    logger.info("TheOver Sides rows loaded: %s", len(df))
    return df

def load_theover_totals(path: Union[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame()
    if hasattr(path, "seek"):
        path.seek(0)

    try:
        df = pd.read_excel(path, sheet_name="TotalsRaw", engine="openpyxl")
    except Exception:
        try:
            if hasattr(path, "seek"):
                path.seek(0)
            df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
        except Exception:
            try:
                if hasattr(path, "seek"):
                    path.seek(0)
                df = pd.read_csv(path)
            except Exception:
                pass

    logger.info("TheOver Totals rows loaded: %s", len(df))
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
    slate_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Transforming TheOver DataFrame ({pick_type_default}) with {len(df)} rows.")

    # Determine columns based on pick_type_default
    is_total = pick_type_default == "TOTAL"

    for _, row in df.iterrows():
        # League normalization
        raw_league = str(row.get("League", "UNKNOWN")).strip().upper()
        if not raw_league or raw_league == "NAN":
            continue

        if "NBA" in raw_league: league = "NBA"
        elif "NFL" in raw_league: league = "NFL"
        elif "NHL" in raw_league: league = "NHL"
        elif "MLB" in raw_league: league = "MLB"
        elif any(x in raw_league for x in ["NCAAB", "CBB", "COLLEGE BASKETBALL"]): league = "NCAAB"
        elif any(x in raw_league for x in ["NCAAF", "CFB", "COLLEGE FOOTBALL"]): league = "NCAAF"
        else: league = raw_league

        # Date - usually not in Excel, default to today
        # Check if row has Date or timestamp
        date_val = slate_date
        # If TheOver provides no timestamp, we rely on fuzzy date matching in the consumer

        # Codes / Teams
        # For Totals: Use AwayKalshi / HomeKalshi if available
        # For Sides: Use AwayTeam / HomeTeam and normalize to code

        home_code = ""
        away_code = ""

        if is_total:
            # Columns: League, HomeTeam, AwayTeam, HomeKalshi, AwayKalshi, Pick, PickCode, Line, WinProbability, Market, Matchup
            hk = str(row.get("HomeKalshi", "")).strip().upper()
            ak = str(row.get("AwayKalshi", "")).strip().upper()
            if hk and hk != "NAN": home_code = hk
            if ak and ak != "NAN": away_code = ak

        # Fallback to team names if codes missing (or for Sides)
        if not home_code:
            ht = str(row.get("HomeTeam", "")).strip()
            if ht and ht != "NAN":
                home_code = team_code_for_league(league, ht)

        if not away_code:
            at = str(row.get("AwayTeam", "")).strip()
            if at and at != "NAN":
                away_code = team_code_for_league(league, at)

        if not home_code or not away_code:
            continue

        # Pick & Line
        raw_pick = str(row.get("Pick", "")).strip()
        pick_team = str(row.get("PickTeam", "")).strip() # Sides specific

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

        final_pick_type = pick_type_default
        final_pick_val = raw_pick

        if "TOTAL" in market:
            final_pick_type = "TOTAL"
            # If line is missing, try to extract from pick
            if line_val is None:
                match = re.search(r'(Over|Under)\s+([0-9]+\.?[0-9]*)', raw_pick, re.IGNORECASE)
                if match:
                    line_val = float(match.group(2))
                    final_pick_val = match.group(1).upper()
        elif "SPREAD" in market or "SIDE" in market or "MONEYLINE" in market:
            final_pick_type = "SIDE"
            # For sides, pick is usually the team name
            # If PickTeam column exists, use that
            if pick_team and pick_team != "NAN":
                final_pick_val = pick_team

        # Canonical Key
        canon_key = generate_canonical_key(league, date_val, away_code, home_code)

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
            # Additional metadata for fuzzy date matching if needed later
            "away_team_raw": str(row.get("AwayTeam", "")),
            "home_team_raw": str(row.get("HomeTeam", ""))
        })

    logger.info(f"TheOver {pick_type_default} rows retained for matching: {len(records)}")
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
        "files_processed": []
    }

    # Debugging / Loading Totals
    totals_df = pd.DataFrame()
    if totals_file:
        try:
            totals_df = load_theover_totals(totals_file)
            stats["raw_totals_rows"] = len(totals_df)
            stats["files_processed"].append("totals_file")
        except Exception as exc:
            logger.error("DEBUG Failed to read Totals: %s", exc)
            totals_df = pd.DataFrame()

    # Debugging / Loading Sides
    sides_df = pd.DataFrame()
    if sides_file:
        try:
            sides_df = load_theover_sides(sides_file)
            stats["raw_sides_rows"] = len(sides_df)
            stats["files_processed"].append("sides_file")
        except Exception as exc:
            logger.error("DEBUG Failed to read Sides: %s", exc)
            sides_df = pd.DataFrame()

    # We do NOT match to master schedule using fuzzy names here anymore.
    # We rely on generated keys.

    # 1. Excel Ingestion
    if not totals_df.empty:
        try:
            processed = _transform_theover_df(totals_df, pick_type_default="TOTAL")
            if not processed.empty:
                dfs.append(processed)
        except Exception as e:
            logger.error(f"Error processing Totals Excel: {e}")

    if not sides_df.empty:
        try:
            processed = _transform_theover_df(sides_df, pick_type_default="SIDE")
            if not processed.empty:
                dfs.append(processed)
        except Exception as e:
            logger.error(f"Error processing Sides Excel: {e}")

    stats["raw_total_rows"] = stats["raw_totals_rows"] + stats["raw_sides_rows"]

    # 2. Text Paste Fallback
    if totals_paste and totals_paste.strip():
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
