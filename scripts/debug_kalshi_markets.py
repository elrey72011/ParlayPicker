"""
Debug Kalshi Markets
Run this to see why a specific game isn't matching with Kalshi.
Usage: python scripts/debug_kalshi_markets.py
"""

import sys
import os
from datetime import datetime
import pytz

# Add parent directory to path to import app_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app_core.kalshi_integrator import KalshiIntegrator, LEAGUE_SERIES_MAP
from app_core.team_name_matcher import TeamNameMatcher

def debug_game_match(kalshi, home_team, away_team, game_date_iso, league="nba"):
    print(f"\n{'='*60}")
    print(f"🕵️  DEBUGGING MATCH: {away_team} @ {home_team}")
    print(f"📅  Target Date: {game_date_iso}")
    print(f"{'='*60}\n")

    # 1. Fetch Markets
    print("1️⃣  Fetching Kalshi Markets...")
    markets = kalshi.get_markets()
    print(f"   -> Found {len(markets)} total active markets.")

    # 2. Setup Comparison
    target_dt = datetime.fromisoformat(game_date_iso.replace("Z", "+00:00"))
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=pytz.UTC)
    
    norm_home = TeamNameMatcher.normalize(home_team)
    norm_away = TeamNameMatcher.normalize(away_team)
    expected_series = LEAGUE_SERIES_MAP.get(league, "")

    print(f"\n2️⃣  Normalization Check:")
    print(f"   Input Home: '{home_team}' -> Normalized: '{norm_home}'")
    print(f"   Input Away: '{away_team}' -> Normalized: '{norm_away}'")
    print(f"   Expected Series Prefix: '{expected_series}'")

    # 3. Analyze Candidates
    print(f"\n3️⃣  Analyzing Candidates (Top Matches):")
    candidates = []

    for m in markets:
        # --- A. Check Series ---
        series = str(m.get("series_ticker") or "").upper()
        series_match = series.startswith(expected_series)

        # --- B. Check Date ---
        m_date_raw = m.get("close_time") or m.get("event_date")
        m_dt = None
        date_match = False
        date_debug = "No Date"
        
        if m_date_raw:
            try:
                if isinstance(m_date_raw, (int, float)):
                    m_dt = datetime.fromtimestamp(m_date_raw / 1000.0, tz=pytz.UTC)
                else:
                    m_dt = datetime.fromisoformat(str(m_date_raw).replace("Z", "+00:00"))
                    if m_dt.tzinfo is None: m_dt = m_dt.replace(tzinfo=pytz.UTC)
                
                # Check date logic (same as integrator)
                diff = (m_dt.date() - target_dt.date()).days
                date_match = abs(diff) <= 1
                date_debug = f"{m_dt.date()} (Diff: {diff} days)"
            except Exception as e:
                date_debug = f"Error parsing: {e}"

        # --- C. Check Names ---
        title = m.get("title") or ""
        subtitle = m.get("subtitle") or ""
        ticker = m.get("ticker") or ""
        
        # Construct the string exactly as the integrator does
        market_full_text = f"{title} {subtitle}"
        market_norm = TeamNameMatcher.normalize(market_full_text)

        # Calculate Scores
        h_s = TeamNameMatcher.similarity_score(norm_home, market_norm)
        a_s = TeamNameMatcher.similarity_score(norm_away, market_norm)
        
        # Calculate Quality (Replicating integrator logic)
        quality = 0.0
        home_in = norm_home in market_norm
        away_in = norm_away in market_norm
        
        if home_in and away_in:
            quality = 1.0
        elif (h_s + a_s) / 2.0 > 0.55:
            quality = (h_s + a_s) / 2.0
        
        # Boost logic
        if "GAME" in ticker.upper() or "SPREAD" in ticker.upper():
            quality += 0.1

        # Save candidate if it has ANY relevance
        # (Matches series OR has > 0.3 name score)
        if series_match or quality > 0.3:
            candidates.append({
                "ticker": ticker,
                "title": title,
                "series": series,
                "series_match": series_match,
                "date_debug": date_debug,
                "date_match": date_match,
                "quality": quality,
                "h_score": h_s,
                "a_score": a_s,
                "market_norm": market_norm
            })

    # Sort candidates by quality score
    candidates.sort(key=lambda x: x['quality'], reverse=True)

    if not candidates:
        print("   ❌ No candidates found matching Series or Name requirements.")
        return

    # Print Top 5
    for i, c in enumerate(candidates[:5]):
        status = "✅ MATCH"
        fail_reasons = []
        
        # Replicate thresholds
        if c['quality'] < 0.65: 
            fail_reasons.append(f"Score {c['quality']:.2f} < 0.65")
        if not c['date_match']: 
            fail_reasons.append(f"Date Mismatch ({c['date_debug']})")
        if not c['series_match']: 
            fail_reasons.append(f"Series Mismatch ({c['series']} != {expected_series})")
            
        if fail_reasons:
            status = "❌ FAIL"
        
        print(f"\n   [{i+1}] {c['ticker']} ({status})")
        print(f"       Title: '{c['title']}'")
        print(f"       Score: {c['quality']:.3f} (Home: {c['h_score']:.2f}, Away: {c['a_score']:.2f})")
        print(f"       Norm:  '{c['market_norm'][:50]}...'")
        if fail_reasons:
            print(f"       Reasons: {', '.join(fail_reasons)}")

if __name__ == "__main__":
    # Initialize Integrator
    k = KalshiIntegrator()
    
    # ---------------------------------------------------------
    # EDIT THESE VARIABLES TO TEST A SPECIFIC GAME
    # ---------------------------------------------------------
    TEST_HOME = "Golden State Warriors"
    TEST_AWAY = "Minnesota Timberwolves"
    # Use format YYYY-MM-DDTHH:MM:SS
    TEST_DATE = datetime.now().strftime("%Y-%m-%dT%H:%M:%S") 
    TEST_LEAGUE = "nba"
    # ---------------------------------------------------------

    debug_game_match(k, TEST_HOME, TEST_AWAY, TEST_DATE, TEST_LEAGUE)
