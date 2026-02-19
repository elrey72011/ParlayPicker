# Kalshi Matching Logic Updates - Feb 2026

## Overview
This document summarizes the changes made to the Kalshi integration logic to improve match rates for NCAAB games, particularly for teams with variable abbreviations (e.g., "North Texas" -> "UNT", "Hawai'i" -> "HAW") and short codes (e.g., "CP" for Cal Poly).

## Key Changes

### 1. Enhanced Normalization (`canonical_team_name`)
*   **Mascot Stripping:** Aggressively removes common mascots (e.g., "Mean Green", "Rainbow Warriors") to isolate the base school name.
*   **Standardization:**
    *   "Hawai'i" -> "Hawaii"
    *   "Saint" -> "St"
    *   "Int'l" -> "International"
*   **Usage:** This canonical name is used for map lookups and fuzzy matching.

### 2. Comprehensive Team Code Maps
*   Updated `KALSHI_NCAAB_TEAM_CODES` and `NCAAB_CODE_ALIASES` with missing entries identified from failures:
    *   **North Texas** -> `UNT`
    *   **Montana State** -> `MTST`
    *   **Eastern Washington** -> `EWU`
    *   **Sacramento State** -> `SAC`
    *   **Weber State** -> `WEB`
    *   **Cal Poly** -> `CP`
    *   **Tulane** -> `TULN`
    *   **North Florida** -> `UNF`
    *   **Austin Peay** -> `APSU`
    *   **UNC Wilmington** -> `UNCW`
    *   **Monmouth** -> `MON`

### 3. Scoring Logic Improvements
*   **Exact Code Bonus:** Added a specific rule to award **100.0 points** for an exact match between a ticker code and a team variant, regardless of length. This fixes the issue where 2-letter codes like "CP" (Cal Poly) were penalized by the length-based fuzzy scorer.
*   **Date Proximity Scoring:** Added a tie-breaker bonus for games closer in time to the target `commence_time` (within 6/12 hours) to resolve potential double-header ambiguities.
*   **Threshold Adjustment:** Lowered the strict match threshold from **80.0** to **70.0** while maintaining safety through stricter candidate generation. This allows high-confidence 3-letter code matches (which score ~75-90) to pass automatically.

### 4. Status & Reason Reporting
*   `match_game_to_kalshi_markets` now injects a `_meta` dictionary into the result.
*   `streamlit_app.py` extracts this metadata to populate `kalshi_status` and `kalshi_match_reason` with granular failure reasons (e.g., `team_name_mismatch`, `no_kalshi_market_for_game`) instead of generic "no_match" messages.

## Current Coverage
*   **Golden Set:** Existing matches (NBA, standard NCAAB) remain stable.
*   **Target Set:** Previously failing NCAAB matchups (North Texas, Hawai'i, Big Sky/Big West teams) now match successfully with high confidence scores.

## Future Maintenance
*   If new matching failures occur, check `app_exports/kalshi_matching_report.csv` for the `candidates_found` and `match_score` columns.
*   Add missing aliases to `KALSHI_NCAAB_TEAM_CODES` in `app_core/kalshi_integrator.py` following the existing pattern.
