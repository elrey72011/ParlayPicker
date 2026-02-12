# Jules Coding Task: Fix ParlayPicker Kalshi Matching & Formula Issues

## Repository
**GitHub**: elrey72011/ParlayPicker  
**Branch**: main  
**Primary Files**: `app_core/kalshi_integrator.py`, `app_core/feature_processing.py`

---

## Context

The ParlayPicker sports betting analytics app integrates with Kalshi prediction markets to enhance pick quality. Currently, only **35 of 68 games (51.5%)** are matching with Kalshi markets when the expected rate should be **85-90%** (58-61 games).

This low match rate is causing:
- Reduced quality scores (88.1/100 instead of 89.5-91/100)
- Fewer Grade A picks (51.7% instead of 56-58%)
- Missing critical market probability data for decision-making

---

## Task Overview

You need to fix 5 critical issues in the Kalshi matching pipeline and formula calculations:

1. **NCAAB Force Match Logic** - Repositioning misplaced code
2. **Match Threshold** - Lowering from 85 to 80
3. **Spread/Total Pagination** - Increasing max_pages from 3 to 10
4. **Differential Calculation Logic** - Fixing zero-value handling
5. **Enhanced Debug Logging** - Adding visibility into matching process

---

## Fix #1: Reposition NCAAB Force Match Logic

### File: `app_core/kalshi_integrator.py`

### Problem
Lines 1336-1354 contain NCAAB force match logic that executes even when `markets` is empty, causing false positive matches.

### Current Location (WRONG):
```
Line 1184-1334: Market classification loop
Line 1336-1354: NCAAB FORCE MATCH ← Executes even if markets=[]
Line 1356+: if target_market check
```

### Required Change
**MOVE lines 1336-1354 to AFTER line 1356** so the force match only runs when a valid target_market exists.

### New Flow Should Be:
```python
# Line 1184-1334: Market classification loop (unchanged)

# Line 1336-1355: Market summary logging (keep here)
logger.info(f"🎯 KALSHI MATCH [{league}]: Event {best_event.get('ticker')} - ...)

# Line 1356-1362: Check if target_market exists
target_market = winner_market
if not target_market and markets:
    target_market = markets[0]

# NEW LOCATION: NCAAB force match logic goes HERE (after target_market check)
if league == 'NCAAB' and target_market:
    # [Insert NCAAB force match code from lines 1336-1354]
    logger.info(f"NCAAB FINAL {best_event.get('ticker')} | markets={len(markets)} | bestscore={best_score} | target={target_market.get('ticker') if target_market else 'NONE'}")
    
    yes_bid = _kalshi_price_norm(target_market, "yes_bid_dollars", "yes_bid")
    yes_ask = _kalshi_price_norm(target_market, "yes_ask_dollars", "yes_ask")
    # ... rest of probability calculation ...
    
    return KalshiMatchResult(
        matched=True,
        kalshi_available=True,
        label=target_market.get('title'),
        probability=final_prob,
        raw_event_id=best_event.get('ticker'),
        league=league,
        reason='ncaab_force_match',
        market_type='force',
        game_date=game_dt_utc
    )

# Line 1364+: Continue with normal winner market logic
if target_market:
    # Calculate prob using _dollars fields...
```

### Verification
After fix, test that NCAAB games without Kalshi markets return `matched=False` instead of `matched=True` with probability 0.5.

---

## Fix #2: Lower Match Score Threshold

### File: `app_core/kalshi_integrator.py`

### Location: Line 995

### Current Code:
```python
MATCH_THRESHOLD = 85
```

### Change To:
```python
MATCH_THRESHOLD = 80  # Lowered from 85 to improve match rate for minor code variations
```

### Rationale
The current threshold of 85 is too strict:
- Perfect match = 100 ✓
- Perfect match with time penalty (100 - 10) = 90 ✓
- Near-perfect match with minor code variation = 80-89 ✗ (currently fails)

Lowering to 80 allows matches where:
- Both teams match correctly (50 + 50 = 100)
- Small time difference penalty (-10 to -20)
- Result: 80-90 score range now passes

### Verification
After fix, check logs for games with scores in 80-84 range that previously failed but should have matched.

---

## Fix #3: Increase Spread/Total Market Pagination

### File: `app_core/kalshi_integrator.py`

### Locations: Lines 1107 and 1146

### Current Code (Line 1107):
```python
series_markets = integrator.get_markets_paginated(
    status=None,
    limit=200,
    max_pages=3,  # ← Only 600 markets max
    extra_params={"series_ticker": spread_series}
)
```

### Change To:
```python
series_markets = integrator.get_markets_paginated(
    status=None,
    limit=200,
    max_pages=10,  # Increased from 3 to fetch up to 2000 markets
    extra_params={"series_ticker": spread_series}
)
logger.info(f"   📊 Spread market pagination: Fetched {len(series_markets)} markets (max_pages=10)")
```

### Repeat Same Change at Line 1146 for Total Markets:
```python
series_markets = integrator.get_markets_paginated(
    status=None,
    limit=200,
    max_pages=10,  # Increased from 3 to fetch up to 2000 markets
    extra_params={"series_ticker": total_series}
)
logger.info(f"   📊 Total market pagination: Fetched {len(series_markets)} markets (max_pages=10)")
```

### Rationale
- Current limit of 3 pages × 200 = 600 markets
- During peak NBA season, KXNBASPREAD may have 1000+ active markets
- Missing markets means missing spread/total data for games

### Verification
Check logs for "Fetched X markets" to confirm pagination is retrieving more markets.

---

## Fix #4: Fix Differential Calculation Logic

### File: `app_core/feature_processing.py`

### Location: Lines 2052-2065

### Current Code:
```python
def safe_diff(h, a):
    try:
        h_val = float(h)
        a_val = float(a)
    except Exception:
        return 0.0
    if abs(h_val) < 1e-6 or abs(a_val) < 1e-6:  # ← WRONG: returns 0 if EITHER is zero
        return 0.0
    return h_val - a_val
```

### Problem
The condition `abs(h_val) < 1e-6 or abs(a_val) < 1e-6` returns 0.0 if EITHER value is zero, even if one team has valid stats.

**Example Failure**:
- Home team: 0.6 win_pct
- Away team: 0.0 (missing stats)
- Current: diff = 0.0 (WRONG - should indicate missing data)
- Expected: diff = NaN or keep 0.6 to show home advantage

### Change To:
```python
def safe_diff(h, a):
    """
    Calculate difference between home and away values.
    Returns 0.0 only if both are zero/missing or if conversion fails.
    Legitimate zero values (e.g., 0% win rate) are valid for calculation.
    """
    try:
        h_val = float(h)
        a_val = float(a)
    except Exception:
        return 0.0
    
    # Check for NaN (pandas NA values)
    if pd.isna(h_val) or pd.isna(a_val):
        return 0.0
    
    # Only return 0.0 if BOTH are exactly zero (indicates missing data for both)
    if h_val == 0.0 and a_val == 0.0:
        return 0.0
    
    # Otherwise calculate legitimate differential (includes cases where one is 0.0)
    return h_val - a_val
```

### Verification
Test with edge cases:
- Home=0.6, Away=0.0 → Should return 0.6 (or -0.6 depending on order)
- Home=0.0, Away=0.0 → Should return 0.0
- Home=0.3, Away=0.7 → Should return -0.4
- Home=NaN, Away=0.5 → Should return 0.0

---

## Fix #5: Add Enhanced Logging for Match Diagnostics

### File: `app_core/kalshi_integrator.py`

### Location: Multiple points in `_match_via_events` function

### Add After Line 970 (before match loop):
```python
# Log search parameters
logger.info(f"   🔍 Match Search Parameters:")
logger.info(f"      Events to scan: {len(events)}")
logger.info(f"      Time window: ±{TIME_WINDOW_HOURS}h")
logger.info(f"      Match threshold: {MATCH_THRESHOLD}")
logger.info(f"      Target time: {game_dt_utc}")
```

### Add in Match Loop (Around Line 1006, enhance existing log):
```python
# Enhanced logging for EVERY potential match attempt (score >= 50)
logger.info(f"   🎲 Evaluating: {ticker}")
logger.info(f"      Raw Codes: away={parsed.get('away')}, home={parsed.get('home')}")
logger.info(f"      Resolved Codes: away={evt_away_code}, home={evt_home_code}")
logger.info(f"      Expected Away Codes: {list(resolved_away)[:3]}")
logger.info(f"      Expected Home Codes: {list(resolved_home)[:3]}")
logger.info(f"      Score Calculation:")
logger.info(f"         - Away Match: {away_match_1} (+{50 if away_match_1 else 0})")
logger.info(f"         - Home Match: {home_match_1} (+{50 if home_match_1 else 0})")
logger.info(f"         - Direct Score: {score_1}")
logger.info(f"         - Swap Score: {score_2}")
logger.info(f"         - Best Score: {match_score}")
if time_diff_hours is not None:
    penalty = 10 if time_diff_hours > TIME_WINDOW_HOURS and league != 'NCAAB' else 0
    logger.info(f"      Time Check: {time_diff_hours:.1f}h diff (penalty: -{penalty})")
logger.info(f"      Result: {'✓ Potential' if match_score >= 50 else '✗ Rejected'} (score={match_score})")
```

### Add After Match Loop (Before best_event check, line ~1015):
```python
# Summary of all attempts
logger.info(f"   📊 Match Summary:")
logger.info(f"      Events scanned: {len(events)}")
logger.info(f"      Best score found: {best_score}")
logger.info(f"      Threshold: {MATCH_THRESHOLD}")
```

---

## Implementation Checklist

### Phase 1: Code Changes (60 min)
- [ ] Fix #1: Move NCAAB force match logic (lines 1336-1354 → after line 1356)
- [ ] Fix #2: Change MATCH_THRESHOLD from 85 to 80 (line 995)
- [ ] Fix #3: Increase max_pages from 3 to 10 (lines 1107, 1146)
- [ ] Fix #4: Update safe_diff function (lines 2052-2065)
- [ ] Fix #5: Add enhanced debug logging (multiple locations)

### Phase 2: Testing (30 min)
- [ ] Run app with current game slate
- [ ] Verify match rate increases to 60%+
- [ ] Check for false positive NCAAB matches
- [ ] Verify spread/total markets are found
- [ ] Confirm differentials calculate correctly

### Phase 3: Validation (30 min)
- [ ] Check Quality Score improvement (should reach 89.5+)
- [ ] Verify Kalshi match count (should be 58-61 matches)
- [ ] Review logs for any remaining unmatched games

---

## Expected Outcomes

### Before Fixes
```
Kalshi Matches: 35/68 (51.5%)
Quality Score: 88.1/100
Grade A Picks: 51.7%
```

### After Fixes  
```
Kalshi Matches: 58-61/68 (85-90%)
Quality Score: 89.5-91/100
Grade A Picks: 56-58%
```

### Improvement
```
+23-26 additional Kalshi matches
+1.4-2.9 quality score points
+4-6% more Grade A picks
```

---

## Success Criteria

✅ **NCAAB games without Kalshi markets return matched=False**  
✅ **Match rate reaches 60%+ (41+ matches out of 68)**  
✅ **False positive rate < 10%**  
✅ **Spread/total markets found for NBA/NHL games**  
✅ **Differential calculations return non-zero for unequal valid inputs**  
✅ **Debug logs show detailed match attempts with scores and reasons**

---

## Common Pitfalls to Avoid

1. **DON'T** just increase threshold and call it fixed - test with real games
2. **DON'T** remove NCAAB force match logic - just reposition it correctly
3. **DON'T** break existing NBA/NFL matching while fixing NCAAB
4. **DON'T** over-log - keep debug logs INFO level, not DEBUG (they're already filtered)
5. **DON'T** forget to update BOTH spread AND total pagination (two separate locations)

---

## Verification Commands

After deploying fixes, run these commands to verify:

```bash
# 1. Check match rate on current slate
python -c "from streamlit_app import *; df = load_all_data(); print(f'Kalshi: {df.kalshi_available.sum()}/{len(df)}')"

# 2. Verify no false positives
python -c "from streamlit_app import *; df = load_all_data(); fp = df[(df.kalshi_available) & (df.kalshi_prob == 0.5)]; print(f'False Positives: {len(fp)}')"

# 3. Check differential calculations
python -c "import pandas as pd; df = pd.read_csv('master_df_raw.csv'); print(df[['feature_diff_win_pct', 'feature_home_win_pct', 'feature_away_win_pct']].describe())"
```

---

## Additional Notes

### Team Code Mappings
If after fixes the match rate is still below 80%, check logs for "TEAM MATCH FAILURE" messages and add missing team codes to:
- `NBA_TEAM_CODE_MAP` (line 186-220)
- `NCAAB_TEAM_CODE_MAP` (line 289-430)  
- `NCAAB_CODE_ALIASES` (line 561-619)

### Performance Considerations
- The pagination increase (3 → 10 pages) adds ~2-3 seconds to initial load
- This is acceptable for the match rate improvement
- Caching (300s TTL) means subsequent loads are instant

### Rollback Plan
If fixes cause issues:
1. Revert `app_core/kalshi_integrator.py` to commit `7ea2336`
2. Revert `app_core/feature_processing.py` to commit `a9c154c`  
3. System continues at 51.5% match rate
