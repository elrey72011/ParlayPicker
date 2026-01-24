# Shotgun Mode: Multi-Parlay Generator

## Overview

**Shotgun Mode** is an automated parlay generation system that creates optimized 2-leg and 3-leg parlay combinations from your picks. It intelligently filters, scores, and categorizes parlays by risk level to maximize expected value while managing correlation and risk.

## Features

### Core Capabilities

- **Automatic Parlay Generation**: Generates 20-50 optimized parlays automatically
- **2-Leg and 3-Leg Parlays**: Supports both parlay types with different risk thresholds
- **Risk Categorization**: Groups parlays into Low, Medium, and High risk tiers
- **EV Optimization**: Ranks parlays by expected value score
- **Correlation Detection**: Ensures picks are from different games
- **CSV Export**: Download parlays for tracking and analysis

### Pick Filtering

Shotgun Mode only uses high-quality picks:

- **Grade A or B only**: Excludes lower quality picks
- **Probability range**: 45-75% (reasonable confidence)
- **No extreme odds**: Filters out moneyline picks with odds >±400
- **Quality-focused**: Prioritizes picks with higher quality scores

### Risk Tiers

#### Low Risk 🟢
- **2-Leg**: >30% combined probability
- **3-Leg**: >18% combined probability
- **Characteristics**: Higher win probability, safer bets, lower payouts
- **Best for**: Consistent returns, conservative betting

#### Medium Risk 🟡
- **2-Leg**: 20-30% combined probability
- **3-Leg**: 10-18% combined probability
- **Characteristics**: Balanced risk/reward, moderate payouts
- **Best for**: Value hunting, balanced approach

#### High Risk 🔴
- **2-Leg**: <20% combined probability
- **3-Leg**: <10% combined probability
- **Characteristics**: Lower win probability, higher potential payouts
- **Best for**: Longshot opportunities, high-risk tolerance

### EV Score Calculation

Parlays are ranked by an Expected Value (EV) score that considers:

1. **Combined Probability**: Base win probability of all legs
2. **Average Quality**: Mean quality score of picks
3. **League Diversity Bonus**: +10% per additional league
4. **Market Diversity Bonus**: +5% per additional market type (Spread/Total/ML)
5. **Timing Spread Bonus**: Up to +15% for staggered start times

**Formula:**
```
EV Score = (combined_prob × avg_quality/100) × league_bonus × market_bonus × timing_bonus
```

## Usage

### In Streamlit App

1. **Run Master Analysis**: Generate picks using the main analysis
2. **Navigate to Shotgun Mode Tab**: Click on the "Shotgun Mode" tab
3. **Enable Auto-Parlay Generator**: Check the checkbox to activate
4. **Configure Settings**:
   - Number of 2-Leg Parlays (5-50, default: 20)
   - Number of 3-Leg Parlays (5-50, default: 20)
5. **Generate Parlays**: Click "Generate Optimized Parlays"
6. **Filter and Sort**:
   - Filter by parlay type (2-leg, 3-leg, or both)
   - Filter by risk tier
   - Sort by EV Score, Probability, Odds, or Quality
7. **Export**: Download parlays as CSV for tracking

### Programmatic Usage

```python
from shotgun_mode import generate_shotgun_mode_parlays
import pandas as pd

# Your picks dataframe
picks_df = pd.DataFrame([...])

# Generate parlays
results = generate_shotgun_mode_parlays(
    picks_df,
    num_2leg=20,
    num_3leg=20
)

# Access results
parlays_2leg = results['2leg']
parlays_3leg = results['3leg']
stats = results['stats']

# Each parlay is a dictionary with:
# - parlay_id: Unique identifier
# - num_legs: 2 or 3
# - legs: List of leg dictionaries
# - combined_probability: Overall win probability
# - expected_odds: American odds format
# - avg_quality: Average quality score
# - ev_score: Expected value score
# - risk_tier: "Low Risk", "Medium Risk", or "High Risk"
```

## Example Output

### 2-Leg Parlay (Low Risk)

```
Parlay ID: 2L_001
Type: 2-Leg
Risk Tier: Low Risk
Combined Probability: 32.1%
Expected Odds: +211
Avg Quality: 95.0
EV Score: 0.3521

Leg 1: NBA - Under 225.5 (Milwaukee vs Denver) | 56.8% | Q:100
Leg 2: NHL - Vancouver Canucks -1.5 | 57.2% | Q:90

Leagues: NBA, NHL (2 unique)
Markets: Total, Spread (2 unique)
```

### 3-Leg Parlay (Medium Risk)

```
Parlay ID: 3L_005
Type: 3-Leg
Risk Tier: Medium Risk
Combined Probability: 14.2%
Expected Odds: +605
Avg Quality: 91.7
EV Score: 0.1487

Leg 1: NCAAB - Under 145.5 (Rutgers vs Indiana) | 55.8% | Q:95
Leg 2: NBA - Phoenix Suns ML | 53.8% | Q:90
Leg 3: NHL - Calgary Flames -1.5 | 54.8% | Q:90

Leagues: NCAAB, NBA, NHL (3 unique)
Markets: Total, Moneyline, Spread (3 unique)
```

## Technical Details

### Files

- `shotgun_mode.py`: Core parlay generation logic
- `shotgun_mode_ui.py`: Streamlit UI components
- `streamlit_app.py`: Integration with main application

### Key Functions

#### `filter_eligible_picks(all_picks_df)`
Filters picks suitable for parlays based on grade, probability, and quality.

#### `generate_2leg_parlays(picks_df, target_count)`
Generates optimized 2-leg parlay combinations.

#### `generate_3leg_parlays(picks_df, target_count)`
Generates optimized 3-leg parlay combinations.

#### `generate_shotgun_mode_parlays(picks_df, num_2leg, num_3leg)`
Main function that orchestrates the entire parlay generation process.

#### `export_parlays_to_csv(parlays_2leg, parlays_3leg, output_path)`
Exports parlays to CSV file.

### Performance

- **Generation Speed**: <5 seconds for 40 parlays
- **Combinatorial Control**: Uses top 50 picks for 3-leg parlays to prevent explosion
- **Memory Efficient**: Processes picks in batches

### Correlation Detection

Parlays are filtered to ensure:
- No picks from the same game
- Independent outcomes for true parlay probability

### Risk Balancing

Parlays are balanced across risk tiers:
- **40% Low Risk**
- **40% Medium Risk**
- **20% High Risk**

## Best Practices

1. **Start Conservative**: Begin with Low and Medium risk parlays
2. **Diversify**: Look for parlays with multiple leagues/markets
3. **Check Timing**: Prefer parlays with staggered start times for live adjustments
4. **Review Quality**: Always check individual leg quality scores
5. **Bankroll Management**: Size bets appropriately for risk tier
6. **Track Results**: Export to CSV and monitor performance over time

## Validation & Testing

### Test Cases

1. **Insufficient Picks**: <4 eligible picks → Warning message
2. **Low Quality Picks**: Only Grade C/D → No eligible picks
3. **Correlated Picks**: All from same game → No parlays generated
4. **Optimal Scenario**: 50+ Grade A/B picks → Full parlay set
5. **Extreme Probabilities**: >75% or <45% → Filtered out

### Success Metrics

- ✅ Generation speed <5 seconds
- ✅ Risk balance approximately 40/40/20
- ✅ Average quality >85/100
- ✅ League diversity <30% same-league parlays
- ✅ Top 10 parlays EV score >0.25

## Future Enhancements

### Version 2 Features

1. **Custom Risk Tiers**: User-adjustable probability thresholds
2. **League Filtering**: Generate NBA-only or cross-league parlays
3. **Same Game Parlays**: Allow correlated picks with special risk adjustment
4. **Bankroll Management**: Suggest bet sizing by risk tier
5. **Historical Performance**: Track success rates and adjust EV scoring
6. **Live Updating**: Regenerate as odds change
7. **Advanced Filters**: Filter by specific leagues, markets, or time windows

## Troubleshooting

### No Parlays Generated

**Possible causes:**
- Not enough eligible picks (need minimum 2 for 2-leg, 3 for 3-leg)
- All picks are correlated (same game)
- Picks don't meet quality criteria (Grade C/D only)
- Probability range too narrow

**Solutions:**
- Run Master Analysis with more games
- Check data quality grades in main results
- Verify picks have diverse games/leagues

### Low Diversity Score

**Issue**: All parlays from same league

**Solution:**
- Include picks from multiple leagues in Master Analysis
- Adjust league selection in sidebar

### Export Fails

**Issue**: CSV export error

**Solution:**
- Check write permissions on output directory
- Verify disk space available
- Check for special characters in team names

## Support

For issues or questions:
- Check the logs in the Streamlit app sidebar
- Review the "Debug" tab for diagnostic information
- File an issue on GitHub repository

## Credits

Developed for ParlayPicker to automate parlay generation and optimize betting strategies using data-driven Expected Value scoring.

---

**Version**: 1.0
**Last Updated**: January 2026
**Compatible with**: ParlayPicker v2.0+
