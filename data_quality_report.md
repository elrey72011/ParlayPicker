# Data Quality Report

## 1. Sentiment Coverage
- Percentage of rows with non-null sentiment values (Home/Away Sentiment): **8.57%**

## 2. Distribution of Sentiment Status
| Status | Count |
|---|---|
| disabled | 175 |

## 3. Implied Win Probability Stats (Calculated)
### Spread Implied Probabilities
|       |   spread_implied_prob_calc |
|:------|---------------------------:|
| count |                175         |
| mean  |                  0.539929  |
| std   |                  0.0452181 |
| min   |                  0.519231  |
| 25%   |                  0.52381   |
| 50%   |                  0.52381   |
| 75%   |                  0.527184  |
| max   |                  0.74026   |

### Total Implied Probabilities
|       |   total_implied_prob_calc |
|:------|--------------------------:|
| count |               175         |
| mean  |                 0.528073  |
| std   |                 0.0121187 |
| min   |                 0.52381   |
| 25%   |                 0.52381   |
| 50%   |                 0.52381   |
| 75%   |                 0.528302  |
| max   |                 0.652778  |

## 4. Data Quality Issues & Suspicious Patterns
- **Suspicious Sentiment State:** All 175 rows have `sentiment_status` as "disabled", yet 8.57% of rows contain non-null sentiment values (often 0.0). This suggests either stale data or a disconnect between the status flag and the data population.
- 14 rows in log dump have match confidence < 70.

## TheOver Log Dump Analysis
- Total Entries: 277
### Match Status Distribution
- **MATCH**: 265
- **FAIL**: 6
- **MISMATCH_PAIR**: 6
