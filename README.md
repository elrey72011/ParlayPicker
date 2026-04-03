# ParlayPicker

ParlayPicker is a robust, data-driven sports betting pipeline designed to generate +EV (Expected Value) betting recommendations. It combines live market odds, statistical baselines, and machine learning predictions to identify actionable edges across multiple sports leagues (NFL, NBA, NHL, MLB, and NCAAB).

## Current Architecture & Operation

The system operates entirely autonomously using the following pipeline:

### 1. Data Ingestion
The system pulls live market odds from **TheOddsAPI** and merges them with statistical baseline data from **TheOver** (via uploaded CSVs or API).

### 2. Identity Resolution
We use strict string sanitization, canonical game keys (`League|Home|Away|Date`), and fuzzy matching (`difflib`/`SequenceMatcher`) to safely resolve team names (e.g., "St. Louis" vs "Saint Louis") and intra-city matchups across different data sources.

### 3. ML Prediction Engine
We use a cached XGBoost `PredictionEngine` to generate win probabilities.
* **Resilience:** If the ML engine fails or feature matrices are empty, the system gracefully and unconditionally falls back to statistical baseline probabilities without crashing.

### 4. Expected Value (EV) Engine
* **Probability Calibration:** We do not blindly trust the ML. We calibrate probabilities using a conservative split (typically 30% Model / 70% Market) to respect efficient markets.
* **Synthetic De-Vigging:** If a market lacks opposing lines, we simulate a standard 4.5% sportsbook vig to calculate true market probability.
* **Sanity Clamps:** Any game showing an Expected Value > 0.40 (40% edge) is flagged as a "Data Error / Suspended Line" and forced to "No Play".

### 5. Pick Filtering & Tiers
Picks are graded into specific statuses based on strict logic:
* `Actionable`
* `High Variance/Speculative` (EV between 0.25 and 0.40)
* `Below Threshold`
* `Fallback / Low Confidence`
* `Missing Line`
* `No Play`

### 6. Triple Filter Ranking
Valid picks are sorted by Tier (S, A, B, C, D) and then strictly ranked by their Expected Value.

### 7. Performance Dashboard
The `results_dashboard.py` auto-grades historical exports. It uses safe boolean extraction to prevent string-matching bugs (e.g., ensuring empty strings don't evaluate to True) and calculates Overall Win Rate, Total Net Profit, and Picks Evaluated for both "Actionable" and "All Picks".

### 8. Infrastructure Resilience
The pipeline is hardened for headless/server deployment. It catches unhandled network timeouts, safely defaults zero-division errors in odds math to `-110`, and skips rate-limited APIs without crashing the main loop.
