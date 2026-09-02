# ParlayPicker

ParlayPicker is a robust, data-driven sports betting pipeline designed to generate +EV (Expected Value) betting recommendations. It combines live market odds, statistical baselines, and machine learning predictions to identify actionable edges across multiple sports leagues (NFL, NCAAF, NBA, WNBA, NHL, MLB, and NCAAB).

## Current Architecture & Operation

The system operates entirely autonomously using the following pipeline:

### 1. Data Ingestion
The system pulls live market odds from **TheOddsAPI** and merges them with statistical baseline data from **TheOver** (via uploaded CSVs or API).

### 2. Identity Resolution
We use strict string sanitization, canonical game keys (`League|Home|Away|Date`), and fuzzy matching (`difflib`/`SequenceMatcher`) to safely resolve team names (e.g., "St. Louis" vs "Saint Louis") and intra-city matchups across different data sources.

### 3. ML Prediction Engine
We use a cached XGBoost `PredictionEngine` to generate win probabilities.
* **Resilience:** If the ML engine fails or feature matrices are empty, the system gracefully and unconditionally falls back to statistical baseline probabilities without crashing.
* **Target Integrity:** The game-winner model is used only for moneylines. Spread and total rows require a separately validated target-specific model; otherwise ML stays unavailable and the blend uses each remaining independent source once.

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

### 9. Gemini Wager Review

Gemini is integrated as a bounded secondary reviewer for both game picks and
player props. When **Require Gemini Review for Bets** is enabled:

* A funded wager must receive a structured Gemini review that selects the exact
  same pick with `MEDIUM` or `HIGH` confidence and no blocking risk flag.
* Gemini output is constrained by a JSON schema and then validated locally;
  incomplete rows are marked `INVALID_RESPONSE`, not reviewed, and held at `$0`.
* Exact-price expected value is authoritative. A missing or non-positive EV can
  never receive Gemini approval, even when the de-vigged probability edge is positive.
* `HIGH` confidence preserves the deterministic stake; `MEDIUM` uses 75% of it.
* Disagreement, low confidence, missing live inputs, invalid JSON, quota/API
  failure, or a missing key holds the wager at `$0`.
* Gemini can never promote a model-rejected row or flip to an opposing pick
  without a separately validated sportsbook line and price.
* Game, all-games, compact, and prop exports carry response-validity, verdict,
  reason, and multiplier fields for a consistent audit trail.

Set either `GOOGLE_API_KEY` or `GEMINI_API_KEY` in Streamlit secrets or the
deployment environment. Never commit the key to this repository.
