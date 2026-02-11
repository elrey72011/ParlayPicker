import pandas as pd
import numpy as np
import logging
from app_core.weights_config import (
    KALSHI_WEIGHT,
    MARKET_WEIGHT,
    ML_MODEL_WEIGHT,
    THEOVER_WEIGHT,
    SENTIMENT_WEIGHT,
)

logger = logging.getLogger("parlaypicker")

def calculate_consensus_for_row(row: pd.Series, market_type: str = "Spread") -> tuple[float, str]:
    """
    Calculates Consensus using weighted blend of Market, Kalshi, Model, TheOver, and Sentiment.
    Uses static weights from app_core.weights_config.

    Args:
        row: The dataframe row containing probability columns
        market_type: "Spread" or "Total"

    Returns:
        (consensus_prob, breakdown_string)
        - consensus_prob: float 0-1
        - breakdown_string: e.g. "56% (M55 / K59 / AI52)"
    """
    # Helper to safely get float
    def _get_f(key):
        val = row.get(key)
        try:
            return float(val) if pd.notnull(val) and val != "" else None
        except:
            return None

    market_type = market_type.capitalize()

    # 1. Map Columns based on Market Type
    p_market = None
    p_kalshi = None
    p_model = None
    p_theover = None
    p_sentiment = None # Sentiment usually applied as adj, but we can try to extract a prob

    if market_type == "Spread":
        # Market
        p_market = _get_f("spread_prob_pick_market")
        if p_market is None: p_market = _get_f("spread_prob_market_based")
        if p_market is None: p_market = _get_f("spread_prob_market")

        # Kalshi — v99 FIX (Bug 1): ONLY use pick-side-adjusted Kalshi probability.
        # NEVER fall back to raw kalshi_prob_spread which is on the YES-contract side,
        # not the pick side. Using the raw value produces wrong-side blending.
        p_kalshi = _get_f("spread_prob_pick_kalshi")

        # AI
        p_model = _get_f("model_spread_prob")

        # TheOver
        p_theover = _get_f("theover_spread_prob")

    elif market_type == "Total":
        # Market
        p_market = _get_f("total_prob_pick_market")
        if p_market is None: p_market = _get_f("total_prob_market_based")
        if p_market is None: p_market = _get_f("total_prob_market")

        # Kalshi — v99 FIX (Bug 1): ONLY use pick-side-adjusted Kalshi probability.
        # NEVER fall back to raw kalshi_prob_total which is on the YES-contract side.
        p_kalshi = _get_f("total_prob_pick_kalshi")

        # AI
        p_model = _get_f("model_total_prob")

        # TheOver
        p_theover = _get_f("theover_total_prob")

    else:
        # Fallback for Moneyline or generic
        p_market = _get_f("Implied_Prob")
        p_kalshi = _get_f("kalshi_prob_for_pick")
        p_model = _get_f("AI_Prob")
        p_theover = _get_f("theover_prob") or _get_f("theover_prob_used")

    # Common Fallbacks
    if p_model is None: p_model = _get_f("AI_Prob")
    if p_theover is None: p_theover = _get_f("theover_prob") or _get_f("theover_prob_used")

    # Sentiment extraction (heuristic from adj)
    # sentiment_adj is usually +/- 0.05. Base is 0.5.
    # So p_sentiment ~ 0.5 + adj.
    sent_adj = _get_f("sentiment_adj")
    if sent_adj is not None:
        p_sentiment = 0.5 + sent_adj

    # 2. Build Sources List using Static Weights
    # FIX: When a source is unavailable (None), set its weight to 0 instead of
    # using 0.5 (neutral). This prevents missing sources from diluting the
    # consensus toward 50%. Weights are redistributed proportionally to
    # available sources via normalization (total_w division).
    sources = []

    # Market
    w_m = MARKET_WEIGHT if p_market is not None else 0.0
    val_m = p_market if p_market is not None else 0.0
    sources.append(("M", val_m, w_m))

    # Kalshi
    w_k = KALSHI_WEIGHT if p_kalshi is not None else 0.0
    val_k = p_kalshi if p_kalshi is not None else 0.0
    sources.append(("K", val_k, w_k))

    # Model
    w_ml = ML_MODEL_WEIGHT if p_model is not None else 0.0
    val_ml = p_model if p_model is not None else 0.0
    sources.append(("AI", val_ml, w_ml))

    # TheOver
    w_to = THEOVER_WEIGHT if p_theover is not None else 0.0
    val_to = p_theover if p_theover is not None else 0.0
    sources.append(("TO", val_to, w_to))

    # Sentiment
    w_s = SENTIMENT_WEIGHT if p_sentiment is not None else 0.0
    val_s = p_sentiment if p_sentiment is not None else 0.0
    sources.append(("S", val_s, w_s))

    # 3. Calculate Weighted Sum
    # Normalize by total available weight to redistribute missing source weights proportionally
    total_w = sum(s[2] for s in sources)
    weighted_sum = sum(s[1] * s[2] for s in sources)

    if total_w == 0:
        return 0.5, "N/A"

    consensus = weighted_sum / total_w
    consensus = max(0.01, min(0.99, consensus))

    # 4. Format
    # Only show non-neutral/valid sources in string to keep it clean, or show all?
    # User likes transparency. Let's show significant ones or all if valid.
    parts = []
    for code, val, w in sources:
        # Only show if original value was not None (meaning we had data)
        # We need to check the original variables again.
        is_valid = False
        if code == "M" and p_market is not None: is_valid = True
        elif code == "K" and p_kalshi is not None: is_valid = True
        elif code == "AI" and p_model is not None: is_valid = True
        elif code == "TO" and p_theover is not None: is_valid = True
        elif code == "S" and p_sentiment is not None: is_valid = True

        if is_valid:
            parts.append(f"{code}{int(val*100)}")

    if not parts:
        breakdown = "No Data"
    else:
        breakdown = " / ".join(parts)

    formatted_str = f"{int(consensus*100)}% ({breakdown})"

    return consensus, formatted_str

def calculate_spread_consensus(row: pd.Series) -> tuple[float, str]:
    """Legacy wrapper for backward compatibility."""
    return calculate_consensus_for_row(row, "Spread")

def build_game_summary_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates ML, Spread, and Total rows into a single row per game with specific summary columns.

    Returns a DataFrame with one row per game and columns:
    [league, Home, Away, Commence (UTC), Commence (Local), Local Date,
     Overall Pick, Overall Prob, Spread, Spread Prob, Total, Total Prob, ML, ML Prob]
    """
    if df is None or df.empty:
        return pd.DataFrame()

    summary_rows = []

    # Helper to safe float conversion for comparison
    def _get_prob(row, col):
        try:
            return float(row.get(col) or 0.0)
        except:
            return 0.0

    # Helper to check if a value is a valid non-zero probability
    def _is_valid_prob(val):
        try:
            if pd.isna(val):
                return False
            f_val = float(val)
            return f_val != 0
        except:
            return False

    # Group by Game Key
    # We use 'league', 'Home', 'Away', 'Commence (UTC)' as the key
    # Ensure these columns exist
    required_cols = ["league", "Home", "Away", "Commence (UTC)"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    grouped = df.groupby(required_cols)

    for name, group in grouped:
        league, home, away, commence = name

        # Base info from the first row of the group
        first_row = group.iloc[0]
        commence_local = first_row.get("Commence (Local)")
        local_date = first_row.get("Local Date")

        summary = {
            "League": league,  # Changed from "league" to "League" to match UI
            "Home": home,
            "Away": away,
            "Commence UTC": commence,  # Changed from "Commence (UTC)" to "Commence UTC" to match UI
            "Commence (Local)": commence_local,
            "Local Date": local_date,
        }

        # ============================================
        # HasKalshiMarket Flag
        # ============================================
        has_kalshi_market = False
        for idx, row in group.iterrows():
            kalshi_matched = row.get("kalshi_matched")
            if kalshi_matched == True or str(kalshi_matched).lower() == "true":
                # Check if at least one Kalshi probability is available
                # Spread
                has_spread = (_is_valid_prob(row.get("kalshi_prob_spread")) or
                              _is_valid_prob(row.get("spread_prob_pick_kalshi")))

                # Total
                has_total = (_is_valid_prob(row.get("kalshi_prob_total")) or
                             _is_valid_prob(row.get("total_prob_pick_kalshi")))

                # Winner / Moneyline (Generic)
                has_ml = (_is_valid_prob(row.get("kalshi_prob")) or
                          _is_valid_prob(row.get("kalshi_prob_used")))

                if has_spread or has_total or has_ml:
                    has_kalshi_market = True
                    break

        summary["HasKalshiMarket"] = has_kalshi_market

        # --- Moneyline (ML) ---
        ml_rows = group[group["Market"] == "Moneyline"]
        ml_pick = None
        ml_prob = None

        if not ml_rows.empty:
            # Choose best ML row based on final_probability or consensus_prob_adj
            best_ml = ml_rows.loc[pd.to_numeric(ml_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]
            ml_pick = best_ml.get("Pick")
            ml_prob = best_ml.get("final_probability") or best_ml.get("consensus_prob_adj")

        summary["ML Pick"] = ml_pick  # Changed from "ML" to "ML Pick" to match UI
        summary["ML Prob"] = ml_prob

        # --- Spread ---
        spread_rows = group[group["Market"] == "Spread"]
        spread_pick = None
        spread_prob = None
        kalshi_spread_prob = None
        spread_market_prob = None
        spread_consensus_prob = None
        spread_consensus_str = None

        if not spread_rows.empty:
            # Choose best spread row
            best_spread = spread_rows.loc[pd.to_numeric(spread_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]

            # Set Spread Pick
            # Fix: Ensure we don't construct "Team 0" or similar invalid strings
            s_line = best_spread.get("Line")
            if best_spread.get("Spread & Pick"):
                spread_pick = best_spread.get("Spread & Pick")
            elif best_spread.get("Pick"):
                try:
                    s_val = float(s_line) if s_line is not None else 0.0
                    if abs(s_val) > 0.001:
                        # Format as float to strip leading zeros ("01" -> "1.0")
                        spread_pick = f"{best_spread.get('Pick')} {s_val:g}"
                    else:
                        spread_pick = None
                except (ValueError, TypeError):
                    spread_pick = None
            else:
                spread_pick = None

            # Set Spread Prob
            spread_prob = (best_spread.get("spread_prob_pick_final")
                           or best_spread.get("spread_prob_adj")
                           or best_spread.get("spread_prob")
                           or best_spread.get("spread_prob_market_based"))

            # Get Kalshi Spread Probability
            kalshi_spread_prob = (best_spread.get("spread_prob_pick_kalshi")
                                 or best_spread.get("kalshi_prob_spread"))

            # Get Market Spread Probability for delta calculation
            spread_market_prob = (best_spread.get("spread_prob_pick_market")
                                 or best_spread.get("spread_prob_market"))

            # --- CALCULATE SPREAD CONSENSUS ---
            spread_consensus_prob, spread_consensus_str = calculate_consensus_for_row(best_spread, "Spread")


        summary["Spread Pick"] = spread_pick  # Changed from "Spread" to "Spread Pick" to match UI
        summary["Spread Prob"] = spread_prob
        summary["Kalshi Spread Prob"] = kalshi_spread_prob
        summary["SpreadConsensusProb"] = spread_consensus_prob
        summary["SpreadConsensus"] = spread_consensus_str

        # Calculate Kalshi vs Market Delta for Spread
        if kalshi_spread_prob is not None and spread_market_prob is not None:
            try:
                delta = float(kalshi_spread_prob) - float(spread_market_prob)
                summary["Kalshi Spread Δ"] = f"{delta:+.1%}" if abs(delta) > 0.001 else "0.0%"
            except (ValueError, TypeError):
                summary["Kalshi Spread Δ"] = None
        else:
            summary["Kalshi Spread Δ"] = None

        # --- Total ---
        total_rows = group[group["Market"] == "Total"]
        total_pick = None
        total_prob = None
        kalshi_total_prob = None
        total_market_prob = None
        total_consensus_prob = None
        total_consensus_str = None

        if not total_rows.empty:
            best_total = total_rows.loc[pd.to_numeric(total_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]

            # Set Total Pick
            # Fix: Avoid "Under 0" and "Under 01" artifacts
            t_line = best_total.get("Line")
            if best_total.get("Total & Pick"):
                total_pick = best_total.get("Total & Pick")
            elif best_total.get("Pick"):
                try:
                    t_val = float(t_line) if t_line is not None else 0.0
                    if abs(t_val) > 0.001:
                        # Format as float to strip leading zeros ("01" -> "1.0")
                        total_pick = f"{best_total.get('Pick')} {t_val:g}"
                    else:
                        total_pick = None
                except (ValueError, TypeError):
                    total_pick = None
            else:
                total_pick = None

            total_prob = (best_total.get("total_prob_pick_final")
                          or best_total.get("total_prob_adj")
                          or best_total.get("total_prob")
                          or best_total.get("total_prob_market_based"))

            # Get Kalshi Total Probability
            kalshi_total_prob = (best_total.get("total_prob_pick_kalshi")
                                or best_total.get("kalshi_prob_total"))

            # Get Market Total Probability for delta calculation
            total_market_prob = (best_total.get("total_prob_pick_market")
                                or best_total.get("total_prob_market"))

            # --- CALCULATE TOTAL CONSENSUS ---
            total_consensus_prob, total_consensus_str = calculate_consensus_for_row(best_total, "Total")

        summary["Total Pick"] = total_pick  # Changed from "Total" to "Total Pick" to match UI
        summary["Total Prob"] = total_prob
        summary["Kalshi Total Prob"] = kalshi_total_prob
        summary["TotalConsensusProb"] = total_consensus_prob
        summary["TotalConsensus"] = total_consensus_str

        # Calculate Kalshi vs Market Delta for Total
        if kalshi_total_prob is not None and total_market_prob is not None:
            try:
                delta = float(kalshi_total_prob) - float(total_market_prob)
                summary["Kalshi Total Δ"] = f"{delta:+.1%}" if abs(delta) > 0.001 else "0.0%"
            except (ValueError, TypeError):
                summary["Kalshi Total Δ"] = None
        else:
            summary["Kalshi Total Δ"] = None

        # --- Overall Pick ---
        # Updated logic: Best Overall should only consider real spread/total picks with:
        # 1. Valid lines (not None/NaN)
        # 2. Probability > 50% (pick has positive expected value)
        # 3. No "no_valid_spread_or_total" warning
        # 4. Confidence bucket is MEDIUM or HIGH (not LOW)

        # Helper to check if probability is in valid range (> 50%)
        def _is_valid_prob_range(p):
            if p is None:
                return False
            try:
                p_float = float(p)
                # FIX: Changed from 50-65% range to just > 50%
                # A 68% spread should be selected over a 37.6% total
                return p_float > 0.50
            except:
                return False

        # Helper to check if row has warnings about invalid lines
        def _has_invalid_line_warnings(game_rows):
            for _, r in game_rows.iterrows():
                warnings = str(r.get("Warnings") or "")
                if "no_valid_spread_or_total" in warnings:
                    return True
            return False

        # Helper to get confidence bucket for a pick
        def _get_confidence(game_rows, market_type):
            for _, r in game_rows.iterrows():
                if str(r.get("Market") or "").upper() == market_type.upper():
                    return str(r.get("Pick_Confidence") or "LOW").upper()
            return "LOW"

        overall_pick = None
        overall_prob = None

        # Build eligible picks list
        eligible = []

        # Check if this game has invalid lines warning
        has_invalid_lines = _has_invalid_line_warnings(group)

        if not has_invalid_lines:
            # Get spread probability and confidence
            s_p = _get_prob(summary, "Spread Prob")
            spread_conf = _get_confidence(group, "Spread")

            # Get total probability and confidence
            t_p = _get_prob(summary, "Total Prob")
            total_conf = _get_confidence(group, "Total")

            # Add spread to eligible if valid
            if (spread_pick and s_p is not None and
                _is_valid_prob_range(s_p) and
                spread_conf in ["MEDIUM", "HIGH"]):
                eligible.append(("SPREAD", spread_pick, s_p))

            # Add total to eligible if valid
            if (total_pick and t_p is not None and
                _is_valid_prob_range(t_p) and
                total_conf in ["MEDIUM", "HIGH"]):
                eligible.append(("TOTAL", total_pick, t_p))

        # Select best pick from eligible list
        if eligible:
            # Pick the one with highest probability
            best = max(eligible, key=lambda x: x[2])
            overall_pick = best[1]
            overall_prob = best[2]

        summary["Best Overall Pick"] = overall_pick  # Changed from "Overall Pick" to "Best Overall Pick" to match UI
        summary["Best Overall Prob"] = overall_prob  # Changed from "Overall Prob" to "Best Overall Prob" to match UI

        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)

def reorder_for_spread_total_focus_v2(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    fixed_front = [c for c in [
        "League", "Home", "Away",
        "Commence UTC", "Commence (Local)", "Local Date",
    ] if c in df.columns]

    summary_block = [c for c in [
        "Best Overall Pick", "Best Overall Prob",
        "Spread Pick", "Spread Prob", "Spread Consensus", "Kalshi Spread Prob", "Kalshi Spread Δ",
        "Total Pick", "Total Prob", "Total Consensus", "Kalshi Total Prob", "Kalshi Total Δ",
        "ML Pick", "ML Prob",
    ] if c in df.columns]

    used = set(fixed_front + summary_block)
    remaining = [c for c in df.columns if c not in used]

    try:
        return df[fixed_front + summary_block + remaining]
    except Exception:
        return df
