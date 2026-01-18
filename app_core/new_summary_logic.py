import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("parlaypicker")

def calculate_spread_consensus(row: pd.Series) -> tuple[float, str]:
    """
    Calculates Spread Consensus using weighted blend of Market, Kalshi, and AI.
    Weights: Market (31.58), Kalshi (36.84), AI (21.05).

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

    # 1. Get Probabilities
    # Market
    p_market = _get_f("spread_prob_pick_market")
    if p_market is None: p_market = _get_f("spread_prob_market_based")
    if p_market is None: p_market = _get_f("spread_prob_market")

    # Kalshi
    p_kalshi = _get_f("spread_prob_pick_kalshi")
    # Fallback to generic Kalshi prob if pick-specific is missing, but only if we can trust it matches the pick
    # For now, rely on pick specific or if we can infer.
    # Actually, spread_prob_pick_kalshi is populated by enrich_with_consensus or similar logic.
    if p_kalshi is None:
        # Check if kalshi_prob_spread exists and if it aligns?
        # Risky without side check. Let's stick to explicit pick prob if possible.
        pass

    # AI
    p_model = _get_f("model_spread_prob")
    if p_model is None: p_model = _get_f("AI_Prob")

    # 2. Weights (from prompt)
    W_M = 31.58
    W_K = 36.84
    W_AI = 21.05

    sources = []
    if p_market is not None: sources.append(("M", p_market, W_M))
    if p_kalshi is not None: sources.append(("K", p_kalshi, W_K))
    if p_model is not None: sources.append(("AI", p_model, W_AI))

    # 3. Calculate
    if not sources:
        # Fallback to final prob if available
        p_final = _get_f("spread_prob_pick_final") or _get_f("final_probability")
        if p_final is not None:
             # Just show final
             return p_final, f"{int(p_final*100)}% (Final)"

        # Log missing data
        logger.debug(f"Missing spread consensus data for {row.get('Home')} vs {row.get('Away')}")
        return 0.5, "N/A"

    total_w = sum(s[2] for s in sources)
    weighted_sum = sum(s[1] * s[2] for s in sources)

    if total_w == 0:
        return 0.5, "N/A"

    consensus = weighted_sum / total_w
    consensus = max(0.01, min(0.99, consensus))

    # 4. Format
    # "56% (M55 / K59 / AI52)"
    parts = [f"{s[0]}{int(s[1]*100)}" for s in sources]
    breakdown = " / ".join(parts)

    formatted_str = f"{int(consensus*100)}% ({breakdown})"

    # Log if we are falling back to 50% despite having data (should be caught by if not sources)

    return consensus, formatted_str

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
        # A game has a Kalshi market if ANY row in the group satisfies:
        # 1. kalshi_matched == True
        # 2. At least one of:
        #    - Spread: kalshi_prob_spread, spread_prob_pick_kalshi
        #    - Total: kalshi_prob_total, total_prob_pick_kalshi
        #    - Winner/ML: kalshi_prob, kalshi_prob_used
        #    is non-null and != 0
        #
        # This flag is critical for the "Markets" badge in the UI.
        # ============================================
        has_kalshi_market = False
        for idx, row in group.iterrows():
            kalshi_matched = row.get("kalshi_matched")
            if kalshi_matched == True or str(kalshi_matched).lower() == "true":
                # Check if at least one Kalshi probability is available
                # We check raw kalshi prob columns and mapped pick columns

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
            spread_pick = best_spread.get("Spread & Pick") or (str(best_spread.get("Pick") or "") + " " + str(best_spread.get("Line") or ""))

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
            spread_consensus_prob, spread_consensus_str = calculate_spread_consensus(best_spread)


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

        if not total_rows.empty:
            best_total = total_rows.loc[pd.to_numeric(total_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]

            total_pick = best_total.get("Total & Pick") or (str(best_total.get("Pick") or "") + " " + str(best_total.get("Line") or ""))

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

        summary["Total Pick"] = total_pick  # Changed from "Total" to "Total Pick" to match UI
        summary["Total Prob"] = total_prob
        summary["Kalshi Total Prob"] = kalshi_total_prob

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
        # "Define overall as the best moneyline pick per game: Overall Pick = same as ML."
        # "If no moneyline row exists, fallback to whichever of Spread or Total has the highest probability for that game."

        overall_pick = ml_pick
        overall_prob = ml_prob

        if overall_pick is None:
            # Fallback
            s_p = _get_prob(summary, "Spread Prob")
            t_p = _get_prob(summary, "Total Prob")

            if s_p > 0 or t_p > 0:
                if s_p >= t_p:
                    overall_pick = spread_pick
                    overall_prob = spread_prob
                else:
                    overall_pick = total_pick
                    overall_prob = total_prob

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
        "Total Pick", "Total Prob", "Kalshi Total Prob", "Kalshi Total Δ",
        "ML Pick", "ML Prob",
    ] if c in df.columns]

    used = set(fixed_front + summary_block)
    remaining = [c for c in df.columns if c not in used]

    try:
        return df[fixed_front + summary_block + remaining]
    except Exception:
        return df
