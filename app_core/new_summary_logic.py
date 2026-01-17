import pandas as pd
import numpy as np

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
        # 2. At least one of kalshi_prob_spread or kalshi_prob_total is non-null and != 0
        #
        # This flag is critical for the "Markets" badge in the UI, which counts
        # games with usable Kalshi markets (independent of sportsbook market counts).
        # ============================================
        has_kalshi_market = False
        for idx, row in group.iterrows():
            kalshi_matched = row.get("kalshi_matched")
            if kalshi_matched == True or str(kalshi_matched).lower() == "true":
                # Check if at least one Kalshi probability is available
                kalshi_prob_spread = row.get("kalshi_prob_spread")
                kalshi_prob_total = row.get("kalshi_prob_total")

                # Check for valid non-zero probabilities
                has_spread = pd.notnull(kalshi_prob_spread) and kalshi_prob_spread != 0
                has_total = pd.notnull(kalshi_prob_total) and kalshi_prob_total != 0

                if has_spread or has_total:
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

        summary["Spread Pick"] = spread_pick  # Changed from "Spread" to "Spread Pick" to match UI
        summary["Spread Prob"] = spread_prob
        summary["Kalshi Spread Prob"] = kalshi_spread_prob

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
        "Spread Pick", "Spread Prob", "Kalshi Spread Prob", "Kalshi Spread Δ",
        "Total Pick", "Total Prob", "Kalshi Total Prob", "Kalshi Total Δ",
        "ML Pick", "ML Prob",
    ] if c in df.columns]

    used = set(fixed_front + summary_block)
    remaining = [c for c in df.columns if c not in used]

    try:
        return df[fixed_front + summary_block + remaining]
    except Exception:
        return df
