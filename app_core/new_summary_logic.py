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
            "league": league,
            "Home": home,
            "Away": away,
            "Commence (UTC)": commence,
            "Commence (Local)": commence_local,
            "Local Date": local_date,
        }

        # --- Moneyline (ML) ---
        ml_rows = group[group["Market"] == "Moneyline"]
        ml_pick = None
        ml_prob = None

        if not ml_rows.empty:
            # Choose best ML row based on final_probability or consensus_prob_adj
            best_ml = ml_rows.loc[pd.to_numeric(ml_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]
            ml_pick = best_ml.get("Pick")
            ml_prob = best_ml.get("final_probability") or best_ml.get("consensus_prob_adj")

        summary["ML"] = ml_pick
        summary["ML Prob"] = ml_prob

        # --- Spread ---
        spread_rows = group[group["Market"] == "Spread"]
        spread_pick = None
        spread_prob = None

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

        summary["Spread"] = spread_pick
        summary["Spread Prob"] = spread_prob

        # --- Total ---
        total_rows = group[group["Market"] == "Total"]
        total_pick = None
        total_prob = None

        if not total_rows.empty:
            best_total = total_rows.loc[pd.to_numeric(total_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]

            total_pick = best_total.get("Total & Pick") or (str(best_total.get("Pick") or "") + " " + str(best_total.get("Line") or ""))

            total_prob = (best_total.get("total_prob_pick_final")
                          or best_total.get("total_prob_adj")
                          or best_total.get("total_prob")
                          or best_total.get("total_prob_market_based"))

        summary["Total"] = total_pick
        summary["Total Prob"] = total_prob

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

        summary["Overall Pick"] = overall_pick
        summary["Overall Prob"] = overall_prob

        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)

def reorder_for_spread_total_focus_v2(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    fixed_front = [c for c in [
        "league", "Home", "Away",
        "Commence (UTC)", "Commence (Local)", "Local Date",
    ] if c in df.columns]

    summary_block = [c for c in [
        "Overall Pick", "Overall Prob",
        "Spread", "Spread Prob",
        "Total", "Total Prob",
        "ML", "ML Prob",
    ] if c in df.columns]

    used = set(fixed_front + summary_block)
    remaining = [c for c in df.columns if c not in used]

    try:
        return df[fixed_front + summary_block + remaining]
    except Exception:
        return df
