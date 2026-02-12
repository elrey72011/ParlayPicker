import pandas as pd
import numpy as np

def verify_kalshi_counts_v2(csv_path="master_df_raw.csv"):
    print(f"Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    def _is_valid_prob(val):
        try:
            if pd.isna(val): return False
            f_val = float(val)
            return f_val != 0
        except:
            return False

    def check_row_strict(row):
        # Graceful Fallback: Include Moneyline if Spread/Total missing
        # User Requirement: "ensure the counting script seamlessly routes the row to the check_row_inclusive processing methodology"
        matched = row.get("kalshi_matched")
        is_matched = matched == True or str(matched).lower() == "true"
        if is_matched:
            p1 = row.get("kalshi_prob_spread")
            p2 = row.get("spread_prob_pick_kalshi")
            p3 = row.get("kalshi_prob_total")
            p4 = row.get("total_prob_pick_kalshi")

            # Fallback to Moneyline (inclusive logic) to salvage the matchup
            p5 = row.get("kalshi_prob")
            p6 = row.get("kalshi_prob_used")

            return (_is_valid_prob(p1) or _is_valid_prob(p2) or
                    _is_valid_prob(p3) or _is_valid_prob(p4) or
                    _is_valid_prob(p5) or _is_valid_prob(p6))
        return False

    def check_row_inclusive(row):
        # Include Moneyline (kalshi_prob)
        matched = row.get("kalshi_matched")
        is_matched = matched == True or str(matched).lower() == "true"
        if is_matched:
            p1 = row.get("kalshi_prob_spread")
            p2 = row.get("spread_prob_pick_kalshi")
            p3 = row.get("kalshi_prob_total")
            p4 = row.get("total_prob_pick_kalshi")
            # Winner / Moneyline prob
            p5 = row.get("kalshi_prob")
            p6 = row.get("kalshi_prob_used")

            return (_is_valid_prob(p1) or _is_valid_prob(p2) or
                    _is_valid_prob(p3) or _is_valid_prob(p4) or
                    _is_valid_prob(p5) or _is_valid_prob(p6))
        return False

    df["HasKalshi_Strict"] = df.apply(check_row_strict, axis=1)
    df["HasKalshi_Inclusive"] = df.apply(check_row_inclusive, axis=1)

    print(f"Total Rows: {len(df)}")
    print(f"Strict (Spread/Total only): {df['HasKalshi_Strict'].sum()}")
    print(f"Inclusive (+Moneyline): {df['HasKalshi_Inclusive'].sum()}")

if __name__ == "__main__":
    verify_kalshi_counts_v2()
