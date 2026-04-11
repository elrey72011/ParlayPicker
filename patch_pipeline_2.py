import re

with open('core/streamlit_pipeline.py', 'r') as f:
    content = f.read()

# Update the consensus assignment in the loop to write to the dataframe
old_loop_start = """
        # Calculate consensus_agreement for the current row before threshold checks
        consensus_agreement = "No Kalshi"
        if pd.notna(kalshi_prob) and float(kalshi_prob) > 0.0:
            gap = win_prob - float(kalshi_prob)
            if gap >= 0.03:
                consensus_agreement = "Agrees"
            elif gap <= -0.03:
                consensus_agreement = "Disagrees"
            else:
                consensus_agreement = "Neutral"
"""

new_loop_start = """
        # Calculate consensus_agreement for the current row before threshold checks
        consensus_agreement = "No Kalshi"
        if pd.notna(kalshi_prob) and float(kalshi_prob) > 0.0:
            gap = win_prob - float(kalshi_prob)
            if gap >= 0.03:
                consensus_agreement = "Agrees"
            elif gap <= -0.03:
                consensus_agreement = "Disagrees"
            else:
                consensus_agreement = "Neutral"
        best.at[idx, "consensus_agreement"] = consensus_agreement
"""

content = content.replace(old_loop_start, new_loop_start)


# Remove the redundant vectorized consensus calculation logic after the loop since we now calculate it inside
old_redundant_logic = """
    if "consensus_agreement" not in best.columns:
        best["consensus_agreement"] = "No Kalshi"
    else:
        # Fill any NA consensus_agreements that may have carried over
        best["consensus_agreement"] = best["consensus_agreement"].fillna("No Kalshi")

    kalshi_prob = _numeric_series(best, "kalshi_probability") if "kalshi_probability" in best.columns else pd.Series([pd.NA]*len(best), index=best.index)
    is_kalshi_available = ((~pd.isna(kalshi_prob)) & (kalshi_prob > 0.0)).fillna(False).astype(bool)
    best["is_kalshi_available"] = is_kalshi_available

    if is_kalshi_available.any():
        blended = best["calibrated_probability"]
        gap = blended - kalshi_prob
        agrees_mask = (is_kalshi_available & gap.ge(0.03)).fillna(False).astype(bool)
        disagrees_mask = (is_kalshi_available & gap.le(-0.03)).fillna(False).astype(bool)
        best.loc[is_kalshi_available, "consensus_agreement"] = "Neutral"
        best.loc[agrees_mask, "consensus_agreement"] = "Agrees"
        best.loc[disagrees_mask, "consensus_agreement"] = "Disagrees"
"""

new_redundant_logic = """
    if "consensus_agreement" not in best.columns:
        best["consensus_agreement"] = "No Kalshi"
    else:
        best["consensus_agreement"] = best["consensus_agreement"].fillna("No Kalshi")

    kalshi_prob = _numeric_series(best, "kalshi_probability") if "kalshi_probability" in best.columns else pd.Series([pd.NA]*len(best), index=best.index)
    is_kalshi_available = ((~pd.isna(kalshi_prob)) & (kalshi_prob > 0.0)).fillna(False).astype(bool)
    best["is_kalshi_available"] = is_kalshi_available
"""
content = content.replace(old_redundant_logic, new_redundant_logic)

with open('core/streamlit_pipeline.py', 'w') as f:
    f.write(content)
