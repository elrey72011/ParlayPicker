import pandas as pd
import numpy as np

def generate_calibration_dataset(
    historical_df: pd.DataFrame,
    probability_col: str = 'final_probability',
    bins: list = None
) -> pd.DataFrame:
    """
    Generates calibration metrics by binning the probabilities and calculating the empirical win rate.

    Args:
        historical_df: Enriched DataFrame with outcomes (e.g., from attach_results).
        probability_col: The column containing the predicted probability.
        bins: List of bin edges. Default is [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0].

    Returns:
        DataFrame containing calibration metrics per League, MarketType, and Probability Bucket.
    """
    if historical_df is None or historical_df.empty:
        return pd.DataFrame()

    df = historical_df.copy()

    if bins is None:
        bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

    # Standardize 'Market' or 'best_pick_type'
    market_col = next((c for c in ['best_pick_type', 'Market'] if c in df.columns), None)
    if not market_col:
        return pd.DataFrame()

    # Filter to recommended picks (valid SPREAD or TOTAL)
    # We might have an 'edge' column to check for "no discernible value"
    if 'edge' in df.columns:
        # Assuming edge > 0 means discernible value, or just exclude text
        df = df[pd.to_numeric(df['edge'], errors='coerce').notna() & (pd.to_numeric(df['edge'], errors='coerce') > 0)]

    df = df[df[market_col].str.upper().isin(['SPREAD', 'TOTAL'])]

    results = []

    # Bucket probabilities
    df['prob_bucket'] = pd.cut(df[probability_col], bins=bins, right=False, include_lowest=True)

    # Process each League x MarketType
    groups = df.groupby(['league', market_col])

    for (league, market), group in groups:
        market_upper = str(market).upper()

        # Calculate result status based on market type
        if market_upper == 'SPREAD' and 'spread_result' in group.columns:
            group['win'] = (group['spread_result'] == 'WIN').astype(int)
            group['valid'] = group['spread_result'].isin(['WIN', 'LOSS'])
        elif market_upper == 'TOTAL' and 'total_result' in group.columns:
            group['win'] = (group['total_result'] == 'WIN').astype(int)
            group['valid'] = group['total_result'].isin(['WIN', 'LOSS'])
        else:
            continue

        # Filter out pushes and N/A
        valid_group = group[group['valid']]

        # Group by bucket
        bucket_groups = valid_group.groupby('prob_bucket', observed=False)

        for bucket, b_group in bucket_groups:
            num_picks = len(b_group)
            if num_picks > 0:
                wins = b_group['win'].sum()
                emp_win_rate = wins / num_picks
                avg_prob = b_group[probability_col].mean()

                results.append({
                    'league': league,
                    'market_type': market_upper,
                    'bucket': str(bucket),
                    'num_picks': num_picks,
                    'wins': wins,
                    'empirical_win_rate': emp_win_rate,
                    'avg_predicted_prob': avg_prob
                })

    return pd.DataFrame(results)
