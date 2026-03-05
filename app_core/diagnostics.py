import pandas as pd
import numpy as np
from typing import Dict, Any

def compute_daily_diagnostics(enriched_df: pd.DataFrame, probability_col: str = 'final_probability') -> Dict[str, Any]:
    """
    Computes daily diagnostics from enriched dataframe containing outcomes.
    Returns a lightweight summary object suitable for UI display.
    """
    if enriched_df is None or enriched_df.empty:
        return {"error": "Empty dataframe"}

    df = enriched_df.copy()

    market_col = next((c for c in ['best_pick_type', 'Market'] if c in df.columns), None)
    if not market_col:
        return {"error": "Missing market column"}

    # Filter to recommended picks that are valid (SPREAD, TOTAL)
    df = df[df[market_col].str.upper().isin(['SPREAD', 'TOTAL'])]
    if 'edge' in df.columns:
        df = df[pd.to_numeric(df['edge'], errors='coerce').notna() & (pd.to_numeric(df['edge'], errors='coerce') > 0)]

    # Prepare result column logic
    def get_result(row):
        m = str(row[market_col]).upper()
        if m == 'SPREAD' and 'spread_result' in row:
            return row['spread_result']
        elif m == 'TOTAL' and 'total_result' in row:
            return row['total_result']
        return 'N/A'

    df['actual_result'] = df.apply(get_result, axis=1)

    # Filter to only WIN/LOSS
    valid_df = df[df['actual_result'].isin(['WIN', 'LOSS'])].copy()
    if valid_df.empty:
        return {"message": "No valid picks with outcomes found"}

    valid_df['is_win'] = (valid_df['actual_result'] == 'WIN').astype(int)

    total_picks = len(valid_df)
    total_wins = valid_df['is_win'].sum()
    overall_win_rate = total_wins / total_picks if total_picks > 0 else 0

    # By League and Market
    groups = valid_df.groupby(['league', market_col])
    group_stats = []
    for (league, market), group in groups:
        w = group['is_win'].sum()
        t = len(group)
        group_stats.append({
            'league': league,
            'market': str(market).upper(),
            'wins': int(w),
            'total': int(t),
            'win_rate': float(w/t),
            'avg_prob': float(group[probability_col].mean())
        })

    # By Probability Bucket
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.0]
    valid_df['prob_bucket'] = pd.cut(valid_df[probability_col], bins=bins, right=False)

    bucket_stats = []
    b_groups = valid_df.groupby('prob_bucket', observed=False)
    for bucket, group in b_groups:
        w = group['is_win'].sum()
        t = len(group)
        if t > 0:
            bucket_stats.append({
                'bucket': str(bucket),
                'wins': int(w),
                'total': int(t),
                'win_rate': float(w/t),
                'avg_prob': float(group[probability_col].mean())
            })

    # Find biggest under/over performers
    anomalies = []
    for stat in group_stats:
        if stat['total'] >= 3: # Need at least a few picks to be an anomaly
            expected = stat['avg_prob']
            actual = stat['win_rate']
            diff = actual - expected

            if diff <= -0.10:
                anomalies.append(f"{stat['league']} {stat['market']} underperformed by {abs(diff)*100:.1f} points (Hit {actual:.1%} vs Expected {expected:.1%})")
            elif diff >= 0.10:
                anomalies.append(f"{stat['league']} {stat['market']} overperformed by {diff*100:.1f} points (Hit {actual:.1%} vs Expected {expected:.1%})")

    for stat in bucket_stats:
        if stat['total'] >= 3:
            expected = stat['avg_prob']
            actual = stat['win_rate']
            diff = actual - expected

            if diff <= -0.10:
                anomalies.append(f"Bucket {stat['bucket']} underperformed by {abs(diff)*100:.1f} points")
            elif diff >= 0.10:
                anomalies.append(f"Bucket {stat['bucket']} overperformed by {diff*100:.1f} points")

    return {
        "overall": {
            "total_picks": int(total_picks),
            "wins": int(total_wins),
            "win_rate": float(overall_win_rate)
        },
        "by_group": group_stats,
        "by_bucket": bucket_stats,
        "anomalies": anomalies
    }
