import pandas as pd
import numpy as np

def calculate_implied_prob(odds):
    if pd.isna(odds):
        return np.nan
    try:
        odds = float(odds)
        if odds == 0: return np.nan
        if odds < 0:
            return (-odds) / (-odds + 100)
        else:
            return 100 / (odds + 100)
    except:
        return np.nan

def analyze():
    # Load data
    try:
        df = pd.read_csv('master_df_raw.csv')
        print(f"Loaded master_df_raw.csv with {len(df)} rows.")
    except Exception as e:
        print(f"Error loading master_df_raw.csv: {e}")
        return

    # 1. % of rows with non-null sentiment values
    # We check 'Home_Sentiment' and 'Away_Sentiment'.

    def is_valid_sentiment(x):
        if pd.isna(x): return False
        if isinstance(x, str):
            if x.lower() in ['nan', 'none', '', 'null']: return False
        return True

    # Check coverage for sentiment columns
    has_sentiment = df.apply(lambda row: is_valid_sentiment(row.get('Home_Sentiment')) or is_valid_sentiment(row.get('Away_Sentiment')), axis=1)
    pct_non_null_sentiment = (has_sentiment.sum() / len(df)) * 100 if len(df) > 0 else 0

    # 2. Distribution of sentiment_status values
    if 'sentiment_status' in df.columns:
        sentiment_status_dist = df['sentiment_status'].value_counts(dropna=False).to_dict()
    else:
        sentiment_status_dist = {}

    # 3. Implied win probability calculated from existing spread/total odds columns
    # We use 'spread_pick_odds' and 'total_pick_odds'.

    if 'spread_pick_odds' in df.columns:
        df['spread_implied_prob_calc'] = df['spread_pick_odds'].apply(calculate_implied_prob)
    else:
        df['spread_implied_prob_calc'] = np.nan

    if 'total_pick_odds' in df.columns:
        df['total_implied_prob_calc'] = df['total_pick_odds'].apply(calculate_implied_prob)
    else:
        df['total_implied_prob_calc'] = np.nan

    # 4. Data quality issues or suspicious patterns
    issues = []

    # Check: Spread Line exists but Odds missing
    if 'spread_pick_line' in df.columns and 'spread_pick_odds' in df.columns:
        missing_odds = df[df['spread_pick_line'].notna() & df['spread_pick_odds'].isna()]
        if len(missing_odds) > 0:
            issues.append(f"{len(missing_odds)} rows have Spread Line but missing Spread Odds.")

    # Check: Total Line exists but Odds missing
    if 'total_pick_line' in df.columns and 'total_pick_odds' in df.columns:
        missing_total_odds = df[df['total_pick_line'].notna() & df['total_pick_odds'].isna()]
        if len(missing_total_odds) > 0:
            issues.append(f"{len(missing_total_odds)} rows have Total Line but missing Total Odds.")

    # Check: Extreme probabilities
    if 'spread_implied_prob_calc' in df.columns:
        extreme_mask = (df['spread_implied_prob_calc'] < 0.01) | (df['spread_implied_prob_calc'] > 0.99)
        if extreme_mask.any():
            issues.append(f"{extreme_mask.sum()} rows have extreme calculated spread implied probabilities (<1% or >99%).")

    # Check: Discrepancy between calculated and existing implied prob (if exists)
    if 'spread_implied_prob' in df.columns and 'spread_implied_prob_calc' in df.columns:
        # Assuming existing column might be text or float
        # Let's try to convert to float
        df['spread_implied_prob_existing'] = pd.to_numeric(df['spread_implied_prob'], errors='coerce')
        # Check difference > 0.01 (ignoring NaNs)
        diff = (df['spread_implied_prob_calc'] - df['spread_implied_prob_existing']).abs()
        large_diffs = diff[diff > 0.01]
        if len(large_diffs) > 0:
            issues.append(f"{len(large_diffs)} rows have mismatch > 1% between calculated and existing spread_implied_prob.")

    # Log Dump Analysis
    log_stats = {}
    try:
        log_df = pd.read_csv('theover_log_dump.csv')
        print(f"Loaded theover_log_dump.csv with {len(log_df)} rows.")
        log_stats['total_rows'] = len(log_df)
        if 'status' in log_df.columns:
            log_stats['status_dist'] = log_df['status'].value_counts().to_dict()

        # Check for low confidence matches if 'confidence' column exists
        if 'confidence' in log_df.columns:
             # Ensure numeric
             log_df['confidence'] = pd.to_numeric(log_df['confidence'], errors='coerce')
             low_conf = log_df[log_df['confidence'] < 70]
             if len(low_conf) > 0:
                 issues.append(f"{len(low_conf)} rows in log dump have match confidence < 70.")

    except Exception as e:
        print(f"Error loading log dump: {e}")
        log_stats['error'] = str(e)


    # Write Report
    with open('data_quality_report.md', 'w') as f:
        f.write("# Data Quality Report\n\n")

        f.write("## 1. Sentiment Coverage\n")
        f.write(f"- Percentage of rows with non-null sentiment values (Home/Away Sentiment): **{pct_non_null_sentiment:.2f}%**\n\n")

        f.write("## 2. Distribution of Sentiment Status\n")
        if sentiment_status_dist:
            f.write("| Status | Count |\n|---|---|\n")
            for k, v in sentiment_status_dist.items():
                f.write(f"| {k} | {v} |\n")
        else:
            f.write("No 'sentiment_status' column found.\n")
        f.write("\n")

        f.write("## 3. Implied Win Probability Stats (Calculated)\n")
        f.write("### Spread Implied Probabilities\n")
        if 'spread_implied_prob_calc' in df.columns:
            desc = df['spread_implied_prob_calc'].describe()
            f.write(desc.to_markdown())
        else:
            f.write("Could not calculate spread implied probabilities.\n")
        f.write("\n\n")

        f.write("### Total Implied Probabilities\n")
        if 'total_implied_prob_calc' in df.columns:
            desc = df['total_implied_prob_calc'].describe()
            f.write(desc.to_markdown())
        else:
            f.write("Could not calculate total implied probabilities.\n")
        f.write("\n\n")

        f.write("## 4. Data Quality Issues & Suspicious Patterns\n")
        if issues:
            for i in issues:
                f.write(f"- {i}\n")
        else:
            f.write("- No obvious data quality issues found in checked areas.\n")

        f.write("\n## TheOver Log Dump Analysis\n")
        if 'error' in log_stats:
            f.write(f"- Error: {log_stats['error']}\n")
        else:
            f.write(f"- Total Entries: {log_stats.get('total_rows', 0)}\n")
            if 'status_dist' in log_stats:
                f.write("### Match Status Distribution\n")
                for k, v in log_stats['status_dist'].items():
                    f.write(f"- **{k}**: {v}\n")

if __name__ == "__main__":
    analyze()
