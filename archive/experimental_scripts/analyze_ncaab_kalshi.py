import pandas as pd

# Load the master_df_raw
df = pd.read_csv('app_exports/master_df_raw - 2026-02-20T145457.295.csv')

print("Columns:", df.columns.tolist())

# Filter for NCAAB and Kalshi_Required == True
ncaab_kalshi = df[(df['league'] == 'NCAAB') & (df['Kalshi_Required'] == True)]

# Compute total count
total_required = len(ncaab_kalshi)

# Compute matched count
matched_count = len(ncaab_kalshi[ncaab_kalshi['kalshi_matched'] == True])

# Identify unmatched games
unmatched = ncaab_kalshi[ncaab_kalshi['kalshi_matched'] == False]

# Use available columns
cols_to_show = ['Home', 'Away', 'Game', 'Market', 'kalshi_available', 'kalshi_matched', 'kalshi_status', 'kalshi_match_reason', 'Kalshi_Required']
# Check for 'Commence' or similar
for col in df.columns:
    if 'Commence' in col:
        cols_to_show.append(col)

unmatched_info = unmatched[cols_to_show]

print(f"Total NCAAB Kalshi Required: {total_required}")
print(f"Matched: {matched_count}")
print("\nUnmatched Games:")
print(unmatched_info.to_string())
