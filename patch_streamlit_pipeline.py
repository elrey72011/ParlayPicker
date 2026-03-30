import re

with open("core/streamlit_pipeline.py", "r") as f:
    content = f.read()

# Replace the ranking and sorting logic. We need:
# primary key = Triple_Filter_Rank (ascending)
# secondary = expected_value (descending)
# tertiary = edge (descending)
# missing/NaN to end

# Let's see what it has
# In `build_best_picks_df` right before returning:
old_sort_block = """        best = best.sort_values(
            by=["Pick_Status", "_rank_sort", "_ev_sort", "_edge_sort"],
            ascending=[True, True, False, False],
            na_position="last"
        ).reset_index(drop=True)

        best = best.drop(columns=["_rank_sort", "_ev_sort", "_edge_sort"], errors="ignore")"""

new_sort_block = """        best = best.sort_values(
            by=["Pick_Status", "_rank_sort", "_ev_sort", "_edge_sort"],
            ascending=[True, True, False, False],
            na_position="last"
        ).reset_index(drop=True)

        best = best.drop(columns=["_rank_sort", "_ev_sort", "_edge_sort"], errors="ignore")"""

# The logic actually looks right there. Why does the rank come out `7, 8, 2, 3, 9, 1, ...` ?
# Ah! It's because before the final sort, it concatenates bucket dfs?
# `Triple_Filter_Rank` gets re-assigned inside each bucket in a loop:
#         for status in status_order:
#             bucket_df = best[best["Pick_Status"] == status].copy()
#             if not bucket_df.empty:
#                 bucket_df = _apply_triple_filter_ranking(bucket_df)
#                 bucket_df["Triple_Filter_Rank"] = range(1, len(bucket_df) + 1)
#                 bucket_dfs.append(bucket_df)
#
# But wait... `_apply_triple_filter_ranking` ALREADY sorts the dataframe inside!
# And then it assigns `Triple_Filter_Rank` = 1..N.
# Then we concat, so now we have rows in order of 1..N for each bucket.
# THEN we sort again by expected_value and edge descending AFTER sorting by Rank? NO, wait.
# Wait, if `Triple_Filter_Rank` is 1..N within the bucket, AND we sort by `_rank_sort` ascending, the output should definitely be 1, 2, 3, 4, 5...
# But why did the user see 7, 8, 2, 3?
# Maybe because `Pick_Status` has a different order?
# Ah, `Triple_Filter_Rank` is also calculated BEFORE the loop:
# pool = _apply_triple_filter_ranking(pool)
# Then we do `pool = pool.sort_values...`
# Then we drop duplicates.
# Then we assign Pick_Status!
# Then at the very end we do:
#        for status in status_order:
#             bucket_df = best[best["Pick_Status"] == status].copy()
#             if not bucket_df.empty:
#                 bucket_df = _apply_triple_filter_ranking(bucket_df)
#                 # Ensure rank is local per bucket 1..N based on tier_score/ev
#                 bucket_df["Triple_Filter_Rank"] = range(1, len(bucket_df) + 1)
#                 bucket_dfs.append(bucket_df)
#
# So `Triple_Filter_Rank` SHOULD be 1..N per bucket.
# But look what happens after the concat:

#    for col in BEST_PICK_COLUMNS:
#        if col not in best.columns:
#            best[col] = pd.NA

#    best = best.drop(columns=["tier_score", "expected_value_sort", "is_unique", "is_kalshi_available", "_status_sort"], errors="ignore")

#    # Final guaranteed sort pass immediately before export
#    if not best.empty:
#        status_order = [
#            "Actionable",
#            "Below Threshold",
#            "Fallback / Low Confidence",
#            "No Play",
#            "Missing Line"
#        ]
#        best["Pick_Status"] = pd.Categorical(best["Pick_Status"], categories=status_order, ordered=True)
#        best["_rank_sort"] = pd.to_numeric(best["Triple_Filter_Rank"], errors="coerce")
#        best["_ev_sort"] = pd.to_numeric(best["expected_value"], errors="coerce")
#        best["_edge_sort"] = pd.to_numeric(best["edge"], errors="coerce")

#        best = best.sort_values(
#            by=["Pick_Status", "_rank_sort", "_ev_sort", "_edge_sort"],
#            ascending=[True, True, False, False],
#            na_position="last"
#        ).reset_index(drop=True)
#
# Let's inspect `_apply_triple_filter_ranking` again.
# Does `_apply_triple_filter_ranking` sort? Yes:
#     final_df = final_df.sort_values(by=['tier_score', '_ev_sort_temp'], ascending=[True, False]).reset_index(drop=True)
#     final_df['Triple_Filter_Rank'] = range(1, len(final_df) + 1)

# WHY is the rank out of order? The ONLY way the rank could be out of order is if something else modifies it OR if `pd.to_numeric(best["Triple_Filter_Rank"])` is messing it up.
# Wait, let's look at the pipeline.
# Run a quick script to see what `best` looks like right before export.
