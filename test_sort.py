import pandas as pd
from core.streamlit_pipeline import build_best_picks_df

# Let's inspect `core/streamlit_pipeline.py` one more time. Wait!
# I noticed this at line 1845:
#    for col in BEST_PICK_COLUMNS:
#        if col not in best.columns:
#            best[col] = pd.NA
#
# Wait, `BEST_PICK_COLUMNS` has `Triple_Filter_Rank`.
# If it is missing, it sets it to pd.NA... Wait, no, it's NOT missing, it was added inside the bucket iteration.
# Look closely at lines 1845-1869:
#    for col in BEST_PICK_COLUMNS:
#        if col not in best.columns:
#            best[col] = pd.NA

#    # Final Cleanup: Drop temporary columns used for processing
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

# wait... WHY would `Triple_Filter_Rank` come out as `7, 8, 2, 3, 9, 1, 10, 11, ...` ?
# Could it be because `pd.to_numeric` is coercing strings and maybe some of them are getting sorted weirdly?
# Wait, the rank is assigned using `range(1, len(bucket_df) + 1)`. These are literally integers!
# If `Triple_Filter_Rank` is `range(1, 18)` inside the "Actionable" bucket...
# and then we sort by `["Pick_Status", "_rank_sort", ...]` where `_rank_sort` is `1, 2, 3...17`.
# Why on earth would it come out as `7, 8, 2, 3`?
# Oh!!!
# I see it now.
# In `build_best_picks_df`:
# Look at line 1589-1590 in `_apply_triple_filter_ranking`:
#    final_df = final_df.sort_values(by=['tier_score', '_ev_sort_temp'], ascending=[True, False]).reset_index(drop=True)
#    final_df['Triple_Filter_Rank'] = range(1, len(final_df) + 1)
# That's perfectly 1..N.
# But then in `build_best_picks_df` inside the bucket iteration:
#    bucket_df = _apply_triple_filter_ranking(bucket_df)
#    bucket_df["Triple_Filter_Rank"] = range(1, len(bucket_df) + 1)
#
# But wait... after concatenating the bucket_dfs...
#        if bucket_dfs:
#            best = pd.concat(bucket_dfs, ignore_index=True)
#
#        best["Pick_Status"] = pd.Categorical(best["Pick_Status"], categories=status_order, ordered=True)
#        best["_rank_sort"] = pd.to_numeric(best["Triple_Filter_Rank"], errors="coerce")
#
# Does `pd.concat` change the order of rows? No.
# What if some rows were duplicates?
# Wait!
# Look at line 1812-1813:
#        bucket_df = _apply_triple_filter_ranking(bucket_df)
#        bucket_df["Triple_Filter_Rank"] = range(1, len(bucket_df) + 1)
# `_apply_triple_filter_ranking` RETURNS A SORTED DATAFRAME based on `tier_score` (ascending) and `_ev_sort_temp` (descending).
# BUT IT DOESN'T SORT BY EDGE!
# And what if `tier_score` computation inside `_apply_triple_filter_ranking` does something unexpected?
# Ah... look closely at what `_apply_triple_filter_ranking` DOES!
# It calculates `tier_score`.
# BUT then look at the final sort pass at line 1860:
#        best = best.sort_values(
#            by=["Pick_Status", "_rank_sort", "_ev_sort", "_edge_sort"],
#            ascending=[True, True, False, False],
#            na_position="last"
#        ).reset_index(drop=True)
#
# Wait! `_rank_sort` is JUST `Triple_Filter_Rank` (which is 1, 2, 3... within the bucket).
# Sorting by `Pick_Status` and then `_rank_sort` ASCENDING will keep the EXACT SAME ORDER as they were returned from `_apply_triple_filter_ranking`!
# Because within a `Pick_Status`, `_rank_sort` is ALREADY 1, 2, 3...N.
# So sorting by `_rank_sort` doesn't change anything, it just preserves the order from `_apply_triple_filter_ranking`!
# BUT the user is saying:
# "Current examples from the latest export:
# Actionable ranks: 7, 8, 2, 3, 9, 1, 10, 11, 4, 12, 5, 6, 13, 14, 15, 16, 17"
# How can it possibly be 7, 8, 2, 3 if `Triple_Filter_Rank` was JUST assigned as `range(1, len(bucket)+1)` ???
# UNLESS...
# The export doesn't show the `Triple_Filter_Rank` we generated in `build_best_picks_df`!
# What if something else overwrites it?
# Let's check where `build_best_picks_df` is called.

