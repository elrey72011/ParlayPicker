import re

def patch_file(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    # Find the _apply_triple_filter_ranking function
    if "def _apply_triple_filter_ranking(df: pd.DataFrame) -> pd.DataFrame:" in content:
        # We need to change how Triple_Filter_Rank is assigned
        # Instead of final_df['Triple_Filter_Rank'] = range(1, len(final_df) + 1)
        # We want to just return the dataframe sorted by tier_score and EV
        # And we will assign the rank per bucket later.
        content = re.sub(
            r"final_df = final_df\.sort_values\(by=\['tier_score', '_ev_sort_temp'\], ascending=\[True, False\]\)\.reset_index\(drop=True\)\n\s+final_df\['Triple_Filter_Rank'\] = range\(1, len\(final_df\) \+ 1\)",
            "final_df = final_df.sort_values(by=['tier_score', '_ev_sort_temp'], ascending=[True, False]).reset_index(drop=True)\n    # Rank is assigned later per bucket\n    final_df['Triple_Filter_Rank'] = range(1, len(final_df) + 1)",
            content
        )

    with open(file_path, "w") as f:
        f.write(content)

patch_file("core/streamlit_pipeline.py")
