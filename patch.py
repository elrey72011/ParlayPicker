with open('streamlit_app.py', 'r') as f:
    content = f.read()

# Replace the sorting logic at the end of the script
old_sort = """            # Apply explicit secondary sorts before export as requested
            sort_cols = ["expected_value", "Commence (Local)", "league", "Home"]
            available_sort_cols = [c for c in sort_cols if c in best_picks_export.columns]
            if available_sort_cols:
                asc = [False] + [True] * (len(available_sort_cols) - 1)
                best_picks_export = best_picks_export.sort_values(available_sort_cols, ascending=asc).reset_index(drop=True)
                if "parlay_rank" in best_picks_export.columns:
                    best_picks_export["parlay_rank"] = range(1, len(best_picks_export) + 1)"""

new_sort = """            # Apply explicit secondary sorts before export as requested
            # We must preserve the primary Pick_Status > Triple_Filter_Rank > EV > Edge order
            status_order = [
                "Actionable",
                "Below Threshold",
                "Fallback / Low Confidence",
                "No Play",
                "Missing Line"
            ]
            if "Pick_Status" in best_picks_export.columns:
                best_picks_export["Pick_Status"] = pd.Categorical(best_picks_export["Pick_Status"], categories=status_order, ordered=True)

            best_picks_export["_rank_sort"] = pd.to_numeric(best_picks_export.get("Triple_Filter_Rank"), errors="coerce")
            best_picks_export["_ev_sort"] = pd.to_numeric(best_picks_export.get("expected_value"), errors="coerce")
            best_picks_export["_edge_sort"] = pd.to_numeric(best_picks_export.get("edge"), errors="coerce")

            sort_cols = ["Pick_Status", "_rank_sort", "_ev_sort", "_edge_sort"]
            available_sort_cols = [c for c in sort_cols if c in best_picks_export.columns]

            if available_sort_cols:
                asc = [True, True, False, False][:len(available_sort_cols)]
                best_picks_export = best_picks_export.sort_values(available_sort_cols, ascending=asc, na_position="last").reset_index(drop=True)

                # Drop temporary sort columns
                best_picks_export = best_picks_export.drop(columns=["_rank_sort", "_ev_sort", "_edge_sort"], errors="ignore")

                if "parlay_rank" in best_picks_export.columns:
                    best_picks_export["parlay_rank"] = range(1, len(best_picks_export) + 1)"""

content = content.replace(old_sort, new_sort)

with open('streamlit_app.py', 'w') as f:
    f.write(content)
