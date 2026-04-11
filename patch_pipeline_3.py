import re

with open('core/streamlit_pipeline.py', 'r') as f:
    content = f.read()


old_metrics = """
        diagnostics_out["market_type_counts"] = final_best_df["market_type"].value_counts().to_dict()
        diagnostics_out["actionable_market_type_counts"] = actionable_market_type_counts

        # Add requested metrics directly to diagnostics_out
        diagnostics_out["actionable_counts_by_league"] = actionable_counts_by_league
        diagnostics_out["actionable_counts_by_market_type"] = actionable_market_type_counts
        diagnostics_out["actionable_counts_by_family"] = actionable_family_counts
        diagnostics_out["actionable_totals_below_floor"] = totals_below_prob_floor
        diagnostics_out["nhl_totals_actionable"] = nhl_totals_actionable
        diagnostics_out["spreads_downgraded_by_divergence"] = spreads_downgraded_by_divergence
        diagnostics_out["spreads_rescued_by_divergence"] = spreads_rescued_by_divergence
"""

new_metrics = """
        diagnostics_out["market_type_counts"] = final_best_df["market_type"].value_counts().to_dict()
        diagnostics_out["actionable_market_type_counts"] = actionable_market_type_counts

        # Add requested metrics directly to diagnostics_out
        diagnostics_out["actionable_counts_by_league"] = actionable_counts_by_league
        diagnostics_out["actionable_counts_by_market_type"] = actionable_market_type_counts
        diagnostics_out["actionable_counts_by_family"] = actionable_family_counts
        diagnostics_out["actionable_totals_below_floor"] = totals_below_prob_floor
        diagnostics_out["nhl_totals_actionable"] = nhl_totals_actionable
        diagnostics_out["spreads_downgraded_by_divergence"] = spreads_downgraded_by_divergence
        diagnostics_out["spreads_rescued_by_divergence"] = spreads_rescued_by_divergence

        # New Consensus and Calibration Tuning Metrics
        diagnostics_out["actionable_counts_by_consensus"] = actionable_df["consensus_agreement"].value_counts().to_dict() if "consensus_agreement" in actionable_df.columns else {}
        diagnostics_out["neutral_downgrades"] = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["Status_Reason"].str.contains("Neutral overlay", na=False))
        ].shape[0]
        diagnostics_out["disagrees_downgrades"] = final_best_df[
            (final_best_df["Pick_Status"] == "High Variance/Speculative") &
            (final_best_df["Status_Reason"].str.contains("Disagrees overlay", na=False))
        ].shape[0]
        diagnostics_out["side_prob_floor_downgrades"] = final_best_df[
            (final_best_df["Pick_Status"] == "Below Threshold") &
            (final_best_df["Status_Reason"].str.contains("Side Win Probability", na=False))
        ].shape[0]
        diagnostics_out["final_actionable_count"] = actionable_df.shape[0]

"""

content = content.replace(old_metrics, new_metrics)

with open('core/streamlit_pipeline.py', 'w') as f:
    f.write(content)
