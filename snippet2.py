                store_decision_trace_sample(
                    league_name,
                    home,
                    away,
                    "total",
                    total_row.get("Pick"),
                    total_row.get("final_probability"),
                    trace_json_str,
                )
                total_row["Eligible_Top_Picks"] = eligible
                total_row = apply_sentiment_defaults(total_row, sentiment_defaults_base)
                rows_out.append(total_row)
                master_stats["market_rows_out"] += 1

        # 1. Create the base Master DataFrame from your processed rows
        master_df = pd.DataFrame(rows_out)

        # Force de-duplication of columns to prevent TypeError crashes
        master_df = master_df.loc[:, ~master_df.columns.duplicated()].copy()

        # 2. Add 'League' column if missing (required for enrichment lookup)
        if 'League' not in master_df.columns:
            master_df['League'] = league

        # 3. CRITICAL: Enrich the whole batch to fill 'feature_diff' columns
        # This fixes the 'Missing feature column' warnings in the logs
        with st.spinner("🚀 Running Batch Feature Enrichment..."):
            master_df = enrich_with_vertex_features(master_df, {league: api_sports_clients.get(league)})

        # 4. BATCH PREDICTION: Call the endpoint once for the whole sheet
        if is_vertex_prediction_configured():
            with st.spinner("🔮 Calling Vertex AI Batch Inference..."):
                from app_core.vertex_ai_endpoint import VERTEX_FEATURE_COLUMNS, predict_win_probabilities

                # 1. Sanitize the feature batch
                inference_df = master_df[VERTEX_FEATURE_COLUMNS].copy()
                for col in VERTEX_FEATURE_COLUMNS:
                    col_data = inference_df[col]
                    if isinstance(col_data, pd.DataFrame): col_data = col_data.iloc[:, 0]

                    # Force numeric, then replace both NaN AND Inf with 0.0
                    num_data = pd.to_numeric(col_data, errors='coerce')
                    inference_df[col] = num_data.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

                # 2. Batch Prediction Call
                probs = predict_win_probabilities(inference_df)
                if probs and len(probs) == len(master_df):
                    master_df["AI_Prob"] = probs
                    master_df["AI_Edge"] = master_df["AI_Prob"] - master_df.get("Implied_Prob", 0.5)
                else:
                    # Safe fallbacks to prevent KeyError: 'AI_Edge' in Optimizer
                    master_df["AI_Prob"] = 0.5
                    master_df["AI_Edge"] = 0.0

        # 4. SHOTGUN ACTIVATION: Use ParlayOptimizer to tier the results
        if ParlayOptimizer:
            optimizer = ParlayOptimizer(model_dir="./models")
            shotgun_picks = optimizer.get_shotgun_picks(master_df)
            st.session_state["shotgun_data"] = shotgun_picks

        # Collapse to one row per game (prefer the first generated row, typically moneyline) for Master View
        # NOTE: master_df now has ALL rows (ML/Spread/Total). We duplicate logic for deduping for the UI view if needed,
        # but the prompt implies we persist the FULL master_df to session state for tabs to use.

        # We need to preserve the sentiment metadata enrichment logic
        sentiment_meta_for_export = sentiment_pack_meta or init_sentiment_meta()
        # Vectorized or simple loop to fill sentiment meta if missing
        # (Assuming enrich_with_vertex_features preserves existing cols, which it does)

        # Deduping logic for "Master View" (one row per game)
        # We'll create a view for display, but keep master_df full for shotgun/optimizer.

        # But wait, the previous code replaced `df` with `deduped_list`.
        # If we overwrite `st.session_state["master_df"]` with the full `master_df`,
        # downstream code expecting 1 row per game might break.
        # However, the user instruction was "Persist to session state for the tabs to use".
        # The tabs (Shotgun) likely need the full rows.
        # The "Master Analysis" tab view logic (later in the file) uses `df` (which was deduped).
        # We should probably assign `df` to the deduped version for the immediate display logic below,
        # but maybe store `master_df_full` or similar?
        # Actually, let's follow the pattern but adapt for the existing `df` variable usage.

        # Apply sentiment meta to master_df
        # (Simulating what the loop did)
        if not master_df.empty:
            master_df["sentiment_sample_status"] = str(sentiment_meta_for_export.get("sentiment_sample_status", "NO_CALL") or "NO_CALL")
            master_df["sentiment_source"] = str(sentiment_meta_for_export.get("sentiment_source", "none") or "none")
            master_df["sentiment_status_counts"] = json.dumps(sentiment_meta_for_export.get("sentiment_status_counts", {"NO_CALL": 1}))
            master_df["sentiment_sample_query"] = sentiment_meta_for_export.get("sentiment_sample_query", "") or ""
            master_df["sentiment_disabled_reason"] = sentiment_meta_for_export.get("sentiment_disabled_reason", "") or ""
            master_df["sentiment_errors_sample"] = sentiment_meta_for_export.get("sentiment_errors_sample", "") or ""
            master_df["sentiment_error_count"] = int(sentiment_meta_for_export.get("sentiment_error_count", 0) or 0)

            # Fill remaining fields if they are missing or null in rows
            if "sentiment_status" not in master_df.columns:
                master_df["sentiment_status"] = None
            master_df["sentiment_status"] = master_df["sentiment_status"].fillna(sentiment_meta_for_export.get("sentiment_status"))

            if "sentiment_confidence" not in master_df.columns:
                master_df["sentiment_confidence"] = None
            master_df["sentiment_confidence"] = master_df["sentiment_confidence"].fillna(sentiment_meta_for_export.get("sentiment_confidence"))
