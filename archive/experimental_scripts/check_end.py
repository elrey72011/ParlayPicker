
            status = str(row.get("kalshistatus", "")).lower()
            if status in ("matched", "strictmatch", "seriesmatch") or row.get("kalshi_matched"):
                return "partial"
            return "none"

        df["Kalshi_Mode"] = df.apply(_map_kalshi_mode, axis=1)

        # -------------------------------------------------------------------------
        # TASK 1: Stats Quality Penalty (NCAAB)
        # -------------------------------------------------------------------------
        def _apply_stats_quality_penalty(row):
            # Only apply if explicitly MISSING (ESPN/REAL are fine)
            if row.get("stats_quality") == "MISSING":
                # Downgrade confidence bucket
                conf = row.get("Pick_Confidence")
                reason = str(row.get("confidence_reason") or "")

                # Strict downgrade logic
                if conf == "HIGH":
                    row["Pick_Confidence"] = "MEDIUM"
                    reason += " (downgraded: missing stats)"
                elif conf == "MEDIUM":
                    row["Pick_Confidence"] = "LOW"
                    reason += " (downgraded: missing stats)"
                elif conf == "LOW":
                    # Already low, just append reason
                    reason += " (stats missing)"
                else:
                    # Default/Unknown -> LOW
                    row["Pick_Confidence"] = "LOW"
                    reason += " (stats missing)"

                row["confidence_reason"] = reason.strip(" |")

                # Dampen probabilities (shrink edge by 50%)
                for col in ["spread_prob_pick_final", "total_prob_pick_final", "Best Overall Prob", "final_probability"]:
                    val = row.get(col)
                    if val is not None and pd.notna(val):
                        try:
                            fval = float(val)
                            # Dampen towards 0.5
                            row[col] = 0.5 + (fval - 0.5) * 0.5
                        except Exception:
                            pass

            # TASK 3: TheOver Transparency
            # Append delta to reason if used
            if row.get("theover_matched") or row.get("theover_used_in_pick"):
                delta = row.get("theover_delta_final_prob") or row.get("theover_delta")
                if delta is not None:
                    try:
                        d_val = float(delta)
                        if abs(d_val) > 0.001:
                            reason = str(row.get("confidence_reason") or "")

                            # Enhanced Text Logic (Requirement B)
                            msg = ""
                            if abs(d_val) <= 0.05:
                                msg = f"TheOver mild boost ({d_val:+.3f})"
                            else:
                                msg = f"TheOver-driven boost ({d_val:+.3f})"

                            row["confidence_reason"] = (
                                reason + f" | {msg}"
                            ).strip(" |")
                    except Exception:
                        pass

            return row

        df = df.apply(_apply_stats_quality_penalty, axis=1)

        # -------------------------------------------------------------------------
        # TASK 5: Update confidence_reason with Sentiment Impact
        # -------------------------------------------------------------------------
        def _update_reason_with_sentiment(row):
            w_sent = float(row.get("wsentiment_used") or 0.0)
            adj = float(row.get("sentiment_adj") or 0.0)

            # Check if sentiment was actually available/valid
            sentiment_valid = row.get("sentiment_available", False)

            # Ensure Pick_Reason_Short exists/synced
            if "Pick_Reason_Short" not in row or pd.isna(row["Pick_Reason_Short"]):
                row["Pick_Reason_Short"] = row.get("confidence_reason", "")

            # Thresholds: Weight > 5% and Adjustment > 1%
            # Also handle "unused" case

            tag = ""
            if not sentiment_valid:
                 # Sentiment disabled or unavailable - explicitly unused?
                 # User said: "If sentiment is effectively unused... prefix to sentiment=unused or just omit it."
                 # Let's omit it if invalid, unless specifically requested.
                 pass
            elif w_sent > 0.05 and abs(adj) > 0.01:
                direction = "Bullish" if adj > 0 else "Bearish"
                tag = f"sentiment={direction}"
            elif sentiment_valid:
                # Valid but neutral/small impact
                tag = "sentiment=neutral"

            if tag:
                # Update confidence_reason
                reason = str(row.get("confidence_reason") or "")
                if tag not in reason:
                    row["confidence_reason"] = (reason + f" | {tag}").strip(" |")

                # Update Pick_Reason_Short
                short = str(row.get("Pick_Reason_Short") or "")
                if tag not in short:
                    row["Pick_Reason_Short"] = (short + f" | {tag}").strip(" |")

            return row

        df = df.apply(_update_reason_with_sentiment, axis=1)

        # Issue 2: Add sentiment_available flag
        # If sentiment_status is 'ok' or 'partial_cached', then available
        if "sentiment_status" in df.columns:
            df["sentiment_available"] = df["sentiment_status"].astype(str).isin(["ok", "partial_cached", "cached"])
        else:
            df["sentiment_available"] = False

        kalshi_markets_count = df["HasKalshiMarket"].sum()

        # Debug logging requested by user
        try:
            total_games_count = len(df)
            kalshi_matched_raw_count = df["kalshi_matched"].fillna(False).astype(bool).sum()
            logger.info(
                "Kalshi summary: total=%s, with_kalshi=%s, kalshi_matched_raw=%s",
                total_games_count,
                kalshi_markets_count,
                kalshi_matched_raw_count
            )
        except Exception as e:
            logger.warning(f"Failed to log Kalshi summary stats: {e}")

        logger.info(f"✅ HasKalshiMarket flag added: {kalshi_markets_count} games have valid Kalshi markets")

        # Fix NaN type field before export (map from Market column)
        if "type" not in df.columns:
            df["type"] = None
        df["type"] = df["type"].fillna(df["Market"].map({"Moneyline": "ML", "Spread": "SPREAD", "Total": "TOTAL"}))

        # NOTE: master_df will be saved AFTER champion selection to ensure both session_state
        # variables hold the collapsed 1-row-per-game dataframe
        logger.info(f"Preparing df ({len(df)} rows) for champion selection...")
