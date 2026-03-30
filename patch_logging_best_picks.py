import re

file_path = "core/streamlit_pipeline.py"

with open(file_path, "r") as f:
    content = f.read()

search_block = """    # Final Validation Logs
    if not best.empty:
        logger.info("--- FINAL PIPELINE VALIDATION AUDIT ---")

        # Pick Status Counts
        if "Pick_Status" in best.columns:
            status_counts = best["Pick_Status"].value_counts(dropna=False).to_dict()
            logger.info("--- FINAL STATUS COUNTS ---")
            for status in status_order:
                count = status_counts.get(status, 0)
                if count > 0:
                    logger.info(f"Validated branch: {status} ({count} rows)")
                else:
                    logger.info(f"Branch not exercised: {status} (0 rows)")

        # Odds Source Counts
        if "odds_source" in best.columns:
            logger.info("--- FINAL ODDS SOURCE CLASSIFICATION ---")
            source_counts = best["odds_source"].value_counts(dropna=False).to_dict()
            for source, count in source_counts.items():
                logger.info(f"odds_source '{source}': {count} total rows")

            # Cross-tabs
            if "Pick_Status" in best.columns:
                logger.info("--- FINAL ODDS SOURCE x PICK STATUS ---")
                for source in best["odds_source"].dropna().unique():
                    source_df = best[best["odds_source"] == source]
                    counts = source_df["Pick_Status"].value_counts().to_dict()
                    filtered_counts = {k: v for k, v in counts.items() if v > 0}
                    logger.info(f"odds_source '{source}' -> {filtered_counts}")

    return best[BEST_PICK_COLUMNS]"""

replace_block = """    # Final Validation Logs (Lightweight terminal/application logging)
    if not best.empty:
        logger.info("--- FINAL PIPELINE VALIDATION AUDIT ---")

        # Pick Status Counts
        if "Pick_Status" in best.columns:
            status_counts = best["Pick_Status"].value_counts(dropna=False).to_dict()
            logger.info("--- FINAL STATUS COUNTS ---")
            for status in status_order:
                count = status_counts.get(status, 0)
                if count > 0:
                    logger.info(f"Validated branch: {status} ({count} rows)")
                else:
                    logger.info(f"Branch not exercised: {status} (0 rows)")

        # Odds Source Counts
        if "odds_source" in best.columns:
            logger.info("--- FINAL ODDS SOURCE CLASSIFICATION ---")
            source_counts = best["odds_source"].value_counts(dropna=False).to_dict()
            for source, count in source_counts.items():
                if pd.notna(source):
                    logger.info(f"odds_source '{source}': {count} total rows")
                else:
                    logger.info(f"odds_source 'NA/MISSING': {count} total rows")

            # Cross-tabs
            if "Pick_Status" in best.columns:
                logger.info("--- FINAL ODDS SOURCE x PICK STATUS ---")
                for source in best["odds_source"].unique():
                    if pd.isna(source):
                        source_df = best[best["odds_source"].isna()]
                        source_label = "NA/MISSING"
                    else:
                        source_df = best[best["odds_source"] == source]
                        source_label = source
                    counts = source_df["Pick_Status"].value_counts().to_dict()
                    filtered_counts = {k: v for k, v in counts.items() if v > 0}
                    if filtered_counts:
                        logger.info(f"odds_source '{source_label}' -> {filtered_counts}")

    return best[BEST_PICK_COLUMNS]"""

content = content.replace(search_block, replace_block)

with open(file_path, "w") as f:
    f.write(content)
