import re

with open('core/streamlit_pipeline.py', 'r') as f:
    content = f.read()

# Add logging points in run_analysis_pipeline
import logging
import inspect

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)

# 1. Total raw rows loaded into analysis
content = patch(
    content,
    """    merged = master_slate.copy()
    print('master_slate len:', len(master_slate))
    logger.info(f"PIPELINE TRACE: master_slate initialized with {len(merged)} rows.")""",
    """    merged = master_slate.copy()
    logger.info(f"PIPELINE AUDIT: [1/9] Total raw rows loaded into analysis (master_slate initialized): {len(merged)}")"""
)

# 2. Rows after sanitization
content = patch(
    content,
    """    # Mandatory Sanitization Layer
    logger.info(f"PIPELINE TRACE: Rows before sanitization: {len(merged)}")""",
    """    # Mandatory Sanitization Layer
    logger.info(f"PIPELINE AUDIT: [2/9] Rows before sanitization: {len(merged)}")"""
)

# 3. Rows patched by sanitization & rows dropped
content = patch(
    content,
    """        dropped = len(merged) - (valid_odds_mask & valid_prob_mask).sum()
        if dropped > 0:
            logger.warning(f"Sanitization layer patched {dropped} rows with extreme/synthetic lines instead of dropping.")""",
    """        dropped = len(merged) - (valid_odds_mask & valid_prob_mask).sum()
        logger.info(f"PIPELINE AUDIT: [3/9] Rows patched by sanitization: {dropped}")
        # Not dropping, we patch instead.
        logger.info(f"PIPELINE AUDIT: [4/9] Rows dropped or excluded after sanitization: 0")
        if dropped > 0:
            logger.warning(f"Sanitization layer patched {dropped} rows with extreme/synthetic lines instead of dropping.")"""
)

# 4. Rows eligible for ML
content = patch(
    content,
    """            if needs_prediction.any():""",
    """            logger.info(f"PIPELINE AUDIT: [5/9] Rows eligible for ML (needs_prediction mask): {needs_prediction.sum()}")
            if needs_prediction.any():"""
)

# 5. Rows actually sent into predict_batch
content = patch(
    content,
    """                logger.info(f"PIPELINE TRACE: Sending {len(enriched_for_prediction)} rows into predict_batch.")""",
    """                logger.info(f"PIPELINE AUDIT: [6/9] Rows actually sent into predict_batch: {len(enriched_for_prediction)}")"""
)

# 6. Rows returned from predict_batch
content = patch(
    content,
    """                logger.info(f"PIPELINE TRACE: Received {len(predictions_list)} predictions from predict_batch.")""",
    """                logger.info(f"PIPELINE AUDIT: [7/9] Rows returned from predict_batch: {len(predictions_list)}")"""
)

# 7. Rows successfully merged back into the analysis dataframe
content = patch(
    content,
    """                logger.info(f"PIPELINE TRACE: Successfully merged {len(predictions_list)} predictions back into master slate.")""",
    """                logger.info(f"PIPELINE AUDIT: [8/9] Rows successfully merged back into the analysis dataframe: {len(predictions_list)}")"""
)

# 8. Rows surviving into best-picks ranking/export
content = patch(
    content,
    """def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:""",
    """def build_best_picks_df(analysis_df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"BEST PICKS AUDIT: Received analysis_df with {len(analysis_df)} rows")"""
)

content = patch(
    content,
    """    pool = analysis_df[_string_series(analysis_df, "market_type").isin(list(VALID_MARKETS))].copy()""",
    """    pool = analysis_df[_string_series(analysis_df, "market_type").isin(list(VALID_MARKETS))].copy()
    logger.info(f"BEST PICKS AUDIT: Rows after VALID_MARKETS filter: {len(pool)}")"""
)

content = patch(
    content,
    """    pool = pool[pool["has_identity"] | game_key_present].copy()""",
    """    pool = pool[pool["has_identity"] | game_key_present].copy()
    logger.info(f"BEST PICKS AUDIT: Rows after identity/game_key filter: {len(pool)}")"""
)

content = patch(
    content,
    """    best = pool.drop_duplicates(subset=["matchup_id"], keep="first").copy()""",
    """    best = pool.drop_duplicates(subset=["matchup_id"], keep="first").copy()
    logger.info(f"BEST PICKS AUDIT: Rows after 'one-per-game' drop_duplicates: {len(best)} (started with {len(pool)})")"""
)

content = patch(
    content,
    """    total_games = int(pool["matchup_id"].nunique(dropna=False))""",
    """    total_games = int(pool["matchup_id"].nunique(dropna=False))
    logger.info(f"PIPELINE AUDIT: [9/9] Rows surviving into best-picks ranking/export: {len(best)}")"""
)

with open('core/streamlit_pipeline.py', 'w') as f:
    f.write(content)
