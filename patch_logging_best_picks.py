import re

with open('core/streamlit_pipeline.py', 'r') as f:
    content = f.read()

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)

content = patch(
    content,
    """    pool["expected_value"] = pd.to_numeric(pool["expected_value"], errors="coerce")

    # 2. Assign calibrated probability with a default
    pool["calibrated_probability"] = _numeric_series(pool, "calibrated_probability", 0.5)

    # 1. Add the ranking metadata
    pool = _apply_triple_filter_ranking(pool)""",
    """    pool["expected_value"] = pd.to_numeric(pool["expected_value"], errors="coerce")

    # 2. Assign calibrated probability with a default
    pool["calibrated_probability"] = _numeric_series(pool, "calibrated_probability", 0.5)

    # 1. Add the ranking metadata
    pool = _apply_triple_filter_ranking(pool)
    logger.info(f"BEST PICKS AUDIT: Rows passed through Triple Filter Ranking: {len(pool)}")"""
)

content = patch(
    content,
    """    best = pool.drop_duplicates(subset=["matchup_id"], keep="first").copy()
    logger.info(f"BEST PICKS AUDIT: Rows after 'one-per-game' drop_duplicates: {len(best)} (started with {len(pool)})")

    # Phase 5: Enforce Thresholds
    # MIN_EDGE_THRESHOLD of 0.01 for high-liquidity markets.
    # Expected Value Floor of 0.005.
    is_postseason = is_postseason_ncaab(best)""",
    """    best = pool.drop_duplicates(subset=["matchup_id"], keep="first").copy()
    logger.info(f"BEST PICKS AUDIT: Rows after 'one-per-game' drop_duplicates: {len(best)} (started with {len(pool)})")
    logger.info("BEST PICKS AUDIT: IMPORTANT - The one-per-game logic drops all but the highest-EV pick per game (e.g. discards the other side and the total).")

    # Phase 5: Enforce Thresholds
    # MIN_EDGE_THRESHOLD of 0.01 for high-liquidity markets.
    # Expected Value Floor of 0.005.
    is_postseason = is_postseason_ncaab(best)"""
)

with open('core/streamlit_pipeline.py', 'w') as f:
    f.write(content)
