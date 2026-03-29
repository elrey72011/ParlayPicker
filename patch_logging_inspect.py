with open('core/streamlit_pipeline.py', 'r') as f:
    content = f.read()

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)

content = patch(
    content,
    """    # We must NOT filter out non-qualifying picks from the output to ensure 1:1 parity between games and picks
    # best = best[valid_edge_mask & valid_ev_mask].copy()

    total_games = int(pool["matchup_id"].nunique(dropna=False))
    logger.info(f"PIPELINE AUDIT: [9/9] Rows surviving into best-picks ranking/export: {len(best)}")""",
    """    # We must NOT filter out non-qualifying picks from the output to ensure 1:1 parity between games and picks
    # best = best[valid_edge_mask & valid_ev_mask].copy()
    logger.info(f"BEST PICKS AUDIT: Non-qualifying picks (edge < 0.01 or EV < 0.005) left intact: {(~valid_edge_mask | ~valid_ev_mask).sum()}")

    total_games = int(pool["matchup_id"].nunique(dropna=False))
    logger.info(f"PIPELINE AUDIT: [9/9] Rows surviving into best-picks ranking/export: {len(best)}")"""
)

with open('core/streamlit_pipeline.py', 'w') as f:
    f.write(content)
