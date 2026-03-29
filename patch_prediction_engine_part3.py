with open('app_core/prediction_engine.py', 'r') as f:
    content = f.read()

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)

# Improve sample group logging
content = patch(
    content,
    """                                    logger.info(f"      [{row.get('league')}] {row.get('matchup_id')} | Pick: {row.get('market_type')} | Mkt: {mkt_prob:.3f} | Kalshi: {kalshi_val:.3f} | Stale: {stale_flag} | Synth: {synth_flags}")""",
    """                                    stats_qual = row.get('stats_quality', 'UNKNOWN')
                                    logger.info(f"      [{row.get('league')}] {row.get('matchup_id')} | Pick: {row.get('market_type')} | Mkt: {mkt_prob:.3f} | Kalshi: {kalshi_val:.3f} | Stale: {stale_flag} | Synth: {synth_flags} | StatsQual: {stats_qual} | Sanitized: {False}")"""
)

with open('app_core/prediction_engine.py', 'w') as f:
    f.write(content)

with open('core/streamlit_pipeline.py', 'r') as f:
    content2 = f.read()

content2 = patch(
    content2,
    """    merged.loc[~valid_odds_mask, "odds_american"] = -110.0
            merged.loc[~valid_odds_mask, "odds_source"] = "fallback_novig"
            merged.loc[~valid_prob_mask, "market_probability"] = 0.5238""",
    """    merged.loc[~valid_odds_mask, "odds_american"] = -110.0
            merged.loc[~valid_odds_mask, "odds_source"] = "fallback_novig"
            merged.loc[~valid_prob_mask, "market_probability"] = 0.5238
            merged.loc[~valid_odds_mask | ~valid_prob_mask, "sanitized_value"] = True
        if "sanitized_value" not in merged.columns:
            merged["sanitized_value"] = False"""
)

with open('core/streamlit_pipeline.py', 'w') as f:
    f.write(content2)
