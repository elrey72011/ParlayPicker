with open('app_core/prediction_engine.py', 'r') as f:
    content = f.read()

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)

content = patch(
    content,
    """                                    logger.info(f"      [{row.get('league')}] {row.get('matchup_id')} | Pick: {row.get('market_type')} | Mkt: {mkt_prob:.3f} | Kalshi: {kalshi_val:.3f} | Stale: {stale_flag} | Synth: {synth_flags} | StatsQual: {stats_qual} | Sanitized: {False}")""",
    """                                    sanitized = row.get('sanitized_value', False)
                                    logger.info(f"ML UNIQUENESS AUDIT:      [{row.get('league')}] {row.get('matchup_id')} | Pick: {row.get('market_type')} | Mkt: {mkt_prob:.3f} | Kalshi: {kalshi_val:.3f} | Stale: {stale_flag} | Synth: {synth_flags} | StatsQual: {stats_qual} | Sanitized: {sanitized}")"""
)

with open('app_core/prediction_engine.py', 'w') as f:
    f.write(content)
