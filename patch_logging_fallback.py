with open('app_core/prediction_engine.py', 'r') as f:
    content = f.read()

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)

content = patch(
    content,
    """            logger.info(f" - Rows using TRUE Live Stats (APIs): {live_stats_pct:.1f}% ({live_stats_count})")
            logger.info(f" - Rows using Strict Historical Reconstruction: {strict_pct:.1f}% ({self._last_metrics['strict_join_rescued']})")
            logger.info(f" - Rows using Fuzzy Historical Reconstruction: {fuzzy_pct:.1f}% ({self._last_metrics['fuzzy_join_rescued']})")
            logger.info(f" - Rows using Split Historical Lookup: {split_pct:.1f}% ({self._last_metrics['split_lookup_rescued']})")

            hybrid_pct = 100.0 if is_flat else 0.0
            logger.info(f"Rows using Hybrid Override: {hybrid_pct:.1f}%")

            sanitized_pct = (self._last_metrics.get('healed_count', 0) / max(total_count, 1)) * 100
            logger.info(f"Rows patched by Sanitization: {sanitized_pct:.1f}%")
            logger.info("=" * 60)""",
    """            logger.info(f" - Rows using TRUE Live Stats (APIs): {live_stats_pct:.1f}% ({live_stats_count})")
            logger.info(f" - Rows using Strict Historical Reconstruction: {strict_pct:.1f}% ({self._last_metrics['strict_join_rescued']})")
            logger.info(f" - Rows using Fuzzy Historical Reconstruction: {fuzzy_pct:.1f}% ({self._last_metrics['fuzzy_join_rescued']})")
            logger.info(f" - Rows using Split Historical Lookup: {split_pct:.1f}% ({self._last_metrics['split_lookup_rescued']})")

            hybrid_pct = 100.0 if is_flat else 0.0
            logger.info(f"Rows using Hybrid Override (Fallback): {hybrid_pct:.1f}%")

            sanitized_pct = (self._last_metrics.get('healed_count', 0) / max(total_count, 1)) * 100
            logger.info(f"Rows patched by Sanitization: {sanitized_pct:.1f}%")
            logger.info("=" * 60)

            # Summarizing for the user explicitly:
            logger.info(f"FALLBACK AUDIT: Live Stats = {live_stats_count}, Historical/Stale = {total_historical_rescued}, Hybrid = {total_count if is_flat else 0}")
            logger.info("=" * 60)"""
)

with open('app_core/prediction_engine.py', 'w') as f:
    f.write(content)
