import re

with open('app_core/prediction_engine.py', 'r') as f:
    content = f.read()

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)

content = patch(
    content,
    """            if is_flat:
                logger.warning(f"XGBoost returned critically flat probabilities ({unique_count}/{total_count}). Overriding with Hybrid Fallback Score.")""",
    """            if is_flat:
                logger.warning(f"ML UNIQUENESS AUDIT: XGBoost returned critically flat probabilities ({unique_count}/{total_count}). Overriding with Hybrid Fallback Score.")"""
)

content = patch(
    content,
    """            else:
                self._last_metrics["hybrid_override_triggered"] = False
                logger.info(f"PIPELINE TRACE: Variance looks good ({unique_count}/{total_count}). Hybrid fallback NOT triggered.")""",
    """            else:
                self._last_metrics["hybrid_override_triggered"] = False
                logger.info(f"ML UNIQUENESS AUDIT: Variance looks good ({unique_count}/{total_count}). Hybrid fallback NOT triggered.")"""
)

with open('app_core/prediction_engine.py', 'w') as f:
    f.write(content)
