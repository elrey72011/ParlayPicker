with open('app_core/prediction_engine.py', 'r') as f:
    content = f.read()

def patch(content, pattern, replace):
    if pattern not in content:
        print(f"FAILED TO MATCH: {pattern[:50]}")
    return content.replace(pattern, replace)

# Make epsilon significant enough to prevent floats from collapsing down to exactly the same 3-decimal values if needed,
# although nunique() is exact. Wait, the problem is 12 out of 320 unique.
# Because the fallback logic only considers market_prob, kalshi, stats.
# If rows have no market odds (-110 default), kalshi (0.5), and no stats, they get EXACTLY the same output.
# The user wants to see WHY they group. So I need to log the duplicates.

content = patch(
    content,
    """                    hybrid_score += epsilon
                    hybrid_score = max(0.01, min(0.99, hybrid_score))

                    chosen_market_probs.append(market_prob if market_prob is not None else 0.5)
                    scores_before_eps.append(score_before_eps)
                    scores_after_eps.append(hybrid_score)
                    final_probs.append(hybrid_score)

                # Variance Diagnostics Logging""",
    """                    hybrid_score += epsilon
                    hybrid_score = max(0.01, min(0.99, hybrid_score))

                    chosen_market_probs.append(market_prob if market_prob is not None else 0.5)
                    scores_before_eps.append(score_before_eps)
                    scores_after_eps.append(hybrid_score)
                    final_probs.append(hybrid_score)

                # Variance Diagnostics Logging
                logger.info(f"ML UNIQUENESS AUDIT: Unique prob count before fallback (raw_unique_count): {self._last_metrics.get('raw_unique_count')}")
                logger.info(f"ML UNIQUENESS AUDIT: Unique prob count after fallback: {len(set(final_probs))}")
"""
)

with open('app_core/prediction_engine.py', 'w') as f:
    f.write(content)
