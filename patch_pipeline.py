import re

with open('core/streamlit_pipeline.py', 'r') as f:
    content = f.read()

# Replace config import
old_import = """
        from app_core.weights_config import (
            BASELINE_MIN_EV, BASELINE_MIN_EDGE,
            TOTAL_OVER_MIN_EV, TOTAL_OVER_MIN_EDGE,
            TOTAL_MIN_WIN_PROB, NHL_TOTAL_MIN_WIN_PROB,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_PROB,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_EV,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_EDGE
        )
"""
new_import = """
        from app_core.weights_config import (
            BASELINE_MIN_EV, BASELINE_MIN_EDGE,
            TOTAL_OVER_MIN_EV, TOTAL_OVER_MIN_EDGE,
            TOTAL_MIN_WIN_PROB, NHL_TOTAL_MIN_WIN_PROB,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_PROB,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_EV,
            SPREAD_DIVERGENCE_OVERRIDE_MIN_EDGE,
            NEUTRAL_ACTIONABLE_MIN_PROB, NEUTRAL_ACTIONABLE_MIN_EV, NEUTRAL_ACTIONABLE_MIN_EDGE,
            DISAGREES_ACTIONABLE_MIN_PROB, DISAGREES_ACTIONABLE_MIN_EV, DISAGREES_ACTIONABLE_MIN_EDGE,
            SIDE_MIN_WIN_PROB
        )
"""
content = content.replace(old_import, new_import)

# Insert consensus agreement calculation before Pick_Status determination
consensus_logic = """
        # Calculate consensus_agreement for the current row before threshold checks
        consensus_agreement = "No Kalshi"
        if pd.notna(kalshi_prob) and float(kalshi_prob) > 0.0:
            gap = win_prob - float(kalshi_prob)
            if gap >= 0.03:
                consensus_agreement = "Agrees"
            elif gap <= -0.03:
                consensus_agreement = "Disagrees"
            else:
                consensus_agreement = "Neutral"
"""

target = """
        is_kalshi_divergence = False
        is_spread_divergence_override = False
"""
content = content.replace(target, consensus_logic + "\n" + target)

# Update threshold logic to incorporate consensus_agreement and SIDE_MIN_WIN_PROB
old_logic = """
            if "total" in market_type.lower():
                # Generic and NHL Total Win Probability floor
                required_prob = NHL_TOTAL_MIN_WIN_PROB if league == "NHL" else TOTAL_MIN_WIN_PROB
                if win_prob < required_prob:
                    status = "Below Threshold"
                    status_reason = f"Fails minimum Win Probability for {'NHL ' if league == 'NHL' else ''}Totals ({required_prob*100}%)"
                # Apply stricter EV/Edge thresholds for total_over
                elif market_type == "total_over":
                    if ev < TOTAL_OVER_MIN_EV or edge < TOTAL_OVER_MIN_EDGE:
                        status = "Below Threshold"
                        status_reason = f"Fails stricter total_over threshold (Edge > {TOTAL_OVER_MIN_EDGE*100}%, EV > {TOTAL_OVER_MIN_EV*100}%)"
                    else:
                        status = "Actionable"
                        status_reason = "Passed all strict filters"
                else: # total_under
                    if ev < BASELINE_MIN_EV or edge < BASELINE_MIN_EDGE:
                        status = "Below Threshold"
                        status_reason = f"Fails minimum Edge ({BASELINE_MIN_EDGE*100}%) or EV ({BASELINE_MIN_EV*100}%) thresholds"
                    else:
                        status = "Actionable"
                        status_reason = "Passed all strict filters"
            else: # Not a total (Sides, Spreads, etc.)
                if ev < BASELINE_MIN_EV or edge < BASELINE_MIN_EDGE:
                    status = "Below Threshold"
                    status_reason = f"Fails minimum Edge ({BASELINE_MIN_EDGE*100}%) or EV ({BASELINE_MIN_EV*100}%) thresholds"
                else:
                    status = "Actionable"
                    status_reason = "Passed all strict filters"
                    if is_spread_divergence_override:
                        status_reason = "Passed all strict filters (Spread divergence override applied)"
        else:
            status = "Actionable"
            status_reason = "Passed all strict filters"
"""

new_logic = """
            # Setup baseline thresholds depending on market family
            req_prob = 0.0
            req_ev = BASELINE_MIN_EV
            req_edge = BASELINE_MIN_EDGE

            is_total = "total" in market_type.lower()
            if is_total:
                req_prob = NHL_TOTAL_MIN_WIN_PROB if league == "NHL" else TOTAL_MIN_WIN_PROB
                if market_type == "total_over":
                    req_ev = TOTAL_OVER_MIN_EV
                    req_edge = TOTAL_OVER_MIN_EDGE
            else:
                req_prob = SIDE_MIN_WIN_PROB

            # Apply consensus overlays
            if consensus_agreement == "Neutral":
                req_prob = max(req_prob, NEUTRAL_ACTIONABLE_MIN_PROB)
                req_ev = max(req_ev, NEUTRAL_ACTIONABLE_MIN_EV)
                req_edge = max(req_edge, NEUTRAL_ACTIONABLE_MIN_EDGE)
            elif consensus_agreement == "Disagrees":
                req_prob = max(req_prob, DISAGREES_ACTIONABLE_MIN_PROB)
                req_ev = max(req_ev, DISAGREES_ACTIONABLE_MIN_EV)
                req_edge = max(req_edge, DISAGREES_ACTIONABLE_MIN_EDGE)

            # Evaluate against combined thresholds
            if win_prob < req_prob:
                if consensus_agreement == "Disagrees":
                    status = "High Variance/Speculative"
                    status_reason = f"Disagrees overlay: Fails min prob {req_prob*100}%"
                else:
                    status = "Below Threshold"
                    if is_total:
                        status_reason = f"Fails minimum Win Probability for {'NHL ' if league == 'NHL' else ''}Totals ({req_prob*100}%)"
                    elif consensus_agreement == "Neutral":
                        status_reason = f"Neutral overlay: Fails min prob {req_prob*100}%"
                    else:
                        status_reason = f"Fails minimum Side Win Probability ({req_prob*100}%)"
            elif ev < req_ev or edge < req_edge:
                if consensus_agreement == "Disagrees":
                    status = "High Variance/Speculative"
                    status_reason = f"Disagrees overlay: Fails min Edge ({req_edge*100}%) or EV ({req_ev*100}%)"
                else:
                    status = "Below Threshold"
                    if is_total and market_type == "total_over" and consensus_agreement not in ("Neutral", "Disagrees"):
                        status_reason = f"Fails stricter total_over threshold (Edge > {req_edge*100}%, EV > {req_ev*100}%)"
                    elif consensus_agreement == "Neutral":
                        status_reason = f"Neutral overlay: Fails min Edge ({req_edge*100}%) or EV ({req_ev*100}%)"
                    else:
                        status_reason = f"Fails minimum Edge ({req_edge*100}%) or EV ({req_ev*100}%) thresholds"
            else:
                status = "Actionable"
                status_reason = "Passed all strict filters"
                if not is_total and is_spread_divergence_override:
                    status_reason = "Passed all strict filters (Spread divergence override applied)"
        else:
            status = "Actionable"
            status_reason = "Passed all strict filters"
"""
content = content.replace(old_logic, new_logic)

with open('core/streamlit_pipeline.py', 'w') as f:
    f.write(content)
