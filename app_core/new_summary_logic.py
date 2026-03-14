import pandas as pd
import numpy as np
import logging
import re
from app_core.weights_config import (
    KALSHI_WEIGHT,
    MARKET_WEIGHT,
    ML_MODEL_WEIGHT,
    THEOVER_WEIGHT,
    SENTIMENT_WEIGHT,
)

logger = logging.getLogger("parlaypicker")

def calculate_consensus_for_row(row: pd.Series, market_type: str = "Spread") -> tuple[float, str]:
    """
    Calculates Consensus using weighted blend of Market, Kalshi, Model, TheOver, and Sentiment.
    Uses static weights from app_core.weights_config.

    Args:
        row: The dataframe row containing probability columns
        market_type: "Spread" or "Total"

    Returns:
        (consensus_prob, breakdown_string)
        - consensus_prob: float 0-1
        - breakdown_string: e.g. "56% (M55 / K59 / AI52)"
    """
    # Helper to safely get float
    def _get_f(key):
        val = row.get(key)
        try:
            return float(val) if pd.notnull(val) and val != "" else None
        except:
            return None

    market_type = market_type.capitalize()

    # 1. Map Columns based on Market Type
    p_market = None
    p_kalshi = None
    p_model = None
    p_theover = None
    p_sentiment = None # Sentiment usually applied as adj, but we can try to extract a prob

    if market_type == "Spread":
        # Market
        p_market = _get_f("spread_prob_pick_market")
        if p_market is None: p_market = _get_f("spread_prob_market_based")
        if p_market is None: p_market = _get_f("spread_prob_market")

        # Kalshi — v99 FIX (Bug 1): ONLY use pick-side-adjusted Kalshi probability.
        # NEVER fall back to raw kalshi_prob_spread which is on the YES-contract side,
        # not the pick side. Using the raw value produces wrong-side blending.
        p_kalshi = _get_f("spread_prob_pick_kalshi")

        # AI
        p_model = _get_f("model_spread_prob")

        # TheOver
        p_theover = _get_f("theover_spread_prob")

    elif market_type == "Total":
        # Market
        p_market = _get_f("total_prob_pick_market")
        if p_market is None: p_market = _get_f("total_prob_market_based")
        if p_market is None: p_market = _get_f("total_prob_market")

        # Kalshi — v99 FIX (Bug 1): ONLY use pick-side-adjusted Kalshi probability.
        # NEVER fall back to raw kalshi_prob_total which is on the YES-contract side.
        p_kalshi = _get_f("total_prob_pick_kalshi")

        # AI
        p_model = _get_f("model_total_prob")

        # TheOver
        p_theover = _get_f("theover_total_prob")

    else:
        # Fallback for Moneyline or generic
        p_market = _get_f("Implied_Prob")
        p_kalshi = _get_f("kalshi_prob_for_pick")
        p_model = _get_f("AI_Prob")
        p_theover = _get_f("theover_prob") or _get_f("theover_prob_used")

    # Common Fallbacks
    if p_model is None: p_model = _get_f("AI_Prob")
    if p_theover is None: p_theover = _get_f("theover_prob") or _get_f("theover_prob_used")

    # Sentiment extraction (heuristic from adj)
    # sentiment_adj is usually +/- 0.05. Base is 0.5.
    # So p_sentiment ~ 0.5 + adj.
    sent_adj = _get_f("sentiment_adj")
    if sent_adj is not None:
        p_sentiment = 0.5 + sent_adj

    # 2. Build Sources List using Static Weights
    # FIX: When a source is unavailable (None), set its weight to 0 instead of
    # using 0.5 (neutral). This prevents missing sources from diluting the
    # consensus toward 50%. Weights are redistributed proportionally to
    # available sources via normalization (total_w division).
    sources = []

    # Market
    w_m = MARKET_WEIGHT if p_market is not None else 0.0
    val_m = p_market if p_market is not None else 0.0
    sources.append(("M", val_m, w_m))

    # Kalshi
    w_k = KALSHI_WEIGHT if p_kalshi is not None else 0.0
    val_k = p_kalshi if p_kalshi is not None else 0.0
    sources.append(("K", val_k, w_k))

    # Model
    w_ml = ML_MODEL_WEIGHT if p_model is not None else 0.0
    val_ml = p_model if p_model is not None else 0.0
    sources.append(("AI", val_ml, w_ml))

    # TheOver
    w_to = THEOVER_WEIGHT if p_theover is not None else 0.0
    val_to = p_theover if p_theover is not None else 0.0
    sources.append(("TO", val_to, w_to))

    # Sentiment
    w_s = SENTIMENT_WEIGHT if p_sentiment is not None else 0.0
    val_s = p_sentiment if p_sentiment is not None else 0.0
    sources.append(("S", val_s, w_s))

    # 3. Calculate Weighted Sum
    # Normalize by total available weight to redistribute missing source weights proportionally
    total_w = sum(s[2] for s in sources)
    weighted_sum = sum(s[1] * s[2] for s in sources)

    if total_w == 0:
        return 0.5, "N/A"

    consensus = weighted_sum / total_w
    consensus = max(0.01, min(0.99, consensus))

    # 4. Format
    # Only show non-neutral/valid sources in string to keep it clean, or show all?
    # User likes transparency. Let's show significant ones or all if valid.
    parts = []
    for code, val, w in sources:
        # Only show if original value was not None (meaning we had data)
        # We need to check the original variables again.
        is_valid = False
        if code == "M" and p_market is not None: is_valid = True
        elif code == "K" and p_kalshi is not None: is_valid = True
        elif code == "AI" and p_model is not None: is_valid = True
        elif code == "TO" and p_theover is not None: is_valid = True
        elif code == "S" and p_sentiment is not None: is_valid = True

        if is_valid:
            parts.append(f"{code}{int(val*100)}")

    if not parts:
        breakdown = "No Data"
    else:
        breakdown = " / ".join(parts)

    formatted_str = f"{int(consensus*100)}% ({breakdown})"

    return consensus, formatted_str

def calculate_spread_consensus(row: pd.Series) -> tuple[float, str]:
    """Legacy wrapper for backward compatibility."""
    return calculate_consensus_for_row(row, "Spread")

def build_game_summary_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates ML, Spread, and Total rows into a single row per game with specific summary columns.

    Returns a DataFrame with one row per game and columns:
    [league, Home, Away, Commence (UTC), Commence (Local), Local Date,
     Overall Pick, Overall Prob, Spread, Spread Prob, Total, Total Prob, ML, ML Prob]
    """
    if df is None or df.empty:
        return pd.DataFrame()

    summary_rows = []

    # Helper to safe float conversion for comparison
    def _get_prob(row, col):
        try:
            return float(row.get(col) or 0.0)
        except:
            return 0.0

    # Helper to check if a value is a valid non-zero probability
    def _is_valid_prob(val):
        try:
            if pd.isna(val):
                return False
            f_val = float(val)
            return f_val != 0
        except:
            return False

    # Issue 4: Helper to get canonical probability from multiple fallback columns
    def _get_canonical_prob(row, market_type):
        """
        Consolidates probability extraction logic.
        Priority:
        1. spread_prob_pick_final (most processed)
        2. spread_prob_adj (sentiment adjusted)
        3. spread_prob (base)
        4. spread_prob_market_based (fallback)
        """
        prefix = market_type.lower() # "spread" or "total"
        # Try pick_final first (canonical)
        p = row.get(f"{prefix}_prob_pick_final")
        if p is not None: return p

        # Fallbacks
        p = row.get(f"{prefix}_prob_adj")
        if p is not None: return p

        p = row.get(f"{prefix}_prob")
        if p is not None: return p

        p = row.get(f"{prefix}_prob_market_based")
        return p

    # Group by Game Key
    # We use 'league', 'Home', 'Away', 'Commence (UTC)' as the key
    # Ensure these columns exist
    required_cols = ["league", "Home", "Away", "Commence (UTC)"]
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    grouped = df.groupby(required_cols)

    for name, group in grouped:
        league, home, away, commence = name

        # Base info from the first row of the group
        first_row = group.iloc[0]
        commence_local = first_row.get("Commence (Local)")
        local_date = first_row.get("Local Date")

        summary = {
            "League": league,  # Changed from "league" to "League" to match UI
            "Home": home,
            "Away": away,
            "Commence UTC": commence,  # Changed from "Commence (UTC)" to "Commence UTC" to match UI
            "Commence (Local)": commence_local,
            "Local Date": local_date,
        }

        # ============================================
        # HasKalshiMarket Flag
        # ============================================
        has_kalshi_market = False
        for idx, row in group.iterrows():
            kalshi_matched = row.get("kalshi_matched")
            if kalshi_matched == True or str(kalshi_matched).lower() == "true":
                # Check if at least one Kalshi probability is available
                # Spread
                has_spread = (_is_valid_prob(row.get("kalshi_prob_spread")) or
                              _is_valid_prob(row.get("spread_prob_pick_kalshi")))

                # Total
                has_total = (_is_valid_prob(row.get("kalshi_prob_total")) or
                             _is_valid_prob(row.get("total_prob_pick_kalshi")))

                # Winner / Moneyline (Generic)
                has_ml = (_is_valid_prob(row.get("kalshi_prob")) or
                          _is_valid_prob(row.get("kalshi_prob_used")))

                if has_spread or has_total or has_ml:
                    has_kalshi_market = True
                    break

        summary["HasKalshiMarket"] = has_kalshi_market

        # --- Moneyline (ML) ---
        ml_rows = group[group["Market"] == "Moneyline"]
        ml_pick = None
        ml_prob = None

        if not ml_rows.empty:
            # Choose best ML row based on final_probability or consensus_prob_adj
            best_ml = ml_rows.loc[pd.to_numeric(ml_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]
            ml_pick = best_ml.get("Pick")
            ml_prob = best_ml.get("final_probability") or best_ml.get("consensus_prob_adj")

        summary["ML Pick"] = ml_pick  # Changed from "ML" to "ML Pick" to match UI
        summary["ML Prob"] = ml_prob

        # --- Spread ---
        spread_rows = group[group["Market"] == "Spread"]
        spread_pick = None
        spread_prob = None
        kalshi_spread_prob = None
        spread_market_prob = None
        spread_consensus_prob = None
        spread_consensus_str = None

        if not spread_rows.empty:
            # Choose best spread row
            best_spread = spread_rows.loc[pd.to_numeric(spread_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]

            # Set Spread Pick
            # Fix: Ensure we don't construct "Team 0" or similar invalid strings
            s_line = best_spread.get("Line")
            if best_spread.get("spread_pick_label"):
                spread_pick = best_spread.get("spread_pick_label")
            elif best_spread.get("Spread & Pick"):
                spread_pick = best_spread.get("Spread & Pick")
            elif best_spread.get("Pick"):
                try:
                    s_val = float(s_line) if s_line is not None else 0.0
                    if abs(s_val) > 0.001:
                        # Format as float to strip leading zeros ("01" -> "1.0")
                        spread_pick = f"{best_spread.get('Pick')} {s_val:g}"
                    else:
                        spread_pick = None
                except (ValueError, TypeError):
                    spread_pick = None
            else:
                spread_pick = None

            # Set Spread Prob (Issue 4: Use helper)
            spread_prob = _get_canonical_prob(best_spread, "Spread")

            # Get Kalshi Spread Probability
            kalshi_spread_prob = (best_spread.get("spread_prob_pick_kalshi")
                                 or best_spread.get("kalshi_prob_spread"))

            # Get Market Spread Probability for delta calculation
            spread_market_prob = (best_spread.get("spread_prob_pick_market")
                                 or best_spread.get("spread_prob_market"))

            # --- CALCULATE SPREAD CONSENSUS ---
            spread_consensus_prob, spread_consensus_str = calculate_consensus_for_row(best_spread, "Spread")

            # Validation (Requirement 1) & Auto-Correction (Issue 2)
            if spread_pick and spread_prob is not None:
                try:
                    sp_val = float(spread_prob)
                    if sp_val < 0.50:
                        logger.error(f"⚠️ CRITICAL: Spread pick {spread_pick} has prob {sp_val:.3f} < 50%!")

                        # FIX: Flip to the alt pick with higher probability
                        alt_pick = best_spread.get("spread_alt_label")

                        # If alt_label missing, try to construct it (basic flip logic)
                        if not alt_pick:
                            # Try simple parsing of pick string
                            # Example: "Lakers -5.5" -> "Warriors +5.5" (needs opponent name)
                            # We have Home/Away in best_spread
                            h_team = best_spread.get("Home")
                            a_team = best_spread.get("Away")

                            if h_team and a_team:
                                # Determine current pick team
                                new_team = None
                                s_pick_str = str(spread_pick)

                                # 1. Try full name match
                                if h_team in s_pick_str:
                                    new_team = a_team
                                elif a_team in s_pick_str:
                                    new_team = h_team
                                else:
                                    # 2. Fallback to token match (safely)
                                    pick_team_match = s_pick_str.split()[0]
                                    # Only flip if token is unique to one team (avoids "New York" ambiguity)
                                    in_home = pick_team_match in h_team
                                    in_away = pick_team_match in a_team

                                    if in_home and not in_away:
                                        new_team = a_team
                                    elif in_away and not in_home:
                                        new_team = h_team

                                if new_team:
                                    # Flip line sign
                                    try:
                                        # Extract line number from string
                                        # "Team -5.5" -> -5.5
                                        # "Team +3" -> +3
                                        match = re.search(r'([+-]?\d+(\.\d+)?)', str(spread_pick))
                                        if match:
                                            old_line = float(match.group(1))
                                            new_line = -old_line
                                            sign = "+" if new_line > 0 else ""
                                            alt_pick = f"{new_team} {sign}{new_line:g}"
                                    except:
                                        pass

                        if alt_pick:
                            spread_pick = alt_pick
                            spread_prob = 1.0 - sp_val  # Flip probability
                            logger.info(f"✅ Auto-corrected to: {spread_pick} ({spread_prob:.1%})")
                        else:
                            logger.error(f"❌ Failed to auto-correct spread pick (no alt label)")

                except (ValueError, TypeError):
                    pass


        summary["Spread Pick"] = spread_pick  # Changed from "Spread" to "Spread Pick" to match UI
        summary["Spread Prob"] = spread_prob
        summary["Kalshi Spread Prob"] = kalshi_spread_prob
        summary["SpreadConsensusProb"] = spread_consensus_prob
        summary["SpreadConsensus"] = spread_consensus_str

        # Calculate Kalshi vs Market Delta for Spread (Issue 5: Added try-except)
        if kalshi_spread_prob is not None and spread_market_prob is not None:
            try:
                delta = float(kalshi_spread_prob) - float(spread_market_prob)
                summary["Kalshi Spread Δ"] = f"{delta:+.1%}" if abs(delta) > 0.001 else "0.0%"
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to calculate Kalshi Spread Δ: {e}")
                summary["Kalshi Spread Δ"] = "Error"
        else:
            summary["Kalshi Spread Δ"] = None

        # --- Total ---
        total_rows = group[group["Market"] == "Total"]
        total_pick = None
        total_prob = None
        kalshi_total_prob = None
        total_market_prob = None
        total_consensus_prob = None
        total_consensus_str = None

        if not total_rows.empty:
            best_total = total_rows.loc[pd.to_numeric(total_rows["final_probability"], errors='coerce').fillna(-1.0).idxmax()]

            # Set Total Pick
            # Fix: Avoid "Under 0" and "Under 01" artifacts
            t_line = best_total.get("Line")
            if best_total.get("total_pick_label"):
                total_pick = best_total.get("total_pick_label")
            elif best_total.get("Total & Pick"):
                total_pick = best_total.get("Total & Pick")
            elif best_total.get("Pick"):
                try:
                    t_val = float(t_line) if t_line is not None else 0.0
                    if abs(t_val) > 0.001:
                        # Format as float to strip leading zeros ("01" -> "1.0")
                        total_pick = f"{best_total.get('Pick')} {t_val:g}"
                    else:
                        total_pick = None
                except (ValueError, TypeError):
                    total_pick = None
            else:
                total_pick = None

            # Set Total Prob (Issue 4: Use helper)
            total_prob = _get_canonical_prob(best_total, "Total")

            # Get Kalshi Total Probability
            kalshi_total_prob = (best_total.get("total_prob_pick_kalshi")
                                or best_total.get("kalshi_prob_total"))

            # Get Market Total Probability for delta calculation
            total_market_prob = (best_total.get("total_prob_pick_market")
                                or best_total.get("total_prob_market"))

            # --- CALCULATE TOTAL CONSENSUS ---
            total_consensus_prob, total_consensus_str = calculate_consensus_for_row(best_total, "Total")

            # Validation & Auto-Correction (Issue 2)
            if total_pick and total_prob is not None:
                try:
                    tp_val = float(total_prob)
                    if tp_val < 0.50:
                        logger.error(f"⚠️ CRITICAL: Total pick {total_pick} has prob {tp_val:.3f} < 50%!")

                        # FIX: Flip to the alt pick
                        alt_pick = best_total.get("total_alt_label")

                        if not alt_pick:
                            # Try flip Over/Under
                            if "Over" in str(total_pick):
                                alt_pick = str(total_pick).replace("Over", "Under")
                            elif "Under" in str(total_pick):
                                alt_pick = str(total_pick).replace("Under", "Over")

                        if alt_pick:
                            total_pick = alt_pick
                            total_prob = 1.0 - tp_val
                            logger.info(f"✅ Auto-corrected to: {total_pick} ({total_prob:.1%})")
                        else:
                            logger.error(f"❌ Failed to auto-correct total pick")

                except (ValueError, TypeError):
                    pass

        summary["Total Pick"] = total_pick  # Changed from "Total" to "Total Pick" to match UI
        summary["Total Prob"] = total_prob
        summary["Kalshi Total Prob"] = kalshi_total_prob
        summary["TotalConsensusProb"] = total_consensus_prob
        summary["TotalConsensus"] = total_consensus_str

        # Calculate Kalshi vs Market Delta for Total (Issue 5: Added try-except)
        if kalshi_total_prob is not None and total_market_prob is not None:
            try:
                delta = float(kalshi_total_prob) - float(total_market_prob)
                summary["Kalshi Total Δ"] = f"{delta:+.1%}" if abs(delta) > 0.001 else "0.0%"
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to calculate Kalshi Total Δ: {e}")
                summary["Kalshi Total Δ"] = "Error"
        else:
            summary["Kalshi Total Δ"] = None

        # --- Overall Pick ---
        # Updated logic: Best Overall should only consider real spread/total picks with:
        # 1. Valid lines (not None/NaN)
        # 2. Probability > 50% (pick has positive expected value)
        # 3. No "no_valid_spread_or_total" warning
        # 4. Confidence bucket is MEDIUM or HIGH (not LOW)

        # Helper to check if probability is in valid range (> 50%)
        def _is_valid_prob_range(p):
            if p is None:
                return False
            try:
                p_float = float(p)
                # FIX: Changed from 50-65% range to just > 50%
                # A 68% spread should be selected over a 37.6% total
                return p_float > 0.50
            except:
                return False

        # Helper to check if row has warnings about invalid lines
        def _has_invalid_line_warnings(game_rows):
            for _, r in game_rows.iterrows():
                warnings = str(r.get("Warnings") or "")
                if "no_valid_spread_or_total" in warnings:
                    return True
            return False

        # Helper to get confidence bucket for a pick
        def _get_confidence(game_rows, market_type):
            for _, r in game_rows.iterrows():
                if str(r.get("Market") or "").upper() == market_type.upper():
                    return str(r.get("Pick_Confidence") or "LOW").upper()
            return "LOW"

        overall_pick = None
        overall_prob = None

        # Build eligible picks list
        eligible = []

        # We no longer skip games with invalid line warnings, allowing them to
        # fall back to ML if needed. We just collect valid spread/total picks.

        # Get spread probability
        s_p = _get_prob(summary, "Spread Prob")

        # Get total probability
        t_p = _get_prob(summary, "Total Prob")

        # Issue 3: Explicit check for Spread prob validity (logging)
        if spread_pick and s_p is not None:
            if not _is_valid_prob_range(s_p):
                # Only log if it's significantly below 50 (e.g. not 49.9 due to float errors)
                if s_p < 0.499:
                    logger.error(f"🚨 CRITICAL BUG: Spread pick {spread_pick} has prob {s_p:.1%} < 50%! Skipping eligibility.")
            else:
                eligible.append(("SPREAD", spread_pick, s_p))

        # Issue 3: Explicit check for Total prob validity (logging)
        if total_pick and t_p is not None:
            if not _is_valid_prob_range(t_p):
                if t_p < 0.499:
                    logger.error(f"🚨 CRITICAL BUG: Total pick {total_pick} has prob {t_p:.1%} < 50%! Skipping eligibility.")
            else:
                eligible.append(("TOTAL", total_pick, t_p))

        # Select best pick from eligible list
        if eligible:
            # Pick the one with highest probability
            best = max(eligible, key=lambda x: x[2])
            overall_pick = best[1]
            overall_prob = best[2]

        # ML Fallback: If no spread/total pick qualified, check Moneyline
        if not overall_pick:
            ml_p = _get_prob(summary, "ML Prob")
            ml_pick_summary = summary.get("ML Pick")
            if ml_pick_summary and ml_p is not None and _is_valid_prob_range(ml_p):
                overall_pick = ml_pick_summary
                overall_prob = ml_p

        summary["Best Overall Pick"] = overall_pick  # Changed from "Overall Pick" to "Best Overall Pick" to match UI
        summary["Best Overall Prob"] = overall_prob  # Changed from "Overall Prob" to "Best Overall Prob" to match UI

        summary_rows.append(summary)

    return pd.DataFrame(summary_rows)

def reorder_for_spread_total_focus_v2(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    fixed_front = [c for c in [
        "League", "Home", "Away",
        "Commence UTC", "Commence (Local)", "Local Date",
    ] if c in df.columns]

    summary_block = [c for c in [
        "Best Overall Pick", "Best Overall Prob",
        "Spread Pick", "Spread Prob", "Spread Consensus", "Kalshi Spread Prob", "Kalshi Spread Δ",
        "Total Pick", "Total Prob", "Total Consensus", "Kalshi Total Prob", "Kalshi Total Δ",
        "ML Pick", "ML Prob",
    ] if c in df.columns]

    used = set(fixed_front + summary_block)
    remaining = [c for c in df.columns if c not in used]

    try:
        return df[fixed_front + summary_block + remaining]
    except Exception:
        return df
