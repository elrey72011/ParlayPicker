from __future__ import annotations

from itertools import combinations

import pandas as pd


import hashlib
import json

def generate_smart_parlays(df: pd.DataFrame, num_rr_candidates: int = 5) -> pd.DataFrame:
    """Generate +EV parlays from the top edge candidates."""
    columns = [
        "parlay_legs", "combined_probability", "combined_decimal_odds", "parlay_ev",
        "legs", "combined_market_prob", "ev_boost_pct", "is_high_correlation",
        "risk_tier", "group_id", "best_payout_book", "Conviction_Score"
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    needed = {"edge", "calibrated_probability", "decimal_odds", "market_probability", "matchup_id", "Conviction_Score"}
    if not needed.issubset(df.columns):
        return pd.DataFrame(columns=columns)

    candidate_bets = df.nlargest(20, "edge").copy()
    candidate_bets = candidate_bets.dropna(subset=["calibrated_probability", "decimal_odds"])
    if candidate_bets.empty:
        return pd.DataFrame(columns=columns)

    # Filter out combinations capping at max combinations or dynamic sizing
    rr_candidates = df.nlargest(num_rr_candidates, "edge").dropna(subset=["calibrated_probability", "decimal_odds"]).copy()

    label_cols = [c for c in ["best_pick", "team", "away_team", "home_team"] if c in candidate_bets.columns]
    records: list[dict[str, object]] = []

    for leg_count in (2, 3, 4):
        if len(candidate_bets) < leg_count:
            continue

        for combo in combinations(candidate_bets.index, leg_count):
            legs = candidate_bets.loc[list(combo)]

            unique_matchups = legs["matchup_id"].unique()
            is_high_correlation = len(unique_matchups) < leg_count

            rho = 0.8

            # Calculate adjusted model probability (Power-Law Adjustment for all)
            adj_prob = 1.0
            for matchup_id in unique_matchups:
                matchup_legs = legs[legs["matchup_id"] == matchup_id]
                if len(matchup_legs) > 1:
                    p_prod = matchup_legs["calibrated_probability"].prod()
                    p_min = matchup_legs["calibrated_probability"].min()
                    adj_prob *= (p_prod ** (1 - rho)) * (p_min ** rho)
                else:
                    adj_prob *= matchup_legs["calibrated_probability"].iloc[0]
            combined_probability = float(adj_prob)

            # Calculate adjusted market probability (Power-Law Adjustment for all)
            adj_market_prob = 1.0
            for matchup_id in unique_matchups:
                matchup_legs = legs[legs["matchup_id"] == matchup_id]
                if len(matchup_legs) > 1:
                    p_prod = matchup_legs["market_probability"].prod()
                    p_min = matchup_legs["market_probability"].min()
                    adj_market_prob *= (p_prod ** (1 - rho)) * (p_min ** rho)
                else:
                    adj_market_prob *= matchup_legs["market_probability"].iloc[0]
            combined_market_prob = float(adj_market_prob)

            combined_decimal_odds = float(legs["decimal_odds"].prod())

            # --- Line Shopping Logic ---
            best_book_name = "novig"
            best_book_odds = float(legs["decimal_odds"].prod())

            # Find the best combined odds from available bookmakers
            bookmakers = ["fanduel", "draftkings", "betmgm"]
            for book in bookmakers:
                book_col = f"odds_american_{book}"
                if book_col in legs.columns:
                    # Convert american odds to decimal, defaulting to 1.0 (invalid) if missing
                    # Since we are multiplying, we need valid lines for all legs in the parlay
                    valid_book_lines = legs[book_col].notna().all()
                    if valid_book_lines:
                        # Helper to convert american to decimal
                        def _amer_to_dec(odds):
                            if pd.isna(odds) or odds == 0: return 1.0
                            if odds > 0: return 1 + (odds / 100.0)
                            return 1 + (100.0 / abs(odds))

                        book_odds = float(legs[book_col].apply(_amer_to_dec).prod())
                        if book_odds > best_book_odds:
                            best_book_odds = book_odds
                            best_book_name = book

            combined_decimal_odds = best_book_odds

            parlay_ev = (combined_probability * (combined_decimal_odds - 1)) - (1 - combined_probability)

            if parlay_ev <= 0:
                continue

            ev_boost_pct = (combined_probability - combined_market_prob) / combined_market_prob if combined_market_prob > 0 else 0.0

            # Conviction Score calculation
            parlay_conviction = float(legs["Conviction_Score"].mean()) if "Conviction_Score" in legs.columns else pd.NA

            labels: list[str] = []
            for _, row in legs.iterrows():
                if "best_pick" in label_cols and pd.notna(row.get("best_pick")) and str(row.get("best_pick")).strip():
                    labels.append(str(row["best_pick"]))
                elif "team" in label_cols and pd.notna(row.get("team")):
                    labels.append(str(row["team"]))
                elif {"away_team", "home_team"}.issubset(label_cols):
                    labels.append(f"{row['away_team']} vs {row['home_team']}")
                elif "away_team" in label_cols:
                    labels.append(str(row["away_team"]))
                else:
                    labels.append("leg")

            # Risk-Tier Categorization
            if leg_count == 2 and parlay_ev > 0.05:
                risk_tier = "Bankroll Builder"
            elif leg_count >= 4 and combined_decimal_odds > 21.0:
                risk_tier = "The Daily Lotto"
            elif combined_decimal_odds > 51.0 and parlay_ev > 0:
                risk_tier = "Moonshot"
            else:
                risk_tier = "Standard"

            records.append(
                {
                    "parlay_legs": " | ".join(labels),
                    "combined_probability": combined_probability,
                    "combined_decimal_odds": combined_decimal_odds,
                    "parlay_ev": parlay_ev,
                    "legs": leg_count,
                    "combined_market_prob": combined_market_prob,
                    "ev_boost_pct": ev_boost_pct,
                    "is_high_correlation": is_high_correlation,
                    "risk_tier": risk_tier,
                    "group_id": pd.NA,
                    "best_payout_book": best_book_name.capitalize() if best_book_name != "novig" else "Novig",
                    "Conviction_Score": parlay_conviction,
                }
            )

    # Automated Round Robin Packages
    if not rr_candidates.empty and len(rr_candidates) >= 2:
        # Generate a unique hash for this specific set of candidate legs
        leg_names = []
        for _, row in rr_candidates.iterrows():
            if "best_pick" in label_cols and pd.notna(row.get("best_pick")):
                leg_names.append(str(row["best_pick"]))
            elif {"away_team", "home_team"}.issubset(label_cols):
                leg_names.append(f"{row['away_team']} vs {row['home_team']}")

        group_hash = hashlib.md5(json.dumps(sorted(leg_names)).encode()).hexdigest()[:8]
        rr_group_id = f"RR_{group_hash}"

        for leg_count in (2, 3):
            if len(rr_candidates) < leg_count:
                continue

            for combo in combinations(rr_candidates.index, leg_count):
                legs = rr_candidates.loc[list(combo)]

                unique_matchups = legs["matchup_id"].unique()
                is_high_correlation = len(unique_matchups) < leg_count

                rho = 0.8

                # Calculate adjusted model probability (Power-Law Adjustment for all)
                adj_prob = 1.0
                for matchup_id in unique_matchups:
                    matchup_legs = legs[legs["matchup_id"] == matchup_id]
                    if len(matchup_legs) > 1:
                        p_prod = matchup_legs["calibrated_probability"].prod()
                        p_min = matchup_legs["calibrated_probability"].min()
                        adj_prob *= (p_prod ** (1 - rho)) * (p_min ** rho)
                    else:
                        adj_prob *= matchup_legs["calibrated_probability"].iloc[0]
                combined_probability = float(adj_prob)

                # Calculate adjusted market probability (Power-Law Adjustment for all)
                adj_market_prob = 1.0
                for matchup_id in unique_matchups:
                    matchup_legs = legs[legs["matchup_id"] == matchup_id]
                    if len(matchup_legs) > 1:
                        p_prod = matchup_legs["market_probability"].prod()
                        p_min = matchup_legs["market_probability"].min()
                        adj_market_prob *= (p_prod ** (1 - rho)) * (p_min ** rho)
                    else:
                        adj_market_prob *= matchup_legs["market_probability"].iloc[0]
                combined_market_prob = float(adj_market_prob)

                # Line Shopping
                best_book_name = "novig"
                best_book_odds = float(legs["decimal_odds"].prod())
                bookmakers = ["fanduel", "draftkings", "betmgm"]
                for book in bookmakers:
                    book_col = f"odds_american_{book}"
                    if book_col in legs.columns and legs[book_col].notna().all():
                        def _amer_to_dec(odds):
                            if pd.isna(odds) or odds == 0: return 1.0
                            if odds > 0: return 1 + (odds / 100.0)
                            return 1 + (100.0 / abs(odds))
                        book_odds = float(legs[book_col].apply(_amer_to_dec).prod())
                        if book_odds > best_book_odds:
                            best_book_odds = book_odds
                            best_book_name = book

                combined_decimal_odds = best_book_odds
                parlay_ev = (combined_probability * (combined_decimal_odds - 1)) - (1 - combined_probability)

                # Still strictly filter negative EV
                if parlay_ev <= 0:
                    continue

                ev_boost_pct = (combined_probability - combined_market_prob) / combined_market_prob if combined_market_prob > 0 else 0.0

                # Conviction Score calculation
                parlay_conviction = float(legs["Conviction_Score"].mean()) if "Conviction_Score" in legs.columns else pd.NA

                labels: list[str] = []
                for _, row in legs.iterrows():
                    if "best_pick" in label_cols and pd.notna(row.get("best_pick")) and str(row.get("best_pick")).strip():
                        labels.append(str(row["best_pick"]))
                    elif "team" in label_cols and pd.notna(row.get("team")):
                        labels.append(str(row["team"]))
                    elif {"away_team", "home_team"}.issubset(label_cols):
                        labels.append(f"{row['away_team']} vs {row['home_team']}")
                    elif "away_team" in label_cols:
                        labels.append(str(row["away_team"]))
                    else:
                        labels.append("leg")

                records.append(
                    {
                        "parlay_legs": " | ".join(labels),
                        "combined_probability": combined_probability,
                        "combined_decimal_odds": combined_decimal_odds,
                        "parlay_ev": parlay_ev,
                        "legs": leg_count,
                        "combined_market_prob": combined_market_prob,
                        "ev_boost_pct": ev_boost_pct,
                        "is_high_correlation": is_high_correlation,
                        "risk_tier": "Round Robin",
                        "group_id": rr_group_id,
                        "best_payout_book": best_book_name.capitalize() if best_book_name != "novig" else "Novig",
                        "Conviction_Score": parlay_conviction,
                    }
                )

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records).sort_values(["risk_tier", "parlay_ev"], ascending=[True, False]).reset_index(drop=True)
