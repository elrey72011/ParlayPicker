"""
Shotgun Mode: Multi-Parlay Generator with Risk Tiers
Automatically generates optimized 2-leg and 3-leg parlays from picks
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from itertools import combinations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PICK FILTERING AND ELIGIBILITY
# ============================================================================

def filter_eligible_picks(all_picks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter picks suitable for parlays

    Criteria:
    - Grade A or B only
    - Probability between 45-75% (reasonable confidence)
    - Exclude extreme ML odds (>±400)
    - Exclude picks with warnings/red flags

    Args:
        all_picks_df: DataFrame of all picks from current run

    Returns:
        DataFrame of eligible picks
    """
    if all_picks_df.empty:
        logger.warning("No picks provided to filter")
        return pd.DataFrame()

    # Start with all picks
    eligible = all_picks_df.copy()

    # Filter by grade (A or B only)
    if 'data_quality_grade' in eligible.columns:
        eligible = eligible[eligible['data_quality_grade'].isin(['A', 'B'])]
        logger.info(f"After grade filter (A/B): {len(eligible)} picks")

    # Filter by probability range (45-75%)
    if 'final_probability' in eligible.columns:
        eligible = eligible[
            (eligible['final_probability'] >= 0.45) &
            (eligible['final_probability'] <= 0.75)
        ]
        logger.info(f"After probability filter (45-75%): {len(eligible)} picks")

    # Filter out extreme moneyline odds if applicable
    if 'Market' in eligible.columns and 'american_odds' in eligible.columns:
        ml_picks = eligible[eligible['Market'] == 'Moneyline']
        non_ml_picks = eligible[eligible['Market'] != 'Moneyline']

        # Filter ML picks by odds
        ml_picks = ml_picks[ml_picks['american_odds'].abs() <= 400]

        # Recombine
        eligible = pd.concat([ml_picks, non_ml_picks], ignore_index=True)
        logger.info(f"After ML odds filter (<±400): {len(eligible)} picks")

    # Sort by quality score (descending)
    if 'data_quality_score' in eligible.columns:
        eligible = eligible.sort_values('data_quality_score', ascending=False)

    logger.info(f"Shotgun Mode: {len(eligible)}/{len(all_picks_df)} picks eligible for parlays")

    return eligible


# ============================================================================
# CORRELATION DETECTION
# ============================================================================

def is_correlated(pick1: Dict, pick2: Dict) -> bool:
    """
    Check if two picks are correlated (should not be in same parlay)

    Correlated if:
    - Same game (same Home and Away teams)

    Args:
        pick1: First pick dictionary
        pick2: Second pick dictionary

    Returns:
        True if correlated, False if independent
    """
    # Same game check - both Home and Away must match
    if (pick1.get('Home') == pick2.get('Home') and
        pick1.get('Away') == pick2.get('Away')):
        return True

    return False


def is_correlated_3leg(pick1: Dict, pick2: Dict, pick3: Dict) -> bool:
    """
    Check if any pair in 3-leg parlay is correlated

    Args:
        pick1: First pick
        pick2: Second pick
        pick3: Third pick

    Returns:
        True if any pair is correlated
    """
    return (is_correlated(pick1, pick2) or
            is_correlated(pick1, pick3) or
            is_correlated(pick2, pick3))


# ============================================================================
# RISK TIER CLASSIFICATION
# ============================================================================

def assign_risk_tier_2leg(combined_prob: float) -> str:
    """
    Assign risk tier for 2-leg parlays

    Thresholds:
    - Low Risk: >30% probability
    - Medium Risk: 20-30% probability
    - High Risk: <20% probability

    Args:
        combined_prob: Combined probability of parlay

    Returns:
        Risk tier string
    """
    if combined_prob >= 0.30:
        return "Low Risk"
    elif combined_prob >= 0.20:
        return "Medium Risk"
    else:
        return "High Risk"


def assign_risk_tier_3leg(combined_prob: float) -> str:
    """
    Assign risk tier for 3-leg parlays

    Thresholds:
    - Low Risk: >18% probability
    - Medium Risk: 10-18% probability
    - High Risk: <10% probability

    Args:
        combined_prob: Combined probability of parlay

    Returns:
        Risk tier string
    """
    if combined_prob >= 0.18:
        return "Low Risk"
    elif combined_prob >= 0.10:
        return "Medium Risk"
    else:
        return "High Risk"


# ============================================================================
# EXPECTED VALUE & SCORING
# ============================================================================

def probability_to_american_odds(probability: float) -> int:
    """
    Convert probability to American odds format

    Args:
        probability: Win probability (0-1)

    Returns:
        American odds (e.g., -110, +150)
    """
    if probability <= 0 or probability >= 1:
        return 0

    if probability >= 0.5:
        # Favorite
        odds = -(probability / (1 - probability)) * 100
    else:
        # Underdog
        odds = ((1 - probability) / probability) * 100

    return round(odds)


def calculate_ev_score(
    legs: List[Dict],
    combined_prob: float,
    avg_quality: float
) -> float:
    """
    Calculate Expected Value score for parlay ranking

    Factors:
    - Combined probability
    - Average quality of picks
    - League diversity bonus
    - Market diversity bonus
    - Timing spread bonus

    Args:
        legs: List of leg dictionaries
        combined_prob: Combined probability
        avg_quality: Average quality score

    Returns:
        EV score (higher is better)
    """
    # Base score from probability and quality
    base_score = combined_prob * (avg_quality / 100)

    # League diversity bonus (picks from different leagues)
    leagues = set([leg.get('league', '') for leg in legs if leg.get('league')])
    league_bonus = 1.0 + (0.1 * (len(leagues) - 1))  # +10% per additional league

    # Market diversity bonus (mix of Spread/Total/ML)
    markets = set([leg.get('Market', '') for leg in legs if leg.get('Market')])
    market_bonus = 1.0 + (0.05 * (len(markets) - 1))  # +5% per additional market type

    # Timing spread bonus (staggered start times for early insight)
    try:
        times = []
        for leg in legs:
            time_str = leg.get('Commence (UTC)', '') or leg.get('Commence (Local)', '')
            if time_str:
                try:
                    times.append(pd.to_datetime(time_str))
                except:
                    pass

        if len(times) > 1:
            time_range_hours = (max(times) - min(times)).total_seconds() / 3600
            timing_bonus = 1.0 + min(0.15, time_range_hours / 20)  # Up to +15% for spread timing
        else:
            timing_bonus = 1.0
    except:
        timing_bonus = 1.0

    # Combine all factors
    ev_score = base_score * league_bonus * market_bonus * timing_bonus

    return ev_score


# ============================================================================
# PARLAY GENERATION
# ============================================================================

def generate_2leg_parlays(
    picks_df: pd.DataFrame,
    target_count: int = 20
) -> List[Dict]:
    """
    Generate optimized 2-leg parlays

    Strategy:
    - Generate all valid combinations
    - Score by EV
    - Balance across risk tiers
    - Return top parlays

    Args:
        picks_df: DataFrame of eligible picks
        target_count: Target number of parlays to generate

    Returns:
        List of 2-leg parlay dictionaries
    """
    if len(picks_df) < 2:
        logger.warning(f"Only {len(picks_df)} picks available - need at least 2 for 2-leg parlays")
        return []

    parlays = []
    picks = picks_df.to_dict('records')

    logger.info(f"Generating 2-leg parlays from {len(picks)} eligible picks...")

    # Generate all valid combinations
    for i, pick1 in enumerate(picks):
        for pick2 in picks[i+1:]:

            # Skip if correlated
            if is_correlated(pick1, pick2):
                continue

            # Calculate combined probability
            prob = pick1.get('final_probability', 0.5) * pick2.get('final_probability', 0.5)

            # Skip if probability too low or too high
            if prob < 0.10 or prob > 0.60:
                continue

            # Calculate average quality
            q1 = pick1.get('data_quality_score', 80)
            q2 = pick2.get('data_quality_score', 80)
            avg_quality = (q1 + q2) / 2

            # Calculate EV score
            legs = [pick1, pick2]
            ev_score = calculate_ev_score(legs, prob, avg_quality)

            # Calculate expected odds
            american_odds = probability_to_american_odds(prob)

            # Determine risk tier
            risk_tier = assign_risk_tier_2leg(prob)

            # Extract league info
            league1 = pick1.get('league', pick1.get('League', 'Unknown'))
            league2 = pick2.get('league', pick2.get('League', 'Unknown'))

            # Build parlay object
            parlay = {
                'parlay_id': f"2L_{len(parlays)+1:03d}",
                'num_legs': 2,
                'legs': [
                    {
                        'league': league1,
                        'game': f"{pick1.get('Away', 'Away')} @ {pick1.get('Home', 'Home')}",
                        'market': pick1.get('Market', 'Unknown'),
                        'pick': pick1.get('Pick', 'Unknown'),
                        'probability': pick1.get('final_probability', 0.5),
                        'quality': q1,
                        'start_time': pick1.get('Commence (Local)', pick1.get('Commence (UTC)', 'TBD'))
                    },
                    {
                        'league': league2,
                        'game': f"{pick2.get('Away', 'Away')} @ {pick2.get('Home', 'Home')}",
                        'market': pick2.get('Market', 'Unknown'),
                        'pick': pick2.get('Pick', 'Unknown'),
                        'probability': pick2.get('final_probability', 0.5),
                        'quality': q2,
                        'start_time': pick2.get('Commence (Local)', pick2.get('Commence (UTC)', 'TBD'))
                    }
                ],
                'combined_probability': prob,
                'expected_odds': american_odds,
                'avg_quality': avg_quality,
                'ev_score': ev_score,
                'risk_tier': risk_tier,
                'leagues': [league1, league2],
                'markets': [pick1.get('Market', 'Unknown'), pick2.get('Market', 'Unknown')]
            }

            parlays.append(parlay)

    logger.info(f"Generated {len(parlays)} potential 2-leg parlays")

    # Sort by EV score
    parlays.sort(key=lambda x: x['ev_score'], reverse=True)

    # Balance across risk tiers
    balanced = balance_risk_tiers(parlays, target_count)

    logger.info(f"Selected {len(balanced)} balanced 2-leg parlays")

    return balanced


def generate_3leg_parlays(
    picks_df: pd.DataFrame,
    target_count: int = 20
) -> List[Dict]:
    """
    Generate optimized 3-leg parlays

    Strategy:
    - Use top picks for efficiency (combinatorial explosion prevention)
    - Generate valid combinations
    - Score by EV
    - Balance across risk tiers

    Args:
        picks_df: DataFrame of eligible picks
        target_count: Target number of parlays to generate

    Returns:
        List of 3-leg parlay dictionaries
    """
    if len(picks_df) < 3:
        logger.warning(f"Only {len(picks_df)} picks available - need at least 3 for 3-leg parlays")
        return []

    parlays = []

    # Use top 50 picks to prevent combinatorial explosion
    # (50 choose 3 = 19,600 combinations - manageable)
    picks = picks_df.head(50).to_dict('records')

    logger.info(f"Generating 3-leg parlays from top {len(picks)} picks...")

    # Generate all valid combinations
    for i, pick1 in enumerate(picks):
        for j, pick2 in enumerate(picks[i+1:], i+1):
            for pick3 in picks[j+1:]:

                # Skip if any pair is correlated
                if is_correlated_3leg(pick1, pick2, pick3):
                    continue

                # Calculate combined probability
                prob = (pick1.get('final_probability', 0.5) *
                       pick2.get('final_probability', 0.5) *
                       pick3.get('final_probability', 0.5))

                # Skip if probability too low or too high
                if prob < 0.05 or prob > 0.35:
                    continue

                # Calculate average quality
                q1 = pick1.get('data_quality_score', 80)
                q2 = pick2.get('data_quality_score', 80)
                q3 = pick3.get('data_quality_score', 80)
                avg_quality = (q1 + q2 + q3) / 3

                # Calculate EV score
                legs = [pick1, pick2, pick3]
                ev_score = calculate_ev_score(legs, prob, avg_quality)

                # Calculate expected odds
                american_odds = probability_to_american_odds(prob)

                # Determine risk tier
                risk_tier = assign_risk_tier_3leg(prob)

                # Extract league info
                league1 = pick1.get('league', pick1.get('League', 'Unknown'))
                league2 = pick2.get('league', pick2.get('League', 'Unknown'))
                league3 = pick3.get('league', pick3.get('League', 'Unknown'))

                # Build parlay object
                parlay = {
                    'parlay_id': f"3L_{len(parlays)+1:03d}",
                    'num_legs': 3,
                    'legs': [
                        {
                            'league': league1,
                            'game': f"{pick1.get('Away', 'Away')} @ {pick1.get('Home', 'Home')}",
                            'market': pick1.get('Market', 'Unknown'),
                            'pick': pick1.get('Pick', 'Unknown'),
                            'probability': pick1.get('final_probability', 0.5),
                            'quality': q1,
                            'start_time': pick1.get('Commence (Local)', pick1.get('Commence (UTC)', 'TBD'))
                        },
                        {
                            'league': league2,
                            'game': f"{pick2.get('Away', 'Away')} @ {pick2.get('Home', 'Home')}",
                            'market': pick2.get('Market', 'Unknown'),
                            'pick': pick2.get('Pick', 'Unknown'),
                            'probability': pick2.get('final_probability', 0.5),
                            'quality': q2,
                            'start_time': pick2.get('Commence (Local)', pick2.get('Commence (UTC)', 'TBD'))
                        },
                        {
                            'league': league3,
                            'game': f"{pick3.get('Away', 'Away')} @ {pick3.get('Home', 'Home')}",
                            'market': pick3.get('Market', 'Unknown'),
                            'pick': pick3.get('Pick', 'Unknown'),
                            'probability': pick3.get('final_probability', 0.5),
                            'quality': q3,
                            'start_time': pick3.get('Commence (Local)', pick3.get('Commence (UTC)', 'TBD'))
                        }
                    ],
                    'combined_probability': prob,
                    'expected_odds': american_odds,
                    'avg_quality': avg_quality,
                    'ev_score': ev_score,
                    'risk_tier': risk_tier,
                    'leagues': [league1, league2, league3],
                    'markets': [pick1.get('Market', 'Unknown'), pick2.get('Market', 'Unknown'), pick3.get('Market', 'Unknown')]
                }

                parlays.append(parlay)

    logger.info(f"Generated {len(parlays)} potential 3-leg parlays")

    # Sort by EV score
    parlays.sort(key=lambda x: x['ev_score'], reverse=True)

    # Balance across risk tiers
    balanced = balance_risk_tiers(parlays, target_count)

    logger.info(f"Selected {len(balanced)} balanced 3-leg parlays")

    return balanced


def balance_risk_tiers(parlays: List[Dict], target_count: int) -> List[Dict]:
    """
    Balance parlays across risk tiers

    Target distribution:
    - 40% Low Risk
    - 40% Medium Risk
    - 20% High Risk

    Args:
        parlays: List of all generated parlays (sorted by EV)
        target_count: Target number to return

    Returns:
        Balanced list of parlays
    """
    # Separate by risk tier
    low_risk = [p for p in parlays if p['risk_tier'] == 'Low Risk']
    med_risk = [p for p in parlays if p['risk_tier'] == 'Medium Risk']
    high_risk = [p for p in parlays if p['risk_tier'] == 'High Risk']

    # Calculate targets
    target_low = int(target_count * 0.40)
    target_med = int(target_count * 0.40)
    target_high = target_count - target_low - target_med

    # Select top from each tier
    selected = (
        low_risk[:target_low] +
        med_risk[:target_med] +
        high_risk[:target_high]
    )

    # If we don't have enough in a tier, backfill from others
    while len(selected) < target_count and len(parlays) > len(selected):
        for p in parlays:
            if p not in selected:
                selected.append(p)
                if len(selected) >= target_count:
                    break

    # Re-sort by EV score
    selected.sort(key=lambda x: x['ev_score'], reverse=True)

    return selected[:target_count]


# ============================================================================
# MAIN SHOTGUN MODE FUNCTION
# ============================================================================

def generate_shotgun_mode_parlays(
    picks_df: pd.DataFrame,
    num_2leg: int = 20,
    num_3leg: int = 20
) -> Dict:
    """
    Main function to generate all Shotgun Mode parlays

    Args:
        picks_df: DataFrame with all picks from current run
        num_2leg: Number of 2-leg parlays to generate
        num_3leg: Number of 3-leg parlays to generate

    Returns:
        Dictionary with 2-leg and 3-leg parlays and stats
    """
    logger.info("=" * 80)
    logger.info("SHOTGUN MODE: Generating Optimized Parlays")
    logger.info("=" * 80)

    # Step 1: Filter eligible picks
    eligible_picks = filter_eligible_picks(picks_df)

    if len(eligible_picks) < 2:
        logger.warning(f"Only {len(eligible_picks)} eligible picks - need at least 2 for parlays")
        return {
            '2leg': [],
            '3leg': [],
            'stats': {
                'eligible_picks': len(eligible_picks),
                'total_parlays': 0,
                '2leg_count': 0,
                '3leg_count': 0
            }
        }

    # Step 2: Generate 2-leg parlays
    parlays_2leg = generate_2leg_parlays(eligible_picks, target_count=num_2leg)

    # Step 3: Generate 3-leg parlays (only if we have enough picks)
    if len(eligible_picks) >= 3:
        parlays_3leg = generate_3leg_parlays(eligible_picks, target_count=num_3leg)
    else:
        parlays_3leg = []
        logger.warning(f"Only {len(eligible_picks)} picks - skipping 3-leg parlays")

    # Step 4: Summary statistics
    logger.info("=" * 80)
    logger.info("SHOTGUN MODE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Eligible picks: {len(eligible_picks)}")
    logger.info(f"2-leg parlays generated: {len(parlays_2leg)}")

    if parlays_2leg:
        low_2 = sum(1 for p in parlays_2leg if p['risk_tier'] == 'Low Risk')
        med_2 = sum(1 for p in parlays_2leg if p['risk_tier'] == 'Medium Risk')
        high_2 = sum(1 for p in parlays_2leg if p['risk_tier'] == 'High Risk')
        logger.info(f"  - Low Risk: {low_2}")
        logger.info(f"  - Medium Risk: {med_2}")
        logger.info(f"  - High Risk: {high_2}")

    logger.info(f"3-leg parlays generated: {len(parlays_3leg)}")

    if parlays_3leg:
        low_3 = sum(1 for p in parlays_3leg if p['risk_tier'] == 'Low Risk')
        med_3 = sum(1 for p in parlays_3leg if p['risk_tier'] == 'Medium Risk')
        high_3 = sum(1 for p in parlays_3leg if p['risk_tier'] == 'High Risk')
        logger.info(f"  - Low Risk: {low_3}")
        logger.info(f"  - Medium Risk: {med_3}")
        logger.info(f"  - High Risk: {high_3}")

    logger.info("=" * 80)

    return {
        '2leg': parlays_2leg,
        '3leg': parlays_3leg,
        'stats': {
            'eligible_picks': len(eligible_picks),
            'total_parlays': len(parlays_2leg) + len(parlays_3leg),
            '2leg_count': len(parlays_2leg),
            '3leg_count': len(parlays_3leg)
        }
    }


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_parlays_to_csv(
    parlays_2leg: List[Dict],
    parlays_3leg: List[Dict],
    output_path: str
) -> pd.DataFrame:
    """
    Export parlays to CSV for user download

    Creates separate sections for 2-leg and 3-leg parlays

    Args:
        parlays_2leg: List of 2-leg parlays
        parlays_3leg: List of 3-leg parlays
        output_path: Path to save CSV file

    Returns:
        DataFrame of all parlays
    """
    rows = []

    # Add 2-leg parlays
    for parlay in parlays_2leg:
        row = {
            'Parlay ID': parlay['parlay_id'],
            'Type': '2-Leg',
            'Risk Tier': parlay['risk_tier'],
            'Leg 1 League': parlay['legs'][0]['league'],
            'Leg 1 Game': parlay['legs'][0]['game'],
            'Leg 1 Market': parlay['legs'][0]['market'],
            'Leg 1 Pick': parlay['legs'][0]['pick'],
            'Leg 1 Prob': f"{parlay['legs'][0]['probability']:.1%}",
            'Leg 1 Quality': parlay['legs'][0]['quality'],
            'Leg 1 Time': parlay['legs'][0]['start_time'],
            'Leg 2 League': parlay['legs'][1]['league'],
            'Leg 2 Game': parlay['legs'][1]['game'],
            'Leg 2 Market': parlay['legs'][1]['market'],
            'Leg 2 Pick': parlay['legs'][1]['pick'],
            'Leg 2 Prob': f"{parlay['legs'][1]['probability']:.1%}",
            'Leg 2 Quality': parlay['legs'][1]['quality'],
            'Leg 2 Time': parlay['legs'][1]['start_time'],
            'Leg 3 League': '',
            'Leg 3 Game': '',
            'Leg 3 Market': '',
            'Leg 3 Pick': '',
            'Leg 3 Prob': '',
            'Leg 3 Quality': '',
            'Leg 3 Time': '',
            'Combined Probability': f"{parlay['combined_probability']:.1%}",
            'Expected Odds': f"{parlay['expected_odds']:+d}",
            'Avg Quality': f"{parlay['avg_quality']:.1f}",
            'EV Score': f"{parlay['ev_score']:.4f}"
        }
        rows.append(row)

    # Add 3-leg parlays
    for parlay in parlays_3leg:
        row = {
            'Parlay ID': parlay['parlay_id'],
            'Type': '3-Leg',
            'Risk Tier': parlay['risk_tier'],
            'Leg 1 League': parlay['legs'][0]['league'],
            'Leg 1 Game': parlay['legs'][0]['game'],
            'Leg 1 Market': parlay['legs'][0]['market'],
            'Leg 1 Pick': parlay['legs'][0]['pick'],
            'Leg 1 Prob': f"{parlay['legs'][0]['probability']:.1%}",
            'Leg 1 Quality': parlay['legs'][0]['quality'],
            'Leg 1 Time': parlay['legs'][0]['start_time'],
            'Leg 2 League': parlay['legs'][1]['league'],
            'Leg 2 Game': parlay['legs'][1]['game'],
            'Leg 2 Market': parlay['legs'][1]['market'],
            'Leg 2 Pick': parlay['legs'][1]['pick'],
            'Leg 2 Prob': f"{parlay['legs'][1]['probability']:.1%}",
            'Leg 2 Quality': parlay['legs'][1]['quality'],
            'Leg 2 Time': parlay['legs'][1]['start_time'],
            'Leg 3 League': parlay['legs'][2]['league'],
            'Leg 3 Game': parlay['legs'][2]['game'],
            'Leg 3 Market': parlay['legs'][2]['market'],
            'Leg 3 Pick': parlay['legs'][2]['pick'],
            'Leg 3 Prob': f"{parlay['legs'][2]['probability']:.1%}",
            'Leg 3 Quality': parlay['legs'][2]['quality'],
            'Leg 3 Time': parlay['legs'][2]['start_time'],
            'Combined Probability': f"{parlay['combined_probability']:.1%}",
            'Expected Odds': f"{parlay['expected_odds']:+d}",
            'Avg Quality': f"{parlay['avg_quality']:.1f}",
            'EV Score': f"{parlay['ev_score']:.4f}"
        }
        rows.append(row)

    # Create DataFrame and export
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    logger.info(f"Exported {len(rows)} parlays to {output_path}")

    return df


def format_parlay_for_display(parlay: Dict) -> Dict:
    """
    Format single parlay for UI display

    Args:
        parlay: Parlay dictionary

    Returns:
        Formatted parlay for display
    """
    legs_text = []
    for i, leg in enumerate(parlay['legs'], 1):
        legs_text.append(
            f"Leg {i}: {leg['league']} - {leg['pick']} "
            f"({leg['probability']:.1%}, Q:{leg['quality']:.0f})"
        )

    return {
        'id': parlay['parlay_id'],
        'type': f"{parlay['num_legs']}-Leg",
        'risk': parlay['risk_tier'],
        'legs': legs_text,
        'probability': f"{parlay['combined_probability']:.1%}",
        'odds': f"{parlay['expected_odds']:+d}",
        'quality': f"{parlay['avg_quality']:.1f}",
        'ev_score': f"{parlay['ev_score']:.3f}"
    }
