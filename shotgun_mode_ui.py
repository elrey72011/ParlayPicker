"""
Shotgun Mode Streamlit UI Components
Display functions for parlay recommendations
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import List, Dict
import logging

from shotgun_mode import export_parlays_to_csv

logger = logging.getLogger(__name__)


def display_shotgun_mode_ui(parlays_2leg: List[Dict], parlays_3leg: List[Dict]):
    """
    Display Shotgun Mode in Streamlit UI

    Args:
        parlays_2leg: List of 2-leg parlays
        parlays_3leg: List of 3-leg parlays
    """
    st.markdown("---")
    st.title("🎯 Shotgun Mode: Optimized Parlays")
    st.markdown("*Automatically generated parlay combinations optimized for expected value*")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("2-Leg Parlays", len(parlays_2leg))

    with col2:
        st.metric("3-Leg Parlays", len(parlays_3leg))

    with col3:
        total = len(parlays_2leg) + len(parlays_3leg)
        st.metric("Total Parlays", total)

    with col4:
        if parlays_2leg:
            avg_prob = sum(p['combined_probability'] for p in parlays_2leg) / len(parlays_2leg)
            st.metric("Avg 2-Leg Prob", f"{avg_prob:.1%}")

    # Filters
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        parlay_type = st.selectbox(
            "Parlay Type",
            ["All", "2-Leg Only", "3-Leg Only"],
            key="shotgun_parlay_type"
        )

    with col2:
        risk_filter = st.multiselect(
            "Risk Tier",
            ["Low Risk", "Medium Risk", "High Risk"],
            default=["Low Risk", "Medium Risk", "High Risk"],
            key="shotgun_risk_filter"
        )

    with col3:
        sort_by = st.selectbox(
            "Sort By",
            ["EV Score", "Probability", "Expected Odds", "Quality"],
            key="shotgun_sort_by"
        )

    # Filter parlays
    all_parlays = []

    if parlay_type in ["All", "2-Leg Only"]:
        all_parlays.extend(parlays_2leg)

    if parlay_type in ["All", "3-Leg Only"]:
        all_parlays.extend(parlays_3leg)

    # Apply risk filter
    filtered = [p for p in all_parlays if p['risk_tier'] in risk_filter]

    # Sort
    if sort_by == "EV Score":
        filtered.sort(key=lambda x: x['ev_score'], reverse=True)
    elif sort_by == "Probability":
        filtered.sort(key=lambda x: x['combined_probability'], reverse=True)
    elif sort_by == "Expected Odds":
        filtered.sort(key=lambda x: abs(x['expected_odds']), reverse=True)
    else:  # Quality
        filtered.sort(key=lambda x: x['avg_quality'], reverse=True)

    st.markdown(f"**Showing {len(filtered)} parlays**")

    if len(filtered) == 0:
        st.warning("No parlays match the selected filters.")
        return

    # Display parlays by risk tier
    for risk_tier in ["Low Risk", "Medium Risk", "High Risk"]:
        if risk_tier not in risk_filter:
            continue

        tier_parlays = [p for p in filtered if p['risk_tier'] == risk_tier]

        if not tier_parlays:
            continue

        # Risk tier header with color
        risk_colors = {
            "Low Risk": "🟢",
            "Medium Risk": "🟡",
            "High Risk": "🔴"
        }

        st.markdown(f"### {risk_colors[risk_tier]} {risk_tier} ({len(tier_parlays)} parlays)")

        # Display parlays in expandable cards
        for parlay in tier_parlays:
            with st.expander(
                f"**{parlay['parlay_id']}** - {parlay['num_legs']}-Leg | "
                f"Prob: {parlay['combined_probability']:.1%} | "
                f"Odds: {parlay['expected_odds']:+d} | "
                f"Quality: {parlay['avg_quality']:.1f}"
            ):
                # Legs table
                legs_data = []
                for i, leg in enumerate(parlay['legs'], 1):
                    legs_data.append({
                        'Leg': i,
                        'League': leg['league'],
                        'Game': leg['game'],
                        'Market': leg['market'],
                        'Pick': leg['pick'],
                        'Probability': f"{leg['probability']:.1%}",
                        'Quality': f"{leg['quality']:.0f}",
                        'Time': leg['start_time']
                    })

                st.table(pd.DataFrame(legs_data))

                # Parlay stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Combined Prob", f"{parlay['combined_probability']:.1%}")
                with col2:
                    st.metric("Expected Odds", f"{parlay['expected_odds']:+d}")
                with col3:
                    st.metric("Avg Quality", f"{parlay['avg_quality']:.1f}")
                with col4:
                    st.metric("EV Score", f"{parlay['ev_score']:.3f}")

                # Diversity info
                unique_leagues = len(set(parlay['leagues']))
                unique_markets = len(set(parlay['markets']))
                st.caption(
                    f"**Leagues:** {', '.join(set(parlay['leagues']))} ({unique_leagues} unique) | "
                    f"**Markets:** {', '.join(set(parlay['markets']))} ({unique_markets} unique)"
                )

    # Export button
    st.markdown("---")
    if st.button("📥 Export All Parlays to CSV", key="shotgun_export_button"):
        try:
            # Create output directory if it doesn't exist
            import os
            output_dir = "/tmp"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            output_path = os.path.join(
                output_dir,
                f"shotgun_parlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )

            export_parlays_to_csv(parlays_2leg, parlays_3leg, output_path)
            st.success(f"✅ Exported to {output_path}")

            # Provide download button
            with open(output_path, 'r') as f:
                csv_data = f.read()

            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"shotgun_parlays_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="shotgun_download_button"
            )

        except Exception as e:
            st.error(f"❌ Error exporting parlays: {e}")
            logger.error(f"Export error: {e}", exc_info=True)


def display_shotgun_mode_summary(stats: Dict):
    """
    Display summary statistics for Shotgun Mode

    Args:
        stats: Statistics dictionary from generate_shotgun_mode_parlays
    """
    st.info(
        f"📊 **Shotgun Mode Summary:** Generated {stats['total_parlays']} parlays "
        f"({stats['2leg_count']} 2-leg, {stats['3leg_count']} 3-leg) "
        f"from {stats['eligible_picks']} eligible picks."
    )


def display_shotgun_mode_help():
    """Display help information for Shotgun Mode"""
    with st.expander("ℹ️ What is Shotgun Mode?"):
        st.markdown("""
        **Shotgun Mode** automatically generates optimized parlay combinations from your picks.

        ### How it works:
        1. **Filters Eligible Picks:** Uses only Grade A/B picks with 45-75% probability
        2. **Generates Combinations:** Creates 2-leg and 3-leg parlays
        3. **Avoids Correlation:** Ensures picks are from different games
        4. **Optimizes for EV:** Ranks parlays by expected value score
        5. **Risk Categorization:** Groups parlays by risk level

        ### Risk Tiers:
        - **🟢 Low Risk:** Higher win probability, safer bets (2-leg: >30%, 3-leg: >18%)
        - **🟡 Medium Risk:** Balanced risk/reward (2-leg: 20-30%, 3-leg: 10-18%)
        - **🔴 High Risk:** Higher potential payout (2-leg: <20%, 3-leg: <10%)

        ### EV Score Factors:
        - Combined probability
        - Average pick quality
        - League diversity bonus (+10% per additional league)
        - Market diversity bonus (+5% per additional market type)
        - Timing spread bonus (up to +15% for staggered start times)

        ### Best Practices:
        - Start with Low/Medium risk parlays for consistent returns
        - Diversify across leagues and markets
        - Review individual leg quality before betting
        - Export to CSV for easy tracking
        """)
