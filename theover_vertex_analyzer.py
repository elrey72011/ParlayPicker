"""
Analyze theover.ai CSV uploads with Vertex AI predictions
Integrates directly into your existing Streamlit app
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional
import logging
from ml_predictions import get_vertex_ai_prediction, is_vertex_ai_enabled

logger = logging.getLogger(__name__)


def enrich_spread_pick_with_stats(row: pd.Series, sportsdata_client=None, apisports_client=None) -> Dict:
    """
    Enrich a single spread pick with team stats
    Handles both column naming conventions (home_team and HomeTeam)
    """
    # Handle both column naming conventions
    home_team = row.get('home_team') or row.get('HomeTeam', '')
    away_team = row.get('away_team') or row.get('AwayTeam', '')
    league = row.get('league') or row.get('League', 'NBA')
    pick = row.get('pick') or row.get('Pick', '')
    line = float(row.get('line') or row.get('Line', 0))
    
    features = {
        'league': league,
        'home_team': home_team,
        'away_team': away_team,
        'pick': pick,
        'line': line,
        'is_home_favorite': line < 0,
        'spread_abs': abs(line),
    }
    
    # Get team stats
    home_stats = get_team_stats(home_team, league, sportsdata_client)
    away_stats = get_team_stats(away_team, league, sportsdata_client)
    
    features.update({
        'home_win_pct': home_stats.get('win_pct', 0.5),
        'away_win_pct': away_stats.get('win_pct', 0.5),
        'home_avg_points': home_stats.get('avg_points', 100),
        'away_avg_points': away_stats.get('avg_points', 100),
        'home_def_rating': home_stats.get('def_rating', 105),
        'away_def_rating': away_stats.get('def_rating', 105),
        'home_last_5': home_stats.get('last_5_wins', 3),
        'away_last_5': away_stats.get('last_5_wins', 2),
    })
    
    return features


def get_team_stats(team_name: str, league: str, client=None) -> Dict:
    """
    Get team statistics
    TODO: Implement with your actual SportsData/API-Sports clients
    """
    # Placeholder - return estimated stats based on league
    if league == 'NBA':
        return {
            'win_pct': 0.55,
            'avg_points': 110,
            'def_rating': 105,
            'last_5_wins': 3,
        }
    elif league == 'NFL' or league == 'NCAAF':
        return {
            'win_pct': 0.55,
            'avg_points': 24,
            'def_rating': 20,
            'last_5_wins': 3,
        }
    elif league == 'NHL':
        return {
            'win_pct': 0.55,
            'avg_points': 3,
            'def_rating': 2.5,
            'last_5_wins': 3,
        }
    else:
        return {
            'win_pct': 0.50,
            'avg_points': 100,
            'def_rating': 100,
            'last_5_wins': 2.5,
        }


def build_vertex_features(enriched_data: Dict) -> List[float]:
    """
    Build feature vector for Vertex AI prediction
    Returns list of features in same order as training
    """
    features = [
        enriched_data.get('home_win_pct', 0.5),
        enriched_data.get('away_win_pct', 0.5),
        enriched_data.get('home_avg_points', 100),
        enriched_data.get('away_avg_points', 100),
        enriched_data.get('home_def_rating', 105),
        enriched_data.get('away_def_rating', 105),
        enriched_data.get('spread_abs', 0) / 20,  # Normalize
        enriched_data.get('home_last_5', 3) / 5,
        enriched_data.get('away_last_5', 2) / 5,
    ]
    
    return features


def calculate_expected_value(ai_prob: float, line: float, pick: str) -> Dict:
    """
    Calculate expected value of a bet
    """
    # For spreads, assume -110 odds (risk $110 to win $100)
    bet_to_win = 100
    bet_to_risk = 110
    
    # Expected value = (prob_win * profit) - (prob_loss * risk)
    ev = (ai_prob * bet_to_win) - ((1 - ai_prob) * bet_to_risk)
    ev_pct = (ev / bet_to_risk) * 100
    
    # Edge = how much better AI thinks it is than 50/50
    edge = ai_prob - 0.5
    
    # Kelly criterion
    kelly = (ai_prob * (100/110) - (1 - ai_prob)) / (100/110)
    kelly_pct = max(0, kelly) * 100
    
    return {
        'ai_probability': ai_prob,
        'expected_value': ev,
        'ev_percentage': ev_pct,
        'edge': edge,
        'edge_percentage': edge * 100,
        'kelly_percentage': kelly_pct,
        'recommendation': 'BET' if ev_pct > 2 else 'PASS',
        'confidence': abs(edge)
    }


def analyze_theover_spreads_with_vertex(
    spreads_df: pd.DataFrame,
    sportsdata_client=None,
    apisports_client=None
) -> pd.DataFrame:
    """
    Analyze theover.ai spreads with Vertex AI predictions
    """
    if not is_vertex_ai_enabled():
        st.warning("⚠️ Vertex AI is disabled. Enable in Settings.")
        return pd.DataFrame()
    
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in spreads_df.iterrows():
        # Get team names (handle both column formats)
        home_team = row.get('home_team') or row.get('HomeTeam', 'Unknown')
        away_team = row.get('away_team') or row.get('AwayTeam', 'Unknown')
        
        status_text.text(f"Analyzing {away_team} @ {home_team}... ({idx+1}/{len(spreads_df)})")
        
        try:
            # Enrich with stats
            enriched = enrich_spread_pick_with_stats(row, sportsdata_client, apisports_client)
            
            # Build features
            features = build_vertex_features(enriched)
            
            # Get Vertex AI prediction
            ai_prob = get_vertex_ai_prediction(features)
            
            if ai_prob is not None:
                # Calculate EV
                ev_data = calculate_expected_value(ai_prob, enriched['line'], enriched['pick'])
                
                # Combine everything
                enriched.update(ev_data)
                results.append(enriched)
            else:
                logger.warning(f"Prediction failed for {home_team} vs {away_team}")
                
        except Exception as e:
            logger.error(f"Error analyzing row {idx}: {e}")
            continue
        
        progress_bar.progress((idx + 1) / len(spreads_df))
    
    status_text.text("✅ Analysis complete!")
    progress_bar.empty()
    status_text.empty()
    
    results_df = pd.DataFrame(results)
    
    # Sort by EV (best bets first)
    if 'ev_percentage' in results_df.columns and len(results_df) > 0:
        results_df = results_df.sort_values('ev_percentage', ascending=False)
    
    return results_df


def show_best_bets_table(results_df: pd.DataFrame):
    """Display best bets in a nice table"""
    
    st.subheader("🎯 Best Betting Opportunities")
    st.caption("Ranked by Expected Value")
    
    # Filter to only positive EV bets
    good_bets = results_df[results_df['recommendation'] == 'BET'].copy()
    
    if len(good_bets) == 0:
        st.info("No positive EV opportunities found in this set.")
        st.write("This could mean:")
        st.write("- The market odds are efficient")
        st.write("- The AI model needs more training data")
        st.write("- Current model is using placeholder team stats")
        return
    
    st.success(f"✅ Found {len(good_bets)} positive EV bets!")
    
    # Show top 10
    for idx, bet in good_bets.head(10).iterrows():
        with st.expander(
            f"{'🟢' if bet['ev_percentage'] > 5 else '🟡'} "
            f"{bet['away_team']} @ {bet['home_team']} - "
            f"Pick: {bet['pick']} {bet['line']:+.1f} - "
            f"EV: {bet['ev_percentage']:+.2f}%",
            expanded=(idx < 3)
        ):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "AI Win Probability",
                    f"{bet['ai_probability'] * 100:.1f}%"
                )
            
            with col2:
                st.metric(
                    "Expected Value",
                    f"{bet['ev_percentage']:+.2f}%",
                    delta=f"{bet['ev_percentage']:+.2f}%"
                )
            
            with col3:
                st.metric(
                    "Edge",
                    f"{bet['edge_percentage']:+.2f}%"
                )
            
            with col4:
                st.metric(
                    "Kelly %",
                    f"{bet['kelly_percentage']:.1f}%",
                    help="Suggested % of bankroll"
                )
            
            # Game details
            st.write(f"**League:** {bet['league']}")
            st.write(f"**Matchup:** {bet['away_team']} @ {bet['home_team']}")
            st.write(f"**theover.ai Pick:** {bet['pick']} {bet['line']:+.1f}")
            
            # Stats
            with st.expander("📊 Team Stats Used"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**{bet['home_team']}**")
                    st.write(f"Win %: {bet['home_win_pct']:.1%}")
                    st.write(f"Avg Points: {bet['home_avg_points']:.1f}")
                    st.write(f"Last 5: {bet['home_last_5']}-{5-bet['home_last_5']}")
                with col_b:
                    st.write(f"**{bet['away_team']}**")
                    st.write(f"Win %: {bet['away_win_pct']:.1%}")
                    st.write(f"Avg Points: {bet['away_avg_points']:.1f}")
                    st.write(f"Last 5: {bet['away_last_5']}-{5-bet['away_last_5']}")
    
    # Full table
    st.subheader("📊 All Positive EV Bets")
    
    display_df = good_bets[[
        'league', 'home_team', 'away_team', 'pick', 'line',
        'ai_probability', 'ev_percentage', 'edge_percentage', 
        'kelly_percentage', 'recommendation'
    ]].copy()
    
    # Format for display
    display_df['ai_probability'] = display_df['ai_probability'].apply(lambda x: f"{x*100:.1f}%")
    display_df['ev_percentage'] = display_df['ev_percentage'].apply(lambda x: f"{x:+.2f}%")
    display_df['edge_percentage'] = display_df['edge_percentage'].apply(lambda x: f"{x:+.2f}%")
    display_df['kelly_percentage'] = display_df['kelly_percentage'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Full Analysis",
        csv,
        f"vertex_ai_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv"
    )
