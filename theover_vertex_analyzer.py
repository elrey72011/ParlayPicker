"""
Analyze theover.ai CSV uploads with Vertex AI predictions
FIXED VERSION - Shows ALL results, not just positive EV
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
    
    # Confidence level based on edge
    if abs(edge) > 0.15:
        confidence_level = 'High'
    elif abs(edge) > 0.08:
        confidence_level = 'Medium'
    else:
        confidence_level = 'Low'
    
    return {
        'ai_probability': ai_prob,
        'expected_value': ev,
        'ev_percentage': ev_pct,
        'edge': edge,
        'edge_percentage': edge * 100,
        'kelly_percentage': kelly_pct,
        'recommendation': 'BET' if ev_pct > 2 else 'PASS',  # Still calculate but don't filter
        'confidence': abs(edge),
        'confidence_level': confidence_level
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
    errors = []
    
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
                error_msg = f"Prediction returned None for {home_team} vs {away_team}"
                errors.append(error_msg)
                logger.warning(error_msg)
                
        except Exception as e:
            error_msg = f"Error analyzing {home_team} vs {away_team}: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg, exc_info=True)
            continue
        
        progress_bar.progress((idx + 1) / len(spreads_df))
    
    status_text.text("✅ Analysis complete!")
    progress_bar.empty()
    status_text.empty()
    
    # Show error summary if any
    if errors:
        with st.expander(f"⚠️ {len(errors)} Prediction Errors", expanded=False):
            for error in errors[:10]:  # Show first 10
                st.write(f"- {error}")
    
    results_df = pd.DataFrame(results)
    
    # Sort by AI probability (most confident first) instead of just EV
    if 'confidence' in results_df.columns and len(results_df) > 0:
        results_df = results_df.sort_values('confidence', ascending=False)
    
    return results_df


def show_best_bets_table(results_df: pd.DataFrame):
    """Display ALL analysis results with ML predictions - NOT filtered by EV"""
    
    st.subheader("🎯 AI Prediction Analysis Results")
    st.caption("All analyzed games - sorted by prediction confidence")
    
    if len(results_df) == 0:
        st.warning("No results from analysis. Check errors above.")
        return
    
    st.success(f"✅ Analyzed {len(results_df)} games with AI predictions")
    
    # Show filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        show_all = st.checkbox("Show all predictions", value=True, key="show_all_preds")
        if not show_all:
            min_confidence = st.select_slider(
                "Minimum confidence",
                options=['Low', 'Medium', 'High'],
                value='Low',
                key="min_conf"
            )
    
    with col2:
        show_positive_ev_only = st.checkbox("Show only positive EV", value=False, key="pos_ev_only")
    
    # Apply filters
    display_df = results_df.copy()
    
    if not show_all:
        conf_map = {'Low': 0, 'Medium': 0.08, 'High': 0.15}
        min_conf_value = conf_map[min_confidence]
        display_df = display_df[display_df['confidence'] >= min_conf_value]
    
    if show_positive_ev_only:
        display_df = display_df[display_df['recommendation'] == 'BET']
    
    if len(display_df) == 0:
        st.info("No games match the selected filters. Adjust filters above.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Games", len(display_df))
    
    with col2:
        positive_ev_count = len(display_df[display_df['recommendation'] == 'BET'])
        st.metric("Positive EV", positive_ev_count)
    
    with col3:
        high_conf = len(display_df[display_df['confidence_level'] == 'High'])
        st.metric("High Confidence", high_conf)
    
    with col4:
        avg_confidence = display_df['confidence'].mean() * 100
        st.metric("Avg Edge", f"{avg_confidence:.1f}%")
    
    st.markdown("---")
    
    # Show detailed cards for each game
    for idx, bet in display_df.head(20).iterrows():  # Show top 20
        # Color code based on EV and confidence
        if bet['ev_percentage'] > 5:
            icon = '🟢'
            color = 'green'
        elif bet['ev_percentage'] > 0:
            icon = '🟡'
            color = 'orange'
        else:
            icon = '🔴'
            color = 'red'
        
        with st.expander(
            f"{icon} {bet['away_team']} @ {bet['home_team']} - "
            f"AI: {bet['ai_probability']*100:.1f}% - "
            f"EV: {bet['ev_percentage']:+.2f}% - "
            f"{bet['confidence_level']} Confidence",
            expanded=(idx < 3)
        ):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "AI Win Probability",
                    f"{bet['ai_probability'] * 100:.1f}%",
                    delta=f"{(bet['ai_probability'] - 0.5) * 100:+.1f}% vs 50/50"
                )
            
            with col2:
                st.metric(
                    "Expected Value",
                    f"{bet['ev_percentage']:+.2f}%",
                    delta="Bet" if bet['ev_percentage'] > 0 else "Pass"
                )
            
            with col3:
                st.metric(
                    "Edge vs Market",
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
            st.write(f"**Confidence Level:** {bet['confidence_level']}")
            
            # Show that these are placeholder stats
            st.info("ℹ️ **Note:** Currently using placeholder team statistics. "
                   "Connect real SportsData/API-Sports clients for accurate predictions.")
            
            # Stats (NOT nested in expander to avoid Streamlit error)
            st.markdown("**📊 Team Stats Used (Placeholder)**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**{bet['home_team']}**")
                st.write(f"Win %: {bet['home_win_pct']:.1%}")
                st.write(f"Avg Points: {bet['home_avg_points']:.1f}")
                st.write(f"Last 5: {bet['home_last_5']:.0f}-{5-bet['home_last_5']:.0f}")
            with col_b:
                st.write(f"**{bet['away_team']}**")
                st.write(f"Win %: {bet['away_win_pct']:.1%}")
                st.write(f"Avg Points: {bet['away_avg_points']:.1f}")
                st.write(f"Last 5: {bet['away_last_5']:.0f}-{5-bet['away_last_5']:.0f}")
    
    if len(display_df) > 20:
        st.info(f"Showing top 20 of {len(display_df)} results. "
               f"Download full results below.")
    
    # Full table
    st.markdown("---")
    st.subheader("📊 Full Results Table")
    
    display_cols = [
        'league', 'home_team', 'away_team', 'pick', 'line',
        'ai_probability', 'ev_percentage', 'edge_percentage', 
        'confidence_level', 'recommendation'
    ]
    
    table_df = display_df[display_cols].copy()
    
    # Format for display
    table_df['ai_probability'] = table_df['ai_probability'].apply(lambda x: f"{x*100:.1f}%")
    table_df['ev_percentage'] = table_df['ev_percentage'].apply(lambda x: f"{x:+.2f}%")
    table_df['edge_percentage'] = table_df['edge_percentage'].apply(lambda x: f"{x:+.2f}%")
    
    # Rename columns for display
    table_df.columns = [
        'League', 'Home', 'Away', 'Pick', 'Line',
        'AI Win %', 'EV %', 'Edge %', 'Confidence', 'Rec'
    ]
    
    st.dataframe(table_df, use_container_width=True, height=400)
    
    # Download
    csv = results_df.to_csv(index=False)
    st.download_button(
        "📥 Download Full Analysis CSV",
        csv,
        f"vertex_ai_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        key="download_analysis"
    )
    
    # Explanation
    with st.expander("ℹ️ Understanding the Results"):
        st.write("""
        **AI Win Probability:** The model's predicted chance of the pick winning
        
        **Expected Value (EV):** Expected profit/loss per $100 bet
        - Positive = Model thinks bet has value
        - Negative = Model says pass
        
        **Edge:** How much the model differs from 50/50
        - Higher edge = More confident prediction
        
        **Confidence Level:**
        - High: Edge > 15%
        - Medium: Edge 8-15%
        - Low: Edge < 8%
        
        **⚠️ Important:** Currently using placeholder team statistics. 
        For real predictions, connect to:
        1. SportsData.io API (for actual team stats)
        2. Your sentiment analyzer (for news/social data)
        3. Trained ML models (from the pipeline)
        """)
