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
    
    Args:
        row: Row from theover.ai spreads CSV
        sportsdata_client: Your SportsData client
        apisports_client: Your API-Sports client
        
    Returns:
        Dict with enriched features
    """
    features = {
        'league': row.get('league', 'NBA'),
        'home_team': row.get('home_team', ''),
        'away_team': row.get('away_team', ''),
        'pick': row.get('pick', ''),
        'line': float(row.get('line', 0)),
        'is_home_favorite': float(row.get('line', 0)) < 0,
        'spread_abs': abs(float(row.get('line', 0))),
    }
    
    # Get team stats (placeholder - customize based on your clients)
    home_stats = get_team_stats(features['home_team'], features['league'], sportsdata_client)
    away_stats = get_team_stats(features['away_team'], features['league'], sportsdata_client)
    
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
    # Placeholder - return estimated stats
    # In production, fetch from your APIs
    
    return {
        'win_pct': 0.55,  # Default to 55% win rate
        'avg_points': 110 if league == 'NBA' else 25,
        'def_rating': 105,
        'last_5_wins': 3,
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
    
    Args:
        ai_prob: AI predicted probability of outcome
        line: Betting line (spread or odds)
        pick: Team/side picked by theover.ai
        
    Returns:
        Dict with EV calculations
    """
    # For spreads, line is typically around -110 for both sides
    # For simplicity, assume -110 odds (risk $110 to win $100)
    
    bet_to_win = 100
    bet_to_risk = 110
    
    # Expected value = (prob_win * profit) - (prob_loss * risk)
    ev = (ai_prob * bet_to_win) - ((1 - ai_prob) * bet_to_risk)
    ev_pct = (ev / bet_to_risk) * 100
    
    # Edge = how much better AI thinks it is than 50/50
    edge = ai_prob - 0.5
    
    # Kelly criterion (fraction of bankroll to bet)
    # Kelly = (probability * odds - (1 - probability)) / odds
    # Assuming -110 odds
    kelly = (ai_prob * (100/110) - (1 - ai_prob)) / (100/110)
    kelly_pct = max(0, kelly) * 100  # Don't bet if negative
    
    return {
        'ai_probability': ai_prob,
        'expected_value': ev,
        'ev_percentage': ev_pct,
        'edge': edge,
        'edge_percentage': edge * 100,
        'kelly_percentage': kelly_pct,
        'recommendation': 'BET' if ev_pct > 2 else 'PASS',  # Need >2% EV
        'confidence': abs(edge)
    }


def analyze_theover_spreads_with_vertex(
    spreads_df: pd.DataFrame,
    sportsdata_client=None,
    apisports_client=None
) -> pd.DataFrame:
    """
    Analyze theover.ai spreads with Vertex AI predictions
    
    Args:
        spreads_df: DataFrame from uploaded CSV
        sportsdata_client: Optional stats client
        apisports_client: Optional stats client
        
    Returns:
        DataFrame with AI predictions and recommendations
    """
    if not is_vertex_ai_enabled():
        st.warning("⚠️ Vertex AI is disabled. Enable in Settings.")
        return spreads_df
    
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, row in spreads_df.iterrows():
        status_text.text(f"Analyzing {row['home_team']} vs {row['away_team']}...")
        
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
