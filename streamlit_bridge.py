"""
Streamlit Bridge Module
Connects the ML pipeline backend with the Streamlit frontend
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PipelineBridge:
    """Bridge between ML pipeline and Streamlit app"""
    
    def __init__(self, output_dir: str = './output', models_dir: str = './models'):
        """
        Initialize bridge
        
        Args:
            output_dir: Directory with pipeline outputs
            models_dir: Directory with trained models
        """
        self.output_dir = Path(output_dir)
        self.models_dir = Path(models_dir)
        
    def load_todays_predictions(self, date: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load predictions from pipeline for given date
        
        Args:
            date: Date string (YYYY-MM-DD), defaults to today
            
        Returns:
            Dictionary with predictions or None if not found
        """
        date_str = date or datetime.now().strftime('%Y-%m-%d')
        betting_card_file = self.output_dir / f'betting_card_{date_str}.csv'
        
        if not betting_card_file.exists():
            logger.warning(f"No predictions found for {date_str}")
            return None
        
        try:
            # Read the CSV file
            df = pd.read_csv(betting_card_file)
            
            # Check if it's the format from parlay_optimizer
            if 'Sport' in df.columns and 'ExpectedValue' in df.columns:
                return self._parse_betting_card(df)
            else:
                # Try to parse as generic format
                return self._parse_generic_format(df)
                
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return None
    
    def _parse_betting_card(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse betting card format from parlay_optimizer"""
        
        predictions = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'single_bets': [],
            'parlays': [],
            'summary': {}
        }
        
        # Parse single bets
        single_bets = df[df.get('BetType', '').notna()].to_dict('records')
        
        for bet in single_bets:
            predictions['single_bets'].append({
                'sport': bet.get('Sport', 'Unknown'),
                'game': f"{bet.get('AwayTeam', '')} @ {bet.get('HomeTeam', '')}",
                'home_team': bet.get('HomeTeam', ''),
                'away_team': bet.get('AwayTeam', ''),
                'bet_type': bet.get('BetType', ''),
                'selection': bet.get('Selection', ''),
                'odds': bet.get('Odds', 0),
                'model_prob': bet.get('ModelProb', 0.5),
                'implied_prob': bet.get('ImpliedProb', 0.5),
                'edge': bet.get('Edge', 0),
                'expected_value': bet.get('ExpectedValue', 0),
                'confidence': bet.get('Confidence', 'Medium')
            })
        
        # Summary stats
        predictions['summary'] = {
            'total_bets': len(predictions['single_bets']),
            'avg_edge': df['Edge'].mean() if 'Edge' in df.columns else 0,
            'avg_ev': df['ExpectedValue'].mean() if 'ExpectedValue' in df.columns else 0,
            'high_confidence_count': len(df[df.get('Confidence', '') == 'High'])
        }
        
        return predictions
    
    def _parse_generic_format(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Parse generic CSV format"""
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'single_bets': df.to_dict('records'),
            'parlays': [],
            'summary': {
                'total_bets': len(df),
            }
        }
    
    def get_model_info(self, sport: str) -> Optional[Dict[str, Any]]:
        """
        Get information about trained models for a sport
        
        Args:
            sport: Sport name (NFL, NBA, etc.)
            
        Returns:
            Model info dictionary or None
        """
        sport_lower = sport.lower()
        model_dir = self.models_dir / sport_lower
        
        if not model_dir.exists():
            return None
        
        # Check for model files
        model_files = list(model_dir.glob('*.pkl'))
        
        if not model_files:
            return None
        
        return {
            'sport': sport,
            'models': [f.stem for f in model_files],
            'model_count': len(model_files),
            'last_updated': max(f.stat().st_mtime for f in model_files),
            'available': True
        }
    
    def load_predictions_for_streamlit(
        self, 
        date: Optional[str] = None,
        sport_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load predictions formatted for Streamlit display
        
        Args:
            date: Date to load predictions for
            sport_filter: Optional sport to filter by
            
        Returns:
            List of prediction dictionaries
        """
        predictions = self.load_todays_predictions(date)
        
        if not predictions:
            return []
        
        single_bets = predictions.get('single_bets', [])
        
        # Filter by sport if requested
        if sport_filter:
            single_bets = [bet for bet in single_bets if bet.get('sport', '').upper() == sport_filter.upper()]
        
        # Format for Streamlit
        formatted = []
        for bet in single_bets:
            formatted.append({
                'Game': bet.get('game', ''),
                'Sport': bet.get('sport', ''),
                'Bet': bet.get('selection', ''),
                'Type': bet.get('bet_type', ''),
                'Odds': self._format_odds(bet.get('odds', 0)),
                'Win Prob': f"{bet.get('model_prob', 0.5):.1%}",
                'Edge': f"{bet.get('edge', 0):.1%}",
                'EV': f"{bet.get('expected_value', 0):.1%}",
                'Confidence': bet.get('confidence', 'Medium'),
                'home_team': bet.get('home_team', ''),
                'away_team': bet.get('away_team', ''),
                'raw_prob': bet.get('model_prob', 0.5),
                'raw_edge': bet.get('edge', 0),
                'raw_ev': bet.get('expected_value', 0)
            })
        
        return formatted
    
    def _format_odds(self, odds: float) -> str:
        """Format American odds for display"""
        if odds >= 0:
            return f"+{int(odds)}"
        else:
            return f"{int(odds)}"
    
    def get_available_dates(self, days_back: int = 7) -> List[str]:
        """
        Get list of dates with available predictions
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of date strings
        """
        dates = []
        for i in range(days_back):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            file_path = self.output_dir / f'betting_card_{date}.csv'
            if file_path.exists():
                dates.append(date)
        
        return dates


def load_pipeline_predictions(date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load pipeline predictions
    
    Args:
        date: Date string (YYYY-MM-DD), defaults to today
        
    Returns:
        Predictions dictionary or None
    """
    bridge = PipelineBridge()
    return bridge.load_todays_predictions(date)


def format_for_streamlit_table(date: Optional[str] = None, sport: Optional[str] = None) -> pd.DataFrame:
    """
    Load predictions and format as Streamlit-ready DataFrame
    
    Args:
        date: Date to load
        sport: Optional sport filter
        
    Returns:
        DataFrame ready for st.dataframe()
    """
    bridge = PipelineBridge()
    predictions = bridge.load_predictions_for_streamlit(date, sport)
    
    if not predictions:
        return pd.DataFrame()
    
    df = pd.DataFrame(predictions)
    
    # Drop raw columns (keep for sorting but hide)
    display_cols = ['Sport', 'Game', 'Bet', 'Type', 'Odds', 'Win Prob', 'Edge', 'EV', 'Confidence']
    return df[display_cols]


def get_pipeline_status() -> Dict[str, Any]:
    """
    Get status of pipeline components
    
    Returns:
        Status dictionary
    """
    bridge = PipelineBridge()
    
    status = {
        'output_dir_exists': bridge.output_dir.exists(),
        'models_dir_exists': bridge.models_dir.exists(),
        'available_dates': bridge.get_available_dates(),
        'models': {}
    }
    
    # Check for models
    sports = ['NFL', 'NBA', 'NCAAB', 'NCAAF', 'NHL']
    for sport in sports:
        model_info = bridge.get_model_info(sport)
        if model_info:
            status['models'][sport] = model_info
    
    status['models_available'] = len(status['models']) > 0
    status['predictions_available'] = len(status['available_dates']) > 0
    
    return status


# Example Streamlit integration code
STREAMLIT_EXAMPLE = """
# Add this to your Streamlit app:

import streamlit as st
from streamlit_bridge import format_for_streamlit_table, get_pipeline_status

# Show pipeline status
st.sidebar.header("🤖 ML Pipeline Status")
status = get_pipeline_status()

if status['models_available']:
    st.sidebar.success(f"✅ {len(status['models'])} sports trained")
else:
    st.sidebar.warning("⚠️ No models found - run training")

if status['predictions_available']:
    st.sidebar.success(f"✅ Predictions available for {len(status['available_dates'])} days")
else:
    st.sidebar.warning("⚠️ No predictions - run pipeline")

# Load and display predictions
st.header("🎯 ML Pipeline Predictions")

sport_filter = st.selectbox("Filter by sport", ['All', 'NFL', 'NBA', 'NCAAB', 'NCAAF', 'NHL'])
sport = None if sport_filter == 'All' else sport_filter

predictions_df = format_for_streamlit_table(sport=sport)

if not predictions_df.empty:
    st.dataframe(predictions_df, width="stretch")
    
    # Show summary
    st.subheader("Summary")
    st.metric("Total Bets", len(predictions_df))
    st.metric("Avg Edge", f"{predictions_df['raw_edge'].mean():.1%}")
else:
    st.info("No predictions available. Run: python main.py")
"""


if __name__ == "__main__":
    # Test the bridge
    print("Testing Pipeline Bridge...")
    
    # Test loading predictions
    predictions = load_pipeline_predictions()
    if predictions:
        print(f"✓ Loaded {len(predictions.get('single_bets', []))} predictions")
        print(f"  Date: {predictions.get('date')}")
        print(f"  Summary: {predictions.get('summary')}")
    else:
        print("✗ No predictions found")
    
    # Test status
    status = get_pipeline_status()
    print(f"\n✓ Pipeline Status:")
    print(f"  Models available: {status['models_available']}")
    print(f"  Predictions available: {status['predictions_available']}")
    print(f"  Sports with models: {list(status['models'].keys())}")
    
    # Test DataFrame format
    df = format_for_streamlit_table()
    if not df.empty:
        print(f"\n✓ Formatted {len(df)} predictions for Streamlit")
        print(df.head())
    else:
        print("\n✗ No predictions to format")
    
    print("\n" + "="*50)
    print("Example Streamlit Integration Code:")
    print("="*50)
    print(STREAMLIT_EXAMPLE)
