"""
Parse theover.ai CSV files
"""
import pandas as pd
import numpy as np
from typing import Dict, List

def parse_spreads_csv(file_path: str) -> pd.DataFrame:
    """Parse spreads CSV from theover.ai"""
    df = pd.read_csv(file_path)
    
    # Clean up
    df = df.rename(columns={
        'League': 'league',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'Pick': 'pick',
        'Line': 'line',
        'Market': 'market'
    })
    
    # Determine favorite/underdog
    df['is_home_favorite'] = df['line'] < 0
    df['abs_spread'] = abs(df['line'])
    
    return df

def parse_totals_csv(file_path: str) -> pd.DataFrame:
    """Parse totals CSV from theover.ai"""
    df = pd.read_csv(file_path)
    
    df = df.rename(columns={
        'League': 'league',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'Pick': 'pick',
        'Line': 'line',
        'Market': 'market'
    })
    
    # Mark over/under
    df['is_over'] = df['pick'] == 'Over'
    
    return df

# Test
if __name__ == "__main__":
    spreads = parse_spreads_csv("22_Nov_Spreads.csv")
    totals = parse_totals_csv("22_Nov_Totals.csv")
    
    print(f"Spreads: {len(spreads)} games")
    print(f"Totals: {len(totals)} games")
    print("\nSample spreads:")
    print(spreads.head())
    print("\nSample totals:")
    print(totals.head())
