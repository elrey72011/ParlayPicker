"""
Example Usage Script
Demonstrates how to use the parlay pipeline components
"""

import pandas as pd
from datetime import datetime

# Import pipeline components
from sports_data_pipeline import SportsDataPipeline
from betting_features import BettingFeatureEngineer
from vertex_training import VertexAITrainingPipeline
from parlay_optimizer import ParlayOptimizer


def example_1_collect_data():
    """Example 1: Collect historical data for NFL"""
    print("="*60)
    print("Example 1: Collecting NFL Historical Data")
    print("="*60)
    
    # Initialize pipeline
    pipeline = SportsDataPipeline(
        api_keys={'NFL': 'YOUR_API_KEY'},
        output_dir='./example_data'
    )
    
    # Fetch data for recent seasons
    seasons = ['2023', '2024']
    df = pipeline.fetch_historical_games('NFL', seasons)
    
    print(f"\nCollected {len(df)} games")
    print(f"Columns: {list(df.columns)[:10]}...")
    print(f"\nSample data:")
    print(df.head())
    
    return df


def example_2_engineer_features():
    """Example 2: Engineer features from raw data"""
    print("\n" + "="*60)
    print("Example 2: Engineering Betting Features")
    print("="*60)
    
    # Load some sample data (or use data from example 1)
    df = pd.read_csv('./training_data/nfl_raw.csv')
    
    # Initialize feature engineer
    engineer = BettingFeatureEngineer('NFL')
    
    # Add individual feature sets
    print("\nCalculating ELO ratings...")
    df = engineer.calculate_elo_ratings(df)
    
    print("Adding recent form...")
    df = engineer.add_recent_form(df, windows=[3, 5])
    
    print("Adding rest days...")
    df = engineer.add_rest_days(df)
    
    print("Creating target variables...")
    df = engineer.create_target_variables(df)
    
    print(f"\nTotal features: {len(df.columns)}")
    print(f"Sample features: {list(df.columns)[50:60]}")
    
    return df


def example_3_train_model():
    """Example 3: Train a single model"""
    print("\n" + "="*60)
    print("Example 3: Training ML Model")
    print("="*60)
    
    # Load ML-ready data
    df = pd.read_csv('./training_data/nfl_ml_ready.csv')
    
    # Initialize training pipeline
    pipeline = VertexAITrainingPipeline(
        project_id='local',
        location='local',
        staging_bucket='local',
        sport='NFL'
    )
    
    # Prepare data for moneyline prediction
    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        df, 
        target_col='Target_HomeML'
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {len(X_train.columns)}")
    
    # Train XGBoost classifier
    result = pipeline.train_xgboost_classifier(
        X_train, X_test, y_train, y_test,
        model_name='NFL_moneyline'
    )
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
    print(f"Log Loss: {result['metrics']['log_loss']:.4f}")
    print(f"AUC: {result['metrics']['auc']:.4f}")
    print(f"Estimated ROI: {result['metrics']['roi_estimate']:.2f}%")
    
    print(f"\nTop 5 Most Important Features:")
    print(result['feature_importance'].head())
    
    return result


def example_4_analyze_bets():
    """Example 4: Analyze individual bets for expected value"""
    print("\n" + "="*60)
    print("Example 4: Analyzing Bets for Edge")
    print("="*60)
    
    # Initialize optimizer with trained models
    optimizer = ParlayOptimizer(
        model_dir='./models/nfl',
        min_edge=0.03
    )
    
    # Load models
    optimizer.load_models('NFL')
    
    # Load today's games (with features)
    games_df = pd.read_csv('./data/todays_games_features.csv')
    
    # Load odds from theover.ai
    odds_df = pd.read_csv('./data/theover/theover_nfl_2024-11-23.csv')
    
    print(f"\nAnalyzing {len(games_df)} games...")
    
    # Analyze for expected value
    bets_df = optimizer.analyze_single_bets(games_df, odds_df)
    
    print(f"\nFound {len(bets_df)} +EV bets")
    print(f"\nTop 5 Bets by Expected Value:")
    print(bets_df.head()[['Selection', 'ModelProb', 'Odds', 'Edge', 'ExpectedValue']])
    
    return bets_df


def example_5_generate_parlays():
    """Example 5: Generate optimal parlays"""
    print("\n" + "="*60)
    print("Example 5: Generating Optimal Parlays")
    print("="*60)
    
    # Use bets from example 4
    bets_df = pd.read_csv('./output/single_bets.csv')
    
    # Initialize optimizer
    optimizer = ParlayOptimizer(
        model_dir='./models/nfl',
        min_edge=0.03
    )
    optimizer.load_models('NFL')
    
    # Generate 3-leg parlays
    print("\nGenerating 3-leg parlays...")
    parlays = optimizer.generate_parlays(
        bets_df,
        parlay_size=3,
        max_parlays=10,
        allow_correlation=False
    )
    
    print(f"Generated {len(parlays)} parlays")
    
    # Display top parlay
    if parlays:
        top_parlay = parlays[0]
        print(f"\nTop Parlay:")
        print(f"Win Probability: {top_parlay['win_probability']:.2%}")
        print(f"Odds: {top_parlay['american_odds']:+.0f}")
        print(f"Expected Value: {top_parlay['expected_value']:.2%}")
        print(f"Kelly Bet Size: {top_parlay['kelly_fraction']:.2%} of bankroll")
        print(f"\nLegs:")
        for i, leg in enumerate(top_parlay['legs'], 1):
            print(f"  {i}. {leg['Selection']} ({leg['Odds']:+.0f})")
    
    return parlays


def example_6_calculate_ev():
    """Example 6: Manual expected value calculation"""
    print("\n" + "="*60)
    print("Example 6: Calculating Expected Value")
    print("="*60)
    
    optimizer = ParlayOptimizer('./models/nfl')
    
    # Example bet: Patriots -150 (favorite)
    model_prob = 0.65  # Model says 65% chance Patriots win
    odds = -150        # Vegas odds
    
    # Calculate EV
    ev = optimizer.calculate_expected_value(model_prob, odds)
    
    print(f"\nBet: Patriots Moneyline")
    print(f"Model Probability: {model_prob:.2%}")
    print(f"Odds: {odds:+d}")
    print(f"Implied Probability: {optimizer.calculate_implied_probability(odds):.2%}")
    print(f"Edge: {(model_prob - optimizer.calculate_implied_probability(odds)):.2%}")
    print(f"Expected Value: {ev:.2%}")
    
    if ev > 0:
        print(f"\n✓ POSITIVE EV - This is a +EV bet!")
        
        # Calculate Kelly
        decimal_odds = optimizer.american_to_decimal(odds)
        kelly = optimizer.calculate_kelly(model_prob, decimal_odds)
        print(f"Kelly Criterion recommends betting: {kelly:.2%} of bankroll")
    else:
        print(f"\n✗ Negative EV - Avoid this bet")


def example_7_compare_models():
    """Example 7: Compare different bet types"""
    print("\n" + "="*60)
    print("Example 7: Comparing Bet Types")
    print("="*60)
    
    optimizer = ParlayOptimizer('./models/nfl')
    optimizer.load_models('NFL')
    
    # Sample game features
    game_features = pd.DataFrame({
        'HomeELO': [1550],
        'AwayELO': [1480],
        'Home_Wins_L5': [4],
        'Away_Wins_L5': [2],
        'HomeRestDays': [7],
        'AwayRestDays': [7],
        # ... more features ...
    })
    
    print("\nPredictions for Patriots vs Jets:")
    
    # Moneyline
    if 'moneyline' in optimizer.models:
        ml_prob = optimizer.predict_game(game_features, 'moneyline')
        print(f"Moneyline: {ml_prob:.2%} chance Patriots win")
    
    # Spread
    if 'spread' in optimizer.models:
        spread_prob = optimizer.predict_game(game_features, 'spread')
        print(f"Spread: {spread_prob:.2%} chance Patriots cover -3.5")
    
    # Totals
    if 'totals' in optimizer.models:
        over_prob = optimizer.predict_game(game_features, 'totals')
        print(f"Total: {over_prob:.2%} chance game goes Over 42.5")


def main():
    """Run all examples"""
    print("="*60)
    print("PARLAY PIPELINE - EXAMPLE USAGE")
    print("="*60)
    
    # Uncomment the examples you want to run:
    
    # example_1_collect_data()
    # example_2_engineer_features()
    # example_3_train_model()
    # example_4_analyze_bets()
    # example_5_generate_parlays()
    example_6_calculate_ev()
    # example_7_compare_models()
    
    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60)


if __name__ == "__main__":
    main()
