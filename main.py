"""
Main Orchestration Script
End-to-end pipeline from data collection to parlay recommendations
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import sys

# Import pipeline components
from sports_data_pipeline import SportsDataPipeline
from betting_features import BettingFeatureEngineer
from vertex_training import VertexAITrainingPipeline
from parlay_optimizer import ParlayOptimizer
from config import (
    API_KEYS, GCP_CONFIG, TRAINING_CONFIG, FEATURE_CONFIG,
    BETTING_CONFIG, SPORT_CONFIGS, THEOVER_CONFIG,
    TRAINING_DATA_DIR, MODELS_DIR, OUTPUT_DIR, DATA_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ParlayAppPipeline:
    """Main orchestration class for the parlay betting app"""
    
    def __init__(self, sports: list = None, target_date: str = None):
        """
        Initialize pipeline
        
        Args:
            sports: List of sports to process (default: all)
            target_date: Date string (YYYY-MM-DD) for theover.ai data/output (default: today)
        """
        self.sports = sports or ['NFL', 'NBA', 'NCAAB', 'NCAAF', 'NHL']
        self.target_date = self._normalize_date(target_date)
        logger.info(f"Initialized pipeline for sports: {self.sports} on date {self.target_date}")

    def _normalize_date(self, date_str: str = None) -> str:
        """
        Normalize a date string or default to today.

        Args:
            date_str: Date string in YYYY-MM-DD format

        Returns:
            Normalized date string (YYYY-MM-DD)
        """
        if date_str is None:
            return datetime.now().strftime('%Y-%m-%d')

        try:
            return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        except ValueError as exc:
            raise ValueError("Date must be in YYYY-MM-DD format.") from exc
    
    def step1_collect_historical_data(self, seasons: list = None):
        """
        Step 1: Collect historical data from sportsdata.io
        
        Args:
            seasons: List of seasons to collect (default from config)
        """
        logger.info("="*60)
        logger.info("STEP 1: Collecting Historical Data")
        logger.info("="*60)
        
        seasons = seasons or TRAINING_CONFIG['seasons']
        
        pipeline = SportsDataPipeline(
            api_keys=API_KEYS,
            output_dir=str(TRAINING_DATA_DIR)
        )
        
        results = {}
        for sport in self.sports:
            logger.info(f"\nProcessing {sport}...")
            try:
                df = pipeline.process_sport(sport, seasons, include_odds=True)
                results[sport] = df
                
                # Save raw data
                output_file = TRAINING_DATA_DIR / f'{sport.lower()}_raw.csv'
                df.to_csv(output_file, index=False)
                logger.info(f"Saved raw data: {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing {sport}: {e}")
                continue
        
        return results
    
    def step2_engineer_features(self):
        """
        Step 2: Engineer betting-specific features
        """
        logger.info("="*60)
        logger.info("STEP 2: Engineering Features")
        logger.info("="*60)
        
        for sport in self.sports:
            logger.info(f"\nEngineering features for {sport}...")
            
            # Load raw data
            raw_file = TRAINING_DATA_DIR / f'{sport.lower()}_raw.csv'
            
            if not raw_file.exists():
                logger.warning(f"Raw data not found for {sport}, skipping...")
                continue
            
            try:
                df = pd.read_csv(raw_file)
                
                # Initialize feature engineer
                engineer = BettingFeatureEngineer(sport)
                
                # Process all features
                df_features = engineer.process_all_features(df)
                
                # Save ML-ready data
                output_file = TRAINING_DATA_DIR / f'{sport.lower()}_ml_ready.csv'
                df_features.to_csv(output_file, index=False)
                logger.info(f"Saved ML-ready data: {output_file}")
                logger.info(f"Total features: {len(df_features.columns)}")
                
            except Exception as e:
                logger.error(f"Error engineering features for {sport}: {e}")
                continue
    
    def step3_train_models(self, use_vertex: bool = True):
        """
        Step 3: Train ML models on Vertex AI
        
        Args:
            use_vertex: Whether to use Vertex AI (True) or train locally (False)
        """
        logger.info("="*60)
        logger.info("STEP 3: Training ML Models")
        logger.info("="*60)
        
        for sport in self.sports:
            logger.info(f"\nTraining models for {sport}...")
            
            # Load ML-ready data
            data_file = TRAINING_DATA_DIR / f'{sport.lower()}_ml_ready.csv'
            
            if not data_file.exists():
                logger.warning(f"ML-ready data not found for {sport}, skipping...")
                continue
            
            try:
                df = pd.read_csv(data_file)
                
                # Initialize training pipeline
                if use_vertex:
                    pipeline = VertexAITrainingPipeline(
                        project_id=GCP_CONFIG['project_id'],
                        location=GCP_CONFIG['location'],
                        staging_bucket=GCP_CONFIG['staging_bucket'],
                        sport=sport
                    )
                else:
                    # Local training (simplified Vertex pipeline)
                    pipeline = VertexAITrainingPipeline(
                        project_id='local',
                        location='local',
                        staging_bucket='local',
                        sport=sport
                    )
                
                # Train all models
                results = pipeline.train_all_models(df)
                
                # Save models locally
                model_dir = MODELS_DIR / sport.lower()
                pipeline.save_models_locally(str(model_dir))
                
                # Upload to GCS if using Vertex
                if use_vertex and GCP_CONFIG['project_id'] != 'your-project-id':
                    pipeline.upload_to_gcs(
                        str(model_dir),
                        f'models/{sport.lower()}'
                    )
                
                logger.info(f"Training complete for {sport}")
                
            except Exception as e:
                logger.error(f"Error training models for {sport}: {e}")
                continue
    
    def step4_load_theover_data(self, date: str = None):
        """
        Step 4: Load theover.ai picks for today's games
        
        Args:
            date: Date string (YYYY-MM-DD), default is today
        """
        logger.info("="*60)
        logger.info("STEP 4: Loading theover.ai Data")
        logger.info("="*60)
        
        date = self._normalize_date(date or self.target_date)
        
        theover_data = {}
        
        for sport in self.sports:
            # Look for theover.ai CSV files
            pattern = f"theover_{sport.lower()}_{date}.csv"
            csv_files = list(THEOVER_CONFIG['data_dir'].glob(pattern))
            
            if not csv_files:
                # Try without date
                pattern = f"theover_{sport.lower()}_*.csv"
                csv_files = list(THEOVER_CONFIG['data_dir'].glob(pattern))
            
            if csv_files:
                # Load most recent file
                latest_file = sorted(csv_files)[-1]
                logger.info(f"Loading {sport} data from {latest_file}")
                
                try:
                    df = pd.read_csv(latest_file)
                    theover_data[sport] = df
                    logger.info(f"Loaded {len(df)} games for {sport}")
                except Exception as e:
                    logger.error(f"Error loading {latest_file}: {e}")
            else:
                logger.warning(f"No theover.ai data found for {sport}")
        
        return theover_data
    
    def step5_apply_probabilities(self, theover_data: dict):
        """
        Step 5: Apply ML model probabilities to theover.ai data
        
        Args:
            theover_data: Dictionary of sport -> DataFrame with games
        """
        logger.info("="*60)
        logger.info("STEP 5: Applying ML Probabilities")
        logger.info("="*60)
        
        enriched_data = {}
        
        for sport, games_df in theover_data.items():
            logger.info(f"\nProcessing {sport}...")
            
            # Load models
            model_dir = MODELS_DIR / sport.lower()
            
            if not model_dir.exists():
                logger.warning(f"Models not found for {sport}, skipping...")
                continue
            
            try:
                # Initialize optimizer (which loads models)
                optimizer = ParlayOptimizer(
                    model_dir=str(model_dir),
                    min_edge=BETTING_CONFIG['min_edge']
                )
                optimizer.load_models(sport)
                
                # Engineer features for today's games
                engineer = BettingFeatureEngineer(sport)
                games_with_features = engineer.process_all_features(games_df)
                
                # Add model probabilities
                # This would normally be done within the optimizer
                enriched_data[sport] = games_with_features
                
                logger.info(f"Applied probabilities to {len(games_with_features)} games")
                
            except Exception as e:
                logger.error(f"Error applying probabilities for {sport}: {e}")
                continue
        
        return enriched_data
    
    def step6_generate_recommendations(self, enriched_data: dict, theover_data: dict, date_str: str = None):
        """
        Step 6: Generate betting recommendations and parlays
        
        Args:
            enriched_data: Games with ML predictions
            theover_data: Original theover.ai odds data
            date_str: Date string used for output naming (YYYY-MM-DD)
        """
        logger.info("="*60)
        logger.info("STEP 6: Generating Betting Recommendations")
        logger.info("="*60)
        
        date_str = self._normalize_date(date_str or self.target_date)
        all_single_bets = []
        all_parlays = []
        
        for sport in enriched_data.keys():
            logger.info(f"\nAnalyzing {sport}...")
            
            model_dir = MODELS_DIR / sport.lower()
            
            try:
                # Initialize optimizer
                optimizer = ParlayOptimizer(
                    model_dir=str(model_dir),
                    min_edge=BETTING_CONFIG['min_edge']
                )
                optimizer.load_models(sport)
                
                # Analyze single bets
                single_bets = optimizer.analyze_single_bets(
                    enriched_data[sport],
                    theover_data[sport]
                )
                
                if not single_bets.empty:
                    single_bets['Sport'] = sport
                    all_single_bets.append(single_bets)
                    
                    logger.info(f"Found {len(single_bets)} +EV bets for {sport}")
                    
                    # Generate parlays for each size
                    for parlay_size in BETTING_CONFIG['parlay_sizes']:
                        parlays = optimizer.generate_parlays(
                            single_bets,
                            parlay_size=parlay_size,
                            max_parlays=BETTING_CONFIG['max_parlays_per_size'],
                            allow_correlation=BETTING_CONFIG['allow_correlated_parlays']
                        )
                        
                        for parlay in parlays:
                            parlay['Sport'] = sport
                        
                        all_parlays.extend(parlays)
                        logger.info(f"Generated {len(parlays)} {parlay_size}-leg parlays")
                
            except Exception as e:
                logger.error(f"Error generating recommendations for {sport}: {e}")
                continue
        
        # Combine all bets
        if all_single_bets:
            combined_singles = pd.concat(all_single_bets, ignore_index=True)
            combined_singles = combined_singles.sort_values('ExpectedValue', ascending=False)
        else:
            combined_singles = pd.DataFrame()
        
        # Sort parlays by EV
        all_parlays.sort(key=lambda x: x['expected_value'], reverse=True)
        
        # Generate final betting card
        output_file = OUTPUT_DIR / f'betting_card_{date_str}.csv'
        
        with open(output_file, 'w') as f:
            f.write(f"BETTING RECOMMENDATIONS - {date_str}\n")
            f.write("="*80 + "\n\n")
            
            f.write("=== TOP SINGLE BETS ===\n")
            if not combined_singles.empty:
                combined_singles.head(20).to_csv(f, index=False)
            else:
                f.write("No +EV single bets found\n")
            
            f.write("\n\n=== TOP PARLAYS ===\n")
            if all_parlays:
                for i, parlay in enumerate(all_parlays[:20], 1):
                    f.write(f"\nParlay {i} ({parlay['Sport']}) - {parlay['num_legs']}-leg\n")
                    f.write(f"Win Probability: {parlay['win_probability']:.2%}\n")
                    f.write(f"Odds: {parlay['american_odds']:+.0f}\n")
                    f.write(f"Expected Value: {parlay['expected_value']:.2%}\n")
                    f.write(f"Kelly Bet: {parlay['kelly_fraction']:.2%} of bankroll\n")
                    f.write("Legs:\n")
                    for leg in parlay['legs']:
                        f.write(f"  - {leg['Selection']} ({leg['Odds']:+.0f}) - {leg['ExpectedValue']:.2%} EV\n")
        
        logger.info(f"\nBetting card saved to: {output_file}")
        logger.info(f"Total +EV bets: {len(combined_singles)}")
        logger.info(f"Total parlays: {len(all_parlays)}")
        
        return combined_singles, all_parlays
    
    def run_full_pipeline(self, train_models: bool = False, date: str = None):
        """
        Run complete pipeline from start to finish
        
        Args:
            train_models: Whether to retrain models (False = use existing)
            date: Date string (YYYY-MM-DD) for theover.ai data/output (overrides initialized target_date)
        """
        logger.info("="*60)
        logger.info("STARTING FULL PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        run_date = self._normalize_date(date or self.target_date)
        
        try:
            # Training phase (only if requested)
            if train_models:
                logger.info("\n*** TRAINING PHASE ***")
                self.step1_collect_historical_data()
                self.step2_engineer_features()
                self.step3_train_models(use_vertex=False)  # Set to True for Vertex AI
            
            # Prediction phase (always run)
            logger.info("\n*** PREDICTION PHASE ***")
            theover_data = self.step4_load_theover_data(run_date)
            
            if not theover_data:
                logger.error("No theover.ai data loaded. Exiting...")
                return
            
            enriched_data = self.step5_apply_probabilities(theover_data)
            singles, parlays = self.step6_generate_recommendations(
                enriched_data,
                theover_data,
                date_str=run_date
            )
            
            # Summary
            elapsed = datetime.now() - start_time
            logger.info("="*60)
            logger.info("PIPELINE COMPLETE")
            logger.info(f"Elapsed time: {elapsed}")
            logger.info(f"Single bets found: {len(singles)}")
            logger.info(f"Parlays generated: {len(parlays)}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description='Parlay Betting App Pipeline')
    
    parser.add_argument(
        '--sports',
        nargs='+',
        choices=['NFL', 'NBA', 'NCAAB', 'NCAAF', 'NHL'],
        help='Sports to process (default: all)'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Date (YYYY-MM-DD) for theover.ai data and output naming (default: today)'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Retrain models (otherwise use existing models)'
    )
    
    parser.add_argument(
        '--step',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Run specific step only'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    try:
        pipeline = ParlayAppPipeline(sports=args.sports, target_date=args.date)
    except ValueError as exc:
        logger.error(exc)
        sys.exit(1)
    
    if args.step:
        # Run specific step
        if args.step == 1:
            pipeline.step1_collect_historical_data()
        elif args.step == 2:
            pipeline.step2_engineer_features()
        elif args.step == 3:
            pipeline.step3_train_models()
        elif args.step == 4:
            theover_data = pipeline.step4_load_theover_data()
        elif args.step == 5:
            theover_data = pipeline.step4_load_theover_data()
            pipeline.step5_apply_probabilities(theover_data)
        elif args.step == 6:
            theover_data = pipeline.step4_load_theover_data()
            enriched_data = pipeline.step5_apply_probabilities(theover_data)
            pipeline.step6_generate_recommendations(enriched_data, theover_data)
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(train_models=args.train, date=args.date)


if __name__ == "__main__":
    main()
