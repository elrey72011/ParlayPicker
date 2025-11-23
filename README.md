# Sports Betting ML Pipeline - Parlay Optimizer

A complete end-to-end machine learning pipeline for sports betting analysis, from data collection through parlay optimization. Uses historical data from sportsdata.io, trains custom ML models on Vertex AI, and generates optimal betting recommendations.

## 🏈 Supported Sports
- **NFL** - National Football League
- **NBA** - National Basketball Association  
- **NCAAB** - NCAA Men's Basketball
- **NCAAF** - NCAA Football
- **NHL** - National Hockey League

## 📋 Features

- **Historical Data Pipeline**: Automated collection from sportsdata.io API
- **Advanced Feature Engineering**: 50+ betting-specific features including:
  - ELO ratings
  - Recent form and rolling statistics
  - Rest days and back-to-back analysis
  - Head-to-head records
  - Pace/tempo metrics
  - Strength of schedule
  - Line movement and public betting percentages
  
- **ML Model Training**: 
  - XGBoost classifiers for moneyline, spread, and totals
  - XGBoost regressors for margin and total points prediction
  - Time-series cross-validation
  - Hyperparameter optimization
  - Vertex AI integration

- **Parlay Optimization**:
  - Expected value calculation
  - Correlation detection
  - Kelly Criterion bet sizing
  - Multi-leg parlay generation

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo>
cd parlay-app

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.py` to set your API keys:

```python
API_KEYS = {
    'NFL': 'your-nfl-api-key',
    'NBA': 'your-nba-api-key',
    'NCAAB': 'your-ncaab-api-key',
    'NCAAF': 'your-ncaaf-api-key',
    'NHL': 'your-nhl-api-key'
}

GCP_CONFIG = {
    'project_id': 'your-gcp-project-id',
    'location': 'us-central1',
    'staging_bucket': 'your-bucket-name'
}
```

Or use environment variables:
```bash
export SPORTSDATA_NFL_KEY="your-key"
export SPORTSDATA_NBA_KEY="your-key"
export GCP_PROJECT_ID="your-project"
```

### 3. Initial Model Training

First time setup - train models on historical data:

```bash
# Train models for all sports
python main.py --train

# Or train specific sports
python main.py --train --sports NFL NBA
```

This will:
1. Collect 3+ years of historical data
2. Engineer 50+ features per sport
3. Train XGBoost models for each bet type
4. Save models locally to `./models/`

### 4. Daily Usage

Once models are trained, run daily predictions:

```bash
# Generate today's picks
python main.py

# Or for specific sports
python main.py --sports NFL
```

## 📁 Project Structure

```
parlay-app/
├── main.py                     # Main orchestration script
├── sports_data_pipeline.py     # Data collection from APIs
├── betting_features.py         # Feature engineering
├── vertex_training.py          # ML model training
├── parlay_optimizer.py         # Parlay generation and optimization
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
│
├── data/                       # Raw data and theover.ai CSVs
│   └── theover/               # Place theover.ai picks here
│       ├── theover_nfl_2024-11-23.csv
│       └── theover_nba_2024-11-23.csv
│
├── training_data/             # Processed training data
│   ├── nfl_raw.csv
│   ├── nfl_ml_ready.csv
│   └── ...
│
├── models/                    # Trained models
│   ├── nfl/
│   │   ├── NFL_moneyline.pkl
│   │   ├── NFL_spread.pkl
│   │   └── ...
│   └── ...
│
└── output/                    # Betting recommendations
    └── betting_card_2024-11-23.csv
```

## 🔄 Pipeline Workflow

### Training Phase (Run Initially or Weekly)

```bash
python main.py --train --sports NFL NBA
```

**Step 1: Collect Historical Data**
- Fetches games, team stats, and odds from sportsdata.io
- Stores in `training_data/`

**Step 2: Engineer Features**
- Calculates ELO ratings
- Creates rolling statistics
- Analyzes head-to-head records
- Adds betting-specific features

**Step 3: Train Models**
- Trains XGBoost models for each bet type
- Performs cross-validation
- Saves models to `models/`

### Prediction Phase (Run Daily)

```bash
python main.py
```

**Step 4: Load theover.ai Data**
- Reads today's picks from `data/theover/`
- Expected format: `theover_{sport}_{date}.csv`

**Step 5: Apply ML Probabilities**
- Runs games through trained models
- Generates win probabilities

**Step 6: Generate Recommendations**
- Analyzes expected value
- Identifies +EV bets
- Constructs optimal parlays
- Outputs to `output/betting_card_{date}.csv`

## 📊 theover.ai CSV Format

Place your theover.ai CSV files in `data/theover/` with this format:

```csv
GameID,DateTime,HomeTeam,AwayTeam,HomeMoneyLine,AwayMoneyLine,PointSpread,HomeSpreadOdds,AwaySpreadOdds,OverUnder,OverOdds,UnderOdds
12345,2024-11-23T13:00:00,Patriots,Jets,-150,+130,-3.5,-110,-110,42.5,-110,-110
```

Required columns:
- `GameID`: Unique game identifier
- `HomeTeam`, `AwayTeam`: Team names
- `HomeMoneyLine`, `AwayMoneyLine`: Moneyline odds
- `PointSpread`: Spread (from home team perspective)
- `HomeSpreadOdds`, `AwaySpreadOdds`: Spread odds
- `OverUnder`: Total points line
- `OverOdds`, `UnderOdds`: Total odds

## 🎯 Running Specific Steps

```bash
# Step 1: Collect data only
python main.py --step 1

# Step 2: Engineer features only
python main.py --step 2

# Step 3: Train models only
python main.py --step 3

# Step 6: Generate recommendations only
python main.py --step 6
```

## ⚙️ Configuration Options

### Betting Strategy (`config.py`)

```python
BETTING_CONFIG = {
    'min_edge': 0.03,              # 3% minimum edge
    'high_confidence_ev': 0.10,    # 10% for high confidence
    'parlay_sizes': [2, 3, 4, 5],  # Parlay leg counts
    'max_parlays_per_size': 10,    # Max parlays per size
    'kelly_fraction': 0.25,        # Use 1/4 Kelly
    'max_bet_size': 0.10,          # Max 10% of bankroll
}
```

### Model Parameters

```python
TRAINING_CONFIG = {
    'xgboost_params': {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        # ... more parameters
    }
}
```

## 📈 Output Format

### Betting Card (`output/betting_card_YYYY-MM-DD.csv`)

**Single Bets Section:**
```csv
Sport,GameID,HomeTeam,AwayTeam,BetType,Selection,ModelProb,Odds,Edge,ExpectedValue,Confidence
NFL,12345,Patriots,Jets,Moneyline,Patriots,0.58,-150,0.07,0.12,High
NBA,67890,Lakers,Celtics,Spread,Lakers -5.5,0.62,-110,0.09,0.15,High
```

**Parlays Section:**
- Each parlay with legs, combined odds, win probability, and expected value
- Kelly Criterion recommended bet size

## 🔧 Troubleshooting

### API Rate Limits
If you hit rate limits, adjust the sleep time in `sports_data_pipeline.py`:
```python
time.sleep(0.5)  # Increase to 1.0 or higher
```

### Missing Models
If you see "Models not found", run training first:
```bash
python main.py --train --sports NFL
```

### GCP Authentication
For Vertex AI, authenticate:
```bash
gcloud auth application-default login
```

### Memory Issues
For large datasets, process sports individually:
```bash
python main.py --train --sports NFL
python main.py --train --sports NBA
```

## 📝 Best Practices

1. **Train Weekly**: Retrain models weekly during season to capture recent trends
2. **Bankroll Management**: Always use Kelly Criterion sizing (configured at 1/4 Kelly)
3. **Edge Threshold**: Start with 3-5% minimum edge
4. **Correlation**: Avoid same-game parlays unless specifically analyzed
5. **Record Keeping**: Track all bets for model validation

## 🧪 Model Performance

Models are evaluated on:
- **Accuracy**: Win rate on test set
- **Log Loss**: Calibration quality
- **ROI**: Simulated betting returns
- **AUC**: Discrimination ability

Check `models/metrics_{sport}.json` for detailed performance metrics.

## 🔐 Security Notes

- Never commit API keys to git
- Use environment variables for sensitive data
- Keep `config.py` in `.gitignore`
- Secure your GCP credentials

## 📚 Advanced Usage

### Custom Feature Engineering

Edit `betting_features.py` to add sport-specific features:

```python
def add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    # Your custom features here
    df['CustomFeature'] = ...
    return df
```

### Hyperparameter Tuning

Use Vertex AI for automated tuning:

```python
from google.cloud import aiplatform

job = aiplatform.HyperparameterTuningJob(...)
```

### Different Models

Try different algorithms in `vertex_training.py`:

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(...)
```

## 📞 Support

For issues:
1. Check logs in `pipeline.log`
2. Verify API keys in config
3. Ensure theover.ai CSVs are formatted correctly
4. Check model files exist in `models/`

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Data from sportsdata.io
- Predictions from theover.ai
- ML framework using XGBoost and scikit-learn
- Cloud infrastructure via Google Cloud Platform

---

**Disclaimer**: This tool is for educational purposes. Always gamble responsibly and within your means. Past performance does not guarantee future results.
