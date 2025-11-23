# Project Overview - Parlay Betting ML Pipeline

## 📦 What You've Got

A complete, production-ready machine learning pipeline for sports betting that:
- Collects historical data from sportsdata.io
- Engineers 50+ betting-specific features
- Trains custom ML models on Vertex AI
- Integrates with theover.ai picks
- Generates optimal parlay recommendations

## 🎯 Your Workflow

```
CSV Upload (theover.ai) 
    ↓
ML Models (Vertex AI)
    ↓
Win Probabilities
    ↓
Expected Value Analysis
    ↓
Optimal Parlays
```

## 📂 File Descriptions

### Core Pipeline Files

**main.py** (400 lines)
- Master orchestration script
- Runs complete pipeline end-to-end
- Command-line interface
- Usage: `python main.py --train --sports NFL`

**sports_data_pipeline.py** (350 lines)
- Data collection from sportsdata.io API
- Historical games, team stats, odds
- Handles all 5 sports (NFL, NBA, NCAAB, NCAAF, NHL)
- Rate limiting and error handling

**betting_features.py** (500 lines)
- Advanced feature engineering
- ELO ratings, rolling stats, rest days
- Head-to-head records, pace metrics
- Strength of schedule, betting indicators
- Target variable creation

**vertex_training.py** (600 lines)
- ML model training on Vertex AI
- XGBoost classifiers and regressors
- Time-series cross-validation
- Model evaluation and storage
- Local and cloud training support

**parlay_optimizer.py** (700 lines)
- Expected value calculation
- Single bet analysis
- Parlay generation and optimization
- Correlation detection
- Kelly Criterion bet sizing

**config.py** (150 lines)
- Centralized configuration
- API keys and GCP settings
- Training parameters
- Betting strategy rules
- Sport-specific configs

### Supporting Files

**requirements.txt**
- All Python dependencies
- Includes XGBoost, scikit-learn, pandas
- Google Cloud libraries
- Easy installation with pip

**README.md** (500 lines)
- Comprehensive documentation
- Installation instructions
- Usage examples
- Configuration guide
- Troubleshooting tips

**QUICKSTART.md**
- 5-minute getting started guide
- Step-by-step setup
- Daily usage workflow
- Common issues and fixes

**examples.py**
- 7 example scripts
- Shows how to use each component
- Good for learning and testing
- Commented code examples

**.gitignore**
- Protects sensitive data
- Excludes API keys and credentials
- Ignores large data files
- Standard Python ignores

## 🔄 Pipeline Stages

### Stage 1: Training (Run Weekly)

**Input:** API keys, historical seasons
**Process:**
1. Fetch 3+ years of game data
2. Calculate 50+ features per game
3. Train 5 models per sport (ML, Spread, Totals, Margin, Total Points)
4. Validate with time-series CV
5. Save models locally and/or to GCS

**Output:** Trained models in `./models/`

**Time:** 30-60 minutes per sport

### Stage 2: Daily Predictions

**Input:** theover.ai CSV with today's games
**Process:**
1. Load today's games and odds
2. Engineer features for each game
3. Run through trained models
4. Calculate win probabilities
5. Compute expected value
6. Generate single bets
7. Construct optimal parlays
8. Apply Kelly Criterion sizing

**Output:** Betting card in `./output/`

**Time:** 1-2 minutes

## 📊 Key Features

### Models Trained (Per Sport)
1. **Moneyline Classifier** - Predict outright winner
2. **Spread Classifier** - Predict spread covering
3. **Totals Classifier** - Predict over/under
4. **Margin Regressor** - Predict point differential
5. **Total Points Regressor** - Predict combined score

### Feature Categories (50+ total)
- **Team Strength**: ELO ratings, win percentages
- **Recent Form**: Rolling wins, points, defense
- **Schedule**: Rest days, back-to-backs, travel
- **Matchups**: Head-to-head records, pace combinations
- **Context**: Home/away, day of week, month
- **Betting**: Line movement, public percentages
- **Advanced**: Strength of schedule, opponent adjustments

### Bet Analysis
- **Edge Calculation**: Model prob vs implied prob
- **Expected Value**: ROI per bet
- **Confidence Levels**: High (>10% EV) or Medium (3-10% EV)
- **Correlation Detection**: Avoid dependent bets
- **Kelly Criterion**: Optimal bet sizing

## 🎲 Example Output

### Single Bet
```
Patriots Moneyline (-150)
├─ Model Probability: 65%
├─ Implied Probability: 60%
├─ Edge: 5%
├─ Expected Value: 8.3%
├─ Confidence: MEDIUM
└─ Kelly Bet: 2.1% of bankroll
```

### 3-Leg Parlay
```
Parlay #1 (+450 odds)
├─ Patriots ML (-150) | 65% | 8% EV
├─ Lakers -5.5 (-110) | 62% | 9% EV  
├─ Over 215.5 (-110) | 58% | 6% EV
├─ Combined Win Prob: 23.4%
├─ Expected Value: 15.2%
└─ Kelly Bet: 3.8% of bankroll
```

## ⚙️ Configuration Options

### Betting Strategy
- Min edge: 3% (configurable)
- High confidence: 10%+ EV
- Parlay sizes: 2, 3, 4, 5 legs
- Max parlays: 10 per size
- Kelly fraction: 1/4 Kelly (conservative)
- Max bet: 10% of bankroll

### Model Training
- Algorithm: XGBoost
- Max depth: 6
- Learning rate: 0.05
- Estimators: 200
- CV folds: 5 (time-series)
- Test size: 20%

### Feature Engineering
- Rolling windows: 3, 5, 10 games
- ELO K-factor: 20
- H2H window: 10 games
- Min games: 1 for rolling stats

## 🚀 Deployment Options

### Option 1: Local (Recommended for Starting)
- Train models on your machine
- Run predictions locally
- No cloud costs
- Full control

### Option 2: Vertex AI
- Train on Google Cloud
- Scalable compute
- Automated retraining
- Production deployment
- Costs $$$

### Option 3: Hybrid
- Train locally
- Store models in GCS
- Use Cloud Functions for predictions
- Balance cost and scale

## 📈 Performance Expectations

### Model Accuracy (Typical)
- Moneyline: 55-58%
- Spread: 52-55%
- Totals: 52-54%

### Betting Performance
- Hit rate: 52-55% (against -110 odds)
- ROI: 3-8% (with proper edge threshold)
- Sharpe ratio: 0.5-1.5
- Max drawdown: 20-30%

**Note:** Past performance doesn't guarantee future results

## 🔐 Security Best Practices

1. **Never commit API keys** - Use .gitignore
2. **Environment variables** - Store keys in .env
3. **GCP credentials** - Keep service account keys secure
4. **Data privacy** - Don't share processed data
5. **Access control** - Limit who can run pipeline

## 🛠️ Maintenance Schedule

### Daily
- Upload theover.ai CSVs
- Run prediction pipeline
- Review and place bets
- Log results

### Weekly  
- Retrain models with latest data
- Review model performance
- Adjust edge thresholds if needed
- Update feature engineering

### Monthly
- Analyze betting results
- Calculate actual ROI
- Compare to model predictions
- Tune betting strategy

### Seasonal
- Major model retraining
- Add new features
- Upgrade dependencies
- Optimize hyperparameters

## 🎓 Learning Resources

### Understanding the Code
1. Start with `examples.py` - Simple demos
2. Read `QUICKSTART.md` - Basic workflow
3. Study `README.md` - Full documentation
4. Review `main.py` - See how it all connects

### ML Concepts
- XGBoost: Gradient boosting for classification/regression
- Time-series CV: Prevents data leakage
- Feature engineering: Transform raw data into predictive signals
- Expected value: Long-term profit/loss calculation
- Kelly Criterion: Optimal bet sizing formula

### Betting Concepts
- Edge: Your advantage over the bookmaker
- Implied probability: Odds converted to win percentage
- Correlation: When bets aren't independent
- Closing line value: Beating the final odds
- Bankroll management: Proper bet sizing

## ⚠️ Known Limitations

1. **Data quality**: Depends on API completeness
2. **Market efficiency**: Edges decrease over time
3. **Correlation**: Hard to model perfectly
4. **Injuries**: Not always reflected in features
5. **Live changes**: Static pregame models only

## 🔮 Future Enhancements

### Short-term
- [ ] Add player injury tracking
- [ ] Include weather data
- [ ] Track line movement
- [ ] Add more sports

### Medium-term  
- [ ] Live betting models
- [ ] Neural network models
- [ ] Automated bet placement
- [ ] Portfolio optimization

### Long-term
- [ ] Deep learning features
- [ ] Alternative data sources
- [ ] Multi-model ensembles
- [ ] Real-time predictions

## 📞 Getting Help

1. **Check logs**: `pipeline.log` has detailed errors
2. **Review examples**: `examples.py` shows usage
3. **Read docs**: `README.md` covers most issues
4. **Verify config**: Check API keys and paths
5. **Test components**: Run examples individually

## 🎯 Success Metrics

Track these to measure performance:

### Model Metrics
- Accuracy, AUC, Log Loss
- Feature importance
- Calibration plots

### Betting Metrics  
- Hit rate (% of winning bets)
- ROI (return on investment)
- Sharpe ratio (risk-adjusted returns)
- Max drawdown (worst losing streak)
- Closing line value (CLV)

### Operational Metrics
- Data freshness
- Model training time
- Prediction latency
- Uptime/reliability

## 📝 Final Checklist

Before going live:

- [ ] API keys configured
- [ ] Models trained and validated
- [ ] theover.ai integration working
- [ ] Betting strategy defined
- [ ] Bankroll allocated
- [ ] Tracking system ready
- [ ] Risk limits set
- [ ] Responsible gambling plan

---

**You're all set! Time to start building your edge.** 🚀

Remember: This is a tool to help find +EV opportunities. Always gamble responsibly and never bet more than you can afford to lose.
