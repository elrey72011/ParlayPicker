# Quick Start Guide - Parlay App

## 🚀 Get Up and Running in 5 Minutes

### Step 1: Setup (2 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API keys in config.py
nano config.py
# Or use environment variables
export SPORTSDATA_NFL_KEY="your-api-key-here"
```

### Step 2: Train Models - FIRST TIME ONLY (30-60 minutes)

```bash
# Train for specific sports
python main.py --train --sports NFL NBA

# This will:
# - Download 3 years of historical data
# - Engineer 50+ features
# - Train XGBoost models
# - Save models to ./models/
```

### Step 3: Daily Usage (1 minute)

```bash
# 1. Place your theover.ai CSV files in data/theover/
#    Format: theover_nfl_2024-11-23.csv

# 2. Generate today's picks
python main.py

# 3. Check output/betting_card_YYYY-MM-DD.csv
```

## 📝 theover.ai CSV Format

Your CSV should look like this:

```csv
GameID,DateTime,HomeTeam,AwayTeam,HomeMoneyLine,AwayMoneyLine,PointSpread,HomeSpreadOdds,AwaySpreadOdds,OverUnder,OverOdds,UnderOdds
12345,2024-11-23T13:00:00,Patriots,Jets,-150,+130,-3.5,-110,-110,42.5,-110,-110
67890,2024-11-23T16:00:00,Cowboys,Giants,-200,+170,-7,-110,-110,48.5,-110,-110
```

## ⚙️ Key Configuration (config.py)

```python
# Minimum edge to consider a bet (3% = aggressive, 5% = conservative)
BETTING_CONFIG = {
    'min_edge': 0.03,  # Start with 3%
    'parlay_sizes': [2, 3, 4],  # Parlay leg counts to generate
}
```

## 📊 Output Explained

**Single Bets:**
- **Edge**: Your advantage over the bookmaker
- **ExpectedValue**: Expected return (%)
- **Confidence**: High (>10% EV) or Medium (3-10% EV)

**Parlays:**
- **Win Probability**: Combined chance of all legs hitting
- **Expected Value**: Expected return on parlay
- **Kelly**: Recommended bet size (% of bankroll)

## 🎯 Best Practices

1. **Start Small**: Use 1/4 Kelly sizing (already configured)
2. **Track Results**: Keep a spreadsheet of all bets
3. **Retrain Weekly**: Run `--train` once per week during season
4. **Min 3% Edge**: Don't bet unless you have 3%+ edge
5. **Avoid Correlation**: Don't parlay same-game bets

## 🔧 Troubleshooting

**"No models found"**
→ Run `python main.py --train --sports NFL` first

**"No theover.ai data"**
→ Place CSV files in `data/theover/` directory

**API rate limit errors**
→ Increase sleep time in sports_data_pipeline.py

**Low memory**
→ Process one sport at a time

## 📈 Example Workflow

### Monday - Train models
```bash
python main.py --train --sports NFL
```

### Sunday Morning - Generate picks
```bash
# 1. Download theover.ai picks to data/theover/
# 2. Run pipeline
python main.py --sports NFL

# 3. Review output/betting_card_2024-11-23.csv
# 4. Place your bets!
```

## 🎲 Sample Output

```
=== TOP SINGLE BETS ===
Patriots ML (-150) - 65% prob, 8% EV, HIGH CONFIDENCE
Lakers -5.5 (-110) - 62% prob, 9% EV, HIGH CONFIDENCE

=== TOP PARLAYS ===
Parlay 1 (3-leg) - +450 odds, 15% EV
  1. Patriots ML (-150)
  2. Bucks -3 (-110)  
  3. Over 215.5 (-110)
Kelly Bet: 2.5% of bankroll
```

## ⚠️ Important Notes

- **Training takes time**: First run will take 30-60 minutes per sport
- **API costs**: sportsdata.io charges per request
- **GCP costs**: Vertex AI has usage fees (can train locally instead)
- **Not guaranteed**: Models provide edge, not certainty
- **Responsible gambling**: Never bet more than you can afford to lose

## 📚 Next Steps

1. ✅ Train your models
2. ✅ Generate daily picks
3. ✅ Track your results
4. ✅ Refine your strategy
5. ✅ Retrain models weekly

---

Need help? Check the full README.md or pipeline.log for errors.

**Good luck! 🍀**
