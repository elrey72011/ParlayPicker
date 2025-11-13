# 🎯 ParlayDesk - AI-Enhanced Sports Betting Analysis

AI-powered parlay finder with machine learning predictions trained on historical data from The Odds API, Kalshi market validation, and live NFL & NHL context from API-Sports.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🚀 Two Versions Available

### **Standard Version** (`streamlit_app.py`)
- ✅ Current odds analysis
- ✅ Parlay combination builder
- ✅ Basic EV calculations
- ✅ Works with standard API subscription

**Use when:** You want quick analysis without ML training

### **Enhanced Version** (`streamlit_app_enhanced.py`) ⭐ **RECOMMENDED**
- ✅ Everything in standard version +
- ✅ Real ML models trained on historical data
- ✅ Gradient Boosting predictions (55-65% accuracy)
- ✅ Historical pattern recognition
- ✅ Feature importance analysis
- ✅ Validated edge detection

**Use when:** You have historical API access and want ML predictions

## 📋 Requirements

### For Standard Version
```bash
pip install -r requirements.txt
```

### For Enhanced Version
- ✅ Python packages (from requirements.txt)
- ✅ The Odds API key with **historical data access**
  - Get yours at: https://the-odds-api.com

## ⚡ Quick Start

### Option 1: Standard Version (Fast)
```bash
# Install dependencies
pip install -r requirements.txt

# Run
streamlit run streamlit_app.py

# Enter your API key in the sidebar
```

### Option 2: Enhanced Version (ML Powered)
```bash
# Install dependencies
chmod +x scripts/setup.sh
./scripts/setup.sh

# Run
streamlit run streamlit_app_enhanced.py

# In the app:
# 1. Enter API key
# 2. Train model (2-5 min)
# 3. Get predictions!
```

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** - Get started in 5 minutes
- **[Enhanced README](docs/README_ENHANCED.md)** - Full ML documentation
- **[Feature Comparison](docs/COMPARISON.md)** - Standard vs Enhanced
- **[File Index](docs/FILE_INDEX.md)** - What each file does

## 🎯 Features

### Standard Version
- 🎲 Multi-sport odds aggregation (NFL, NBA, MLB, NHL, etc.)
- 📊 Parlay combination builder (2-leg, 3-leg, 4-leg)
- 💰 Expected Value (EV) calculations
- 🛰️ API-Sports NFL & NHL live data integration
- 🌐 Embedded API-Sports league widget for cross-sport research
- 📈 Real-time odds from The Odds API

### Enhanced Version (Additional)
- 🧠 **ML Model Training** on 7-180 days of historical data
- 📊 **Gradient Boosting Classifier** with feature engineering
- 🎯 **55-65% Prediction Accuracy** (validated on test data)
- 💡 **Real Edge Detection** (ML probability vs market odds)
- 📈 **Historical Analysis** tab with insights
- 🔄 **Local Caching** for faster retraining
- 📉 **Feature Importance** rankings

## 🔬 How the ML Works

```
Historical Odds → Feature Engineering → Model Training → Validation
        ↓                    ↓                  ↓            ↓
   (Past games)        (11+ features)    (Gradient Boost)  (58% acc)
        ↓                    ↓                  ↓            ↓
Current Odds → Extract Features → Predict → Compare to Market → Edge!
```

**Example Pattern Learned:**
```
"Home favorites at -300 with -7.5 spread in NFL: 
 Market says 75%, ML model says 78% based on 147 similar games
 → 3% edge detected!"
```

## 📊 Sample Output

```
🟢 💰 #1 | AI Score: 45.2 | AI EV: +8.5%

AI Metrics:
├─ Confidence: 72% (high)
├─ AI EV: +8.5% (excellent value)
├─ Model Accuracy: 58.3%
└─ Edge: +7% over market

Parlay Legs:
├─ Lakers ML: Market 58% → AI 65% (7% edge!)
├─ Bills -3.5: Market 52% → AI 59% (7% edge!)
└─ Over 225: Market 50% → AI 53% (3% edge!)

Payout: +280 ($100 → $380)
Expected Value: +$23.80 per $100 wagered
```

## 🎓 Understanding Results

### Confidence Icons
- 🟢 **High (>70%)**: Strong ML signal, model very confident
- 🟡 **Moderate (50-70%)**: Good opportunity, reasonable confidence
- 🟠 **Lower (<50%)**: Higher risk, less certain

### Expected Value
- 💰 **High +EV (>10%)**: Excellent value
- 📈 **Positive +EV (0-10%)**: Good value, profitable long-term
- 📉 **Negative -EV (<0%)**: Poor value, avoid

## ⚙️ Configuration

Create a `.streamlit/secrets.toml` file (optional):
```toml
[odds_api]
api_key = "your-api-key-here"
```

Or enter your API key directly in the sidebar.

To enable NFL live data integration, add your API-Sports token under the `NFL_APISPORTS_API_KEY` secret:

```toml
# .streamlit/secrets.toml
NFL_APISPORTS_API_KEY = "your-nfl-api-sports-token"
```

To enable NHL live data integration, add your hockey token under the `NHL_APISPORTS_API_KEY` secret:

```toml
# .streamlit/secrets.toml
NHL_APISPORTS_API_KEY = "your-nhl-api-sports-token"
```

The app automatically picks up those keys from Streamlit secrets. If the secrets
aren't defined it falls back to the `NFL_APISPORTS_API_KEY`, `NHL_APISPORTS_API_KEY`,
`APISPORTS_API_KEY`, or `API_SPORTS_KEY` environment variables so existing deployments
keep working without additional configuration.

## 🛠️ Development

```bash
# Clone the repo
git clone https://github.com/yourusername/parlaydesk.git
cd parlaydesk

# Install dependencies
pip install -r requirements.txt

# Run standard version
streamlit run streamlit_app.py

# Run enhanced version
streamlit run streamlit_app_enhanced.py
```

## ⚠️ Important Notes

### API Costs
- Standard version: ~1 API call per sport per refresh
- Enhanced version: 
  - Training: 2 × days_back (e.g., 180 calls for 90 days)
  - After training: Same as standard
- Historical data costs extra - check The Odds API pricing

### Model Performance
- **58% accuracy is good!** (vs 50% random guessing)
- Even 60% accuracy means 40% losses
- Edge detection helps find value, not guarantees
- Always use proper bankroll management

### Responsible Gambling
- ⚠️ Never bet more than you can afford to lose
- ⚠️ ML predictions are estimates, not certainties
- ⚠️ Past performance doesn't guarantee future results
- ⚠️ Use for education and entertainment

## 📈 Roadmap

- [ ] Live odds tracking with WebSocket
- [ ] Player injury data integration
- [ ] Weather data for outdoor sports
- [ ] Advanced bankroll management tools
- [ ] Portfolio tracking and analytics
- [ ] Automated bet slip generation
- [ ] Discord/Telegram bot integration
- [ ] Deep learning models (LSTM for sequences)

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- **The Odds API** - For comprehensive sports betting data
- **Streamlit** - For the amazing web app framework
- **scikit-learn** - For ML capabilities

## 📞 Support

- **Issues**: Open a GitHub issue
- **Docs**: Check the `/docs` folder
- **API Help**: https://the-odds-api.com/liveapi/guides/

## ⭐ Star This Repo!

If you find this useful, please star the repo! It helps others discover the project.

---

**Disclaimer**: This tool is for educational and entertainment purposes only. Sports betting involves risk. Never bet more than you can afford to lose. Not financial advice.
