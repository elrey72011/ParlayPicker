# 🎯 ParlayDesk - AI-Enhanced Sports Betting Analysis

AI-powered parlay finder with machine learning predictions trained on historical data from The Odds API, Kalshi market validation, and live NFL & NHL context from API-Sports.

> **What's new:** the primary Streamlit app now bundles the historical-machine-learning workflow that previously lived in the "enhanced" build. Provide your The Odds API and API-Sports keys and the app will auto-build logistic models from recent API-Sports schedules, blend them with Kalshi and sentiment signals, and surface the combined analysis throughout the UI.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

## 🚀 One Unified Streamlit App

`streamlit_app.py` now includes the full feature set—live odds exploration, Kalshi validation, API-Sports insights, and on-demand historical machine learning. Enable or disable components from the sidebar without switching builds.

**Highlights**

- ✅ Real-time odds aggregation across supported books
- ✅ Historical ML models trained automatically when Odds API + API-Sports keys are supplied
- ✅ Sentiment, weather, social, and sharp money overlays
- ✅ Kalshi prediction-market blending with fallback handling
- ✅ Custom parlay builder, EV calculators, and export tools
## 📋 Requirements

### Base Installation
```bash
pip install -r requirements.txt
```

> 💡 **Tip:** scikit-learn remains optional—the app now ships with a lightweight
> NumPy-powered logistic regression fallback, so historical ML predictions still
> train even in minimal environments (including Streamlit Cloud) without the
> extra dependency.

### Optional Data Sources
- ✅ The Odds API key with **historical data access** for ML training
  - Get yours at: https://the-odds-api.com
- ✅ API-Sports tokens for NFL/NHL live data overlays
- ✅ NewsAPI, weather, social, or Kalshi credentials for deeper context

## ⚡ Quick Start

### Run the App
```bash
# Install dependencies
pip install -r requirements.txt

# Launch Streamlit
streamlit run streamlit_app.py

# Configure API keys from the sidebar or .streamlit/secrets.toml
```

## 📚 Documentation

- **[Quick Start Guide](docs/QUICKSTART.md)** – Step-by-step setup for the unified app
- **[Enhanced README](docs/README_ENHANCED.md)** – Archived deep dive into the historical ML pipeline
- **[Feature Comparison](docs/COMPARISON.md)** – Legacy breakdown of pre-merge builds (kept for reference)
- **[File Index](docs/FILE_INDEX.md)** – What each file does

## 🎯 Features

### Core Features
- 🎲 Multi-sport odds aggregation (NFL, NBA, MLB, NHL, etc.)
- 🤖 Automatic logistic-regression predictions trained on recent API-Sports schedules (no manual training step)
- 🗂️ Multi-season backfill automatically taps prior campaigns (e.g., 2024 data) whenever the latest window is sparse
- 📊 Parlay combination builder (2-leg, 3-leg, 4-leg)
- 💰 Expected Value (EV) calculations
- 🛰️ API-Sports NFL & NHL live data integration
- 🌐 Embedded API-Sports league widget for cross-sport research
- 📈 Real-time odds from The Odds API blended with Kalshi validation

### Advanced Extras
- 🔁 *Legacy experiments:* gradient-boosting prototypes remain for comparison, but the main app now auto-trains logistic models.
- 🧪 Optional notebooks for trying alternative models or wider historical windows
- 🧮 Advanced feature-engineering templates to extend the ML pipeline further

## 🔬 How the ML Works

```
API-Sports Schedules + The Odds API → Feature Engineering → Logistic Pipeline → Blended Probabilities
              ↓                               ↓                        ↓                      ↓
     (Records, form, trends)        (11 numerical features)   Impute → Scale → Train    65% ML • 25% market • 10% sentiment
              ↓                               ↓                        ↓                      ↓
 Current Odds → Build Feature Vector → Predict → Compare to Market → Edge!
```

When the current season hasn't produced enough completed games (such as early in the offseason), the builder automatically
backfills with earlier campaigns—including the full 2024 schedules for NFL and NHL—so the logistic model still trains on a
balanced dataset before influencing the parlay analysis. If the live feeds remain sparse even after those backfills, the
trainer tops up the dataset with a small synthetic sample so the logistic model stays calibrated; the Streamlit status panel
calls out how many "booster" rows were injected alongside the real games.

If scikit-learn isn't installed the builder seamlessly drops to an internal
logistic regression trainer that mirrors the same feature engineering pipeline
using NumPy. You'll still see the model source and training-row counts in the UI
so it's clear when the simplified engine is in play.

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
├─ Model Source: Historical ML (276 training rows)
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

### Temporarily disabling ML

Open the **AI Settings** expander in the sidebar and click **“🔌 Disable ML for this session”** to turn off the historical
machine-learning models. The app will immediately fall back to odds, sentiment, Kalshi, and live data signals without
building training datasets. Click **“⚡ Re-enable ML predictions”** at any time to bring the models back.

To enable NHL live data integration, add your hockey token under the `NHL_APISPORTS_API_KEY` secret:

```toml
# .streamlit/secrets.toml
NHL_APISPORTS_API_KEY = "your-nhl-api-sports-token"
```

To stream NBA context, supply the basketball token under `NBA_APISPORTS_API_KEY`:

```toml
# .streamlit/secrets.toml
NBA_APISPORTS_API_KEY = "your-nba-api-sports-token"
```

The app automatically picks up those keys from Streamlit secrets. If the secrets
aren't defined it falls back to the `NFL_APISPORTS_API_KEY`, `NBA_APISPORTS_API_KEY`,
`NHL_APISPORTS_API_KEY`, `APISPORTS_API_KEY`, or `API_SPORTS_KEY` environment variables so existing deployments
keep working without additional configuration.

## 🛠️ Development

```bash
# Clone the repo
git clone https://github.com/yourusername/parlaydesk.git
cd parlaydesk

# Install dependencies
pip install -r requirements.txt

# Run the unified app
streamlit run streamlit_app.py
```

## ⚠️ Important Notes

### API Costs
- Odds API calls scale with the number of sports you request (≈1 per sport per refresh)
- Historical training triggers additional Odds API + API-Sports calls during the first build or when caches expire
- Historical data costs extra—check The Odds API pricing before enabling ML

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
