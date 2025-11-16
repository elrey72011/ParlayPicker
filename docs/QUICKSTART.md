# Quick Start Guide - 5 Minutes to AI Predictions

> **Update:** The enhanced build has been merged into `streamlit_app.py`. Use the main app for both quick analysis and historical ML workflows; legacy references to `streamlit_app_enhanced.py` are for archival purposes only.

## ⚡ Super Fast Setup

### 1. Install Dependencies (30 seconds)
```bash
chmod +x setup.sh
./setup.sh
```

OR manually:
```bash
pip install streamlit pandas numpy requests pytz scikit-learn
```

### 2. Get Your API Key (2 minutes)
1. Go to https://the-odds-api.com/account/
2. Copy your API key
3. Verify you have **historical data access** enabled

### 3. Launch the App (10 seconds)
```bash
streamlit run streamlit_app.py
```

### 4. Train Your First Model (2 minutes)
1. Enter your API key in the sidebar
2. Select a sport (start with NFL or NBA)
3. Choose 30 days of historical data
4. Click "📊 Load & Train Model"
5. Wait 1-2 minutes while it fetches data and trains

### 5. Get Predictions! (Immediate)
1. Select sports you want to bet on
2. Click "🔍 Find AI-Optimized Parlays"
3. See ML-powered predictions with:
   - AI probability estimates
   - Model confidence scores
   - Edge over market odds
   - Historical accuracy

## 🎯 What You'll See

### After Training
```
✅ Model trained! Accuracy: 58.3%

Training Stats:
- Games analyzed: 487
- Date range: 2024-09-01 to 2024-10-01
- Model accuracy: 58.3%
```

### In Your Predictions
```
🟢 💰 #1 | AI Score: 45.2 | Odds: +280 | AI EV: +8.5%

AI Score: 45.2
AI Confidence: 72%
AI EV: +8.5%
Payout: +280

📋 Parlay Legs:
Pick: Lakers ML
Market Prob: 58%
AI Prob: 65%      ← Model thinks higher chance than market!
AI Edge: 7%       ← Edge over market
Model Acc: 58.3%  ← Overall model performance
```

## 🎓 Understanding Your Results

### Confidence Levels
- **🟢 High (>70%)**: Model is very confident, strong historical pattern
- **🟡 Moderate (50-70%)**: Good opportunity, reasonable confidence  
- **🟠 Lower (<50%)**: Higher uncertainty, proceed with caution

### Expected Value (EV)
- **💰 High +EV (>10%)**: Excellent value, model sees major edge
- **📈 Positive +EV (0-10%)**: Good value, theoretically profitable long-term
- **📉 Negative -EV (<0%)**: Poor value, avoid these bets

### AI Score
The overall ranking that combines:
- Expected value (how profitable)
- Model confidence (how certain)
- Edge over market (how much advantage)

Higher score = Better opportunity!

## 🔥 Pro Tips

### 1. Start Small
- Train on 30 days first
- Test predictions without betting
- Track accuracy on paper

### 2. Focus on Quality
- Look for 🟢 + 💰 (high confidence + positive EV)
- Avoid 📉 even if high confidence
- Higher AI score is better

### 3. Retrain Regularly
- Weekly retraining captures recent trends
- More data generally = better model
- But 30-60 days is usually optimal

### 4. Understand Limitations
- 58% accuracy is good (vs 50% random)
- Even 60% means 40% losses
- Use for identifying value, not guarantees
- Past patterns don't guarantee future results

## ⚠️ Common Mistakes to Avoid

### ❌ Don't:
- Bet more because "AI says so"
- Ignore negative EV picks
- Trust low confidence picks
- Forget to retrain regularly
- Bet more than you can afford to lose

### ✅ Do:
- Focus on high confidence + positive EV
- Track your actual results vs predictions
- Retrain with fresh data weekly
- Use proper bankroll management
- Treat as educational tool

## 🆘 Troubleshooting

### "Failed to fetch historical data"
→ Check your API key has historical access enabled
→ Verify subscription at https://the-odds-api.com/account/

### "Model accuracy seems low"
→ 55-60% is actually good! Market is efficient
→ Try 60-90 days of data for better patterns

### "Training is slow"
→ Each day = 2 API calls, be patient
→ Data is cached for faster subsequent loads

### "No predictions available"
→ Make sure model is trained first
→ Check API key has current odds access too

## 📊 Sample Session

```bash
# 1. Install
./setup.sh

# 2. Run
streamlit run streamlit_app_enhanced.py

# 3. In the app:
#    - Enter API key
#    - Select NFL
#    - Choose 30 days
#    - Click train
#    - Wait 2 minutes

# 4. See results:
✅ Model trained! Accuracy: 58.3%

# 5. Get predictions:
#    - Select NFL + NBA
#    - Click find parlays
#    - See AI-ranked opportunities

# 6. Best pick example:
🟢 💰 Bills ML + Lakers ML
AI Score: 48.7
Confidence: 74%
AI EV: +9.2%
Odds: +305
```

## 🎯 Your First Bet (Practice)

1. **Train model** on 30 days
2. **Find parlay** with:
   - 🟢 High confidence
   - 💰 Positive EV
   - AI Score > 40
3. **Paper trade** - track without betting
4. **Monitor** actual results
5. **Compare** AI predictions to outcomes
6. **Learn** from differences

## 🚀 Next Steps

After your first session:
1. ✅ Read full README_ENHANCED.md
2. ✅ Review COMPARISON.md to understand improvements
3. ✅ Track predictions in a spreadsheet
4. ✅ Retrain weekly with fresh data
5. ✅ Gradually increase historical data to 60-90 days

## 💡 Remember

This tool helps you **identify value** using machine learning, but:
- It's not magic
- Sports betting is gambling
- Even good predictions lose sometimes
- Use responsibly
- Never bet more than you can afford to lose

## 🎉 You're Ready!

Run the setup script, get your API key, and start finding value with real ML predictions!

```bash
./setup.sh
streamlit run streamlit_app_enhanced.py
```

Good luck! 🍀

---

Questions? Read the full documentation in README_ENHANCED.md
