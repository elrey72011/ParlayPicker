#!/bin/bash
# run_daily_pipeline.sh

echo "🚀 Starting Daily ParlayPicker Pipeline..."

# 1. Backfill most recent results and closing odds (NCAAB focus)
echo "📊 Backfilling last 2 days of box scores..."
python collect_historical_data.py --sports NCAAB NBA NHL --days 2

# 2. Run the main Streamlit application or pick generator
echo "🎰 Generating Best Picks for Today..."
# If running as a one-time script:
python daily_picks.py
# Or if launching the dashboard:
# streamlit run streamlit_app.py

echo "✅ Pipeline Complete."
