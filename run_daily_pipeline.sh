#!/bin/bash
# run_daily_pipeline.sh

echo "🚀 Starting Daily ParlayPicker Pipeline..."

# 1. Backfill most recent results and closing odds (NCAAB focus)
echo "📊 Backfilling last 2 days of box scores..."
python collect_historical_data.py --sports NCAAB NBA NHL --days 2

# 1b. Auto-grade YESTERDAY's slate from final scores (ESPN) BEFORE today's picks are
#     generated — at this point the most-recent export in app_exports/ is the prior
#     run's card, whose games are now complete. This is the step that actually FEEDS
#     the earned-Actionable loop: it writes data/backtest_exports/<date>.csv that the
#     bucket/calibration refits below consume, so buckets can EARN Actionable from
#     realized results. No manual recap needed; guarded so missing finals never break
#     pick generation. (Idempotent: re-grading the same slate just overwrites its file.)
PRIOR_EXPORT=$(ls -t app_exports/best_picks_export*.csv 2>/dev/null | head -1)
if [ -n "$PRIOR_EXPORT" ]; then
  echo "🧮 Auto-grading prior slate from final scores: $PRIOR_EXPORT ..."
  python scripts/grade_from_scores.py "$PRIOR_EXPORT" || \
    echo "   (auto-grade skipped — finals unavailable yet; non-fatal)"
fi

# 2. Run the main Streamlit application or pick generator
echo "🎰 Generating Best Picks for Today..."
# If running as a one-time script:
python daily_picks.py
# Or if launching the dashboard:
# streamlit run streamlit_app.py

# --- Earned-Actionable validation loop ----------------------------------------
# "Actionable" is gated on PROVEN buckets (core/empirical_tiers.py: realized win
# rate >= 55% over >= 25 graded picks, and Kalshi Agrees). For that gate to stay
# honest and current it must be fed (a) closing-line value and (b) freshly graded
# results. These steps wire that loop; each is guarded so a missing input never
# breaks pick generation.

# 2b. Capture closing lines for today's card (run as close to first game as the
#     schedule allows — CLV is the only pre-grade signal of genuine edge).
LATEST_EXPORT=$(ls -t app_exports/best_picks_export*.csv 2>/dev/null | head -1)
if [ -n "$LATEST_EXPORT" ]; then
  echo "🕒 Capturing closing lines for $LATEST_EXPORT ..."
  python scripts/capture_closing_lines.py --export "$LATEST_EXPORT" || \
    echo "   (closing-line capture skipped — odds unavailable; non-fatal)"
fi

# 3. Refresh the empirical calibration the earned-Actionable gate reads, from the
#    accumulated graded slates. Re-fits realized bucket win rates so a totals/sides
#    bucket can EARN Actionable once it proves >55% over >=25 graded picks (and a
#    bucket that bleeds is demoted). No-ops cleanly if no graded history yet.
if ls data/backtest_exports/*.csv >/dev/null 2>&1; then
  echo "📈 Refreshing empirical bucket stats from graded slates..."
  python scripts/fit_bucket_stats.py data/backtest_exports data/calibration/bucket_stats.json || \
    echo "   (bucket-stats refit skipped; non-fatal)"
  # Refit the isotonic calibration table from the SAME graded slates. The empirical
  # overlay reads both this AND the bucket stats; if only the buckets are refit they
  # drift apart (a bucket can read >=55% while a stale calibration still maps those
  # picks down — exactly the contradiction seen on the 17 Jun over slate), so refit
  # them together to keep the earned-Actionable gate coherent.
  echo "📈 Refreshing probability calibration from graded slates..."
  python scripts/fit_calibration.py data/backtest_exports data/calibration/effective_prob_calibration.json || \
    echo "   (calibration refit skipped; non-fatal)"
else
  echo "📈 No graded slates in data/backtest_exports yet — skipping calibration refresh."
  echo "   This fills automatically as step 1b grades each day's finals. To backfill a"
  echo "   past slate now:  python scripts/grade_from_scores.py <export.csv> [YYYY-MM-DD]"
fi
# ------------------------------------------------------------------------------

echo "✅ Pipeline Complete."
