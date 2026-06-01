# Backtest exports (graded Strategy Lab dumps)

Each `YYYY-MM-DD.txt` is the text rendering of a daily **Best Picks Strategy
Lab** export (graded with W/L), captured from Google Drive. They are the input
fixtures for `scripts/backtest_low_line_over.py`, which measures how the MLB
sub-8.0 `total_over` guardrail (`low_line_over_guardrail`) would have performed.

Format: header-driven (column order varies by day; some days use `%`, some use
decimals). Each file embeds a `Totals` W-L row that the loader cross-checks
against the parsed rows, printing a `[WARN]` on mismatch.

Notes:
- `2026-05-27` is intentionally omitted (slate was ungraded, 0-0).
- `2026-05-24` trips the validation warning because that day's sheet had a
  broken `#REF!` total formula (embedded 8-6); the row-level parse (10-6) is
  the correct count.

To extend the backtest, drop more graded exports here as `.txt` (Drive
rendering) or `.csv`/`.xlsx` (normal export with the same column names) and
re-run `python3 scripts/backtest_low_line_over.py data/backtest_exports`.
