# Refresh performance follow-up — September 18, 2026

The supplied 17:46 log measured 306.870 seconds for input/evidence initialization, 269.675 seconds for odds/statistics/model work (including receipt collection), 26.897 seconds for selection, 94.168 seconds for Gemini, and 6.983 seconds for saving evidence.

Changes:
- Prediction evidence recovery uses the Drive adapter's bounded parallel, duplicate-checked reads. Record decoding, identity checks, and transactional import remain required.
- Routine game refresh captures and verifies receipt backups but defers historical receipt settlement and its second backup. Catch up MLB receipt history retains settlement. Remote recovery is retained to protect first-receipt identity across restarts.
- Receipt stages and prediction evidence restore tables log elapsed times without payloads or credentials.
- Gemini service timeout/5xx rows are not immediately retried as incomplete reviews. Successful but incomplete reviews retain their targeted retry.

The repository bucket-statistics inputs end August 28. They cannot justify a fresh September 18 artifact. No evidence dates, thresholds, validation flags, probabilities, or wager authority were changed. Rebuilding requires verified recent exports with the required market/consensus fields; MLB receipt outcomes alone are not a compatible bucket dataset.

After deployment, run one refresh and inspect PERFORMANCE timings. Use Catch up MLB receipt history for historical receipt settlement. Public results grading remains separate. Production speedup has not yet been measured.
