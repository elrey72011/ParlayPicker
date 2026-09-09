# Analysis performance

Team-stat enrichment requests only the leagues in the current normalized slate, including the previous-season fallback. Explicit callers that omit the new league filter retain their all-league behavior. Model eligibility, quote freshness and wager checks are unchanged.

Strategy Lab no longer grades on render. It reads the session's Results pipeline output. Use Refresh Strategy Lab scores or Results > Refresh / Backfill Final Scores for updated scores. A new session needs an explicit refresh; no timer automatically queries providers. Uploaded/editable recap data still follows its existing Results workflow.

During Master Analysis, a progress message names the active stage. Workspace > Diagnostics > Last analysis stage timings shows elapsed seconds for input/evidence initialization, odds/stats/model analysis, market enrichment, Gemini/wager checks, props, and saving evidence. Logs also contain PERFORMANCE records for stages and each fetched league/season. Timings do not contain credentials or raw uploaded data. Failures may leave the last stage incomplete; these are not a complete request profiler.

Validation uses mocked provider calls and Streamlit reruns. Production speedup is not yet measured. After deployment, run once with the normal inputs and review stage timings; then navigate to Preview & Publish and verify no Strategy Lab grading request is triggered. Gemini retries and provider latency can still contribute to runtime.
