# Gemini structured review

Today shows six primary columns: game, pick, odds, estimated win probability,
estimated EV, and wager status. Expanded details retain ranking and approval
explanations. Pick Details shows the saved Gemini agreement and explanation.

## Configuration

Enable **Settings & research → Require Gemini Review for Bets**. Existing
GOOGLE_API_KEY / GEMINI_API_KEY configuration remains in effect. No additional
sports-provider calls are introduced by this change.

Optional Streamlit secret (or environment variable, which takes precedence):

```toml
PARLAYPICKER_GEMINI_DAILY_REQUESTS = 20
PARLAYPICKER_GEMINI_REVIEW_MODEL = "gemini-2.5-flash"
PARLAYPICKER_GEMINI_THINKING_BUDGET = 0
PARLAYPICKER_GEMINI_MAX_OUTPUT_TOKENS = 8192
```

This limits structured review HTTP requests shared by game and prop reviews per
UTC day. Each batch contains up to 12 selections; retries and failed requests
count. Zero disables new requests. An exact model/prompt batch is reusable for
10 minutes; price, date, context, prompt, thinking budget, output limit, or model
changes invalidate its key. Cached reviews preserve their original timestamp.
No credential probe request is made.

The synchronous path accepts only `gemini-2.5-flash` and
`gemini-2.5-flash-lite`. Flash remains the default. Thinking is disabled by
default because this is a constrained structured classification task; set `-1`
for dynamic thinking or a bounded nonnegative token budget only as part of a
measured evaluation. Invalid model/budget/output settings fail back to the safe
defaults.

Only rows already marked `production_eligible` by deterministic checks are sent
to the synchronous Gemini API. Other card rows remain available for research,
are labeled `SKIPPED`, make no online model request, and cannot be promoted by
Gemini. The gate still narrows eligibility only.

Accounting and cache live beside the prediction database in
`gemini_review_usage.sqlite3`. Accounting is atomic across local sessions and
processes, but is NOT an account-wide billing cap or a durable cross-deployment
quota. It resets if local storage is replaced and is not included in Drive
snapshot backup. Other legacy Gemini entry points are outside this limit.
Missing reviews still fail closed when the wager review requirement is enabled.

The same database records one redacted metric row per provider attempt or cache
hit: model, batch size, latency, cache-hit flag, prompt/output/thought/cached/
total token counts, finish reason, and error type. Prompt text, API keys, and
response content are not stored in the telemetry table.

## Context and review evidence

The existing market/model probabilities remain available. Additional context is
accepted only for probable_pitchers, lineups, injuries, and weather, with matching
`<category>_source` and `<category>_recorded_at` fields no more than one hour old
and not in the future. The exact candidate may supply these fields before the
canonical export strips enrichment. Missing/unverified context is explicitly
listed in the prompt. This does not add a new injury/lineup/weather feed; current
providers without provenance remain unavailable to the reviewer.

The structured response must cite supplied context keys and exactly list missing
facts. Unknown or duplicate evidence references, unknown missing categories, or
an incomplete missing-category list invalidate the review for approval.
Free-text explanations remain model output, not independently established
facts. Review agreement, model, original timestamp, input hash, context,
explanation and evidence fields follow the exact ticket through terminal
selection and are copied to immutable prediction snapshots.

## Offline research batches

Broad declined-row or reconciliation review uses the separate asynchronous Batch
API path, never the live wager gate. It defaults to `gemini-2.5-flash-lite`,
thinking disabled, and the same strict structured schema. Prepare a JSON array of
bounded research payloads, then submit and inspect a job:

```powershell
python scripts/gemini_research_batch.py submit --input research-rows.json --display-name sept-review
python scripts/gemini_research_batch.py status --name batches/REPLACE_WITH_JOB_NAME
```

Override a research run with `--model gemini-2.5-flash` or set
`PARLAYPICKER_GEMINI_RESEARCH_MODEL`. Inline jobs are deliberately capped at 20
requests of 12 rows each (240 rows); split larger studies or move them to a
file-backed batch. Batch results are research artifacts and have no code path to
live wager authorization.

## Evaluation

Results → Gemini prospective review comparison uses the graded selected-candidate
ledger. It requires review timestamps before game start, a review input hash,
and an agree/disagree verdict. The earliest reviewed selection per game/model is
counted once. All reviewed model selections are compared with the overlapping
Gemini-agreement subset; model versions are separate. Win rates exclude pushes
and unresolved games. This is descriptive forward evidence, not causal proof of
improvement, actual betting returns, or grading Gemini's opposing selection.
Historical exports without provenance are excluded. Collect and grade new
pregame runs before drawing conclusions. Gemini does not alter numerical
probabilities or promote a failed wager.

For a model comparison, hold the input JSON and configuration fixed, submit one
research batch with Flash and one with Flash-Lite, then compare schema-completion
rate, agreement/hold distribution, token usage, latency, and eventually graded
pregame outcomes. Do not change the live default from cost or anecdotal examples
alone; require prospective coverage and outcome evidence.
