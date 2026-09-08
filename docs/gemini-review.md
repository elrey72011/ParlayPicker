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
```

This limits structured review HTTP requests shared by game and prop reviews per
UTC day. Each batch contains up to 12 selections; retries and failed requests
count. Zero disables new requests. An exact model/prompt batch is reusable for
10 minutes; price, date, context, or prompt changes invalidate its key. Cached
reviews preserve their original timestamp. No credential probe request is made.

Accounting and cache live beside the prediction database in
`gemini_review_usage.sqlite3`. Accounting is atomic across local sessions and
processes, but is NOT an account-wide billing cap or a durable cross-deployment
quota. It resets if local storage is replaced and is not included in Drive
snapshot backup. Other legacy Gemini entry points are outside this limit.
Missing reviews still fail closed when the wager review requirement is enabled.

## Context and review evidence

The existing market/model probabilities remain available. Additional context is
accepted only for probable_pitchers, lineups, injuries, and weather, with matching
`<category>_source` and `<category>_recorded_at` fields no more than one hour old
and not in the future. The exact candidate may supply these fields before the
canonical export strips enrichment. Missing/unverified context is explicitly
listed in the prompt. This does not add a new injury/lineup/weather feed; current
providers without provenance remain unavailable to the reviewer.

The structured response can cite supplied context keys and list missing facts.
Unknown evidence references invalidate the review for approval. Free-text
explanations remain model output, not independently established facts. Review
agreement, model, original timestamp, input hash, context, explanation and evidence
fields are retained in exports and copied to immutable prediction snapshots.

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
