# Re-lock change review and confirmation

Base: d0b330a5f71cb63f49a37229a03b7850ca820530 (current main when work began).
Branch: codex/relock-change-confirmation.

Removed locks are loaded read-only into the owner session. The comparison helper chooses the latest removal for the immutable lock identity and verifies its archived-lock hash. Every selected re-lock displays the prior and current selection, market, line, price, sportsbook and analysis/quote timestamps; absent metadata stays Not recorded.

NORMAL price/line refresh: comparison and final review, no checkbox. WARNING selection change: acknowledgment. HIGH market-family switch: market acknowledgment. CRITICAL team reversal overrides other severity: acknowledgment plus exact RELOCK. The final dialog lists comparisons, offers Cancel first, and requires Confirm and save new lock. Mixed batches require every re-lock acknowledgment. Fresh first-time locks retain their existing button workflow.

Acknowledgments are keyed to the removed record and exact current candidate. The dialog is bound to the selected set and entire reviewed package. Storage re-reads removals and checks the supplied review tokens before writing; changed removal history or candidate content requires new review. Existing eligibility checks still run at save acceptance time. Existing active locks retain first-write semantics.

Comparison code never ranks, selects, reprices or changes probabilities. Favorite changes are claimed only from unique paired moneyline quotes with matching provider namespace/event, book and timestamp; they are never inferred from the selected team's name or run-line sign. Historical packages normally lack that evidence, so no favorite claim appears. The details expander lists saved values without causal claims.

Success messages retain previous/new selections. The locked table displays the strongest prior-lock change classification. Corrections invalidate the removal cache. No historical record is modified; archived locks and tombstones remain intact.

Validation:
- Full suite during implementation: 2,460 passed.
- Final focused comparison, UI, lock storage and eligibility regressions: 42 passed (includes four later-added favorite/acknowledgment tests).
- Production-safety suite: 545 passed.
- Production compilation and diff checks: passed.

Tests cover normal/within-family/both family directions/team reversal, multi-change precedence, unavailable/conflicting favorite evidence, side-by-side UI, acknowledgment invalidation, dialog Cancel/save, immutable removed records, exact current quote persistence and stale/started/invalid eligibility regression. Existing isolated UI fixtures now declare empty restored removal history; their assertions remain unchanged.

No selection, ranking, blend, quote freshness, lock identity, wagering, Kelly, maturity or model/calibration policy changes. No sportsbook execution. No deployment, commit or push performed. Unrelated audit documents, article files and download script left untouched.
