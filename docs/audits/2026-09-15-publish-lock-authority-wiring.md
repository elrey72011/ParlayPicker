# Publish/lock authoritative candidate wiring — 2026-09-15

HEAD BEFORE: `7c3c25ad031c443e10fd9c3a62e11438bef11442` (current main; #2284 and #2285 merged).
HEAD AFTER: commit containing this report on `codex/publish-lock-authority`; exact SHA in PR handoff.

## Verified root cause and fix

`streamlit_app._run_pipeline` uses `capture_run(..., authoritative_candidates=True)` to assign the persisted export run and snapshot identities to both the candidate frame and saved final card. It stores returned candidates in `diagnostics["candidate_authority_df"]` and replaces `best_picks_df` with `saved_card`. The earlier reporting audit is not updated to that same run identity.

Both `render_publish_panel` call sites still passed `candidate_audit_df`: the no-analysis/history branch and the normal analysis branch. The publication card is an export projection: its current column list retains `provider_quotes` but omits `spread_line` and `total_line`. Therefore, its fallback cannot bind an exact quote when the supplied reporting candidates are excluded by the strict export-run match. This produces “No ranked candidate evidence; refresh analysis” and “Quote unavailable” despite the captured candidate having a valid exact quote.

OLD PUBLISH CANDIDATE SOURCE: `candidate_audit_df`.
NEW PREFERRED PUBLISH CANDIDATE SOURCE: nonempty DataFrame `candidate_authority_df`.
LEGACY FALLBACK: `candidate_audit_df` when authoritative candidates are absent, empty, or not a DataFrame.

A single `_publication_candidates(diagnostics)` helper supplies both call sites. It returns the original frame object as-is. No fields, identities, ranks, prices or timestamps are copied from another frame, inferred or repaired. `per_game_board` run/game matching remains unchanged.

## Approved narrow quote-check exception

Inspection and a direct reproduction found a separate pre-existing defect: `exact_book_quote` delegated to a matcher that checked market/line/price/book but did not compare explicit candidate and quote provider identities. A candidate with one provider event/namespace could therefore receive a Novig quote carrying a conflicting event/namespace.

The owner explicitly authorized an exception to the handoff's prohibition on quote-binding changes: reject explicit identity conflicts only, with no new requirement for legacy missing IDs.

`exact_book_quote` now filters its temporary matching copy when both candidate and quote contain nonempty `provider_event_id` AND `provider_namespace`. A differing event ID or namespace excludes that quote. If either side lacks a complete scoped identity, the original legacy behavior remains. The original candidate frame and stored quote payload are not mutated. Existing book, market, line, price, timestamp and age checks remain in place; the shared evidence matcher, source policies and lock identity are unchanged.

## Regression evidence

The new tests use the real `begin_run` and authoritative `capture_run` with temporary SQLite storage, preserve the saved identity, and project out family-specific line fields to reproduce the relevant publication export shape. Both MLB spread and total cases go through `per_game_board`, `build_package` and `lock_audit`.

- PRE-CAPTURE MISMATCH REPRODUCED: PASS. Earlier reporting run fails to supply the candidate and returns the documented unavailable reason.
- POST-CAPTURE AUTHORITATIVE MATCH: PASS. Saved card and preferred candidates share snapshot, export run, candidate and matchup identities.
- NOVIG QUOTE RETAINED: PASS. Exact offered line/price and provider update timestamp remain discoverable.
- ELIGIBLE NOW: PASS. Same-day fresh pregame spread/total candidates are lockable without becoming production wagers.
- ALREADY LOCKED UNCHANGED: PASS. Real `History.lock_picks` with an in-memory storage client preserves the first saved selection/price when a changed package tries to relock the same identity.
- STARTED UNCHANGED: PASS. Checking after scheduled start reports Started and yields no lock candidate.
- STALE QUOTE UNCHANGED: PASS. Old quote reports Stale quote and yields no lock candidate.
- STALE ANALYSIS UNCHANGED: PASS. Old analysis reports Stale analysis and yields no lock candidate.
- OTHER DATE UNCHANGED: PASS. Future-date game reports Other date and yields no lock candidate.
- INVALID EXACT QUOTE FAIL-CLOSED: PASS. Wrong line, price, book, provider ID, namespace, invalid timestamp and future quote time are excluded.
- LEGACY FALLBACK: PASS. Missing/empty/non-DataFrame authoritative input falls back without ambiguity from DataFrame truth evaluation.
- LEGACY MISSING PROVIDER IDs: PASS. Missing either identity field on either candidate or quote preserves prior behavior.
- ALL PUBLISH CALL SITES: PASS. AST coverage asserts both paths invoke the frame selector; the existing actual no-analysis Streamlit branch still grades and republishes saved history. That test harness imports the new helper; no assertions were removed.

## Scope and validation

FULL TESTS: 2,440 passed (138.87 seconds); final export-shape fixture refinement rerun: all 39 wiring tests passed.
PRODUCTION SAFETY: 545 passed using the current CI test list.
PRODUCTION COMPILE: PASS (current CI command).
GIT DIFF CHECK: PASS.
Focused publish, board, private projection, evidence and workflow tests: 191 passed.

No live website publication or lock mutation was performed during development. Hosted CI and deployment remain separate from local validation. Existing pandas warnings are not addressed by this change.

FILES CHANGED:

- streamlit_app.py — shared frame selector and both publish call sites.
- app_core/per_game_boards.py — explicitly authorized provider-conflict check only; no event/run matching relaxation.
- tests/test_publish_authority_wiring.py — real capture/board/lock regressions and call-site coverage.
- tests/test_simple_publication_workflow.py — import the production helper into its extracted-branch harness.
- docs/audits/2026-09-15-publish-lock-authority-wiring.md — this report.

UNRELATED CHANGES: NONE. Pre-existing untracked Medium article files remain excluded. Selection, ranking, probabilities, Gemini, Kelly, maturity, wager policy, production eligibility, fallback policy, freshness thresholds, lock identity, public-history storage, evidence immutability, closing capture, MLB challenger and receipt collector are unchanged.
