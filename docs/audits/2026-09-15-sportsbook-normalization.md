# Sportsbook normalization verification — 2026-09-15

HEAD BEFORE: `7147b7429ffafab140867ab867362f0e3bf74017` (main, merged #2279).
HEAD AFTER: the commit containing this report on `codex/canonical-sportsbook-labels`; exact hash supplied in the PR handoff.

## Reproduction and root cause

The real `build_best_picks_df` private authority frame, with no pre-populated `book` or `quote_bookmaker`, bound a raw `draftkings` quote as `quote_bookmaker=draftkings`. The terminal adapter retained `book=draftkings`; `supported_quote` returned false because it compared against `DraftKings`, clearing exact-price eligibility and causing `invalid_exact_price`.

Before edits, the new real-path regression failed at the supported-quote assertion: 1 failed, 12 deselected, 50 warnings, 2.61 seconds. After edits the same path passes.

## Canonicalizer and binding

Shared implementation: `app_core/public_quote_policy.py::canonical_book_label`.

| Provider key | Canonical label |
| --- | --- |
| novig / novig_us | Novig |
| draftkings | DraftKings |
| fanduel | FanDuel |
| betmgm | BetMGM |

Known names accept casing and surrounding whitespace. Already canonical labels remain unchanged. Unknown strings remain unknown (trimmed); non-string values become empty. DK, FD, MGM, provider namespaces and other books do not authorize.

`ensure_authoritative_quote_binding` normalizes existing verified bindings without rebinding and new successful bindings before returning. Present book/sportsbook/quote_source aliases use the same canonical identity. Matching uses a normalized working copy; the original provider_quotes payload remains unchanged. Provider namespace, event ID, exact line, price and original quote time are preserved. Existing integrity checks still run.

## Policy preservation

- Novig retains the existing normal quote-policy behavior.
- DraftKings, FanDuel and BetMGM remain football fallbacks only for NFL/NCAAF; MLB, NBA, WNBA, NHL and NCAAB remain blocked for these books.
- Unsupported books remain unsupported.
- If quote_time_basis is present, the special branch still requires NCAAF + DraftKings + espn_observed. NFL, other books, other bases and a null basis fail that branch.

## Real raw-provider end-to-end test

PASS: `test_raw_provider_no_book_reaches_terminal_policy` runs the real builder, private authority projection, binding, adapter, prepare_live and finalize_live_wagers.

- Pre-populated book field used: NO.
- Raw provider book: draftkings.
- Bound quote_bookmaker: DraftKings.
- Adapted book: DraftKings.
- supported_quote: true.
- invalid_exact_price present: NO.
- Synthetic funded amount with all isolated fixture gates satisfied: $2.50.
- Raw provider payload unchanged: PASS.

This is hermetic test evidence, not validation or activation of any real sport policy.

## Fail-closed regressions

PASS: ambiguous quote, exact line/price mismatch and missing binding data (`test_prediction_evidence`); stale quote, missing producer validation and UNVALIDATED policy (`test_real_projection_still_fails_closed`); Moneyline exclusion and Gemini hard veto (`test_live_wager_contract`). No threshold, freshness, model/calibration, deployment-state, bankroll/exposure, parlay or execution changes.

## Validation

| Check | Result |
| --- | --- |
| Focused tests | 179 passed, 0 failed, 569 warnings; 13.24s |
| Full pytest -q | 2280 passed, 0 failed, 9828 warnings; 104.82s |
| Exact CI production-safety command | 531 passed, 0 failed, 1430 warnings; 33.92s |
| Exact CI production compile command | PASS |
| git diff --check | PASS |

Warnings remain; this change does not address the repository's existing pandas warnings. CI-equivalent tests ran locally on Windows using the configured Python runtime, test dependencies and Node runtime. Hosted CI runs after push.

## Files changed

- app_core/public_quote_policy.py
- app_core/prediction_evidence.py
- tests/test_prediction_evidence.py
- tests/test_candidate_authority_projection.py
- docs/audits/2026-09-15-sportsbook-normalization.md

Unrelated changes in this patch: NONE. The two pre-existing untracked Medium article files were left untouched and excluded. No wager authority was activated.
