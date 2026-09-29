# W01–W32 acceptance matrix

Statuses describe evidence available at local implementation capture. Final PR-head Linux CI and PostgreSQL results are reported separately after the PR runs.

| ID | Status | Evidence / boundary |
|---|---|---|
| W01 | PASS | Remote `main`, local base, and anchor all `fd81b534dce193efa95ce910a447a9d6e27feb92`; deployed package reports source `7600b449fea5936d143717a2e32a0c4363dabace`; open PR inventory contains no recent overlapping current-wagers work; six owner untracked files remain excluded. |
| W02 | PASS | Exact public `board-data.json` and `version.json` hashes are in `source-runtime-publication-manifest.json`; deployed source and timestamps are pinned. Historical full-candidate reconstruction is explicitly incomplete and not replaced by a fixture. |
| W03 | PASS | `test_w03_w04_eleven_negative_rows_remain_research_after_expiry`: 11 negative research rows, zero current wagers, history preserved. |
| W04 | PASS | `test_w04_a_new_clock_does_not_freshen_or_approve_negative_value` plus the -190 case in `test_w13_w14_shared_push_value_examples`; package equality proves source timestamps were not rewritten. |
| W05 | PASS | `test_w05_numeric_value_without_authority_is_not_a_wager`: positive numerical fields with rejected model/calibration and no authority/allocation remain blocked. |
| W06 | PASS | `test_w06_w10_w11_private_trace_maps_eligible_candidate_and_deduplicates` and `test_w06_w18_fully_eligible_fixture_passes_then_expires`: an explicitly synthetic fully eligible fixture reaches Overall output and current usability. |
| W07 | PASS | New board diagnostics preserve recorded producer rejection before current expiry; immutable v1 diagnostics are accepted without rewrite. W03/W04 preserve the original reason and W06 preserves an expired approval as saved evidence. |
| W08 | PASS | Trace primary counts include exactly one value per deduplicated candidate (including `NONE`); overlapping counts are separate; stages carry `executed`; exact selected denominator/reason/current counts are in `current-wagers-original-vs-current-blockers.json`. |
| W09 | PASS | Every available stage reports candidate and unique-game counts separately. Selected-only replay keeps the full-candidate denominator and IDs `null` while recording 11 known output rows (`test_w28_selected_only_replay_keeps_candidate_count_unknown`). |
| W10 | PASS | Exact duplicates collapse; quote ID, book, price, and selection changes remain distinct (`test_w10_distinct_quote_book_price_and_selection_do_not_collapse`). Every non-pass stage records a reason. |
| W11 | PASS | Fully eligible candidate maps to its actual Overall output even with duplicate index labels; finalist and packaging stages are asserted. Policy-permitted exclusions remain explicit. |
| W12 | PASS | `tests/test_post2357_row_contract_isolation.py` and `tests/test_post2352_probability_closure.py` cover singleton/mixed/reordered/partitioned/duplicate-index invariance and keep tie-breaking outside probabilities/EV. |
| W13 | PASS | Shared price-value and actual lean/gate/export/subscriber route coverage: `tests/test_post2356_priced_value_consistency.py`, `tests/test_post2355_integration_verification.py`, and the 269-test dedicated run. The probability-first public path now uses the shared helper. |
| W14 | PASS | `test_w13_w14_shared_push_value_examples` asserts `.5175/.10/.3825`, break-even `.45`, edge `.0675`, mean EV `.135`, and fixed-push conservative EV `.09`. |
| W15 | PASS | Existing Post-2355/Post-2356 semantics suites and controlled-trial tests reject illegal half-point push, unsupported/missing semantics, malformed mass, and quote conflict. The only legacy compatibility is a deterministically parsed half-point with push fixed at zero. |
| W16 | PASS | Post-2355 integration tests retain exact predictor/market/calibration consumer binding; the new trace reports rejected/missing model-calibration execution as block/unknown and never creates authority. |
| W17 | PASS | Existing scope/isolation regressions (including WNBA/NBA separation) remain green in production safety; no league alias or scope rule changed. |
| W18 | PASS | `evaluate_release` uses `package_age_minutes`; schema-v5 uses the current 30-minute policy and supported legacy/history schemas retain their policy. Publication time is informational and never resets quote/analysis ages. |
| W19 | PASS | `test_w19_future_and_exact_start_fail_closed` plus existing quote-freshness tests cover exact start, future/missing timestamps, and operator-dwell expiry with existing `>` TTL and `<=` start boundaries. |
| W20 | PASS | Changed price/line cannot inherit frozen authority (`test_w20_price_change_never_inherits_frozen_authority`, `test_w20_changed_line_and_w22_expired_current_authority_block`); Post-2356 tests cover required repricing through `core.price_value`. No new quote was fabricated. |
| W21 | PASS | Release checks compare immutable selection/market/line/price/book/quote/start contracts. Altered payloads block; a clock or reserialization never creates a new analysis timestamp or transfers approval. |
| W22 | PASS | Missing/expired quote requirements and expired current authority block locally; `test_w20_changed_line_and_w22_expired_current_authority_block` and public-quote failure tests cover the fail-closed path. No provider request was attempted. |
| W23 | PASS | Existing publication-only retry, lock, history, uncertainty, and idempotence regressions remain in the 563-test production-safety/full-suite scope; new preflight runs before writes and before SFTP upload. `test_w23_publication_boundaries_recheck_without_writes` proves no call/write after expiry. |
| W24 | PASS | Existing hash/sidecar mismatch tests plus `test_w24_hosted_parity_still_rejects_expired_actionable_content`; executed public browser journey clears approved display after elapsed expiry. |
| W25 | NOT_RUN | UTC quote/analysis/build/publication/observation intervals are measured in the timeline artifact. A correlated actual run with machine, queue, operator, upload, host-verification, and notification spans was not authorized/recovered, so no processing-duration claim is made. |
| W26 | PASS | Public allowlist/rounding/browser regressions pass; private trace is stored in session/download only and excluded from `build_package`, rendered HTML, and public JSON. |
| W27 | PASS | `tests/test_controlled_trial.py` and `tests/test_controlled_trial_integration.py` retain separate review/value/freshness/allocation checks and reject negative or unsupported value paths. No trial was activated. |
| W28 | PASS | `scripts/trace_current_wagers.py` replays local files only and records no provider call/authority creation; CLI artifact and selected-only tests pass; fixtures are labeled synthetic/controlled and do not qualify models. |
| W29 | NOT_RUN | No fresh provider-backed run, quota expenditure, production authority check, remote mutation, or publication was authorized. The recovered public package is read-only historical evidence, not a new actual run. |
| W30 | PASS | `current-wagers-readiness-dependencies.json` records model qualification as unknown and census/staging/provider refresh as not executed. Old counts and fixtures are not presented as current readiness. |
| W31 | NOT_RUN | Local evidence: 108 focused, 269 dedicated, 563 production-safety, and both real local browser journeys pass. Full Linux application shards and both subscriber/PostgreSQL workflows await the final PR head; local concurrent shard runs are explicitly non-authoritative. |
| W32 | PASS | Protected-file diff is empty; existing test expectations unchanged; no activation, billing/sales, calibration replacement, historical rewrite, production record, publication, provider spend, or wager. Local guard's only failure is the known Windows CRLF byte-hash mismatch; Linux PR guard remains required. |

## Exact selected-board counts

- Saved Overall denominator: 11.
- Saved status: `PASS=11`.
- Saved public qualification reasons: `model EV is not positive=7`; `Alternative selection...estimated EV is not positive=2`; `missing push probability=2`.
- Saved negative EV: 11/11.
- Immutable selected-diagnostic v1 primary reason: `STRICT_TRACE_UNAVAILABLE=11`.
- Current Overall primary reason at `2026-09-29T16:01:29Z`: `QUOTE_EXPIRED=11`.
- Current overlapping reasons: `QUOTE_EXPIRED=11`, `ANALYSIS_EXPIRED=11`.
- Saved actionable rows: 0; current wagers: 0.
- Full candidate denominator and omitted-qualifier result: `null` / `UNKNOWN` because the matching candidate audit was unavailable.
