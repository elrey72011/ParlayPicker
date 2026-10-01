# Public probability trace and display correction — October 1, 2026

## Finding and evidence limits

A matching saved PASS wager contract with null conservative probability/EV overwrites a valid per-game research probability/EV in `app_core.public_board.pick_record`. The unchanged-main export → package builder path independently reproduces that behavior. Conservative fields are authoritative and must remain null; this change adds a separate, price-bound `research_display` object rather than supplying a fallback to approval fields.

The captured public package does **not** prove that this overwrite caused the October 1 screen: none of its 13 Overall records contains a wager contract. All 13 published probabilities and EVs are null. Matching producer exports, complete candidate audit and inference/calibration records were not recovered in bounded read-only discovery. Historical causation remains **UNVERIFIED**. Main HEAD is not inferred to be the producer runtime.

Captured HTML, its directly referenced board JSON and version JSON agree:
- Main/base and served source metadata SHA: `98015f64c6ca3f5df6e68ce0acbb27a8d0ecbd36`.
- Board/build hash: `c65d137b279076192c799bf5c238fbbde2047db6941296901c718a41c5eb67ca`.
- Built: `2026-10-01T19:39:47.100237+00:00`; published: `2026-10-01T19:39:56.875091+00:00`.
- Original analysis timestamp: `2026-10-01T19:25:32.042478+00:00`; exact export-run ID and producer SHA are unknown.
- Source fingerprint: `f3a330e73a930ade5fc166b5b9e735d4ea747b7349bffd9c86267994aa665add`.
- Selected-row hash: `a4d93d221a4969c40a41b68268293ecbbbd921f7af314773a5ce51faffca2b68`.
- 13 Overall / 13 Sides / 13 Totals entries; published selected-game denominator 13; full-candidate count unknown.
- Capture: `2026-10-01T21:05:58.901658+00:00`; HTML SHA-256 `2aeae72084896630d838c63ebcc5f80248c5bdd88d7c042b76d48949e91890eb`.

Five representative picks/prices match the supplied incident clues. The Library screenshot could not be installed locally: the current materialization helper fails on Windows because `os.setxattr` is unavailable, including the single bounded retry. Its pixels were not inspected. The capture is therefore separately identified comparison evidence, not conclusively the screenshot's exact publication.

## Saved decision reasons and unavailable estimates

| Overall selection | Saved public decision reason |
| --- | --- |
| Seattle +1.5 | model EV is not positive |
| Columbus +1.5 | model EV is not positive |
| New York +1.5 | model EV is not positive |
| Nashville +1.5 | model EV is not positive |
| San Jose +1.5 | model EV is not positive |
| Atlanta +1.5 | model EV is not positive |
| Philadelphia +1.5 | model EV is not positive |
| Chicago +1.5 | model EV is not positive |
| Vancouver +1.5 | model EV is not positive |
| Las Vegas -4.5 | missing or invalid push probability |
| Over 37.5 (Pittsburgh/Cleveland) | calibrated edge below 2.0% safety margin (+0.9%) |
| Over 55.5 (Western Kentucky/New Mexico State) | alternative not finalized |
| North Texas -2.5 | model EV is not positive |

Each displayed estimate is null in serialized JSON; the earlier point of loss cannot be determined without the exact preceding exports. Mutually exclusive first-loss attribution: upstream missing 0 confirmed, legitimate rejection 0 confirmed, export loss 0 confirmed, display loss 0 confirmed, unresolved **13**. These are attribution counts, not assertions that a possible cause did not occur.

Saved primary producer reasons reconcile once per Overall row: 10 nonpositive model EV, one missing/invalid push, one recorded producer rejection, one alternative not finalized. Strict trace is unavailable for all 13; trial outcomes, model/calibration references and full candidate count remain unknown. Overlapping blockers are non-additive. At a frozen 19:40 UTC clock all 13 are saved research PASS, not quote-expired. At 20:11 UTC all 13 quotes are expired. Later expiry does not explain the original fresh-screen decision.

Private artifacts contain all 39 published view records and nine stages per record (351 stage entries), joined to Overall diagnostics by the exact public-row digest. View rows without verified candidate/event identities are not joined by team names. The existing `scripts/trace_current_wagers.py` reports `HISTORICAL_CANDIDATES_UNAVAILABLE`; no alternate approval engine was introduced.

## Precise push disagreement and correction

The builder already calls `price_value_display.display(p, odds, ev, push_probability=saved_push)`. The validator previously called the same function without that push argument after serialization.

For a PASS contract with null authoritative p/EV and explicit push **0**, the builder produces null break-even/edge (no compatible authoritative p/EV), while the old validator recomputes a sportsbook-only break-even of 0.5238095238095238 at -110 and rejects the package. For a compatible integer-line research estimate conditional p=0.575, push=0.1, +100, the exporter emits unconditional p=0.5175, EV=0.135, break-even=0.45 and edge=0.0675; omitting push cannot reproduce that basis.

The additive `price_push_probability` metadata preserves the exact builder argument for the validator. Validation still recomputes the same price math, rejects invalid push types/ranges, rejects conflicting research/display push, and rejects inconsistent EV. Neither exporter probability/EV calculations nor gate/stake inputs were changed. Legacy packages without the metadata retain their existing validation path.

## Display schema and authority separation

`research-display-v1` requires recorded event, candidate, export-run, quote, price, analysis/start, market/selection/line, model target, period and settlement rules. Conflicting identities, context/home-win targets, missing target/provenance and failed inference cannot supply a cover/total estimate. Saved inference status remains UNKNOWN when absent.

The original exporter value is captured before contract assignment. Research EV is accepted only when the existing price-value math agrees on probability, exact price and push; conservative EV is never borrowed. Valid zero and negative EV survive. Explicit nonzero push on half-point markets is rejected; missing push is compatible only on the existing half-point legacy path. Integer markets need recorded compatible semantics. CSV exports carry canonical JSON display data, preserving rejection reasons through CSV conversion.

The browser labels research, approved and controlled-trial estimates separately, explains availability and the saved wagering blocker separately, and shows PASS as “not approved.” The existing layout and current-wager filters remain in place. The additive object is not read by ranking, funded parlays, history/lock classification, trial gating or subscriber release.

## Validation and review

Actual-path regressions exercise the per-game exporter, package builder, validator, serialized board assets and installed headless browser with frozen clocks and fake transport. They cover null contracts, absent/invalid/zero values, negative EV, exact-target conflicts, push semantics, approved/trial values and stakes, expiry/start, all three views, unknown candidate counts, legacy packages, missing inference and downstream authority separation.

Frozen before/after inputs are identical (SHA-256 `55b6bc90aa9a589b1d1a63dbefae819d549d78be106363f5cf78e60ac2b2798f`), at `2026-10-01T19:40:00+00:00`. Before: estimate/EV unavailable. After: explicitly research 60.0%, EV 14.5%, edge 7.6%, break-even 52.4%. Both retain PASS, null authoritative p/EV, zero stake and zero current wagers. These are synthetic local previews, not a repaired live publication.

Broader local regressions passed 314 cases and encountered 19 setup errors in `test_sport_market_gate.py`: pytest treats its imported `setup` function as xunit setup and passes a module where a path is expected. The same 19 errors reproduce on the unchanged base. Existing test expectations and protected scope tooling were not modified. Exact final-revision check results and CI links are recorded in the PR and private evidence bundle.

The correction uses a separate checkout. Prior fixes #2362 and #2364 are ancestors of the base; #2370 and the immutable `7c4fe71c7b9bd1a7ae73f8ba04b5e9a79d720eea` checkout remain untouched. PR workflows were inspected: this branch/PR triggers test and offline verification jobs, not publishing or provider acquisition jobs. No workflow dispatch, live release, merge or deployment is part of this change.

## Missing evidence and separate publication verification

The smallest bounded historical follow-up is to supply already retained finalized three-view CSVs, finalized producer rows, the complete candidate audit, exact quote/model/calibration outputs and saved strict/trial/review/allocation traces for the above analysis timestamp and selected candidate IDs. Dashboard downloads and publication preview are session/browser downloads; no matching durable output was found in the inspected local holdings. Do not substitute September 29 records or rerun acquisition.

The actual dependency that would permit approved wagers cannot be established from null published estimates and incomplete authority traces. Saved nonpositive EV, invalid push and edge-threshold rejection remain blockers; this display correction grants no authority.

After a separate owner-authorized merge and publication: verify the generated package locally against its retained producer inputs; confirm decisions/stakes are unchanged; obtain the existing HTML and its directly referenced version/board once; require matching hashes/build IDs; inspect research versus approved/trial labels, saved reasons and fresh/expired counts. Do not refresh producer inputs as part of verification.
