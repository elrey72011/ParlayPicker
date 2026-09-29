# Post-#2356 priced-value consistency review

## Revision and scope

- Audit/base revision: `263cab0e1e4c4e7dc49d4906b43dea857f5f0347`.
- Implementation revision: `b5fb3a5e153d827970c8f7a05058a96ff307c843`.
- Final PR revision: pending the evidence commit and final CI.
- Branch: `codex/post2356-priced-value`.
- Overlap review: open pull requests were inspected before implementation; no open pull request contained the post-#2356 priced-value correction. Later main changes were preserved.
- Owner work: the original checkout's untracked files were left untouched; implementation occurred in an isolated managed worktree.

This change closes the local priced-value inconsistency only. It does not qualify a model, deploy staging, approve a product, activate a market, enable billing or sales, replace calibration, create a production record, or authorize a wager.

## Correction

The final priced value is now computed once with `core.price_value.price_value` from an explicit contract: final unconditional win mass, supported push mass, exact decimal quote, and an optional conservative win mass with declared semantics. The production gate consumes those values rather than deriving no-push break-even or treating `p_win / break_even - 1` as per-total-stake expected return.

For the supplied non-identity example, the actual lean and gate path now returns:

| Quantity | Value |
|---|---:|
| Calibrated conditional win probability | 0.575 |
| Final unconditional win mass | 0.5175 |
| Push mass | 0.1000 |
| Loss mass | 0.3825 |
| Unconditional break-even win mass | 0.4500 |
| Absolute edge | 0.0675 |
| Mean EV per total unit staked | 0.1350 |
| Fixed-push conservative win mass | 0.4950 |
| Fixed-push conservative EV per total unit staked | 0.0900 |
| Legacy upstream model EV diagnostic | 0.2000 |

The value `0.5175 / 0.45 - 1 = 0.15` is retained only as a negative test: it is not labeled mean EV. `EV` remains a documented legacy alias of upstream model EV; `Upstream_Model_EV` is the explicit diagnostic field, while `Mean_EV_Per_Unit` and `Calibrated_EV` carry the final mean per-total-stake value.

The actual selection, public export, strict decision filter, parlay consumer, and subscriber projection are exercised for the same candidate, quote, calibration content identity, probabilities, and values. That trace is an isolated integration fixture, not empirical qualification or hosted evidence. An unreviewed post-calibration bucket tilt receives its own transform identity and loses production calibration authority.

## S01-S18 acceptance matrix

| ID | Status | Evidence |
|---|---|---|
| S01 | PASS | Base, implementation SHA, overlap review, isolated worktree, and preserved owner changes are recorded above and in `ci-execution-and-protected-scope.json`. |
| S02 | PASS | The new actual lean/gate regression failed before correction because `Final_P_Win` was absent and incompatible no-push values remained; the exact test and failure are recorded in `final-value-parity-tests.json`. |
| S03 | PASS | Actual consumer regression and runtime trace prove `.5175/.10/.3825`, break-even `.45`, edge `.0675`, and mean EV `.135`. |
| S04 | PASS | The production-gate regression proves ratio `.15` is not the unconditional per-total-stake EV `.135`. |
| S05 | PASS | The no-push regression preserves mean EV `.20` and conservative EV `.10` at decimal odds 2. |
| S06 | PASS | Positive/zero/negative EV, integer/half-point lines, explicit zero/nonzero push, missing/invalid push, and invalid/mismatched odds are covered without changing thresholds. |
| S07 | PASS | Final mean and conservative EV fields are distinct from `Upstream_Model_EV`; the legacy `EV` alias is explicitly labeled. |
| S08 | PASS | The validated 12-scope runtime trace invokes all five actual consumers per scope with matching quote/artifact/value identity; unsupported bucket tilt is separately identified and blocked. |
| S09 | PASS | Numerical corrections retain the existing edge floor and every identity, chronology, qualification, review, exposure, market, and commercial control. All traced recommendations remain non-bettable. |
| S10 | PASS | Existing wrong-scope/predictor, rejected-artifact, mutation, fallback, and mixed-scope regressions remain in the targeted/full suites; the post-transform case adds a new fail-closed regression. |
| S11 | PENDING_FINAL_PR_CI | Expanded local regression results are recorded. Final-revision application, production-safety, paid, completion PostgreSQL, protected-scope, and browser results will be recorded after PR CI. Local browser execution was not claimed because Playwright is absent. |
| S12 | PASS | No protected performance/storage/lock/publication file, historical record, scope guard, or existing test expectation was modified. |
| S13 | BLOCKED_EXTERNAL_AUTHENTICATED_CENSUS_INCOMPLETE | The 12-scope ledger contains exact frozen plan identities and field-level `UNKNOWN` reasons. The newest authenticated run is incomplete and predates the final revision; fixture and legacy counts are excluded from qualification. |
| S14 | NOT_RUN_EXTERNAL_BLOCKER | No GitHub environment/deployment/Pages site or local deployment configuration was available. Host, deployed/served revision, migration, image, endpoint, and TLS evidence remain unknown with owners named. |
| S15 | NOT_RUN_EXTERNAL_BLOCKER | Actual OIDC, Stripe sandbox, entitlement, results/expiry, and denial journeys were not run; local tests are not relabeled as hosted evidence. |
| S16 | PARTIAL_LOCAL_ONLY_HOSTED_NOT_RUN | Local deterministic publication/alert coverage exists, but hosted retry, provider failure, backup/restore, rollback, load, LCP, and authenticated timing checks were not run. |
| S17 | PASS_TRACKING_ONLY | All deployment, product, policy, rights, processor, marketing, pilot, and final-authorization decisions are independently tracked as pending/not started/not granted. No approval or elapsed pilot history was invented. |
| S18 | PASS | `ship-sell-gate-report.json` separates code correctness, hosted staging, market qualification, commercial approval, pilot, and owner authorization. No state implies another. |

## Protected-scope report

Protected performance, evidence storage, public history, stage timing, prediction evidence, and lock files changed: **none**. Historical calibration/model artifacts, releases, predictions, settlements, cohorts, and billing records changed: **none**. Existing expectations, validation thresholds, scope-guard baselines, and market activation policy changed: **none**.

The local Windows scope-guard byte check sees CRLF working-tree bytes where its stored tooling hash describes LF bytes. The committed blobs were not changed; final Linux protected-scope CI is the authoritative check.

## Evidence boundary and remaining decisions

The current market census is intentionally `INCOMPLETE`, with all twelve markets `UNVALIDATED` and zero markets evidenced as qualified or activated. The latest authenticated research workflow was cancelled after 60 minutes, requested only MLB/NCAAF/NFL, ran on pre-final SHA `72aba3152a87739107ef2e4e9b35f8c4f263024f`, and did not emit a finished per-market census. Its mixed restored-record count is not treated as observations or qualification.

Staging remains unverified because no authorized host, registry, secret configuration, OIDC test tenant, Stripe test configuration, or isolated PostgreSQL deployment was available. The next external step belongs to the deployment/operator owners listed in `staging-environment-and-journey-report.json`. The next empirical step is a new explicitly read-only authenticated six-sport inventory on the final revision; the normal readiness loader is not authorized merely as a census because it can reconcile and back up data.

Commercial and launch decisions remain with the named owners in `owner-launch-decision-register.md`. The pilot has 0 observed days and 0 observed slates. Live effects remain `NOT_PERFORMED`.
