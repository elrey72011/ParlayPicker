# Remaining production-readiness implementation audit

Baseline: `f348cf06176cbddcf9160177710fe13a636572e3` (merged PR #2331).

## MLB receipt workflow failure and repair

The latest failed main run, [35919428090](https://github.com/elrey72011/ParlayPicker/actions/runs/35919428090), never created a job or an artifact. GitHub rejected the workflow at line 19, column 34: `${{ runner.temp }}` was used in job-level `env`, where the `runner` context is unavailable. The failure class is `WORKFLOW_CONFIG_ERROR`; no receipt restore, provider call, reconciliation, backup, or audit ran.

The workflow now sets the evidence path from `$RUNNER_TEMP` within the shell step. Its runtime order is configure, restore, reconcile, grade, backup, fresh manifest verification, then audit. It writes a machine-readable success report or classified failure report and requires an audit artifact. Provider retries are bounded to transient transport failures, HTTP 429, and selected 5xx responses. Auth, integrity conflicts, and uncertain backup writes are not retried as if they succeeded. Final scores are bound to the stored MLB response and its observation time; original pregame receipts remain immutable.

The audit reports total receipts, unique games, settled and pending games, independent spread/total units, Eastern slates, outcome-available slates, maximum feasible training units, possible chronological splits, and blockers. It never grants training authority. **No live audit artifact or successful authorized workflow run exists for this patch yet**: the corrected workflow is awaiting merge onto the default branch and configured GitHub Actions secrets/variables. Fixture tests do not substitute for a live receipt audit.

## Former 31 local failures

The original 247-case selection was rerun with its incomplete Python environment and reproduced **31 failed / 216 passed**. Every failure was an import-time environment error. A clean environment installed from `requirements.txt` plus pytest, with macOS `libomp` for the pinned XGBoost wheel, reran the identical selection: **247 passed**. No affected test was removed or skipped; no product gate was changed to make it pass.

| Test | Failure class | Root cause | Disposition | Verification |
| --- | --- | --- | --- | --- |
| `tests/test_public_assets.py::test_preview_fingerprint_tracks_production_sources[publishing/board.html]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_public_assets.py::test_preview_fingerprint_tracks_production_sources[publishing/site.css]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_public_assets.py::test_preview_fingerprint_tracks_production_sources[publishing/site.js]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_public_assets.py::test_preview_fingerprint_tracks_production_sources[app_core/public_site_shell.py]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_public_assets.py::test_preview_fingerprint_tracks_production_sources[scripts/publish_board.py]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_live_candidate_first_selects_qualified_runner_up` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_portfolio_cannot_override_canonical_moneyline_or_pass` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_verified_family_populates_without_candidate_self_assertion[spread-spread_home]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_verified_family_populates_without_candidate_self_assertion[total-total_under]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change0]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change1]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change2]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change3]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change4]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change5]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change6]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change7]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change8]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change9]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_invalid_family_context_cannot_promote[change10]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_unverified_study_cannot_supply_family[missing]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_unverified_study_cannot_supply_family[sport]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_unverified_study_cannot_supply_family[hash]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_unverified_study_cannot_supply_family[expired]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_eligible_alternative_and_canonical_order` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_terminal_authority_passes_absolute_period_caps[daily]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_live_wager_contract.py::test_terminal_authority_passes_absolute_period_caps[weekly]` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_results_parlay_reconciliation.py::test_nfl_denver_kansas_city_schedule_and_no_completed_games` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `xgboost`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_results_parlay_reconciliation.py::test_explicit_update_is_idempotent_and_recomputes_ledger` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_results_parlay_reconciliation.py::test_batch_timeout_transport_is_preserved` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |
| `tests/test_results_parlay_reconciliation.py::test_outage_cap_survives_final_portfolio_allocation` | `DEPENDENCY_MISSING` | Incomplete local environment lacked `streamlit`; environment issue. | Installed declared dependencies; installed macOS `libomp` for XGBoost. | Pass in the same 247-case selection. |

The supported Linux GitHub Actions environment had already passed the post-merge full-suite run. Local macOS needs the native OpenMP runtime to load `xgboost==2.1.4`; installing the Python wheel alone is insufficient on this host.

## Implementation and evidence boundaries

`app_core/parlay_ticket_quotes.py` defines exact ticket quote capabilities and a provider protocol. The default provider is explicitly unconnected, so no live executable parlay price was obtained. The owner-confirmed route requires a retained exact-ticket artifact and owner attestation; its source is labeled separately and it cannot mint validation evidence. Synthetic providers in tests prove binding and failure behavior only.

`app_core/parlay_validation.py` adds prospective plan, candidate, outcome, report, and deployment review records to the same append-only parlay SQLite database. Product plans are independent. Validation reports retain zero stake and activation pending even after a hypothetical pass. No real frozen prospective cohort or validated SGP/Cross-Game joint model has been supplied. All three products therefore remain `UNVALIDATED`, production ineligible, and at `$0`.

| Product | Current validation state | Evidence or activation blockers |
| --- | --- | --- |
| Standard | `UNVALIDATED` | No connected exact ticket quote, frozen prospective cohort, product validation artifact, or owner activation and exposure authority. |
| Same Game | `UNVALIDATED` | Standard blockers plus no validated same-event joint/correlation model; independent multiplication is prohibited. |
| Cross Game | `UNVALIDATED` | Standard blockers plus no validated SGP component blocks, cross-component dependence and shared-factor model, or final joint calibration. |

The public product funnel now reports nested stages and exclusive exits. Netlify status checks hosted HTML, `board-data.json`, and `version.json` against the reviewed package before declaring the deployment ready; the existing SFTP path already performed hosted asset comparison. No production host URL/deployment has been reconciled in this task, so live hosted parity remains unverified.

## Schema, quote capability, and validation reporting

The quote migration adds nullable provider, response, source type, exact-binding hashes, and raw-evidence hash columns to `parlay_quote`. Existing rows keep NULL for unknown facts. `ProviderCapabilities` declares Standard, SGP, Cross-Game, exact-ticket pricing, expiration, provider ticket ID, and settlement-rule support; `NoConnectedTicketProvider` supports none. A successful quote attempt carries the exact canonical provider response bytes or owner artifact bytes and their SHA-256. The validation store retains that material in append-only, foreign-key-backed source tables, with separate acceptance, result, and comparable closing-source records. No leg-odds product is promoted to a verified ticket.

Validation plans freeze product/model and sport/market scope, training source availability, chronological validation/holdout windows, independent-unit minimums, probability/value thresholds, quote/CLV/ROI policies, and deployment criteria. Candidate snapshots preserve ticket and component hashes, model/quote timing, probabilities, blockers, and deployment state. Reports include Brier score, log loss, calibration bins/error, all-candidate coverage, effective event-cluster units, predicted EV versus realized selection return, accepted-wager ROI only with acceptance and settlement sources, price rejection rate, and CLV only with replayable comparable closing evidence. WIN/LOSS/PUSH/VOID/PENDING/NEEDS_REVIEW remain visible. A passing artifact still requires a separate deployment review and remains `ACTIVATION_PENDING` with `$0` stake.

## Verification

| Check | Result |
| --- | --- |
| Former 31 failure selection in clean dependency environment | 247 passed; all 31 previously failing cases passed. |
| Full local Python suite | 2,768 passed, 15 skipped for unavailable Node, 38 subtests passed; zero failures. |
| Browser-backed pytest with bundled Node and Chrome | 82 passed, zero skipped. |
| Canonical CI shard 1 / shard 2 locally | 1,540 passed / 1,228 passed; zero failures. |
| Canonical production-safety selection | 553 passed. |
| Focused parlay, controlled-trial, quote, wager, MLB, and publication selection | 256 passed, 38 subtests passed. |
| Full MLB receipt selection | 220 passed. |
| Standalone public-site suites | Three Chrome Playwright and two pure Node suites passed. |
| Compile and whitespace | `compileall` and `git diff --check` passed. |

The local environment used Python 3.11 with all declared Python dependencies plus pytest and macOS `libomp`. GitHub CI uses Python 3.12 and will provide the independent remote check for this pull request. The 15 full-suite skips were optional Node checks, not any of the former 31 failures; all 15 ran and passed in the browser-backed selection. The only test fixture update was a browser test using dates before the public-results start and an obsolete navigation selector; it now tests the current public period and navigation without relaxing outcome checks.
