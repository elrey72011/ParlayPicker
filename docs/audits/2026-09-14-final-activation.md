# Final activation implementation — working notes

AUTHORITATIVE IMPLEMENTATION FILE: parlaypicker_final_activation_codex_prompt_with_precedence.md (internal title: parlaypicker_final_activation_codex_prompt.md)
REFERENCE-ONLY FILES: 2026-09-14-live-wager-contract.md; 2026-09-14-sport-deployment-assessment.md
OLDER PROMPTS: superseded unless explicitly incorporated.
HEAD BEFORE: 772d909735879db73105613565de0ca9c5e36ce9
Baseline: 2,119 passed, 9,261 warnings, 96.80 seconds.

| Component | Active path | Authority / missing work |
|---|---|---|
| Live selection | streamlit_app -> finalize_live_wagers | Present; explicit maturity and manually supplied exposure |
| Evidence | begin_run / capture_run / materialize | Present; complete strict activation schema and closing records missing |
| Bundle | artifact_manifest | Present; frozen artifact upper bound is not original training provenance |
| Quotes | bind_quote | Present; exact provider timestamp required |
| Calibration | effective_prob_calibration / selector_validation | Present; pooled flag is not sport authority |
| Sport gates | SportPolicy / candidate_decision | Present; independent ceilings, no automatic maturity |
| Closing | capture_closing_lines / ncaaf_closing | Research collectors; strict shared-store linkage missing |
| Grading | public_history / mlb_event_matcher / record_scores | Present; append-only scores, preserve deterministic identity |
| Remote | evidence_remote restore/sync | Present; configured=false locally, folder/service-account dependency |
| Exposure | finalize_live_wagers config | Ledger missing |
| Parlays | production_parlays | Qualification present; actual ticket confirmation/sizing missing |
| History | public_history/public_board | Immutable compatibility must remain |

Discrepancies: references correctly describe prior work, but the earlier pooled-calibration provenance assessment is not proof about remote evidence. Remote configuration is absent in this environment. Activation requires additional verified evidence, not source-code label changes.

## Implemented in this run

- Explicit policy artifact build, verification and owner activation commands, with hash binding, per-sport expiry enforcement and production rejection of test-only validation.
- Automatic candidate maturity from verified prior validation, complete version matching, conservative eligibility and uncertainty/regime requirements. Same-slate validation cannot inform maturity. Private maturity reason/version/input hashes are saved without relaxing public schemas.
- Append-only owner exposure ledger, explicit bankroll/caps configuration, fresh snapshots, turnover and open-risk tracking, and live adapter integration. Recommendations alone create no commitment. Every underlying parlay game/team consumes exposure.
- Strict candidate enrichment and exclusion reports; immutable development-plan freeze receipts; whole-slate chronological boundaries; sport/market/version isolation; calibration, conservative residual, CLV, return and slate-resampled risk diagnostics against frozen criteria.
- Append-only exact closing observations replicated through the existing remote store; original outcome observation times preserved during materialization.
- Owner readiness/policy/ledger panel behind the existing publishing-token check. No provider requests during panel rendering.
- Explicit actual-ticket confirmation command. Qualified tickets remain $0 until the owner confirms a price and the validated joint/stake policy and exposure checks pass. The implemented joint bound is Frechet's lower bound, not an invented correlation coefficient.

## Verification

HEAD BEFORE / AFTER: 772d909735879db73105613565de0ca9c5e36ce9 (uncommitted changes).
Baseline: 2,119 passed, 9,261 warnings, 96.80 seconds.
Final full suite: 2,137 passed, 9,261 warnings, 96.89 seconds.
Activation acceptance suite: 18 passed. Tests use explicitly synthetic evidence; none was written into active policy.

Engineering proof: $1,000 synthetic bankroll produces **$2.50** normally and **$1.00** during the configured outage. Tested hard failures remain $0. Owner-confirmed positive actual parlay price passes the hermetic recommendation path; unconfirmed and nonpositive-EV prices remain $0.

The reproducible proof package is `output/activation-proof.json` and `output/activation-proof.md`. It captures literal executed commands, exit codes and outputs, including fail-closed configuration results.

## Real authority and blockers

- Active store: `data/prediction_evidence/evidence.sqlite3`.
- Local prediction snapshots: 0; score revisions: 0; strict closing observations: 0.
- Remote storage: not configured in this environment. `PARLAYPICKER_DRIVE_FOLDER_ID` and `PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT` are absent; `.streamlit/secrets.toml` does not exist locally. This does not establish that Streamlit Cloud has no private evidence.
- All six sports: UNVALIDATED, strict_n=0, effective_n=0. No frozen development plan or admissible holdout exists locally.
- Active real policy: absent. Candidate artifact contains zero-authority defaults only; it was not activated.
- Bankroll/limits ledger: unconfigured. Snapshot generation correctly fails; no bankroll was invented.
- REAL_SPORT_AUTHORITY_ACTIVE=false; funded real candidates=0; total recommended real stake=$0.
- No actual owner ticket price was entered; no real actionable parlay was produced.
- Automated sportsbook execution: absent from the added paths. No login, transaction or placement was performed.

No actual frozen development cutoff exists, so the existing selector report was not run with an invented cutoff. No admissible current export/closing input was available for a provider closing run. Those are missing inputs, not successful validations.

## Limits of this delivery

The engineering proof demonstrates the funded recommendation and fail-closed paths. It is not a claim that every research validation requested in the activation prompt is empirically complete. Hierarchical prior/weight comparisons, uncertainty-interval coverage, ML-context ablation, Premium 75% evidence and challenger comparisons require admissible versioned predictions and frozen study choices; this run has no such dataset. The implemented diagnostics and frozen criteria must not be represented as completed empirical studies. The existing research model/provenance outputs still need genuine original metadata and conservative uncertainty outputs to produce admissible records; missing fields are preserved as missing.

The strict closing collector accepts verified provider records via `--verified-quotes`; the older automatic collector remains research-oriented. No historical close was inferred. Remote evidence recovery and deployment were not performed. No commit or push was requested in this run.

## Operator commands

1. Run live analysis as usual; it captures candidate evidence through `capture_run`. Resolve the strict report's missing original provenance at the producing model/provider; do not fill it with current or filename times.
2. Closing observations, during the valid pregame window:
   `python scripts/capture_closing_lines.py --export "app_exports/best_picks_export*.csv" --verified-quotes provider-quotes.json --database data/prediction_evidence/evidence.sqlite3`
   The JSON must carry original provider timestamps and exact game/sport/market/book/provider event IDs. It is not a manual substitute for missing historical quotes.
3. Append results: `python scripts/prediction_evidence.py refresh --database data/prediction_evidence/evidence.sqlite3`. For independently verified owner scores use the existing `import-scores --scores final-scores.csv` command.
4. Freeze development choices before holdout: `python scripts/validate_sport_deployment.py --database data/prediction_evidence/evidence.sqlite3 --sport NFL --freeze-plan nfl-development-choices.json --output output/sport-validation`. The choices must include actual development cutoff, versions, market family and frozen tier/maturity/exposure criteria. The command records the actual freeze time; it cannot backdate it.
5. Validate: `python scripts/validate_sport_deployment.py --database data/prediction_evidence/evidence.sqlite3 --all-sports --output output/sport-validation`.
6. Build/review: `python scripts/build_wager_policy.py --validation-dir output/sport-validation --output output/wager-policy-candidate.json`, then `python scripts/verify_wager_policy.py --policy output/wager-policy-candidate.json --validation-dir output/sport-validation`.
7. Owner activation only after legitimate validation: `python scripts/activate_wager_policy.py --candidate output/wager-policy-candidate.json --destination data/policies/active_wager_policy.json --confirm`. The authenticated owner panel provides the same action.
8. Configure actual bankroll/limits in the owner panel, or `python scripts/exposure_ledger.py configure --ledger data/exposure/exposure.sqlite3 --input bankroll-and-limits.json --confirm`. Required fields: status=CONFIGURED, bankroll, unit_value, currency and all five fraction caps. No nonzero defaults are supplied.
9. Record a manually placed wager: `python scripts/exposure_ledger.py record --ledger data/exposure/exposure.sqlite3 --input actual-placement.json --confirm`. Required COMMITTED record: bet_id, source_snapshot_id, sportsbook, stake_dollars and legs with sport/game_id/team_ids/market/selection/line/odds. Settlement is a new SETTLED/VOID/CANCELLED record with bet_id; never edit a commit. This action records actual exposure even if the owner independently placed an unattractive wager.
10. Confirm actual ticket price: `python scripts/confirm_parlay_price.py --package saved-package.json --ticket-id ACTUAL_SAVED_ID --decimal-odds ACTUAL_PRICE --sportsbook ACTUAL_BOOK --policy data/policies/active_wager_policy.json --ledger data/exposure/exposure.sqlite3 --database data/prediction_evidence/evidence.sqlite3 --output output/confirmed-ticket.json --confirm`. Use actual values; no real confirmation was executed here.
11. Reproduce engineering/configuration proof: `python scripts/run_activation_proof.py`. Real dry-run and test-only results remain separate.

PROMPT PRECEDENCE CHECK:
- authoritative file read: PASS
- live-wager reference read: PASS
- deployment reference read: PASS
- older prompts treated as superseded: PASS
- repository/reference discrepancies: no active ledger or automatic maturity existed initially; existing closing collection was research-only, and current capture's legacy artifact-time upper bound does not establish original training provenance.

Automatic review rejected an expired-policy allowance and a public-schema relaxation during development. Neither rejected edit was applied. Strict per-sport expiry and the exact public schema were retained; private records carry the added diagnostics.

Production dry-run evaluates only the latest saved cohort in one allocation pass; it does not add recommendations from repeated historical snapshots. The command proof was rerun after this restriction.
