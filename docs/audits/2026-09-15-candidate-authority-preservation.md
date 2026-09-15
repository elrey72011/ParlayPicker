# PR #2277 authority-field preservation follow-up

HEAD BEFORE: `e4a1c42adcd0e535e5fc34de13c7b6be96be47d2` (current main).
HEAD AFTER: the commit containing this report on `codex/preserve-candidate-authority`.

## Reproduction and root cause

Before production edits, `test_real_projection_retains_authority` called the real
`build_best_picks_df` and failed: the supplied `team_ids=['home','away']` became
absent in `candidate_audit_df`. Evidence: `test-results/auth-before.txt`.
The reporting-only column allowlist dropped authority inputs. A second narrow
projection also omitted the source `game_id` from the final template. Reporting
synchronization can repair a selected row, so that reporting frame is not suitable
as the immutable candidate input to terminal authority.

| Fields | Supplied upstream in reproduction | In old audit | Required downstream | Action |
|---|---|---|---|---|
| team_ids, game_id, sport, selection, line | Yes | No | Identity/allocation | Preserve private source values; retain template game/candidate IDs |
| book, quote_time, start, exact_quote_verified, identity_verified | Yes | No | Verification/freshness | Preserve |
| model_validated/version/trained_through/available_at | Yes | No | Producer validation | Preserve without inference |
| calibration_validated/version/available_at | Yes | No | Producer validation | Preserve without inference |
| evidence_snapshot_id/frozen_at/effective_sample_size/version | Yes | No | Maturity/evidence | Preserve |
| conservative_probability, mean_probability, calibration_uncertainty | Yes | No | Conservative gates | Preserve |
| critical_feature_error, current_regime_conflict, prior_clv_lower | Yes | No | Maturity/data integrity | Preserve |
| slate_id, selection_policy_version, sport_policy_version | Yes | No | Study binding | Preserve |
| push/alternate flags, provider identifiers | When supplied | Incomplete | Exact market/quote identity | Explicit private contract |

## Implementation and schema boundary

Option B: build `candidate_authority_df` from the exact expanded pool before any
reporting repair, with one explicit allowlist reusing evidence FIELDS and maturity
INPUTS. Keep `candidate_audit_df` as owner reporting. Both receive the same
candidate IDs. Producer IDs are retained; missing candidate keys use a stable
hash of exact game/market/selection/line/price/book/time/provider identity. Such a
key does not assert event verification. No row-order or probability join is used.

The live app sends the private frame through real `prepare_live`, terminal
`finalize_live_wagers`, and `capture_run`. Capture explicitly selects candidates
by ID and preserves prepared authority facts instead of replacing them with
reporting adjustments or rebinding their quotes. Legacy capture callers retain
their existing behavior. Per-run snapshot metadata and hashes are still recorded.

Final templates retain source game/candidate identifiers so allocator group keys
refer to the same event. Missing team IDs still allocate zero, now with the
explicit diagnostic `missing_stable_team_ids`. This is the sole change to
`wager_decisions.py`; allocation arithmetic is unchanged.

Public wager-contract fields are unchanged. Arbitrary input columns are not
copied into the private allowlist. No secrets or active configuration were added.

## Authority fields required downstream

Canonical transport contract (evidence fields, maturity fields, adapter aliases,
quote identity and allocation inputs):

```
snapshot_id export_run_id candidate_id game_id matchup_id sport season slate_id event_date game_start_utc home_team_id away_team_id home_team away_team market_type selection line american_odds decimal_odds sportsbook odds_source odds_recorded_at quote_verified prediction_generated_at model_version model_available_at model_trained_through calibration_version calibration_available_at evidence_version evidence_frozen_at selection_policy_version sport_policy_version raw_model_probability calibrated_probability sport_calibrated_probability hierarchical_probability conservative_probability market_probability fair_market_probability edge conservative_edge expected_value conservative_ev calibration_uncertainty effective_evidence_size historical_prior_weight current_season_weight ml_context_probability ml_spread_alignment kalshi_probability consensus_agreement gemini_review_status gemini_reviewed_at gemini_input_hash identity_verified data_quality_status push_semantics_verified candidate_maturity production_eligible production_bet_amount created_process_id payload_hash candidate_rank_before_gate candidate_rank_after_gate selected_as_best_pick best_available_candidate_count decision_bundle_version provider_event_id provider_namespace candidate_pool_complete odds_american book start quote_time exact_quote_verified model_validated calibration_validated evidence_snapshot_id critical_feature_error mean_probability evidence_effective_sample_size prior_clv_lower current_regime_conflict validated_evidence_family push_probability alternate alternate_quote_verified unresolved_material_news gemini_status team_ids league best_pick quote_bookmaker quote_source quote_timestamp commence_time game_time_est market_line_used market_line_source spread_line total_line quote_binding_verified maturity line_consistency_flag line_event_identity_match_flag degraded_feature_subset_flag provider_quotes provider_ids gamePk event_id schedule_week probability_semantics win_probability_unconditional loss_probability_unconditional model_probability gemini_error gemini_stake_multiplier gemini_gate_reason
```

## Acceptance evidence

- AUTH-01: PASS, stable team IDs survive the real builder with no reinjection.
- AUTH-02: PASS, real builder and real app pipeline allocate **$2.50** under the
  hermetic test-only provisional policy and fresh $1,000 exposure fixture.
- AUTH-03: PASS, absent team IDs remain absent and allocate $0 with explicit reason.
- AUTH-04: PASS, otherwise-valid expanded runner-up is selected/funded when the
  research winner fails identity verification.
- AUTH-05: PASS, prepared evidence IDs, timestamps, probabilities, model/calibration
  versions and private identity/quote facts persist under the same candidate IDs.
- AUTH-06: PASS, both spread candidates remain in preparation/capture and IDs
  reconcile with the reporting audit. Existing unsupported-market, preselection
  line-integrity and missing-matchup filters remain unchanged.
- AUTH-07: PASS, complete Moneyline source row is excluded by the real builder.
- AUTH-08: PASS, missing model validation and false calibration validation remain $0.
- AUTH-09: PASS, stale quote remains $0.
- AUTH-10: PASS, UNVALIDATED sport remains $0.
- Contract regression checks adapter/gate input names and real supplied-field
  preservation. Exact-book/time keys are stable under row reordering.
- Public/private boundary: PASS, team IDs remain outside reporting/public schema.

## Validation on final code

- Focused: **184 passed, 0 failed, 1,110 warnings, 17.52 seconds**.
- Full suite: **2,194 passed, 0 failed, 9,688 warnings, 112.16 seconds**.
- Exact CI production-safety command: **448 passed, 0 failed, 1,430 warnings, 34.63 seconds**.
- CI production compilation: **PASS**.
- `git diff --check`: **PASS**.
- Logs: `test-results/auth-verified-focused.txt`, `auth-full-final.txt`,
  `auth-verified-safety.txt`, `auth-verified-compile.txt`.

## Safety and unrelated changes

No thresholds, weights, calibration artifacts, Kelly fractions, deployment
states, Gemini policies, parlay rules, bankroll defaults, scheduler settings,
active policies, locks, results or public content changed. No validation facts
were inferred and no sportsbook execution was introduced. Synthetic test funding
is engineering evidence only, not validation for real wagers.

UNRELATED CHANGES: NONE. Existing untracked article files were left untouched.
