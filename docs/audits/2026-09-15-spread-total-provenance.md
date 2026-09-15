# Spread/Total model and calibration provenance — 2026-09-15

## Architecture finding

ACTUAL SPREAD/TOTAL PRODUCER: deterministic, configured multi-signal blend, including a fixed-parameter target-specific score distribution where supported, followed by optional empirical selection calibration.
TRAINED SPREAD/TOTAL MODEL PRESENT: NO trained artifact with a defensible fit cutoff and availability receipt is loaded by this path.
MODEL ARTIFACT: none for this purpose. `app_core/market_probability_model.py` implements `score-distribution-v1` with fixed `_LEAGUE_PARAMS` for MLB/WNBA. This is an algorithm/source label, not an immutable trained-model artifact ID.
MODEL TRAINING CUTOFF SOURCE: absent.
MODEL AVAILABILITY SOURCE: absent.

CALIBRATION ARTIFACT: `data/calibration/effective_prob_calibration.json`.
CALIBRATION FIT CUTOFF SOURCE: absent in the checked-in artifact; new writer derives it from the maximum dated graded slate actually used in the final PAV refit.
CALIBRATION AVAILABILITY SOURCE: absent in the checked-in artifact; new writer records actual aware UTC completion/write time after fitting.

HEAD BEFORE: `d005074c839782ac7ebbca2cecb4ad1ac225cc93` (main after merged #2282).
HEAD AFTER: commit containing this report on `codex/spread-total-provenance`; exact hash in PR handoff.

## Complete producer trace

1. `core/streamlit_pipeline.py::run_analysis_pipeline` enriches candidates with scoring statistics and raw market/Kalshi/TheOver signals. It calls the home-win prediction engine and `app_core.market_probability_model.predict_market_probabilities` separately.
2. `_retire_game_winner_model_from_unsupported_markets` removes the home-win XGBoost/fallback output from spread/total rows. The target-specific replacement uses fixed normal-CDF residual scales, home advantage, scoring means, form adjustments and reliability shrinkage. It loads no fitted parameter artifact and emits `ml_probability_source`, not a training receipt. Unsupported leagues or missing stats stay unavailable.
3. Spread and total `model_probability` consumes that target-specific output, with existing injury/weather/context adjustments. `compute_blended_probability` combines it with de-vigged market probability, Kalshi, TheOver and sentiment using `app_core/weights_config.py` constants. Missing signals are redistributed. Existing MLB debiasing follows. `_apply_analysis_calculations` uses the same configured blend for the analysis view.
4. `scripts/fit_blend_weights.py` fits/evaluates weights and prints suggested configuration; it does not publish a trained bundle consumed by the live blend. The live constants have no recorded final training cutoff/availability provenance. A Git SHA, source hash or deployment timestamp cannot fill this gap.
5. In `build_best_picks_df`, fresh bucket evidence can invoke `empirical_selection_probabilities`. That consumes the exact table returned by `load_calibration`, then applies existing bounded bucket effects and complementary-pair normalization. Its result is `selection_probability_used`. The upstream column named `calibrated_probability` is still the configured blend; the names must not be confused.
6. This change attaches verified calibration facts only at that actual empirical selection boundary and explicitly records `calibration_probability_field=selection_probability_used`. It does not claim that the isotonic artifact calibrated the upstream model or every other probability column.
7. `authority_projection` produces `candidate_authority_df` before reporting repair. The private evidence schema now also carries/hashes `training_cutoff_basis`, `calibration_trained_through` and the calibration probability-field scope.
8. `streamlit_app._run_pipeline` binds exact quotes, calls `prepare_live`, then `finalize_live_wagers`, then authoritative `capture_run`. Capture preserves supplied facts and writes immutable CSV payloads to `evidence.sqlite3`. It was not changed in this PR and still refuses to substitute decision-bundle/freeze metadata for missing model facts.
9. `load_snapshots` verifies immutable payload hashes. `build_readiness` checks model/timing facts; `activation_validation.reasons` additionally requires calibration identity and availability before prediction. Neither validator was changed or weakened.

## Current artifacts and field decisions

The checked-in calibration has n_graded=1155, source=data/backtest_exports, fitted_on=2026-08-29, and validation.promotable=true. However, validation.train_end and test_start are BOTH 2026-08-16 00:00:00+00:00. The existing default loader therefore rejects it for same-slate overlap. It also lacks a version digest, final-fit cutoff and aware availability timestamp. Its day-only fitted_on and holdout train_end are not substituted for these missing facts. The artifact remains untouched.

| Field | Factual source / behavior |
| --- | --- |
| model_version | Missing for the real Spread/Total producer; no trained predictive bundle identity is proven. |
| model_trained_through | Missing; fixed parameters and live scoring features do not establish training data/cutoff. |
| model_available_at | Missing; no completed trained-artifact availability receipt. |
| training_cutoff_basis | Missing in real producer; supplied genuine facts now survive private projection. |
| calibration_version | New artifacts: SHA-256 of all artifact contents excluding only meta.calibration_version. |
| calibration_trained_through | New artifacts: maximum slate_date among graded rows with interior probabilities actually accepted by final PAV fitting, including final-refit holdout rows. |
| calibration_available_at | New artifacts: actual UTC time immediately after fitting, before write. |
| source / validation | Existing actual source and chronological holdout diagnostics preserved. |

Fields now populated conditionally: calibration_version, calibration_trained_through, calibration_available_at at the boundary that actually consumes a verifiable artifact. No model fields or model_validated/calibration_validated flags are invented. With the repository's current artifacts, model and calibration authority provenance remains missing, so production remains blocked.

## Calibration transport and fail-closed behavior

`CalibrationTable` remains list-compatible and contains the exact loaded artifact alongside its knots. This avoids separately re-reading a changed file to obtain metadata. `calibration_provenance` requires digest equality, unmodified knots, source, approved chronological holdout, aware valid fit/availability timestamps and nonfuture availability. Missing, invalid, future, tampered or unapproved provenance produces no authority metadata. It never authorizes a sport or sets validation flags.

The existing numerical loader behavior is preserved: explicit paths remain research-compatible; the default still requires promotion and a strictly later holdout. A legacy chronological table without new metadata can retain existing numerical behavior but provides NO verified provenance; missing-calibration authority gates remain blocked. No probability values, formulas, bucket weights or calibration promotion criteria changed. The current checked-in overlapping artifact is already rejected numerically.

## Tests and acceptance

- Real calibration writer: final refit cutoff exceeds validation.train_end and equals the last fitted slate; completion timestamp is bounded by the actual invocation; digest reproduces from artifact contents.
- Valid/invalid artifacts: digest tampering, changed knots, missing metadata, invalid/future timestamps, unapproved and overlapping holdout cases cannot supply provenance. Unapproved default artifacts still cannot load.
- Real Spread/Total selection boundary: loaded synthetic artifact facts reach both market families, private projection, prepare_live/finalize_live_wagers and authoritative capture_run → load_snapshots unchanged.
- Synthetic supplied model facts retain training_cutoff_basis and remove model_provenance_missing. Invalid/future model facts remain blocked. Future calibration availability produces calibration_not_available.
- Missing model/calibration facts stay missing even with a nonempty decision_bundle_version. No synthetic bundle provenance: PASS.
- Target-specific real producer test confirms numeric output and configured blending without falsely attaching trained-model provenance.
- Plain knots and metadata-carrying knots produce identical calibrated numbers.
- Existing immutable snapshot recapture/update protections remain exercised; old snapshots and artifacts were not rewritten.

AUTHORITATIVE SNAPSHOT ROUND TRIP: PASS.
FAIL-CLOSED MISSING PROVENANCE: PASS.
NO SYNTHETIC BUNDLE PROVENANCE: PASS.
OLD SNAPSHOTS UNCHANGED: PASS.

## Validation

- Focused regression suite: 264 passed.
- Full test suite: 2,329 passed.
- Current CI production-safety suite: 545 passed.
- Current CI production compilation: PASS.
- git diff --check: PASS.
- Validation ran locally on Windows; hosted CI will run on the PR.

## Files changed

- core/probability_calibration.py
- scripts/fit_calibration.py
- core/streamlit_pipeline.py
- app_core/candidate_evidence_schema.py
- tests/test_calibration_promotion.py
- tests/test_candidate_authority_projection.py
- tests/test_run_readiness.py
- docs/audits/2026-09-15-spread-total-provenance.md

UNRELATED CHANGES: NONE. Pre-existing untracked Medium article files excluded. No active policy artifact, model training run, production calibration refit, sportsbook execution or wager authority activation was introduced.
