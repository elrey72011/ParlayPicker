# NCAAF research experiment v1

Protocol fixed before the first 2025 evaluation. Train on 2023, calibrate on 2024, evaluate once on 2025. Do not select parameters, subsets, thresholds, or models after inspecting 2025 results and still call that season untouched. Future changes require a new prospective evaluation period.

Use the collection's same-season, strictly greater-than-seven-day lagged features. Require at least three prior scoring and yardage games for both teams and a known neutral-site flag. Use the same eligible rows for every model. Minimum 50 eligible games per split; report exclusions. These features have unverified historical publication/correction timestamps and remain research-only.

Fit separate home-margin and total regressions with fixed ridge penalty 10. Standardize seven inputs using 2023 only: home/away points for, home/away points against, home/away total yards, neutral-site indicator. Fit an unpenalized intercept. No hyperparameter search. Baselines are a 2023 target mean and a scoring blend: each team's expected score is the average of its own prior scoring and its opponent's prior points allowed.

For all three models, use 2024 only to estimate additive residual bias and residual standard deviation (minimum one point). Total centers are floored at zero. Use a discretized Gaussian, truncated below zero for totals, to represent integer scores and explicit push mass. This is a modeling assumption, not proof of calibration. Freeze model parameters and their hash before evaluating 2025.

Report 2025 MAE, RMSE, empirical coverage of nominal 90% Gaussian intervals, and home-win conditional-on-no-tie Brier score, log loss, accuracy and reliability bins. Margin zero is explicitly excluded from binary metrics and counted. Winner accuracy is across all eligible games, not approved bets or a 75% wagering objective. No odds, spread-cover accuracy, closing-line value or ROI claims. Nominal intervals describe model uncertainty, not confidence intervals on performance differences. No activation in the live market model.

Save the protocol, training/calibration data fingerprints, frozen parameters, checkpoint fingerprint, metrics and per-game predictions. Keep user datasets and generated artifacts out of the source PR. A later run on the same holdout is reproduction, not new independent validation.

## Reproduce and view

After loading the completed historical checkpoint in Streamlit, click **Run fixed NCAAF research evaluation** within **NCAAF Historical Collection**. Download the research ZIP containing the report, frozen artifact, and per-game predictions. No requests, fitting or evaluation run on normal page reruns. Results from a different checkpoint are hidden.

Alternatively, run `python scripts/run_ncaaf_research.py <collection.zip> --output <local-output-directory>`. This rebuilds features from the checkpoint and never trusts the ZIP's feature CSV as model input. Generated reports and artifacts are local research outputs; the live market model does not load them.

The JSON report includes RMSE, discrete central-90% interval coverage and reliability bins in addition to the concise Markdown table. Integer intervals can have coverage above their nominal level. The stored source hash identifies the implementation used; fingerprints and coefficients allow reproduction checks.
