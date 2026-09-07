# Fixed MLB research protocol v1

This protocol is fixed before inspecting test metrics: 2023 trains; 2024 calibrates a mean bias and residual RMS scale; 2025 is the final retrospective test. At least 100 eligible games are required per split. Rebuild features from a complete collector checkpoint; do not accept separately edited feature files.

Compare three models on identical eligible games: a training-mean constant baseline, a lagged scoring blend (home scoring plus opponent allowed runs, divided by two), and ridge regression with alpha 10. Ridge uses six features: home/away runs scored per game, runs allowed per game and win percentage. Training alone determines scaling and coefficients. Calibration alone determines each model's bias and residual scale (minimum one run). No test-driven model or threshold search is performed.

Use discrete normal margin probabilities, conditioning winner probabilities on a non-tied final margin. Total distributions truncate below zero. Report margin/total MAE and RMSE, nominal 90% interval coverage, winner accuracy, Brier score, log loss and reliability bins. These are research assumptions, not a validated MLB scoring law. No odds, spread/total bet results or returns are inferred. The fit function receives no test rows; hash the artifact before evaluation.

```powershell
python scripts/run_mlb_research.py output/mlb-history/checkpoint.json --output output/mlb-research-v1
```

Outputs: report.md, report.json, artifact.json and predictions.json. This is a local CLI workflow with no Streamlit or production-model changes. Checkpoint, training, calibration and source hashes support reproducibility.

Historical corrections/publication times remain unverified. Excluded source games create gaps in team averages. Once evaluated, 2025 cannot serve as an untouched test for subsequent tuning. Future promotion requires new prospective evidence with dated executable quotes; no result here automatically enables wagers.
