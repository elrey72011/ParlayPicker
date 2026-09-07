# MLB starter development workflow

Collect actual starter identities and all pitching appearances from official boxscores for 2023–2024 only. Resume the same command until audit.json reports collection_complete. Roughly 4,859 requests are needed. Each batch is capped; no API key is required.

```powershell
python scripts/collect_mlb_pitcher_history.py output/mlb-history/checkpoint.json --output output/mlb-pitchers --max-requests 100
python scripts/evaluate_mlb_pitcher_development.py output/mlb-history/checkpoint.json output/mlb-pitchers/pitcher-checkpoint.json --output output/mlb-pitchers/development.json
```

Do not modify the source checkpoint during collection; its hash is pinned. Network errors stop the batch and previously saved records survive. Excluded boxes are not retried automatically; audit them before comparison. This is a local workflow, not a Streamlit feature or automatic Drive backup.

Features use only same-season appearances whose game completion precedes the target cutoff, including relief appearances and following a pitcher across trades. Require three prior appearances and at least 27 outs. ERA/K9/BB9/WHIP use outs, not decimal innings. Missing or ambiguous starters and invalid stats are excluded, never filled. Actual starters are known retrospectively; their pregame announcement time is NOT established. Historical corrections and missing source games remain limitations.

The fixed development comparison fits ridge alpha 10 on 2023 and compares team-only versus team-plus-starter margin/total MAE and RMSE on identical 2024 games. Neither is tuned on 2025, which has already been evaluated. This is development evidence, not independent validation, and does not estimate betting accuracy or promote models. A later frozen prospective cohort is required to evaluate any proposed improvement.
