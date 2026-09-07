# MLB historical research collection

Run from the repository root with the project Python dependencies installed:

```powershell
python scripts/collect_mlb_history.py --seasons 2023 2024 2025 --output output/mlb-history --max-requests 20
```

Repeat the identical command to resume. Each batch uses at most the requested number of HTTP requests (1–500). A complete three-season collection needs roughly 7,300 individual game requests plus schedules. No API key is required. This is a local CLI workflow; it does not add a Streamlit button or automatically upload to Drive.

The output contains checkpoint.json, features.json, targets.json and audit.json. Keep the checkpoint to resume. Network failures stop the batch with an error; previously saved games survive. Do not run concurrent collectors against the same directory. Back up the directory after completion. To retry excluded or corrected games, use a fresh output directory; checkpoints intentionally retain their collected version.

Review collection_complete, pending_games, missing_schedules and collection_exclusions before any evaluation. A partial collection produces partial features and is not training-ready. Completeness means all returned schedule IDs were processed, not that every game is usable or that provider coverage is independently verified.

Only final regular-season games with complete play timestamps and valid scores are accepted. The feature cutoff is the earlier of scheduled start and first recorded play. Source games must have their last recorded play strictly before the cutoff, including games suspended and resumed later. Doubleheaders retain separate game IDs. Features use only same-season completed games, require at least ten per team, and retain the exact source game IDs. Outcomes are a separate file.

Final-play time is an event-time proxy, not proof of when a corrected result was published. Historical corrections remain unverified. All outputs remain research-only. No odds, starting pitchers, injuries or production eligibility are inferred.

Next: after auditing complete coverage, predeclare chronological train/calibration/test periods and compare constant and lagged-scoring baselines with a candidate on identical eligible games. This collector does not train or promote a model or change betting thresholds.
