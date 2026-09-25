# MLB exact-market production-readiness audit

The `MLB receipt reconciliation` workflow has a manual `readiness_audit` option. It restores the immutable MLB receipt backup, verifies every record hash and manifest dependency, rereads the published manifests, and inventories canonical prospective, native MLB, saved prediction, and repository research evidence. This option is read-only against the Shared Drive. It uploads twelve sanitized JSON reports named in the task brief. A failed restore or readback produces a classified failure report and no readiness claim.

Run the same audit from an authenticated environment with:

```bash
python scripts/run_mlb_production_readiness.py \
  --database-dir /tmp/mlb-readiness/data \
  --output-dir /tmp/mlb-readiness/reports \
  --source-commit "$(git rev-parse HEAD)"
```

The only admissible training source is the native immutable pregame receipt chain. Each selected receipt is rechecked against its retained MLB schedule, Odds API event and exact book/line/price, prior final-game feeds, and final result feed. The fixed `mlb-one-game-market-home-over-v1` rule chooses the saved home Run Line or Over Total receipt first, then the opposite selection if necessary. One MLB `gamePk` contributes at most one independent observation per market. Provider IDs, team IDs, scheduled start, and MLB game number are checked so doubleheaders cannot collapse. Other books and reprices remain evidence, without increasing independent `n`.

Exact targets are `COVER/PUSH/NO_COVER` for Run Line and `OVER/PUSH/UNDER` for Total. Integer pushes are retained; voids and uncertain finals are blocked. The existing MLB ridge challenger estimates win conditional on a decided bet and does not model pushes. The older paired-score forecaster estimates margin and total points. Both are classified as wrong targets for this audit and keep research-only status.

## Frozen feature and split policy

`mlb-receipt-asof-exact-market-v1` allows six prior-game scoring summaries, the exact selected line, and the implied chance from the exact selected price. Each source must have been observed no later than the quote and before first pitch. Historical starter, bullpen, lineup, weather, season-end and closing-price features are excluded until a versioned as-of source exists. TheOver `ModelHitRate` is never a game probability; blank `WinProbability` remains blank.

Whole games are ordered by scheduled first pitch and assigned across both markets to development, selection validation, calibration candidate, and untouched research holdout in 50/16.7/16.7/16.7 percent proportions. The prespecified floors are 240/80/80/80 independent games, two seasons, and at least 30 examples of each decided class in development. Eight features plus intercept imply up to about 18 three-class logits, so 240 development games gives roughly 13 games per parameter before class imbalance. The three later 80-game windows limit the instability of model selection, calibration and final descriptive scores. These are engineering safeguards, not a guarantee of statistical power. A half-point line has structural push probability zero with integer MLB scores; integer-line modeling additionally requires ten observed development pushes. Every earlier outcome must be available before the next cohort's selected quote. The plan is declared before fitting and does not use random final splitting.

If a scope clears the evidence gate, a later exact-scope modeling run must compare a smoothed three-class base rate and stored-price benchmark with a regularized exact-scope candidate using multiclass Brier, log loss, calibration error, coverage, class balance, push rate, hit rate and nonpush AUC where defined. This readiness audit never converts the existing binary challenger into a three-class model. When the gate is not met, baseline/model scores remain explicitly unevaluated, model and calibration IDs remain null, and prospective validation is blocked.

Existing prospective validation plans are inventoried without changing frozen windows. Near-start quotes are not certified closes. Research records remain `UNVALIDATED`, production ineligible, and zero stake. No activation, production parlay leg, or wager operation is called.
