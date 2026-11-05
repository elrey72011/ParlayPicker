# Market readiness, 2026-11-05 (roadmap C0)

**Source.** The read-only workflow **Market readiness (read-only)** ([run 41988807649](https://github.com/elrey72011/ParlayPicker/actions/runs/41988807649), commit `e618e2cd`).

- It restored the canonical prospective-evidence store from the Shared Drive: **75,795 records**, each checked against Drive's SHA-256.
- It wrote nothing to Drive (`writes_refused: 0`).
- The first, uncached restore ([run 41970355476](https://github.com/elrey72011/ParlayPicker/actions/runs/41970355476)) took 2 h 22 min. With the verified cache it now takes about 4 min.

## Result: no market is in validation

All **16 frozen plans** are `UNVALIDATED`. That is 12 v1 plans, plus 4 football v2 plans for NFL and NCAAF spread and total.

| | All 16 plans |
|---|---|
| Model bound | **none** (`MISSING_MODEL`) |
| Calibration bound | **none** (`MISSING_CALIBRATION`) |
| Prospective predictions | **0** |
| Verified entry prices / certified closes | **0 / 0** |
| Deployment state | `UNVALIDATED`, stake 0 |

Additional blockers:

- **Football v2 plans:** `FULL_SLATE_COVERAGE_UNVERIFIED`, and an open confirmation window.
- **v1 plans:** open holdout windows, and 200 more settled events needed.

## Where the evidence actually is

| Canonical table | Rows |
|---|---|
| `prospective_football_event` | NCAAF 2,823 · NFL 170 |
| `prospective_football_quote` | NCAAF 22,889 · NFL 6,475 |
| `prospective_football_result` | NCAAF 319 · NFL 93 |
| `prospective_football_settlement` | NCAAF 18,175 · NFL 5,321 |
| `prospective_football_training_row` | NCAAF 10,184 · NFL 3,331 |
| `prospective_football_coverage` / `_cycle_coverage` | NCAAF 602 / 104 · NFL 33 / 110 |
| `prospective_football_team_identity` | NCAAF 687 · NFL 32 |
| `prospective_reconciled_fact` / `_source` | MLB 1,468 / 2,044 · NFL 859 / 25 · NCAAF 24 / 11 |
| `prospective_validation_plan` | 16 (NFL 4, NCAAF 4, the other four sports 2 each) |
| `prospective_event`, `_quote`, `_close`, `_result`, `_model`, `_calibration`, `_prediction`, `_validation_artifact`, `_deployment_review` | **0** |
| `prospective_football_theover` | 0 |

## What this means

1. **Football is where the evidence is.** NCAAF has 10,184 training rows and 319 settled results; NFL has 3,331 and 93. That is enough to train and freeze a model, but none has been bound to a plan.
2. **No predictions are being recorded against any plan.** Even a perfect model could not qualify today, because nothing writes its pregame predictions into `prospective_prediction`.
3. **G1 confirmed: no certified closes.** `prospective_close` is empty for every sport.
4. **G2 confirmed, including football v2:** no plan has a model or calibration.
5. **NBA, NCAAB and NHL have nothing in the canonical store.** Their captures stay in the per-sport research stores, and the research models have nothing to ingest. The research scheduler's timeouts (fixed by #2459) likely contributed. A `with_sources` run would show the per-sport stores.
6. **MLB** exists only as reconciled facts from the pregame receipts, with no canonical events or predictions.

## Next steps (roadmap)

- **C2 first, for NCAAF:**
  1. train on the football training rows;
  2. register the model and calibration;
  3. freeze a new plan version bound to them, under owner decision D11;
  4. record pregame predictions for each captured NCAAF event.
- **C1:** certified closes into `prospective_close`.
- **Football coverage:** verify full-slate coverage.
- **Scheduler:** confirm that the research scheduler now completes cycles with the cache.
