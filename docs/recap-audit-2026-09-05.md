# September 5 recap audit

Inputs: September 6 uploads `selected_best_available_results (1)`,
`graded_candidate_audit (8)`, `candidate_results_ledger (9)`, and
`player_prop_performance_recap (7)`. Game selections originate from export run
`20260905T184406Z`. Raw user exports are not committed.

## Observed results

| Scope | Wins | Losses | Decided hit rate |
|---|---:|---:|---:|
| Selected game card | 50 | 50 | 50.0% |
| MLB games | 11 | 4 | 73.3% |
| NCAAF games | 39 | 46 | 45.9% |
| Player props | 61 | 40 | 60.4% |

Four game rows were ungraded; 12 props were void. These are excluded from the
win/loss denominators. The selected results exactly match the audit's selected
candidate keys. Independent arithmetic checks reproduce all 377 decided candidate
grades and all 101 decided prop grades from the supplied scores/statistics and lines;
this does not independently verify score-provider
accuracy. All game rows were Best Available / Pass, with no approved wagers or
positive stakes, and all prop recap rows had `source_funded=False`.

The full candidate pool went 189–188. It includes opposing sides and totals and
cannot serve as the selector's accuracy scorecard.

## Findings and corrections

1. Louisiana–Lamar appeared twice: `Louisiana Ragin Cajuns` from the primary feed
   and `Louisiana` from ESPN fallback. Both selected Lamar +20.5 and lost 38–7.
   Whole-name aliases now share the existing Louisiana identity, preventing this
   duplicate at feed merging and candidate grouping. Other Louisiana schools remain
   distinct. Existing downloaded recaps are not rewritten: counting this game once
   gives 50–49 (50.5%), far below 75%.
2. Recap summaries labeled selection ranking scores as probabilities and used the
   `empirical_bucket_blend` score in expected-wins significance tests. These scores
   now remain visible as ranking scores but are excluded from that calculation.
   This preserves observed wins and losses and prevents unsupported regression
   claims based on a ranking-only expectation.

## Why the card was near 50%, and what 75% would require

NCAAF contributed 85 of 100 settled selections, 46 of the 50 losses, and lacked
independent target-specific ML probabilities on those rows. Eighty of the 100
settled game selections had ranking scores at or below 55%. The average score
was 0.5314, but this is not a calibrated expected hit rate. The coverage selector
still chooses a best available alternative when no wager qualifies; it is not
designed to make every game a high-confidence bet.

One MLB slate's 11–4 record does not establish a sustainable 75% rate. Nor should
the winning prop categories be selected retrospectively to manufacture that rate.
No threshold or model weights were fitted to yesterday's outcomes in this fix.

A defensible 75% target requires a separately evaluated, selective card: develop
target-specific NCAAF spread/total probabilities, validate calibration and thresholds
on chronological held-out slates, and report both hit rate and selection coverage.
Include price/return metrics so higher hit rate is not purchased with unprofitable
odds. The current one-slate evidence cannot validate such an improvement.
