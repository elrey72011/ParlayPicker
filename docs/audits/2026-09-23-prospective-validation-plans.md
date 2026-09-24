# Six-sport prospective validation plan release v1

The reviewed policy source is `app_core/prospective_validation_plans.py`. It
declares one version 1 plan for each of the 12 exact sport and market families.
The policy is fixed before its future holdout starts. Installing it in a
canonical database assigns `frozen_at` at the database boundary and returns
the immutable database artifact hash. It does not read outcomes, register a
model or calibration, create a validation artifact, authorize stakes, or place
wagers.

## Frozen scope and chronology

Every market uses the same prospective calendar windows. Model fit results must
have been available no later than **2026-10-31 23:59:59 UTC**. Validation event
starts are **2026-11-01 through 2027-10-31 UTC**; untouched holdout event starts
are **2027-11-01 through 2028-10-31 UTC**. The database records the actual
freeze time and refuses a first freeze on or after 2027-11-01 UTC. Training
features and results must be available by their respective prediction and fit
cutoffs. Model selection may use training and validation evidence only.

The first release intentionally has `model_id=NULL` and `calibration_id=NULL`
for every market. This is an honest model scope declaration, not a placeholder
model or identity calibration. Fitted exact-market identities require an
explicit version 2 plan before that version's validation cohort is used. A
version must supersede the preceding exact-market plan; an old plan cannot be
rewritten. If the proposed calendar proves infeasible because legal evidence
is insufficient, a new prospectively frozen version must set new future windows.

| Sport/market | Plan ID | Policy source SHA-256 |
| --- | --- | --- |
| NFL/SPREAD | `prospective-nfl-spread-2026-09-23-v1` | `554aa9225e23f559ed8c5227749bdcae00aadb6dbe4ab8adb7381de7f23dbc48` |
| NFL/TOTAL | `prospective-nfl-total-2026-09-23-v1` | `7955ebdd9a560a7672fb80dd41b6e5744d8422263739b273d5210fa1d7799881` |
| NCAAF/SPREAD | `prospective-ncaaf-spread-2026-09-23-v1` | `786c2439765ddab55513826952b7e961e7e9508060d71790bb63fc2452fa11b2` |
| NCAAF/TOTAL | `prospective-ncaaf-total-2026-09-23-v1` | `a34953a970563337b6ef73ea4f74598f038fdaa46e805724fd5c2dfcfb64ba0c` |
| NBA/SPREAD | `prospective-nba-spread-2026-09-23-v1` | `fb5f9d702f18a6aa77097a3a2191afee6ce7c6f02edc6d353597fa8d82abcac5` |
| NBA/TOTAL | `prospective-nba-total-2026-09-23-v1` | `6fecc581d87dfcc0a0a5e8b83b62022bc2299457b3dff3b5aa86fba282e182d9` |
| NCAAB/SPREAD | `prospective-ncaab-spread-2026-09-23-v1` | `5153fee1920504c2bb057c1d359c37851c8b6feba0318136755161a1ef501ec7` |
| NCAAB/TOTAL | `prospective-ncaab-total-2026-09-23-v1` | `fb3818f106f4ea8e214bd3ce6513e18e1117b3deba963d90dc58a421a1c1bdcb` |
| MLB/RUN_LINE | `prospective-mlb-run_line-2026-09-23-v1` | `2b779afa39b6e55f5f300fe0826c848075c71ad57dea1525d309af761acf50c0` |
| MLB/TOTAL | `prospective-mlb-total-2026-09-23-v1` | `4e709fcde589a10cc2c3c1f78af228106caa4b6aab2d9e671a73c3bed7d26d76` |
| NHL/PUCK_LINE | `prospective-nhl-puck_line-2026-09-23-v1` | `6f5d935b1d9f74200ebb8873edca3a2e8dbc454659ba02e7779a48823da5652a` |
| NHL/TOTAL | `prospective-nhl-total-2026-09-23-v1` | `1e4b4d72d434289114ceb25fda6903a8c1f2a85bf327e0a33bc167c119934445` |

The policy source hashes bind the declared input before `frozen_at` and the
runtime `source_commit` are added. The actual canonical artifact hashes are
computed from the complete stored methodology; obtain all 12 from the freeze
command's output after authenticated restore. Repeated installation checks the
policy source hashes, returns the current plan's original `frozen_at` and
artifact hash, and cannot create duplicates. A reviewed later version can
supersede v1 without changing v1. A later code commit with unchanged policy
does not rewrite the original source commit.

## Minimum evidence and promotion policy

Each validation cohort and each holdout cohort needs at least **200 distinct
settled events and 200 effective decided events** in that exact sport and
market. The event is the independent unit. Additional books, reprices,
opposing selections, and repeated snapshots do not add units; a corrected
result updates the outcome revision and does not add a game. The evaluator
counts the first eligible prediction per event and reports raw predictions
separately. Spread and total, and run line or puck line and total, remain
independent families; neither may borrow sample size from another.

Both cohorts must independently meet Brier score at most **0.24**, log loss at
most **0.68**, expected calibration error at most **0.08**, and decided-outcome
coverage at least **90%**. These fixed proper-score ceilings are tighter than
an uninformative 50/50 Brier score of 0.25 and log loss of about 0.693. They
are predeclared screening thresholds, not evidence that any current model has
an edge. Predictions also require a fitted, exact-scope calibration artifact.

All included entries need an exact verified provider event, selection, line,
book, price, and pregame timestamp. The price coverage threshold is **100%**.
Certified, replayable, comparable closing prices are required for at least
**80%** of the independent events; this requirement currently blocks
validation because near-start research snapshots are not certified closes.
Missing closes report CLV as unavailable. The hypothetical one-unit paper ROI
must be at least zero in both cohorts. Accepted-wager ROI remains unavailable
without actual accepted wager and settlement evidence; no paper result is
presented as a realized wager return.

WIN and LOSS enter decided probability scoring. PUSH and VOID return the paper
unit and are excluded from decided scoring. PENDING and NEEDS_REVIEW block
validation. A half-point line has zero push probability unless an explicit
verified settlement exception applies. Probability scoring uses the decided
win probability `P(win)/(P(win)+P(loss))`, while the three unconditional
probabilities must still sum to one.

A passing report alone cannot promote a market. An immutable validation
artifact, independent deployment review, exact-market model and calibration,
and separate owner activation and exposure authority are still required.
This release targets only `PROVISIONAL_VALIDATED`, and all 12 markets start
`UNVALIDATED`, `production_eligible=false`, at `$0` stake.

## Installation after authenticated restore

Run this from the repository root against the **restored canonical database**,
using the full Git commit SHA of the reviewed running source:

```text
python -m scripts.freeze_prospective_plans --database /path/to/prospective-evidence.sqlite3 --source-commit FULL_40_CHARACTER_GIT_SHA
```

`--dry-run` prints only policy IDs and source hashes and does not write. The
normal command prints all 12 plan IDs, actual freeze times, source commits,
policy hashes and canonical artifact hashes. Preserve that sanitized JSON in
the run audit and verify its remote backup/read-back before claiming remotely
durable plans. Local plan rows alone do not prove the authenticated remote
store contains them.
