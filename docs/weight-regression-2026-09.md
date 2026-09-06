# Weight regression analysis: August 29–September 5, 2026

## Method and scope

Eight candidate-audit exports supplied 986 rows across 261 event identities.
The analysis excludes 35 events with unknown/non-pregame timing, removes the
Louisiana duplicate fallback event, and excludes nine events whose alternative
candidates were not all graded WIN/LOSS. Remaining evidence is 828 candidates
across 216 games. Pushes and ungraded alternatives are not imputed as losses.

Fit window: August 29–September 2, 92 games / 358 candidates.
Evaluation window: September 3–5, 124 games / 470 candidates.
Whole games and dates stay together; each game has equal total fitting weight,
regardless of how many candidate alternatives it contains. The two ridge-logistic
specifications use fixed regularization C=1. The blend constrains weights to be
nonnegative and sum to one, renormalizing over available signals.

These dates were already inspected during prior work. This is a chronological
historical evaluation, not a pristine prospective holdout. Comparing several
specifications also introduces model-selection uncertainty. No live weights or
saved production model are modified by this analysis.

## Candidate selection results on the later dates

| Formula | Wins–losses | Accuracy | Candidate log loss |
|---|---|---|---|
| Exported selector | 64–60 | 51.6% | 0.6889 |
| Market probability only | 66–58 | 53.2% | 0.6885 |
| Ridge logistic: market + ML + TheOver | 63–61 | 50.8% | 0.6881 |
| Ridge logistic: signals + existing ranking score | 64–60 | 51.6% | 0.6876 |
| Fitted nonnegative signal blend | 69–55 | 55.6% | 0.6885 |

All selectors choose one candidate from each of the same 124 events. Log loss is
averaged with equal event weights over all alternatives; the exported selector's
ranking scores provide only a diagnostic comparison, not calibrated probabilities.
The blend changed 11 losses to wins and six wins to losses. An exploratory paired
exact test gives p=0.332; it does not establish a reliable improvement and does not
account for dependencies across games on the same slate.

The fitted three-signal formula is:

`p = (0.61656 * market + 0.38344 * independent_ML) / sum(available weights)`

TheOver's fitted weight is zero in this sample. This is a candidate for further
validation, not a recommended production replacement. Earlier expanding fits
varied from 76.8% market / 23.2% TheOver to 100% market before settling on the final
market/ML mix. Such instability argues against precise fixed production weights.

On the later dates the blend produced MLB 25/38 (65.8%) and NCAAF 44/86 (51.2%).
The exported selector produced MLB 25/38 and NCAAF 39/86. Thus all five net added
wins came from NCAAF; this did not improve MLB's aggregate record. Only five blend
selections exceeded 60%, and three won. None reached 65%, 70% or 75%.

## Full production-signal fit

Exact run/league/team/pick joins recovered all 92 training and 124 evaluation
selected rows with oriented `blend_in_*` values. This allows fitting the original
four inputs but cannot evaluate alternative-pick identity: older candidate audits
omit the full oriented inputs for unselected alternatives.

The existing unconstrained-in-practice simplex optimizer concentrated nearly all
weight on TheOver, which was present in only eight training rows. Tiny residual
weights carry the missing-TheOver rows through renormalization, so that boundary
solution must not be read as a recommendation for 100% TheOver everywhere.
Evaluation log loss worsened to 0.7052 from the selected rows' exported-score
benchmark of 0.6857. Its six selections at 75% or higher went 3–3.

## Changes and next evaluation

The candidate audit now preserves oriented Kalshi, market, TheOver and ML blend
inputs plus blend tier for both selected and rejected alternatives. This removes
an avoidable data gap for future weight fitting without inventing missing signals.

`scripts/analyze_candidate_weights.py` reproduces the candidate regressions with
explicit input pattern, training cutoff and optional JSON output. Regression tests
verify that changing later outcomes cannot alter fitted weights, repeated downloads
do not increase sample counts, and started/incomplete games are excluded.

Keep current live weights pending new evidence. Freeze a candidate specification
before collecting new slates and compare accuracy, coverage, proper scoring metrics
and realized returns at offered odds. Separate sport/market weights require larger
samples than this eight-slate collection. The current evidence does not support a
75% claim or disabling TheOver globally.
