# Private NFL calibration evidence preparation

Offline software only. No fitting, source registration, acquisition, qualification,
activation, wagering, deployment or merge. Frozen version 10 and all predecessor
policies, workflows, source catalogs, probability formulas and artifacts are unchanged.

`app_core.nfl_calibration_evidence.build_dataset` reads the existing private
capture/export SQLite evidence with `research_replay.read_export`. It verifies
original snapshot, source, export and package hashes, reuses the NFL input validator
and numerical replay, and exports private JSON plus SHA-256. It never initializes
or repairs an input database. The command is `python -m
scripts.prepare_nfl_calibration_evidence --database LOCAL_DB --specification
PRIVATE_SPEC --specification-sha256 HASH --output NEW_PRIVATE_FILE`. The output is
created exclusively and cannot overwrite prior evidence. No public/remote route
or new production artifact reader is added.

The specification declares export IDs, exactly one stage (`raw_model`,
`original_blend`, or `ui_refresh`) and its full predictor/pipeline binding. It
supplies separate independently reviewed exact-source/settlement decisions and
proposed assignments. Review file hash pinning establishes file identity, not
reviewer competence, source authenticity or scientific approval. All passing rows
are SOFTWARE_ADMISSIBLE_PROPOSAL; they cannot authorize fitting or evaluation.
The current `core.probability_calibration.inspect_calibration_artifact` and schema
version are reused for diagnostics. No calibrator is fitted or replaced.

Every observation retains the original exact event, named selected side, signed
line, price/book, operator/product decision, period/rules, quote and inference
clocks; raw probability, original blend and validated refresh are distinct. The
input binding includes ordered features, full dependency receipts and availability,
runtime, predictor callables, consumed artifact hashes/configuration and, where
applicable, blend or refresh code/configuration. The legacy calibrated_probability
name denotes a research blend here and is not a genuine scoped calibration.

Verified settlement document/outcome/availability must bind the exact original
contract and the separate review. Final scores alone do not qualify a market
settlement. A reviewed canonical-game crosswalk is required for independent-game
counts. Repeated exports do not multiply rows; duplicate offers/variants/opposite
sides do not add games. Proposed selected-contract sampling and season/week groups
are required. Home/away teams, side, line/sign and kickoff remain explicit. Approved
assignment declarations are preserved separately and cannot be overwritten by a
proposal. Prior inspected outcomes block validation/holdout regardless of a new
uninspected assertion. These are necessary software checks, not custody approval.

## Inspected retained historical evidence

The owner's local prediction database has zero snapshots. The named September 15
native NFL record is one score-only game (Denver at Kansas City), with no original
inference, ordered features or exact-offer settlement/source review. It supplies
zero admissible observations to all four proposed roles. The retained September
30 readiness ledger lists 16 NFL/SPREAD inventory games, but zero canonical model,
calibration and prediction rows. It does not contain the complete canonical record
bytes needed for this target. Its 16 games therefore supply zero *demonstrated*
admissible development/calibration/validation/holdout observations; the larger
canonical population's actual admissible counts remain UNKNOWN. These inventories
are overlapping evidence classes and must not be summed.

The separately documented 2015–2022 development (2,079 games), 2023 residual-scale
calibration (272), and 2024–2025 evaluation (544) concern the rejected historical
scoring regression. Historical schedule prices lack original observation clocks;
features differ from the consumed live predictor. These counts cannot be moved to
this calibration binding. The 2024–2025 outcomes have been inspected and can never
be relabeled untouched. A byte-preserving read-only audit of the three named local
inputs is retained privately by this task, including their hashes and exclusions.

## Chronological evaluation specification — proposal only

1. Model development may use only reviewed historical or prospective inputs whose
   features were available before each original quote/inference. Demonstrate original
   model availability and out-of-sample exclusions before assigning any calibration
   role. Existing Stage 2 assignments remain separate: whole-event kickoff ties,
   55/15/15/15 and independent-event floors 200/60/60/60 are unchanged.
2. Freeze the target, source/operator/product and settlement interpretation, predictor,
   ordered feature/missingness contract and chosen calibration input stage. Earlier
   development outcome availability must strictly precede every calibration quote.
   Calibration fitting requires separate approval of objective, solver, support and
   disjoint original predictions. The prior proposed Platt design remains a proposal;
   this task makes no objective/solver decision and invokes no fitting.
3. Freeze genuine fitted calibration and uncertainty/reader identities before
   validation. All earlier outcomes must be available before any later-partition
   quote, including when an intermediate partition is empty. Keep whole games,
   simultaneous kickoff groups and revised outcomes together. Report attempted,
   failed, excluded, duplicate and decided denominators plus season/week/team/side/
   sign/line support. Do not convert event deduplication into a dependence-adjusted
   statistical effective sample size. Dependence and multiplicity decisions need
   scientific review.
4. Untouched holdout requires an independent custodian's sealed sampling/artifact
   manifest and access history fixed before outcomes. Any previous inspection or
   role conflict blocks the claim. This builder's owner-visible output is a readiness
   proposal; protected holdout outcome custody needs separate accepted tooling and
   review before an outcome-bearing export is shared with developers.

Frozen V1 is preserved: fit availability by 2026-10-31T23:59:59Z; validation
[2026-11-01,2027-11-01); holdout [2027-11-01,2028-11-01). Both evaluation cohorts
require 200 distinct settled and 200 effective decided events, Brier ≤ .24,
log loss ≤ .68, weighted ten-bin ECE ≤ .08, decided coverage ≥90%, exact entry
coverage 100%, certified comparable close coverage ≥80%, paper unit ROI ≥0 and
zero unresolved integrity/outcome failures. Football V2's registered precision,
class-support, chronological and calculated sample floors also remain unchanged.
Novig FVS cannot be priced as a refund or with binary EV. Any successor target,
calendar or value design is a separately reviewed proposal; no qualification date
follows from this specification, synthetic fixtures or CI.

## Verification and dependency

Synthetic tests traverse native intake, actual analysis, UI refresh, terminal PASS,
capture, export and private replay before building a dataset. They accept one
synthetic proposed calibration observation from two opposing offers, preserve all
stages and reject mismatched bindings, altered feature order, future predictor
availability, missing/rehashed conflicting settlement, rights gaps, corrupt exports,
approved-role conflicts and used-outcome holdout contamination. Network is blocked.
The NHL draft is independent on the same verified main and shares the scope guard;
the owner must reconcile the second draft with actual merged main and create fresh
bindings/seal/CI before a second merge. No merge is performed by this task.
