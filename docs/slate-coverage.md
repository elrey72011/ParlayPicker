# Independent slate coverage v1

Coverage is informational. Existing wagering actions, approval readers, scientific
requirements, exposure limits and zero-stake PASS behavior remain authoritative.
No schedule is downloaded by this reader, date selector or reconciliation command.
MLB development remains paused. The reusable inventory schema enables no provider
or model for any other league.

## Bounded findings

Verified main is `e8e36e303bb663442702023cad2b2939854cf411`, the actual #2400 merge.
Ordered parents are `13cb72786d5b2e46987fd779b5318be0ffe55b52` and
`fb4b66b224669009d20b45f1f7df1c27a7158ed5`. Its tree is
`7197b4c3389e1c11b0c31d14951dc1679e9495eb`, identical to the reviewed seal.
Post-merge scope, production safety, workflow, shard 1 and operations checks passed.
Shard 2 was cancelled; the full-suite aggregate failed for that cancellation.
This is not a passing full-suite baseline and no workflow was rerun.

The labelled example in `docs/examples/slate-coverage-synthetic.json` reconciles
five independent synthetic events: zero APPROVED, one policy PASS, four UNVERIFIED,
one separate orphan, and no missing/duplicate/extra decision IDs. One evaluated
market remains PASS inside a game with unresolved required markets. COMPLETE here
describes the synthetic inventory index, not evaluation or any real slate.

Before this change:

- `core/streamlit_pipeline.py::_fetch_live_odds_dataframe` continues on empty
  provider results and filtered-empty games. Provider health distinguishes request
  failures and empty successes, but those exits have no surviving game row.
  Downstream normalization and candidate filters can also remove games.
- `app_core/football_identity_capture.py::collect` already makes bounded identity
  reads, but discarded their unmatched schedule observations. This change retains
  the same observations and sanitized request outcomes without adding requests.
  The existing public `fetch_live_odds_dataframe` attaches the retained receipts;
  the normalizer's original provider-health source anchors stay intact.
- `app_core/game_coverage.py::unranked_games` derives its denominator from the
  candidate audit. It retains missing selections but cannot discover games lost
  before candidate generation. Candidate audits remain useful observations, not
  independent schedules.
- `streamlit_app.py` fed surviving selections to Today/Pick Details while
  publication appended audit-derived placeholders. The two paths differed. An
  exception could fall back to selections alone. Coverage mismatches now fail
  explicitly instead of silently constructing a partial publication frame.
- `app_core/per_game_boards.py::per_game_board` and
  `app_core/public_board.py::build_package` enumerate supplied rows; neither could
  recover absent schedule events. They now carry the reconciled informational
  decisions through the three exports and public package. The browser retains
  unavailable schedule-only cards and keeps coverage separate from wager status.
- `core/run_readiness.py::build_readiness` used the candidate denominator and
  returned early for an empty audit. Independent coverage is now attached before
  that return. Its grading readiness remains a distinct contract.

The retained October 6 trace (224,507 bytes; SHA-256 `ca0dd283c2c436b4341dbc0e89972566f06d958fff167b1fb178c99671319c18`)
has no NFL rows; its 48 candidates
(36 NHL, eight MLB, four NCAAF) do not establish a complete schedule. The older
current-wager audit similarly describes selections, not omitted games. Native
NCAAF schedule completeness is established only by the existing FBS/FCS
scoreboard/index requirements in `app_core/ncaaf_schedule.py`. Existing NFL
identity queries are bounded, quote-derived UTC dates; their observations are
PARTIAL even after a successful response. Absent retained independent inventories
are UNAVAILABLE, and inaccessible hosted storage remains UNKNOWN. No original
packet is executed or reclassified as scientifically accepted.

## Inventory and decision contract

`slate-inventory-v1` records league, source, selected ISO Eastern date,
`America/New_York`, aware original observation clock, completeness basis,
status/reasons and canonical events. Events retain named home/away teams and IDs,
original aware start, provider IDs and known revisions. The content digest binds
the inventory; projections retain the original source digest and window.
Different event IDs retain repeated matchups and doubleheaders. Named orientation
and start must agree even when an ID is present. Ambiguous joins create orphan
receipts and blockers, never a guessed schedule match.

COMPLETE requires the source's recorded window/index completeness requirements
and verified inventory clock. PARTIAL records limits, interruptions or missing
facts. UNAVAILABLE contains no usable events. An explicitly COMPLETE empty index
can reconcile as a legitimately empty day; an empty candidate audit cannot.
Inventory completeness and completed market evaluation are different questions.

The declared default required evaluation scope is selected home spread, selected
away spread, total over and total under. A caller may explicitly declare a smaller
scope; that scope is retained in every decision. No implicit scope reduction
occurs when candidates are missing. FBS/FCS inventory membership does not change
Stage 1's frozen FBS-only policy. A policy exclusion resolves only its declared
scope and needs its exact policy ID and reason; it supplies no approval. Its
`scope` must name the applicable market or list exact market names. A separately
recorded `stage1` cohort exclusion cannot resolve the research market scope.

Game aggregation:

1. APPROVED if any candidate has an exact current finalized strict approval,
   positive finalized stake, matching event/offer/run identity, and current
   original inference/quote clocks through existing approval readers. Saved
   labels, legacy metrics, positive EV, research estimates and synthetic fixtures
   cannot create authority. Controlled-trial status remains separate.
2. Otherwise PASS only if every required market is resolved by an observed
   rejection without an incomplete prerequisite, or an explicit referenced
   policy exclusion resolves the declared scope.
3. Otherwise UNVERIFIED. A PASS market remains visible when another market is
   unresolved. A resolved candidate cannot lend any field to another candidate.

Read-only coverage checks have a documented order: schedule identity, pregame,
provider request, odds match, quote clock, candidate generation, model evidence,
finalization. These describe retained observations, not invented execution of the
wagering pipeline. Actual finalizer receipts separately retain market authority,
observed maturity blockers, candidate-contract failures in returned order and
selection-identity safety. Their original run, evaluation clock and exact ticket
bind the receipt. Individual passes inside an aggregate contract are not inferred.
Finalization's original run is retained separately as `finalization_run_id` when
the existing capture mechanism subsequently records a different export clock.
That clock change does not replace original inference or quote timestamps.
`actual_gate_results` and `first_actual_failure` expose the actual caller order;
`evaluated_offer` appears only for an exactly bound evaluation receipt.
Skipped gates are NOT_EVALUATED; absent evidence is UNKNOWN. All observed failures
and the first in each recorded order survive. Missing prerequisites produce
UNVERIFIED, including missing/incompatible model evidence. An eligible but unfunded
candidate does not acquire an invented rejection or approval.

The report strictly reconciles scheduled canonical IDs with decisions: missing,
duplicate or extra IDs raise `COVERAGE_INTERNAL_RECONCILIATION_MISMATCH`. Provider,
candidate and final rows without a verified schedule match remain separate orphan
receipts. Equality never promotes PARTIAL/UNAVAILABLE to COMPLETE.

Coverage-only placeholders retain identity, clocks and reasons, never selection,
quote, probability, EV or stake. They have no inference timestamp or wagering
contract. Existing evaluated candidates and private original fields remain intact.
Public coverage uses an explicit allowlist and excludes raw dependencies. The
public browser demotes a saved coverage APPROVED label when its actual wager
predicate is no longer current; saved bytes remain unchanged.

## Use without analysis

Advanced controls → Coverage Eastern date selects the independent retained slate.
Changing it reconciles retained diagnostics only; no acquisition or analysis is
triggered. Today, Pick Details and the publication preview show Independent slate
coverage and JSON/CSV downloads. Public packages carry the identical report and
decision rows. Legacy other-league display remains outside the declared inventory.

The adjacent command opens no database and performs no network access:

```sh
python scripts/reconcile_slate.py --inventory retained-inventory.json \
  --date 2026-10-07 --as-of 2026-10-07T16:00:00Z \
  --run-id 20261007T160000.000000Z --output coverage.json
```

Optional candidate/final CSVs, original provider JSON and private finalizer receipt
JSON bind to the same run. `--required-market` records an explicit evaluation scope;
`--policy-exclusions` accepts retained policy references, not new authority.
It emits JSON plus CSV. Exit 1 is invalid/mismatched
internal reconciliation; 2 is PARTIAL; 3 is UNAVAILABLE; 0 is internally reconciled
COMPLETE inventory, regardless of the number of approved wagers.

## Separate model, source and scientific work

The original recovered NCAAF model record
`881833a03169d8ff3cfdc7cfb6000a174ccb545b1d087657169fe06d5f8a1153`
has unchanged 2,850 original bytes and hash. It derives from model `6d7291fd…`
with unchanged artifact `9334f2a4…`, parameters, protocol and train/calibration
hashes. Its recovery explicitly has `production_eligible=false`. Source runtime
`bd837b0b…` became `d5acdd2c…` under the reviewed NFL-alias-only transition.
Current five-component source digest is `8283b60a…`: NCAA research/history/math
components are unchanged, but NCAA identity now includes McNeese, LSU and
Gardner-Webb aliases; the shared mapper preserves Rangers/Islanders names. The
source digest is not a hosted runtime attestation. Python/NumPy and dynamic alias
state are not established by this model record.

V1's reader rejects the recovery envelope (`NCAAF_MODEL_SCHEMA`) and then the
current runtime differs (`NCAAF_FROZEN_RUNTIME_MISMATCH`). Recommend a separately
reviewed versioned exact compatibility envelope preserving original record,
recovery/predecessor metadata and hashes, plus event-specific identity review and
a distinct current-reader binding. Do not strip recovery, replace hashes or grant
a blanket exception. The seven-day lag, three-game minimum, named orientation,
neutral-site facts and unvalidated integer push assumptions remain frozen.

| Blocker | Kind | Smallest next evidence/action | Responsible owner |
|---|---|---|---|
| Independent NFL schedule not retained | Operational evidence | Existing canonical selected-date inventory with source/window/index and original clock; inaccessible hosted contents stay UNKNOWN | Producer owner |
| NCAAF recovered model incompatible | Software/model compatibility | Separate versioned compatibility review of the exact original model and event identity changes | Model maintainer and independent reviewer |
| Missing original features/model dependencies/clocks | Research evidence | One existing complete native observation chain; preserve missing historical facts | Producer owner |
| Unaccepted listing/product, period, rules and permitted use | External source evidence | Offer-specific authoritative provider/operator response and its referenced document | Source owner/provider |
| Calibration/validation/holdout and qualification incomplete | Scientific | Existing governing chronological protocol and independent acceptance; no fit or threshold change here | Scientific reviewer |
| Current activation or wagering authority absent | Authority | Separate explicit owner authorization through existing gates after qualification | Owner |

No number of fixtures, reconciled rows, displayed estimates or passing software
checks grants scientific qualification, a qualification date or wagering authority.
