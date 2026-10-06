# Private NCAAF spread and total replay contracts

This scope adds a private reader/export adapter and synthetic actual-caller
harness to the existing NCAAF prospective workflow. Normal board inference,
public display, sources, request limits, the frozen Stage 1 cohort, fitting,
model parameters, qualification and wager authority remain unchanged. NFL and
the merged MLB/NHL corrections are preserved.

## Targets and probability stage

`ncaaf-selected-full-game-discrete-spread-v1` represents the named selected
home/away margin plus that side's signed handicap. A positive result wins,
zero pushes, and a negative result loses. `ncaaf-full-game-discrete-total-v1`
is separate: over uses total minus line; under uses line minus total. Both
require independently reviewed full-game terms including overtime. Regulation
and partial-game offers are rejected. Only integer and half-point lines are
supported. A winner classifier cannot supply either target.

The adapter consumes the existing `ncaaf-research-v1` frozen margin and total
models. Its seven ordered features are home/away points for, home/away points
against, home/away yards, and the known neutral-site flag. Ridge standardization,
coefficients, residual bias and scale remain those of the consumed artifact;
constant and scoring baselines retain their own explicit model identities.
No parameters or calibrator are fitted here.

For integer lines, the existing reader integrates the discretized score mass
at the push threshold. This is positive model-assumed push mass, rather than a
continuous distribution's zero mass. It is not empirically validated push
calibration. Half-point push mass is structurally zero. Unconditional win,
push and loss remain separate from conditional win given no push. The raw
artifact probabilities and original prospective EV are retained; no original
board blend or UI refresh is claimed. A Novig offer cannot use this binary
win/push/loss payoff contract because the existing FVS restriction still applies.

## Original capture, private export and explicit replay

Use an existing immutable records download or coherent owner SQLite snapshot.
`read_records` uses `mode=ro&immutable=1`, rejects pending WAL, and never calls
the store's initializing connector. Missing access does not mean an empty store.
`export_observation` accepts an exact capture/event/model/market/line/book/price/
quote-time selection plus a separate source review. Export creates no source
registration or approval. Source review binds the original capture and model
record, artifact, inputs, event, quote, features and target hashes, listing,
operator/product, permitted private use, effective interval and review clock.
Period/rules facts must already exist in the original captured offer. A later
assertion or generic rule template cannot repair missing historical facts.

Export verifies immutable record hashes, exact named CFBD/odds identity, kickoff,
pregame capture, original quote and retrieval clocks, artifact/runtime/reader
identity, target schema and feature reconstruction. Original allowlisted CFBD
inputs and their dependency receipts accompany the private packet. Full raw
dependencies and private numbers are never added to a public package or GitHub.
The packet remains research-only with zero live stake. Prospective v1 retains a
capture-finished clock but no exact per-model inference clock. Export preserves
the finish clock and explicitly marks the original inference time missing; it
does not substitute the finish timestamp. A subsequent separately reviewed
prospective capture design must record that clock for stronger input lineage.

`replay` is explicit numeric execution through the installed existing
`centers`/`probabilities` reader. It verifies the export contract, raw three-way
probabilities and original binary EV. Caller authorization for execution is
required separately from static inspection. The task's authentic historical ZIPs
are inspected statically only; they are never executed by this harness. Every
acceptance test is marked `SYNTHETIC` and captures with a local fake response
through the actual `ncaaf_prospective.capture` caller, then reads, serializes,
exports and replays the result under blocked sockets/DNS/HTTP. No fit is used.

## Readiness and custody

Existing requirements remain the fixed 2023 training, 2024 residual calibration
and once-evaluated 2025 holdout, minimum 50 eligible games per split, minimum
three same-season scoring and yardage histories for both teams, and strictly
greater-than-seven-day lag. The historical publication/correction timestamps
remain unverified; a successful replay does not improve that status. Evaluated
2025 outcomes cannot become untouched. Any new tuning requires a separately
reviewed future chronological validation/untouched custody design.

Prospective inputs must be genuinely observed before inference, at most 24 hours
old; exact quote clocks must be at most 15 minutes old; capture must finish
before kickoff within the existing seven-day window. Original game IDs, team
IDs, season/week, starts, final scores, yardage records, neutral-site facts and
their retrieval clocks are needed. Compatible frozen artifacts need original
training/calibration fingerprints and an independently reviewed out-of-sample
and source-permission bridge. Those fingerprints alone do not demonstrate
historical point-in-time availability or scientific acceptance.

The full board inventory remains ESPN FBS plus FCS. The native CFBD research
collection and frozen Stage 1 FBS-only capture cohort are different inventories;
FCS policy exclusion is not provider omission. Current quotes do not prove full
schedule coverage. No new refresh or capture is requested by this software PR.

## Review boundary

The successor binds verified actual merged main, exact additive files and the
guard wrapper. A subsequent policy-only seal binds the implementation commit
and tree. Predecessor guards are reconstructed byte-for-byte, and their policies,
original baseline, frozen version 10, scientific requirements, workflows and
existing tests remain protected. Negative fixtures cover identity/quote changes,
duplicate games/offers, altered features, missing/future/stale dependencies,
runtime/target/reader conflicts, missing source/listing/rights/period semantics,
Novig payoff restrictions, corrupt/re-signed packets and unchanged default-board
unavailability. CI and these fixtures confer no qualification or wagering authority.
