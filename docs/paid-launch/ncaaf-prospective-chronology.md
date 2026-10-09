# Explicit NCAAF prospective admission chronology

This bounded successor starts from verified main
`bb696fc7f8da436c9f7c9658756a254f8fdef29f` (reviewed #2406 tree
`f5850dd4c84ea899e624f36322e999c6e71c8287`). Its 13 post-merge checks passed;
those unchanged baseline receipts are distinct from this candidate's fresh CI.
Owner-reported preview success does not prove a hosted revision or an authentic
research observation. Original private planning and failure receipts stay local.

## Reproduced contradiction and explicit successor

The unchanged `ncaaf-compatible-prospective-inputs-v2` reader requires an exact
quote hash and `source_review.reviewed_at <= quote.recorded_at`. A labelled
synthetic quote at 12:00:00Z, local observation at 12:00:01Z, exact verification
at 12:00:02Z, independent acceptance at 12:00:03Z and intended inference at
12:00:04Z rejects with `NCAAF_COMPAT_SOURCE_REVIEW_CLOCK_CONFLICT`. The original
rejection and packet were saved before implementation. V2 cannot represent the
separate observation and acceptance receipts. Its file, behavior and all old
packets remain unchanged.

Explicitly selected `ncaaf-compatible-normal-inputs-v2` instead contains an
original `ncaaf-compatible-prospective-inputs-v3` observation and the existing
native dependency objects. Its static reader copies the frozen common v2
model/event/feature/dependency checks and adds `ncaaf-prospective-admission-v1`.
It never strips recovery metadata, repairs a v2 packet or backdates a review.
Results use `ncaaf-compatible-normal-result-v2`; dispatch through the existing
caller, capture/export, private replay, per-game and coverage readers is explicit.

## Clock and trust contract

| Receipt | Meaning and required ordering |
|---|---|
| `terms_review.reviewed_at` | Independently trusted advance review of the operator/product/listing scope, rule edition, rule and rights document SHA256s and collection/private-retention/research/public-derived-output permissions. Strictly before local quote collection. Terms may be reviewed before their effective start. |
| `quote.recorded_at` | Original provider market-last-update clock. Supported only with the independently reviewed `provider_clock_field=recorded_at` and `provider_clock_meaning=provider_market_last_update`. It is never relabelled as local observation. |
| `quote_observation.observed_at` | Actual local observation of the exact original quote SHA256, no earlier than its provider clock. Unknown/missing meaning or time remains unavailable. |
| `offer_verification.verified_at` | Strictly after observation. Covers the exact quote, observation, advance-terms and event-mapping review hashes and effective rule edition. A nonempty verifier identity is retained. |
| `acceptance.accepted_at` | Trusted independent acceptance, no earlier than verification or event-mapping review. Reviewer must differ from the offer verifier; the trusted catalog binds the exact receipt, not a self-asserted independence flag. |
| `as_of` | V3 admission checkpoint, no earlier than acceptance and no later than inference. This new meaning does not change v2 semantics. |
| `inference_time` | Genuinely new clock from the existing normal caller. Acceptance must strictly precede it. All receipts are rechecked at this clock, including effective terms, freshness and pregame restrictions. |

The advance terms' effective interval must cover provider quote through inference;
its end is exclusive. Original 900-second quote freshness, seven-day prospective
event window, pregame restriction and future-clock rejection remain unchanged.
CFBD advance permission must separately precede every native batch's recorded
capture clock and remain effective at inference. It contains no future response
or packet hashes. Existing 24-hour input age,
three histories per team and strict seven-day history lag remain unchanged.

Terms bind the applicable listing/product/operator, full-game period and binary
win/push/loss overtime settlement. The exact quote hash additionally binds named
event orientation, selected target, signed line, American price and original
quote identity/clock. Changed prices, lines, listings, products, clocks or events
invalidate the earlier verification and/or acceptance. Only half-point selected
side spreads and separately targeted half-point totals are supported. Novig/FVS,
integer pushes, other models and unreviewed aliases remain unavailable.

Acceptance's `subject_sha256` covers the complete v3 payload excluding only
`source_review.acceptance`; its `subject_version` is exactly v3. The receipt's
own digest is bound in a separate trusted catalog. This avoids circular hashes
and binds the original model/predecessor, ordered features, dependency hashes,
mapping and all earlier receipts. Rehashing an attack cannot update trusted
acceptance. Features and dependency availability must precede or equal acceptance;
an accepted subject cannot contain not-yet-available inputs. A v2 packet cannot
borrow v3 admission.

Production `ACCEPTED_TERMS_REVIEWS`, `ACCEPTED_ADMISSIONS`, original event/source
catalogs, `ACCEPTED_DEPENDENCY_PERMISSIONS`, `ACCEPTED_DEPENDENCY_ADMISSIONS`
and exact packet/public-output catalog remain empty. Staging an owner
packet performs existing bounded static inspection only; it creates no review,
catalog entry, storage, request or inference. Synthetic accepted entries exist
only under monkeypatch in labelled tests. No accepted-source registration occurs.

## CFBD review closure: advance permission is not an exact-byte review

At original reviewed head `0dfcfd71d1d16f39511f742bed80075e73996d23`,
`dependency_source_review` simultaneously required exact `dependency_hashes`
and `reviewed_at` before every batch's `retrieved_at`. The actual normal caller
rejected a truthful 12:00:02Z exact review of 12:00:00Z captured synthetic bytes
with `NCAAF_DEPENDENCY_ADVANCE_PERMISSION_CLOCK_CONFLICT`, before inference.
That original rejection and packet are retained privately. Advance review
cannot know previously unknown response bytes; its receipt must not be amended
or backdated after capture.

Only the explicit successor uses `ncaaf-prospective-dependency-admission-v1`
inside the existing trusted packet's `dependency_source_review`. Its separate
immutable receipts are:

| Receipt | Binding and chronological meaning |
|---|---|
| `permissions_review` | Independently trusted before capture. Provider `cfbd`, allowed endpoints `games`/`games/teams`, season, effective interval, rights edition/document SHA256, collection/private-retention/feature-research/public-derived permissions and known native clock meaning. No dependency or future packet hashes are allowed. |
| Native `retrieved_at` | Actual local native-batch capture clock, with explicitly reviewed meaning `local_native_batch_capture`. Original native bytes and clocks are unchanged. This does not claim original provider wire-response custody. |
| `dependency_verification` | Strictly after every captured object and no earlier than every included fact's availability/review clock; binds `permissions_sha256` to the unchanged advance receipt, exact ordered dependency hashes and versioned pre-admission `subject_sha256`. Existing byte integrity, canonical identity, scope, history and feature checks remain required. |
| Dependency `acceptance` | Separately trusted independent receipt binds the exact verification digest and pre-admission subject/version. Reviewer differs from verifier. No earlier than verification, no later than the complete offer admission, and strictly before actual inference. |

The permission's effective interval covers all captured objects through actual
inference, with an exclusive end. Unknown clocks, untrusted permissions or
acceptance, altered linkage, future inputs, legacy-shaped receipts, out-of-scope
seasons/endpoints, and rehashed attempts to manufacture trust reject before
normal inference. An old v2 packet cannot borrow successor admission.

The labelled fixture creates and trusts permission at 11:00:00Z before producing
previously unknown native objects at 12:00:00Z; it verifies them at 12:00:02Z,
independently accepts at 12:00:03Z, then calls normal inference at 12:00:04Z.
It proves the advance receipt and digest remain unchanged through normal caller,
private capture/export/replay, static reader and browser rendering. Exact
dependency hashes first occur in later verification, never in advance review.
The original v2 reader and its dependency-review behavior remain unchanged.

Dependency subject `ncaaf-prospective-dependency-subject-v1` includes the exact
original model, event/offer, available feature/dependency facts, earlier source
receipts and native object bytes. It excludes only the later offer-acceptance
receipt and actual admission checkpoint `as_of`, plus the enclosing observation
digest that depends on them. This hash projection does not alter any packet or
receipt. The existing trusted final-packet catalog separately binds the complete
original packet, including those later clocks, before inference. A regression
constructs the dependency subject while acceptance and checkpoint are still
absent and proves its digest survives their later addition unchanged. An exact
verifier never needs a future final-packet hash. Altered prior facts or bytes
still change the subject and invalidate earlier verification/acceptance.

### Dependency-subject availability closure

At reviewed head `26558147dec815c72da947f0782d4b4f23289e8e`, the actual normal
caller accepted a fully valid labelled synthetic packet with dependency
verification at 12:00:00.5Z, although its hashed subject included observation
at 12:00:01Z, feature availability at 12:00:01.5Z and offer verification at
12:00:02Z. One call to the unchanged numerical function occurred. No other
gate rejected that sequence. Its original before-correction receipt remains
private and unchanged; this is not authentic historical inference.

The successor now rejects this sequence before numerical inference with
`NCAAF_DEPENDENCY_SUBJECT_FUTURE_FACT`. Known, parseable clocks for the provider
quote, mapping review, feature availability, each dependency's availability,
advance terms review, local observation, exact-offer verification, original
model creation and predecessor creation must all be no later than dependency
verification. Model clocks are read through the existing exact static model
reader; hash or lineage failures keep their existing reasons. Native captures
remain strictly earlier than verification. The valid 12:00:02Z verification
may cover facts available at 12:00:02Z; later independent acceptance at
12:00:03Z and inference at 12:00:04Z remain separate.

No acceptance or checkpoint hash/clock is added to the dependency subject.
Target kickoff and terms' future effective end describe applicability, not
when evidence became available. Their existing pregame/effectivity checks
remain unchanged. This check does not manufacture unrecorded schedule/provider
response clocks or replace independent mapping/source review. Tests cover
future and missing included-fact clocks, rehashed/trusted synthetic receipts,
the unchanged v2 route, immutable input/review objects, precise UNVERIFIED
coverage, and the unchanged caller/capture/export/display probabilities.

## Original shard-3 report reconciliation

No test rerun was used to reconcile original application CI run `37967106189`,
job `113944081861`. The original log says **2,012 passed tests and 38 passed
subtests**. The XML summary's **2,050 = 2,012 + 38** counts those passed subtests;
the XML has 2,012 top-level testcase nodes and the immutable assignment manifest
has 2,012 selected/completed identities. Subtests do not create another 38
shard-assigned tests. The initial reporting concern is explained, not an
omission or duplication. Raw reports remain unchanged.

- Original ZIP SHA256: `fd36a33009f3597d11afffcaccfc068aedcab54f2d895fb00c84b96b4aa445d3`.
- Original log SHA256: `bf5bc83e462668c8f38d04343aaf3af17f50cf5b3ce9b05498b2db1bf685e71e`.
- XML SHA256: `419f1d0aebd7ed6e14b5f184e7a20e6e8d7dd3266143b14be847eb7829a8fadb`.
- Manifest SHA256: `03ab77e4a7d7260607c1b102f655803d68c4083e8f13551fcc2ede0b8550843d`.

This reuses the original passing run only as predecessor/reporting evidence.
Changed implementation and bindings require complete fresh final-head CI.

## Custody and remaining boundaries

Original model `881833a03169d8ff3cfdc7cfb6000a174ccb545b1d087657169fe06d5f8a1153`,
predecessor `6d7291fd464a9a367ba060ccf7bd526708bdba7b5809657930e1616f026173c0`,
artifact `9334f2a4904e9cdb88f0b50563c058b846647d876a72e46a0548035a27c2d25e`,
recovery, protocol, parameters, historical runtime declarations and hashes remain
unchanged. Current reader/source hashes and actual Python/NumPy/runtime are
retained separately. Research mathematics, ordered features, scientific
calendars and qualification requirements remain frozen.

The existing private capture retains the exact original nested packet/native
dependency bytes and separate admission receipts plus genuinely new inference
receipt. Capture/export/replay uses existing mechanisms; browser rendering is
static and repeats no inference. Raw native uncalibrated probability, original
blend and any UI-refresh receipt remain separate. No new UI-refresh receipt is
invented. Public output excludes private admission/model/dependency objects;
value-display restrictions and actual wagering gates remain unchanged. Failed
successor inputs retain precise UNVERIFIED coverage blockers and wagering PASS
at zero stake, with no borrowed probability, quote, selection or authority.

This makes chronology workable, not real collection ready. Original provider
response-byte acquisition (native batches currently retain cleaned projections),
bounded pilot-runner isolation, unreviewed Tulane labels, applicable authentic
mapping/source/rights documents, independent admission and owner collection
authorization remain separate unresolved workstreams. No historical probability
was rebuilt. Software fixtures supply no calibration, scientific qualification,
approved wagering or launch date.
