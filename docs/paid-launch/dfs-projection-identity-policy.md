# DFS projection and player identity draft

This bounded draft starts at main `af7b3dcbc3fe9ca7161b886c65074be3d739bae8`,
tree `98a117dbc7f6977bafed53624898d95a4e72f334`. Authorization covers preparing
the fixes, offline regressions and exact successor bindings. It does not authorize
merging, deploying, submitting contest entries or changing model weights.

## Projection and identity behavior

Finite supplied projections, including zero and signed negative fantasy points,
replace historical FPPG. A blank or missing value has `ProjectionStatus=missing`:
the salary parser may use finite, explicitly labeled historical FPPG, and a separate
upload retains the prior labeled score. Malformed text, infinity and overflow have
`ProjectionStatus=invalid`, `ProjectionSource=invalid_projection` and unavailable
points. They cannot revive a high FPPG fallback. Actual DataFrame NA values denote
missing input; CSV literals such as `NaN` remain text and are invalid. Nonfinite
average FPPG and salary values are also unusable. Missing and invalid upload counts
are separate from applied projections and unmatched identities.

Projection IDs are normalized before joining. Duplicate nonblank projection IDs
are rejected before filtering values, including repeated IDs with blank or invalid
scores. Exact IDs disambiguate equal names. An exact ID with a contradictory
nonblank normalized name is rejected. Name fallback requires one source row and
one pool row with that normalized name, and at least one missing ID. Two known,
different IDs cannot match by name. Identity errors reject the upload instead of
silently selecting a row or discarding its evidence. This deliberately requires
correction of ambiguous feeds before a lineup can be generated from them.

Both optimizers constrain assignments by canonical DraftKings ID across rows and
slots. They also count portfolio overlap by that identity. ID is read from the
explicit field or the trailing `Name + ID`; contradictory fields are rejected.
When both are absent, normalized name supplies a conservative uniqueness key.
`Unique Players` and `Lineup Key` use the same canonical identities; keys are
namespaced with `id:` or `name:`. Duplicate salary rows can supply alternative
eligible assignments, but only one assignment for that player can be selected.

Roster slots, the salary cap, NFL QB pass-catcher stack and DST conflict rule,
MLB hitter team cap and pitcher/hitter conflict rule, existing status handling,
MLB starter/confirmed-hitter rules and the requested diversity setting remain
in force. This patch does not introduce live availability data or claim that an
early blank status proves a player will play.

## Exact successor guard

The original baseline manifest, v2 and v3 policies, existing DFS tests, retained
clock correction, workflows, provider integration and scientific inputs stay
unchanged. The guard's original and v2/v3 logic before its CLI entry point stays
byte-identical. V4 adds a distinct verifier and selects it only for a committed
`docs/paid-launch/launch-scope-policy-v4.json`.

Implementation commit A4 has the approved starting main as its only parent and
changes exactly six paths: the DFS module, its panel copy, the new projection
regressions, scope guard, new guard regressions and this policy document.
Policy-only commit B4 has A4 as its only parent
and adds only the v4 JSON. That seal binds the base commit/tree, A4 commit/tree,
every before/after blob, original manifest SHA-256, prior v3 policy and clock-test
blobs, tooling SHA-256 values and explicit retained evidence blobs. The guard also
pins the reviewed DFS module, panel and regression blobs independently of the
policy, preventing a changed implementation from being accepted by rehashing it.
The complete successor guard bytes have an independent SHA-256 binding with
the embedded digest normalized to 64 zeroes. This avoids a circular checksum
while rejecting a changed extension or CLI even if its policy is re-sealed.

An exact CI merge must have parents `[starting main, B4]` and B4's tree. A changed
base, extra commit, extra seal path, edited baseline, existing-test edit, model or
calibration change, tooling mutation, runtime shadow or altered reviewed DFS blob
fails. CRLF checkout conversion is reported against committed LF bytes without
rewriting baseline identities. A future revision requires a new exact reviewed
implementation and seal. The approval reference records draft authorization and
cannot grant merge approval.

## Offline evidence

The new DFS counterexamples block network connections. Against starting-main
code, 32 of 35 new cases fail and three pass; after correction all 35 pass. The
27 existing DFS/parser/panel tests pass unchanged. The new Git-fixture tests
exercise candidate and CI ancestry, policy-only seals, immutable prior evidence,
dirty/staged/committed mutations, re-sealing attacks, runtime shadows and CRLF
conversion. Prior policy tests are also rerun unchanged. Test artifacts and exact
candidate verification are recorded in the PR, outside the implementation seal.

The historical audit is provided separately as a local companion and is excluded
from this public PR. These offline contracts establish software behavior, not
historical loss causation or out-of-sample predictive quality. No weights are
tuned, historical predictions regenerated with later information, or recent
results reused as validation.
