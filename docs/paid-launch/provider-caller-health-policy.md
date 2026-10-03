# Provider caller health: bounded implementation and successor policy

This draft addresses the actual Streamlit provider caller at starting main
`057abaf8214268d99ba9d1d8ba672997a5aef81f`, tree
`9bc0b53a2e1055918c399a36d5a4de65094bd540`. Owner authorization covers preparation
of this reviewable draft. Merge requires separate review and authorization.

## Application boundary

`fetch_live_odds_dataframe` keeps its DataFrame return contract, including healthy
rows when another sport fails. The new `provider_health` attribute records each
requested sport's primary-provider outcome, verified typed HTTP status, received
game-list count, processing outcome and controlled fallback failures. A successful
empty list is distinct from authentication failure, exhausted rate limiting,
other HTTP failure, timeout, transport failure, invalid response and absent
configuration. Filtering all games does not relabel a successful response as a
provider failure. Received counts describe the client's returned game list, not
proof of provider coverage. Returned rows can include an existing fallback while
the original primary-provider failure remains visible.

The real ESPN FCS helper now returns a list-compatible `ProviderGames` object
with an additive, sanitized `provider_outcome` receipt. Legacy list iteration,
equality, JSON game serialization and the helper's positional signature are
preserved. The caller reads the receipt before merging games and records
`fallback_outcomes.ncaaf_fcs`; actual failures additionally populate
`fallback_errors.ncaaf_fcs`. A primary successful empty response stays
`SUCCESS_EMPTY` when the fallback fails. Successful empty fallback results are
recorded separately from authentication, timeout and transport failures. A
healthy NBA response plus an ESPN FCS failure yields aggregate `PARTIAL_FAILURE`.
Plain lists from older adapters have no receipt and are not assigned a fabricated
fallback outcome.

`run_analysis_pipeline` captures these attributes before DataFrame transformations
and adds a sanitized projection to its diagnostics. The existing Streamlit
analysis handler already stores those diagnostics for each refresh. Readiness
report construction preserves the projection; the current-analysis readiness
panel displays it even when no candidates exist. Saved-snapshot mode does not
borrow current provider health. Every refresh replaces the outcome snapshot.

The affected caller logs only fixed categories, sport keys and verified status
codes. Exception text, response bodies and request URLs are not interpolated.
The existing ESPN FCS helper's exception log uses the same controlled fields.
Its failure still returns an empty game list, with a sanitized outcome receipt.
Other fallback adapters can still handle failures internally; absent receipts
are not proof that those requests succeeded. The panel displays controlled
fallback outcomes and HTTP codes; its download preserves those same facts.
The panel and JSON download re-project stored diagnostics through the same
controlled schema. This information does not supply probabilities, alter prices,
or grant eligibility or wager authority.

## Immutable evidence and exact candidate identity

The original baseline manifest remains unchanged. Policy v2, the exact reviewed
clock-test correction, the paid-launch workflow and the subscriber scope assertion
remain unchanged. No new exception for an existing test is introduced.

The provider caller, readiness builder and readiness panel are not listed as
protected files in the original manifest. They nevertheless share code with
scientific and authority decisions. Policy v3 reconstructs exactly the reviewed
diagnostic replacements from starting-main blobs and rejects every other change
in those three files, including a re-sealed authority or staking override.
The fallback file is likewise bound to the reviewed diagnostic receipt and
logging replacements. Parsing, event identity, prices and observation timestamps
remain unchanged.

The original guard and v2 validation logic remain byte-identical before their
CLI entry point. The new entry point selects v3 only when its policy is committed.
The original guard still runs. Its raw findings are retained, with approved
integration changes and the retained clock-test exception reported separately
from unauthorized changes. Existing-test edits, protected model/calibration
changes, altered baselines, unauthorized tooling, runtime shadows, extra policy
fields and incorrect candidate ancestry remain failures.

The implementation commit **A3** has only the approved starting main as parent and
changes exactly these nine paths:

- `core/streamlit_pipeline.py`
- `core/run_readiness.py`
- `app/ui/readiness_dashboard.py`
- `app_core/provider_health.py`
- `app_core/espn_ncaaf_odds.py` (only the exception diagnostic)
- `scripts/check_launch_change_scope.py`
- `tests/test_provider_caller_health.py`
- `tests/test_provider_health_scope_policy.py`
- `docs/paid-launch/provider-caller-health-policy.md`

A second commit **B3**, parent A3, adds only
`docs/paid-launch/launch-scope-policy-v3.json`. That seal records A3's commit, tree,
exact before/after blobs, successor tooling hashes and unchanged evidence blobs.
It also binds the approved starting commit/tree and prior policy and clock blobs.
The CLI verifies the seal's single-parent, policy-only shape. A CI or eventual
merge commit must have parents `[starting main, B3]` and B3's exact tree. There is
no circular commit dependency: B3 binds the already completed A3, and B3's own
identity is verified through ancestry and its policy-only change set. Git blob
identity is compared with checkout bytes; CRLF-only conversion is reported
without changing baseline hashes. An approval-reference string is a review
record, not a substitute for GitHub review or owner merge approval. The revised
policy is explicitly versioned `paid-launch-provider-health-v3-r1`; it incorporates
the owner-authorized structured ESPN FCS outcome review correction in PR #2376.
The prior draft commit remains recoverable; the revised implementation and
policy-only seal are prepared again from the same approved starting main.

## Verification and remaining acceptance gates

New offline tests exercise the actual provider caller, real client retry contract
with mocked HTTP, analysis/readiness propagation, refresh clearing and the actual
readiness panel. Credential canaries in exception messages, request URLs and
response bodies must be absent from affected logs and diagnostics. Network entry
points are blocked. Real Git fixture tests cover exact PR/CI seals, CRLF conversion
and adversarial scope, policy, tooling, baseline, existing-test and runtime changes.
Real-helper caller cases cover HTTP 403, timeout, transport failure, successful
empty results, invalid payloads, healthy NBA-row preservation, unchanged primary
outcomes, readiness display/download and clearing a prior fallback error on
refresh. List compatibility is checked independently of diagnostic propagation.

These results cover the exercised software contracts. They do not establish live
provider availability, deployed Streamlit operation, scheduling or publication
durability, scientific qualification, commercial READY status, or guaranteed
winning selections. The application must still PASS when no qualified selection
exists. Event-ID/doubleheader correction, publication durability, feature-order
correction and scientific development remain separate work items.
