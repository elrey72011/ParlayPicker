# Bounded Drive history and evidence loading draft

Starting main is befe6a09cfcec3e6dbe01f53b92602828c3a0db7, tree
d2aa023d4e5bc1770fbbc6c8f284c21ec4e87e07. Its merged DFS and NCAAF
corrections remain immutable. The current-main scope guard passed, and all main
CI checks completed successfully before this successor was sealed.

## Caller and behavior

The completed-analysis branch in streamlit_app.py calls
render_publish_panel(..., lazy_history=True). render_history() offers an
explicit Load saved history action; display and preview reruns do not restore
Drive history. Empty-session publication recovery retains the existing automatic
load behavior. Required begin_run() and snapshot loading still initialize
prediction evidence. The optional evidence-status panel reads local status;
explicit Restore and sync performs full byte verification.

restore_history() now calls History.history_phase() through _read_kinds()
and DriveStore.read_verified_prefixes(). One complete folder inventory supplies
deployments, confirmations, packages, scores, imports, locks, removals, grading
runs, prop statistics and prop imports. Publication packages must match their
confirmation hashes. Only a missing confirmation in a verified complete phase
permits hosting-status recovery; corrupt/conflicting/incomplete reads fail closed.
If confirmation recovery writes change membership, a second, fresh phase follows.

Active locks are rebuilt from original locks minus matching removal hashes.
Fresh action inventories bypass inventory/read-phase coalescing. Existing lock
pre-action, create-only exact read-back, post-write reconciliation and publication
checks remain intact. Display records cannot substitute for these checks.

Equivalent in-flight history, inventory, media and explicit evidence reads share a Future within their authenticated identity,
Drive/folder, namespace and site scope. Completed membership is never retained
as authority. Read callers receive separate parsed objects. Immutable bytes are
reused only when their SHA-256 matches freshly listed provider metadata; every
duplicate participates in conflict checking. Missing checksums require downloads,
including null checksums and on warm reads. Malformed checksums are rejected. Downloads validate available checksums even when no
disk cache is requested. Malformed, incomplete and repeated-token listings fail
closed. Injected sessions without authenticated credentials are isolated by
instance; parallel authenticated workers retain the parent's scope.

Evidence initialization is keyed by configured authenticated identity, folder,
namespace, site, absolute database path and file generation. Generation is
checked after acquiring the initialization lock, avoiding duplicate restores when
concurrent callers first create the database. Successful explicit restores also
satisfy initialization. Replacement, scope changes and failed initialization/full
verification invalidate it. Explicit restores and full verification remain
available. A scope or existing database generation change during restore rejects initialization and cannot relabel older display records. Incremental backup receipts additionally include the actual client
storage identity.

The three evidence restore entry points were analysis initialization, snapshot
loading and the optional evidence panel. All previously called restore_once().
The supplied aggregate logs do not identify whether the warm restores crossed
process, scope or database generations; this draft does not claim an exact
historical attribution. The regression reproduces the concurrent missing-file
generation race and verifies one initialization for three concurrent callers.

## Offline measurements

All new behavior fixtures and the benchmark block DNS, socket connections,
requests and urllib. The benchmark invokes the original main UI restore function
and the new actual UI restore function against independent cold caches and
39,000 synthetic objects, using 1,000-object pages and eight matching records.

| Synthetic operation | Full inventories | Listing requests | Media downloads | Verified byte reuse | Local wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original UI restore | 8 | 314 | 9 | 0 | 0.673927 |
| Original subsequent lock display | 1 | 39 | 0 | 0 | 0.091530 |
| Batched cold UI restore | 1 | 39 | 8 | 0 | 0.130759 |
| Warm explicit restore | 1 | 39 | 0 | 8 | 0.132491 |
| Explicit full verification | 1 | 39 | 8 | 0 | 0.130626 |
| Unchanged optional display reruns | 0 | 0 | 0 | session display reuse | no provider work |

The original 314 requests comprise 312 full-folder pages and two exact-name
lookups. Eight matching records is a synthetic fixture size, not a production
download estimate. Equivalent concurrent history phases use one inventory and
eight downloads; equivalent concurrent reads of one media object use one download.
Separate authority phases still discover fresh membership.

The owner-supplied runtime summary reports 23 full inventories at 36â€“41 seconds
each, two zero-match inventories, three evidence restores (two warm zero-import
restores), and approximately 18 minutes across analysis/history/locks/publication.
Those are observed aggregate inputs, not a controlled before/after live benchmark.
Multiplying 23 by 36â€“41 gives 828â€“943 seconds of inventory spans, which may overlap;
it is not elapsed wall time. Nested and overlapping spans must not be summed.
The local synthetic numbers above measure wall time directly and exclude live
provider/Drive latency. No live after-runtime speedup is claimed.

## Market-stage instrumentation

The separate market-enrichment/candidate-selection stage now emits correlated
component spans for Kalshi enrichment, ML probability joining, consensus,
candidate selection, export preparation and review-candidate preparation.
Calls, ordering, inputs, outputs, timeout behavior and prediction rules are
unchanged. The enclosing stage supplies the common trace; component durations
explain its cost and must not be added to the enclosing duration. The historical
approximately 70-second cost cannot be attributed further without a subsequent
authorized runtime capture. No live provider operations or prediction rerun were
performed for this draft.

## Exact successor

Implementation A has starting main as its sole parent and exactly the paths in
DRIVE_PATHS. Seal B adds only
docs/paid-launch/launch-scope-policy-drive-history-v1.json. Its policy binds the
base and implementation commits/trees, before/after blobs, original baseline,
prior v4 policy, unchanged evidence and complete tooling digests. An independent
normalized successor-guard digest and exact reviewed implementation blobs reject
resealing attacks. The scope module is verified before importing it. CI must have
ordered parents [starting main, B] and B's exact tree. Dirty, staged, committed,
extra-seal, extra-commit, wrong-parent/tree, runtime-shadow and hook mutations fail.

The complete previous guard is reconstructed byte-for-byte using its frozen CLI.
The DFS and NCAAF scope fixtures adapt only source reads. Frozen inverse hunks
and independent digests reconstruct exact prior application/fixture bytes in
shallow CI checkouts. The Drive corruption fixture adapts only its cache target
path. Reverse reconstruction must reproduce original test bytes; every prior
assertion is unchanged. The synthetic benchmark includes the exact original UI
restore function and its digest, without depending on historical Git objects.
The default invocation also retains the previously approved clock correction
only after its exact unchanged blob is verified, preserving the subscriber
assertion without altering it.
Original baseline/policies/workflows/subscriber scope, scientific/authority
inputs, merged NCAAF/DFS behavior tests, frozen version 10, prior deliverables and
owner work are preserved. Raw logs and private evidence do not enter this PR.

No merge, deployment, live provider/Drive operation, fitting, threshold,
qualification, authority or financial change is authorized.
