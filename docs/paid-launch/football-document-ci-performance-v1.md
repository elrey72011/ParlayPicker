# Document CI reconstruction and advisory timing successor

This correction is separate from the credential-scanner and fixture corrections.
It changes no acquisition, numerical, admission or operational behavior.

## Retained failure and measured work

Head `43925c920a07c475f50ccfec0b366ffd963cfb8c`, application run
`38638272550`, exhausted shard 6's 30-minute budget while progressing.
Its absent final XML/manifest is incomplete coverage, not a pass or a proven hang.
All original failures, successful artifacts and spent operational evidence remain.

Completed run `38626230037`, artifact `11926041783`, contains passing, complete
49/56/42-case results for the unchanged football-catalog, MLB/NHL-research and
source-intake files: 279.199 / 341.974 / 228.344 testcase seconds respectively.
Their former costs were 255.942850 / 312.643600 / 213.981650 seconds. Only these
three rows are refreshed, with the existing 15% advisory variance. Testcase sums
include fixture setup/call/teardown; installation, collection, session reporting
and artifact upload are separate overhead. Cancelled log intervals are not
completed measurements. Frozen revision 3 and its provenance remain unchanged;
new provenance is recorded in separate successor metadata.

The bounded local 12-case profile found repeated removal of known successor
blocks from a 6,042,118-byte immutable guard. The old procedure repeatedly copied
the full body while deleting blocks. The successor reconstructs the same
5,577,215-byte intermediate from 11 retained spans in one join, checking both
exact input and output SHA-256. It then executes the original interpretation
declarations and parent delegate checks. Unknown/changed input takes the original
path and retains its rejection. No validation decision is cached; the existing
memo caps and live context checks are unchanged.

The local 12-case sample reduced adapter self time, but did not establish a total
wall-time improvement (142.702 seconds before, 156.396 for the integrated
candidate). A main-versus-PR sample is also confounded by the already-approved
source-intake fixture maintenance correction; its assertions and identities are
unchanged. Neither observation is a causal whole-job speedup claim.

The authorized Ubuntu comparison completed at diagnostic commit
`fcb0ccc40695aedf34d5cbd54afd89a4cf0e181e`, run `38646925182`, attempt 1.
Both jobs passed all 147 identities with no failures or skips. Source/test
hashes, identity order, Python 3.12.15, dependency lock and Ubuntu runtime records
match. The candidate is the isolated exact-source projection prototype on
43925c92, not the final sealed head. Application transports remained blocked.

| Measurement (seconds) | Baseline | Projection |
| --- | ---: | ---: |
| Profiled workload | 1072.787 | 986.312 |
| Collection | 11.507 | 12.104 |
| Fixture/setup phase | 18.952 | 17.726 |
| Test call phase | 1041.498 | 955.414 |
| Repository copying (subset) | 9.632 | 9.627 |
| Non-cat-file Git elapsed (subset) | 19.687 | 20.495 |
| Reconstruction Python self time | 358.100 | 321.900 |
| Hashlib self time | 134.272 | 106.202 |
| Compilation self time | 0.767 | 1.100 |
| Dependency installation | 12 | 15 |
| Artifact upload | 1 | 2 |
| Whole job | 1096 | 1016 |

The workload saved 86.476 seconds; the whole job saved 80 seconds, or 4.44% of a
30-minute budget for this three-file workload. This is not a per-shard or
full-suite speedup guarantee. Both jobs made 5561 Git launches, 182854 stream
header operations and 6764 contents operations. Cat-file process lifetimes
include Python between reads and must not be reported as Git CPU. Fixture/copy,
Git and reconstruction measurements are subsets of phases; do not sum overlapping
categories. Validation still executes; its call phase includes these operations.
Peak RSS was 284580/293444 KiB. Existing memo statistics and bounds match exactly:
6 entries and 30620004 retained bytes per memo, with 393/512 context fields.

The advisory refresh uses the compatible unprofiled completed XML described above,
not profiler-inflated durations or cancelled progress. It separates the three
formerly co-located files onto shards 6, 3 and 5. The deterministic 7697-identity
preview preserves all 7673 prior identities and adds 24 focused cases. It forecasts
1622.263 testcase seconds plus 100 seconds of setup/reporting overhead per job:
28m42s worst job, approximately 78 seconds headroom. The overhead reserve exceeds
the retained completed-job maximum of 85.854 seconds. A mixed-revision completed
XML cross-check, without subtracting optimization savings, estimates a worst job
of 1622.942 seconds including overhead; it is also an estimate, not a final-head
measurement. Final CI must establish actual coverage and performance.

## Exactness and preservation

Schema 68 seals the implementation against head 43925c92 and verified main
5b9e5fe8. Removing this successor reconstructs the preceding guard byte for byte.
The optimized prefix contains only exact immutable source transforms. Mutable
repository state, parent declarations and validation results are never memoized.
Focused tests retain changed-source, changed-parent, changed-delegate,
cross-repository, dirty/staged/committed/resealed and policy-only-seal rejection.
Existing test files/assertions, six whole-file shards, scheduler, workflows,
network blocking and 30-minute limits remain unchanged.

One fresh complete final-head CI cycle is permitted only after the comparison
and deterministic assignment preview establish credible headroom, with explicit
setup/reporting overhead. Readiness requires all 16 checks and six complete
execution artifacts, every collected identity exactly once and no new skips.
Estimates are not runtime guarantees. No authentic acquisition or inference is
authorized by this correction.
