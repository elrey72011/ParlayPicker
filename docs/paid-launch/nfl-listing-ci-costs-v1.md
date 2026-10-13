# Measured NFL listing CI cost successor

This follow-up changes advisory whole-file costs only. It preserves the NFL
listing implementation at 028080969ed341e5d92b7445352deac8ba2ae288, all original
tests and assertions, the scheduler, workflows and 30-minute required-job limit.
It reduces neither test work nor runner cost by assertion.

## Completed evidence and bounded comparison

Actual main is e7f3d6ac8ddd8a38a8bc082809e823bcef16c42c. Its completed CI run
38316289647 used attempt 2 for shards 1/3 and retained attempt 1 for successful
shard 2. PR run 38322189449 tested merge
0a22bc0433784eb0f62b59d6764b9a28ef90d46f of head
028080969ed341e5d92b7445352deac8ba2ae288. Shards 1/2 completed; shard 3 was
cancelled and produced no final XML/manifest. That missing execution remains
unverified. Console progress is not a completed whole-file timing measurement.

The two new files completed all 50 cases in shard 1: acquisition 36 cases,
0.309 seconds; listing scope 14 cases, 9.674 seconds. These are XML testcase
sums (pytest setup/call/teardown), excluding collection/report/upload. Their
175-second missing-file fallback was conservative, not a measured cost.

[Diagnostic run 38327494974](https://github.com/elrey72011/ParlayPicker/actions/runs/38327494974)
at e2e3226026003005fb3d2bb46c1c0619cd3b85d3 compared main and the PR with
identical Ubuntu 24.04, Python 3.12.14, Git executable, dependencies and
instrumentation. Each ran the same 63 identities: nine original slate/canonical/
football cases, three exact predecessor cases and all 51 NFL input scope cases.
All 63 passed on each revision and both artifacts uploaded and passed hash/CRC
verification. Network blocking and independent mutable repository copies stayed
active. The dependency-lock SHA256 is
7d0c8b8325237c57b258f66f3de4074f496efd1aa813cb5bc8818c75b0d7b49e.

| Inclusive category (seconds) | Main | PR |
| --- | ---: | ---: |
| Profiled total | 232.175 | 244.140 |
| Collection | 5.952 | 6.299 |
| Pytest setup | 11.850 | 12.046 |
| Pytest call | 213.792 | 225.250 |
| Pytest teardown | 0.016 | 0.015 |
| Repository copying, subset of setup | 4.145 | 4.240 |
| Compile self CPU | 0.567 | 0.557 |
| Hash self CPU | 18.484 | 20.066 |
| Guard self CPU | 76.554 | 84.511 |
| High-water RSS (KiB) | 246172 | 239684 |

Fixture, copy, Git and reconstruction categories overlap phases and must not be
summed. Persistent cat-file process lifetimes (204.109/215.301 seconds) include
Python validation between stream operations; they are not Git execution CPU.
Reconstruction hit/miss counts are identical: timing 121066/1793, odds
192196/1569; each retained 29572180 bytes, below the existing 32 MiB limit.
Context fields rose from 272/358 to 295/381, within the unchanged bound.
There is modest observed successor overhead (5.15% total under cProfile), not
evidence of a hang, cache collapse or fixture regression. One paired run cannot
separate all runner variation or prove future runtimes.

The local, separately profiled 12-case comparison also passed unchanged cases:
205.022 seconds main versus 204.558 seconds PR. It is not Ubuntu evidence.

Earlier diagnostic run 38326701393 failed in instrumentation: cache inspection
mistook pytest's dynamic MarkDecorator for a cache. Its logs/XML/pstats and
successful partial artifact uploads are preserved separately, and are not used
as passing application validation. The corrected helper was verified locally
and its single replacement run was separately owner-approved.

## Correction and exact preservation

Refresh only 55 inherited scope-file entries and the two new listing-file
entries from completed main/PR XML. Keep all earlier sources, incomplete
observations and other file entries unchanged. The maximum compatible completed
cost plus the existing 15% variance margin is labelled estimated. It does not
turn cancelled progress into a completed measurement. Provenance binds run,
attempt, branch head, tested checkout, artifact, XML and assignment hashes.

Using the unchanged 7077-node collection, the previous assignment distributes
maximum retained comparable testcase costs as 1216.150 / 1717.768 / 1625.123
seconds. Updating only the two new files still gives 1354.255 / 1486.725 /
1718.061 seconds. Updating the demonstrated scope costs gives 1517.574 /
1500.076 / 1541.391 seconds. This is a retrospective assignment preview, not
new-head execution or a runtime guarantee. New successor tests must be included
in the final collection preview and actual final-head reconciliation.

No guard optimization or mutable validation-result cache is introduced. Scope
plumbing reconstructs the exact predecessor guard/profile from exact reviewed
bytes and rejects changed bytes. The initial implementation/seal and policy
remain unchanged. A new implementation parented by the old seal is followed by
one policy-only seal; main's baseline hash is not reset.

The original cancelled run and one-second shard-2 headroom remain recorded.
Imbalance and stale advisory costs are measured. The exact historical share of
runner variation in the cancelled shard remains UNKNOWN. Final readiness still
requires all 13 final-head checks, uploads and complete identity reconciliation.

Passing CI establishes listing-acquisition software validation, not authentic
listing applicability, source acceptance, predictive accuracy or wagering
qualification. No private Rams evidence or result is changed.

