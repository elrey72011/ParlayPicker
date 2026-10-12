# Bounded pure source reconstruction memo

This separately reviewable correction follows sealed head
`a561e6a1018d2106c33f934b879ff2169f50e7e7`; main is
`457a7f3e43abfbd95de64e4d9c79e2ca82bdfb21`. Every earlier implementation,
seal, failure and predecessor binding remains retained. No baseline is reset.

## Completed diagnosis

One push-only diagnostic run, [38305791913](https://github.com/elrey72011/ParlayPicker/actions/runs/38305791913),
used only two new files on branch `codex/scope-performance-probe`, commit
`cd3663853bd666cd84ed62ec2c4025c8cd48fa6d`. All three Ubuntu 24.04 jobs
passed the same 16 original tests and uploaded complete XML, profiles and
progress records. Python 3.12.14, complete dependency locks and Git executable
hashes matched. No retry occurred. Each job had a 30-minute limit and a
20-minute profiler deadline with flushed partial output. These diagnostic files
are not part of this PR or its required workflows.

| Revision | Elapsed | Collection | Setup | Test calls | Git launches | Repository copies | Compile self |
|---|---:|---:|---:|---:|---:|---:|---:|
| Main 457a7f3 | 84.727s | 3.795s | 9.601s | 70.721s | 23,631 | 1.591s | 0.471s |
| Before optimization 732e179 | 85.099s | 4.108s | 9.397s | 71.064s | 23,631 | 1.532s | 0.421s |
| Existing implementation a561e6a | 59.999s | 4.534s | 7.830s | 47.187s | 1,997 | 1.089s | 0.369s |

Timings include identical cProfile instrumentation. Inclusive fixture, copy,
Git and reconstruction costs overlap phases; they must not be summed.
Persistent cat-file lifetime includes intervening Python validation. Stream
operations, ordinary process durations, compilation, hashes, reporting and
artifact upload are retained separately in private receipts/raw logs.

The prior Windows nine-case improvement (501.030s to 132.935s) is not an
Ubuntu shard forecast. The completed Ubuntu comparison improved only 29.5%.
Linux's Git launch/wait cost was already much lower. Current guard functions
accounted for 29.178s self CPU, compared with 13.380s before optimization;
source adapters were invoked hundreds of thousands of times. Compile and
copy costs were smaller. This supports a source-reconstruction correction,
not a timeout increase, assertion removal or fixture-copy cache. Continued
progress in previous cancelled shards does not establish a hang. Attribution
for their unrecorded remainder stays uncertain; no completed XML is invented.

A narrowed local prototype preserved all nine original cases, reducing elapsed
132.935s to 121.247s and call-phase time 79.757s to 73.183s in the same pinned
Windows environment. The earlier broader prototype (119.155s) and conservative
single-adapter prototype (127.882s) are also retained. These measurements do
not promise future Ubuntu runtimes: prototype collection was warm, so the
elapsed comparison is not a cold-session improvement claim.

The checked cold-session candidate `46e7d34f148c739641e33667855ffe7274674bf2`
passed the same nine original tests in 129.742s (collection 6.182s, setup
46.802s, calls 76.288s), against a561's 132.935s (6.486s / 46.220s / 79.757s).
The two targeted source adapters' self time fell from 6.826s to 0.185s and
their calls from 112,878 to 5,663. Git launches remain 1,567; no work is
removed from test collection or validation. The memos recorded 48,145 hits,
525 misses, five retained entries each and 29,572,180 counted bytes each.

A hardening candidate `d826e2f20b552e2ec85dcfb916411e6057d5e621` took
141.167s. Its interpreted per-field checks caused measurable overhead; it
is retained, not presented as an improvement. The correction uses a compiled
read-only snapshot of the same reviewed fields, freshly evaluated on every
call. All code, default, closure and mutable-declaration checks remain live.
Final required CI, artifact upload and identity reconciliation are necessary;
the local elapsed improvement is only 2.4%, with no future-runtime guarantee.

## Cache contract

Only `_nfl_odds_timing_base_source` and `_nfl_odds_base_source` are candidates.
They return immutable reconstructed source bytes. Their unchanged delegates
continue to validate source hashes and exact predecessor bytes on misses.
No modules, globals, catalogs, HEAD/index/filesystem facts, policy results,
validation decisions or mutable fixtures are cached.

Each memo retains at most eight entries and 32 MiB of counted input/output
bytes. Each input/output is bounded at 8 MiB. Two memos therefore retain at
most 64 MiB of source payload, plus bounded key/context bookkeeping. The new
envelope's two-entry reconstruction cache has a conservative 32 MiB cap;
earlier frozen caches stay unchanged. Unknown/oversized inputs and exceptions
are never admitted to the memo.

Keys include the exact input bytes, callable and source-module location.
Referenced static declarations and reconstruction delegates are traced from
the reviewed functions, including captured functions. At every call, primitive
declarations are re-read and mutable declaration containers are snapshotted
(256 KiB / 4,096 elements / 64 nesting levels). A changed delegate, unrecognized
context, I/O dependency or oversized context bypasses caching. Function/context
discovery is capped at 512 fields. Function code, defaults, keyword defaults
and captured delegates/primitives are also checked. The generated getter
contains only fixed accesses to that reviewed metadata graph, never source
or provider input. Different repositories/modules cannot lend
globals or decisions. Frozen reconstruction and validation assertions remain
unchanged; dirty, staged, committed and resealed attacks must still reject.

The helper's consumed file bytes are hash-verified at every real validation
entry. An isolated historical bootstrap does not need adjacent new files.
The new policy binds its exact implementation against the previous seal,
then a policy-only seal binds the implementation without circular hashes.

Acquisition, model mathematics, Rams scenario, protected files/tests, existing
test assertions, required workflows, scheduler, advisory timing profile and
30-minute required-job timeout are unchanged. CI makes software reviewable;
it provides no source admission, live-offer acceptance or wagering authority.
