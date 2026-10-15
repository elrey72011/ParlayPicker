# NCAAF socket correction: measured CI timing follow-up

Immediate predecessor is sealed #2425 head
`2b3e65213d9f0870e45ac26cc078403de8e52211`; main remains
`8f92728d7538afc1a9a6a842fbee5f172e241ec7`. The socket correcting commit
`e3790664b7c3a023b27a4762927788fa0969a11c`, its application bytes and all 48
new regression identities remain unchanged. This successor changes advisory CI
file costs, exact scope machinery, focused new tests and documentation only.

## Retained failure and diagnosis

Run 38673099731 attempt 1 passed five application shards. Shard 2 exhausted its
30-minute job limit while progressing; its upload was skipped and no final XML
or execution manifest exists. Its aggregate failed. The original five artifacts,
raw logs and missing-coverage result remain retained; progress is not passing
coverage. Dependency installation was 39 seconds, application step 1,768 seconds,
and reporting did not run. Job accounting includes 19 seconds beyond 30 minutes
for cancellation/cleanup, not extra authorized test time.

The interrupted slate-audit log span is 492.87 seconds across 44 observed passes
out of 110 selected identities. This includes setup/teardown between lines and
is not a completed file measurement. Component attribution for the entire failed
shard remains unresolved. No hang, socket-code runtime regression or runner
variability is inferred from that span.

Nine unchanged synthetic scope assertions in matched local Python 3.12.14 ran
on main in 159.40 seconds (setup 48.07, call 99.69), and on the sealed PR in
175.28 seconds (setup 47.11, call 114.16). Both launched 1,570 Git processes.
Repository-copy totals were 3.37 and 3.70 seconds. Git process lifetimes include
Python work between cat-file stream operations and must not be added to other
phase totals. Compilation, hashing, reconstruction, validation and reporting
were separately retained in cProfile/phase receipts.

A rejected exact-reconstruction prototype passed the same assertions but took
180.53 seconds. Its extra function graph exceeded an inherited 512-field context
bound and disabled one cache. It is not adopted; no cache bound, helper,
validation decision or mutable fixture changes are included in this correction.

The one authorized isolated Ubuntu comparison is run 38678139543, diagnostic
commit `a165710f563b24315badf97ecc28ea727d6cb0a1`. Its two-file diff SHA-256 is
`e3b3527075144790b8298097f6e4c86d7bb865d66a6b10ecc4f55e4f93e63dff`.
Both 16-test jobs passed under Ubuntu 24.04/Python 3.12.15, identical pinned
dependencies (lock SHA-256
`7d0c8b8325237c57b258f66f3de4074f496efd1aa813cb5bc8818c75b0d7b49e`),
30-minute ceilings, 1,200-second profiling deadlines and no retries.
Main took 117.95 seconds; PR took 118.27. Setup was 22.57/22.51 seconds, calls
86.19/86.18, collection 8.44/8.64, repository copies 0.849/0.844, and both
launched 1,850 Git processes. Both had identical pure-cache hit/miss counts and
29,572,180 retained bytes per cache; high-water RSS was 278,472/289,644 KiB.
This bounded subset establishes no material PR-introduced slowdown. It does not
prove the cause of the whole-shard timeout. Instrumentation overhead is included.
The diagnostic branch is separate; required release workflows are unchanged.

## Evidence-supported correction

Twelve advisory rows are refreshed from hash-bound completed whole-file XML and
assignment manifests for reviewed #2424, post-merge main and completed #2425
shards. Every referenced test-file blob matches the current test bytes. No
cancelled span or profiling subset supplies a scheduling cost. The selection is
material scheduler-cost errors of at least 60 seconds, plus provenance for both
new socket files. For example listing scope's old 11.13-second advisory cost
contrasts with a completed 66.83-second testcase sum; the new socket lifecycle
file's 119-second fallback contrasts with a completed 1.85-second sum.

The existing methodology uses the maximum compatible completed testcase sum
plus an explicitly estimated 15% variance. Collection, installation, reporting
and uploads remain separate. A 100-second setup/reporting allowance is an
estimate, compared with retained actual job steps. Frozen profile revision 3,
historical sources and every untouched row remain exact; successor metadata
records this correction separately. No runtime savings are claimed: this is
workload distribution based on corrected advisory weights.

Before correction all four expensive inherited scope files were assigned to
shard 2. The preview after material corrections spreads slate-audit, calibration,
DFS and remote-canonical scopes across four shards. Exact final collected
identity count, ordered assignment, profile/manifest hashes, loads, headroom
estimates and new-test overhead are retained before the one fresh final-head
CI cycle. Estimates are not a performance guarantee. Final results, artifacts,
tested-tree equality and exact-once reconciliation are reported in the PR
validation receipt; missing execution never counts as passing coverage.

## Preservation

No test identity/assertion is removed or weakened, including dirty/staged/
committed/resealed rejection and live repository validation. No application,
model, admission, history, freshness, credential, attempt, scheduler, workflow,
six-shard cardinality, timeout, protected requirement or acceptance catalog is
changed. Scope reconstruction binds the exact prior profile and guard; a later
policy-only seal binds this implementation without circular hashes.

The scheduled NFL document task, its checkout/runtime/dependencies/configuration,
encrypted custody, WNBA work, Army–FAU authorization, originals and spent journals
are read-only preservation subjects. No local intensive tests run during its
October 15 10:00–10:15 Eastern window. No acquisition, inference, trust change,
Drive operation, scheduling, merge, deployment, publication or wagering is included.
