# October 11 advisory timing correction

This is a separate CI scheduling change following #2416's diagnostics seal
`10503e1d84d5cac93e3b404414001367a794af11`. Actual main remains
`0a19dae206d80b482a516e0b780275cfd8278a93`. Diagnostics implementation,
original scope bindings, authentic failed capture and spent journal are unchanged.
The existing v1 scheduler consumes revision 2 of the same advisory profile.
No scheduler, test assertion, workflow, timeout or production gate changes.

The final profile is revision 3. The first refresh at seal
`832ca1625ead7159fb70846eeef47c87a27f79b7` (run 38170314113) completed shards
2/3 in 27m30s/19m55s. Shard 1 recorded all 2,318 selected outcomes before
cancellation at 30m10s, with no assertion failure, but retained no completed
XML/manifest. Its `full-suite` correctly failed. Windows standard/full-corpus
and aggregate passed separately. These results remain historical, not final-head
coverage. The unfinished interval for compatibility scope was 491.773s between
its first/last outcomes versus the earlier 364.920s completed testcase sum.

Revision 3 incorporates completed same-code XML from refresh shards 2/3 and
**estimated** inter-outcome intervals from cancelled shard 1. These intervals
include preceding teardown/first-test setup, but terminal teardown/reporting is
UNKNOWN; observing all node IDs does not establish completed JUnit measurements.
All recorded outcomes match that candidate's selected identities in order,
accounting for two exact existing synthetic canary IDs masked by GitHub in the
console. Raw logs remain unchanged; this timing comparison supplies no completed
outcome evidence for the cancelled job.
The maximum completed or estimated basis receives the same 15% advisory margin.
Older source identities and incomplete status remain unchanged. New profile
regressions distinguish the newly completed diagnostics measurement from its
original, still-incomplete observation. No original test assertions changed.
All three shards must execute again on one final sealed head/profile.

## Retained evidence

| Source | Physical shard | Installation | Collection/startup* | Test execution** | Reporting*** | Job |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Main run 38150437562, attempt 1 | 1 | 36s | 15.5s | 1462.2s | 0.6s | 25m29s |
| Main run 38150437562, attempt 1 | 3 | 34s | 14.7s | 888.2s | 0.5s | 15m51s |
| Main run 38150437562, attempt 1 | 2, cancelled | 40s | 18.4s | at least 1738.8s | UNKNOWN | 30m16s |
| PR run 38158137485, attempt 1 | 1, cancelled | 38s | 16.5s | at least 1747.8s | UNKNOWN | 30m17s |
| PR run 38158137485, attempt 1 | 2 | 30s | 14.0s | 1171.6s | 0.4s | 20m29s |
| PR run 38158137485, attempt 1 | 3 | 42s | 18.4s | 1405.1s | 0.4s | 24m38s |
| PR run 38158137485, attempt 2 | 1, cancelled | 42s | 18.2s | at least 1744.2s | UNKNOWN | 30m15s |

* Test-step start to collection summary; includes interpreter/plugin startup.
** Collection summary to last recorded test outcome. Completed JUnit testcase
sums, separately used in the profile, include setup/call/teardown; these wall
intervals are not interchangeable with those sums.
*** Last outcome to pytest summary, excluding artifact upload (about 1s on
completed jobs). Cancellation has no finished reporting measurement.

Completed XML covers 393 of the 436 collected files. Exact public run, attempt,
API head, tested checkout, artifact and XML/manifest/log hashes accompany profile
sources. Main used its actual commit. PR used merge
`c4fd8175dd1a74bf72c50433d2f99b744d72a190`, distinct from branch head.
Raw artifacts/logs are retained locally unchanged. Subtests are not added to
collected node identities when reconciling manifests and XML.

## Findings and correction

No assertion failure precedes the cancellations. Both PR cancellations still
progress through `test_stale_research_carrier.py`, at different tests. The
largest gap between recorded outcomes is under 12s in each cancelled job.
The logs do not demonstrate a hanging test or browser-startup failure.

Scope files are substantial: completed testcase sums include 364.920s for
`test_ncaaf_compatibility_scope.py`, 257.896s for `test_slate_audit_scope.py`, and
231.204s for `test_football_research_scope.py` (old estimate 150.5691s).
Their fixtures construct/copy synthetic Git repositories and repeatedly verify
exact frozen bindings. No specific fixture correctness defect is demonstrated.
Runner variance/resource contention remains an unproven explanation for timing
differences; redistribution cannot establish or repair a runtime regression.

Newer NFL files previously fell back to 3.5s per test: the scoring acquisition
file cost 154s in assignment but completed in 1.572s; owner research cost 178.5s
but completed in 22.064s. Misestimates changed which expensive files clustered.

Revision 2 uses maximum retained completed file testcase sums plus an explicitly
estimated 15% variance allowance. Such costs are labelled `estimated`, with the
completed measurements retained separately. Cancelled all-node progress spans
use a labelled estimated first-test allowance plus variance; they are never
called completed measurements. Remaining missing/older observations retain
their existing fallback/provenance. Old source identities remain unchanged.
New files still receive max(30s, 3.5s per test); additional tests add 3.5s each.
These are scheduling assumptions, not upper bounds.

The unchanged scheduler sorts whole files by cost/path and balances three
deterministic buckets, retaining original within-file collection order. New
regressions prove complete, disjoint assignment and truthful provenance.
Local collection manifests and final-head executed manifests/XML provide the
actual proof, rather than the synthetic estimate test alone.

All three application jobs must run afresh under one final head/profile.
Older passed shards cannot satisfy that reconciliation. Report actual counts
and durations separately from projections. Total test work/runner cost is not
reduced. Full-suite, production safety, blocked network, paid-launch and other
workflows, 30-minute limits and every prior assertion remain unchanged.

This change supplies no collection permission, source acceptance, predictive
qualification or wagering authority. The authentic acquisition failure's root
cause remains UNKNOWN. No retry or replacement capture window is proposed.
