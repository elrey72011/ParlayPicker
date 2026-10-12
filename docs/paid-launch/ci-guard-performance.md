# Exact guard performance successor v1

The paired offline probe at main 457a7f3e43abfbd95de64e4d9c79e2ca82bdfb21
and #2418 head 732e179b8ae976c9d87f963998635909978810d3 runs nine identical,
unchanged slate-audit, canonical-download and football-research scope cases.
Both cover the valid seal/ordered merge, dirty/staged/committed/resealed edits
and altered policy. External transports remain blocked. Python, Git, packages,
process-only Git long-path setting and test bytes are pinned identically.

| Probe | Elapsed | Git launches | Repository copying | Test-call phase |
|---|---:|---:|---:|---:|
| Main | 493.145s | 17,081 | 4.302s | 421.190s |
| PR before | 501.030s | 17,081 | 4.267s | 428.816s |
| Same PR + measured prototype | 188.349s | 3,852 | 4.275s | 117.950s |

Every probe passes the same nine cases. These include all four mutation stages
within each hostile-edit case. Counts overlap, not 27 distinct tests. Main's
session preparation is 61.810s; all setup is 66.139s. PR setup is 66.223s;
prototype setup is 65.282s. Compilation/execution, predecessor reconstruction,
hashing, live validation, directory traversal and reporting are recorded in
separate raw profiles. Fixture and Git timings overlap their enclosing phases;
they must not be summed. Long-lived batch-worker lifetime includes intervening
Python validation and is separate from measured query round trips. The initial
profiling-artifact preparation time is UNKNOWN; pytest reporting is included
in session residuals and recorded profiling functions. The raw profiles remain
local; no private sports evidence or original provider bytes enter this diff.

The identical launch counts show an inherited subprocess bottleneck. The small
main/PR time difference does not establish a regression or explain runner
variation. Windows measurements are not Linux runtime forecasts. Fresh exact
head CI, including upload and identity reconciliation, is still required.

## Versioned optimization

`scripts/scope_git_batch.py` creates a lazy, read-only `git cat-file
--batch-command` worker inside each validation invocation. Exact supported
HEAD/full-commit blob expressions use live info/contents requests. Every
request is read again; there is no object, HEAD, filesystem or decision cache.
Original Git operations remain the fallback for unsupported expressions,
missing objects, trees and objects beyond the internal stream bound. Original
body bytes are never truncated or rewritten. Contents are checked against
their Git blob identity. Mutating Git commands and checkout/dirty/index checks
continue through their original implementation.

The worker belongs to one resolved repository and validation context. Nested
checks may share that worker only in that same invocation/repository. A changed
ROOT cannot use another repository's stream. Contexts reset and workers close
on success or exception, with a bounded close/kill path. Existing session
fixtures and mutable per-test repository copies are unchanged.

The append-only guard layer strips only its exact reviewed bytes to recover
the entire prior guard, with its SHA-256 verified. Its pure reconstruction
cache holds at most two successful byte-to-byte results; input is capped at
8 MiB before caching, and output is smaller than input, bounding retained
input/output bytes at 32 MiB. Keys contain complete source bytes plus reviewed
source/parent identities. It carries no ROOT, Git state, module namespace,
acceptance, validation result or mutable policy. Changed source, oversized
source and cross-repository tests prevent contamination. All pre-existing
reconstruction/cache behavior remains unchanged.

No acquisition code, test file/assertion, scheduler, advisory profile,
workflow, 30-minute limit, model, source/scientific rule or wagering gate is
modified. Earlier implementation/seal policies and baseline hashes remain
immutable. Exact successor bindings and a subsequent policy-only seal authorize
only these performance paths. They grant no acquisition or wagering authority.
