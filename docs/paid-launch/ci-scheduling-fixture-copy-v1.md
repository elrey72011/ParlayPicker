# CI scheduling fixture copy consistency

The synthetic session template in tests/test_ci_scheduling_scope.py now disables
only its own automatic Git maintenance and completes synchronous
git gc --prune=never before any copy. This is the same three-call safeguard
released in #2430 for tests/test_readiness_dashboard_scope.py. Every original
statement and assertion remains; no copy error or required object is ignored.

Controlled real Git packing after the actual copy enumerates loose object
paths reproduces the setup failure while preserving the source HEAD/tree and
integrity. The corrected actual fixture survives that operation. Missing and
corrupt required packs still reject through the original copy/guard path.
The process responsible for the historical Ubuntu deletion remains UNKNOWN.

The exact successor binds the merged predecessor, comparison manifest, reviewed
source bytes and policy-only seal. Historical #2432 scope tests receive its exact
inputs; older historical projections continue from their exact predecessors.
New current-scope tests and real CLI processes receive the live tree. Cold module
lookup and both standalone launch paths remain required validation.

No acquisition, model, workflow, nine-shard configuration, advisory profile,
scheduler, timeout, storage, runtime pin, attempt journal or operator authority
changes. This correction supplies no release clearance until the applicable
final-head and subsequent owner-merge checks actually pass.

