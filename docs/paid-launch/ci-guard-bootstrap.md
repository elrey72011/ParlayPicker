# Isolated guard bootstrap preservation

The completed shard-3 report at 39296ffa records one unchanged test failure:
test_legacy_bootstrap_needs_no_successor_binding. Loading a copied guard alone
incorrectly read scope_git_batch.py at import time. The exact offline test
reproduces FileNotFoundError before validation; no original assertion changed.

This successor defers the exact-byte, hash-verified transport load until an
actual validation entrypoint. Isolated immutable predecessor reconstruction
needs no adjacent transport, binding or policy file. This grants no exception:
actual validation still fails for missing/changed helper, binding or policy.
Helper bytes are read and verified at every validation entry, even after a
module has been installed. No mutable decision, ROOT, HEAD or policy is cached.
The additional pure reconstruction cache holds at most two exact-byte subjects,
each capped at 8 MiB. Its conservative input/output limit is 32 MiB; together
with the prior three new caches the conservative bound is 128 MiB. Existing
predecessor caches are unchanged. The transport, numerical code, all original
tests/assertions, profile, scheduler, workflows and timeouts are unchanged.

The prior failed CI/XML/logs and seals are retained. Fresh complete final-head
CI, artifact upload and executed-identity reconciliation remain required.
