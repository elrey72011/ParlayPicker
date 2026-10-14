# Fixture and provenance closure following the measured CI correction

The single authorized cycle at `0a5d76ae84519b8f9d2f437128e74410146ad058`
finished application work inside the 30-minute budget but exposed distinct fixture
and provenance errors. Original logs, XML, manifests and sealed identities remain.
This local successor is not release-validated; no additional CI authority is
inferred from the preparation authorization.

1. Eight new performance-reference cases required predecessor Git history absent
   from the unchanged depth-1 application checkout. The actual fixture reproduced
   the same exit-128 failure in a current-only synthetic repository. It now uses
   the existing independently hash-bound predecessor reconstruction, requiring
   SHA-256 `7fc3b110224763bc8845e733be4e2bcd0aa4e9ab014fb4ea01021ab717379832`
   before compiling the identical frozen reference. No history is fetched.
2. The new completed timing source omitted `checkout_commit`. Original run
   `38626230037` checkout logs establish the tested merge
   `c2f3a6eb3d583b407f34c3b1f85051b6d5c15a2c`, distinct from API/branch head
   `56da7ac68caccbba56d018d22de4dc3f211e019d`. Only that new provenance is
   completed; costs, historical observations and frozen revision 3 are unchanged.
3. The historical PDF-cost revision assertion was reading the newest cost rows.
   It receives an isolated, exact-hash historical profile through the existing
   immutable reconstruction mechanism. Every original assertion and test identity
   is retained; current-profile tests continue to validate the refreshed costs.
   Unknown or changed profiles cannot receive the historical view.
4. The remote-canonical template lost loose Git-object paths during copy. A real
   Git/copy2 synthetic interleaving reproduced 154 missing entries. Its historical
   deleting actor remains UNKNOWN. Only that synthetic template gains local
   `gc.auto=0`, `maintenance.auto=false` and synchronous `git gc --prune=never`
   before copies. HEAD/tree/fsck and dirty/staged/committed/resealed rejection
   remain intact. Errors are not caught or suppressed.

Focused regressions cover history-free fixture operation, attributed checkout,
historical-input identity/idempotence/tampering, isolated copies, original assertion
AST equality, and real canonical-template integrity. Exact reversible bindings
preserve schema 68 and all earlier source. No mutable validation decision is cached.

Acquisition/scanner behavior, model mathematics, histories, clocks, custody,
authorizations, six whole-file shards, scheduler, advisory cost values, required
workflows, network blocking and 30-minute limits remain unchanged. No provider,
document, Drive, inference, publication, merge or deployment operation is included.

Pushing this candidate to #2424 would trigger another complete CI cycle. The prior
one-cycle authorization is spent; the additional exact-head cycle must be approved
separately. Local tests do not establish merge readiness or authentic operation.
