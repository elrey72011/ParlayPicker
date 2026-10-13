# Immutable reconstruction dependency successor

The cost-only candidate faa92abe introduced a reproduced scope hot-path
regression: its 43711-character profile-history dictionary became a live
container in the generic pure-reconstruction context. Every warm memo lookup
froze that entire dictionary. Existing immutable facts need no such container
traversal; live validation decisions must still be reevaluated.

The correction uses exact serialized bytes for the reviewed reconstruction
subject. The fixed profile path and exact after-hash are checked before parsing.
The exact predecessor profile hash is checked after reconstruction. Unknown or
changed bytes remain untouched. Parsing is not cached. Live policy validation
checks the serialized subject against its binding and the existing declaration
on every validation, including changed declarations, bytes and hashes.

The existing reconstruction memo, its limits, and source/code-context keys remain
unchanged. Only its new unnecessary mutable-container dependency is removed.
No mutable validation decision is cached. Original dictionaries, policies,
bindings, test files, profile, scheduler, workflows, acquisition and mathematics
remain unchanged. The exact predecessor guard is reconstructed byte-for-byte.

The first cost candidate and stopped validation are retained separately. Stopping
that run follows a reproduced new cause; it is not a timeout retry or passing
validation. The original #2419 timeout evidence and the separately approved
completed Ubuntu comparison remain preserved.

The code-only local probe blocks network and uses the actual memo/reconstruction
path in the same Python 3.12.14 executable (SHA256
10d845f50a2af64e3500bb2fcb348b5bc98a75d8ddada63e45ba1da6a1fc79d1).
One hundred warm odds-reconstruction calls took 0.079212 seconds before and
0.003099 seconds after this change, a 96.1% reduction in that microprobe. Both
recorded 100 cache hits/one miss. The 43711-character mutable container disappeared
from the hot context; the unchanged 365-character predecessor declaration remains.
Context fields are 341/437, within the existing 512-field bound. These are local
microprobe measurements, not full-suite timing predictions or sports inference.
Final-head CI and executed-identity reconciliation are still required.

Successor tests cover exact reconstructed bytes, unknown/changed sources, live
payload/declaration/hash rejection, the actual hot context, all ancestor matchers,
cache bounds, and dirty/staged/committed/resealed attacks. Scientific and wagering
requirements are untouched; this correction supplies no authority.
