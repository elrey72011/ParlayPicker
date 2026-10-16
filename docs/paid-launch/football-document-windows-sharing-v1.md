# Bounded Windows qualification lock synchronization

Owner-authorized exact patch3ff95cfff814783420547ef1b6e72834ee164162a975facd7db539afc9c4989d,
proposal590ca919f05a6c567554040c421145c97291d97b9a8e80b9d13afe6af36360f7.

Replace only the test's200ms timer with a five-second-bounded event wait. Signal
only after the actual replacement raises OSError; re-raise it unchanged.
Preserve every assertion, real Windows child handle, production retry/deadline,
journal counters and existing persistent-lock check. No synthetic error is
injected into the corrected fixture. A missing lock must still fail its original
assertTrue(errors). Cleanup joins the bounded releaser and real holder child.

The historical CI assertion and errors=[] are retained. Its exact scheduling is
UNKNOWN; a separately labelled synthetic delayed-replacement experiment proves
the timer race mechanism, not the historical schedule.

The actual Windows qualification test passed normal/delayed paths, while missing
lock and forced-failure cases rejected with child cleanup and temporary removal.
The original persistent-lock case retained its1.5-second production retry budget.
Validation used separate pinned test libraries and immutable application7c4fe71,
not operational data. Native Python3.12.14 differs from release3.12.15; final CI
provides the release-environment proof.

Fresh scope80 restores the exact scope79 source/policy and projects only the
exact reviewed fixture bytes for historical consumers. All original assertion
ASTs remain identical. Immutable source transformations are exact-content-bound;
live validation decisions are not cached.512 fields and8MiB bounds are unchanged.

#2426 remains diagnostics only; the authentic Odds guide trigger remains UNKNOWN.
No acquisition, inference, operational process/runtime/configuration/task/security
changes, merge or deployment. Preserve armed Memphis–Tulane process4988 and pins.
