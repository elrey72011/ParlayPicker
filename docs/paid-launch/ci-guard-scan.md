# Exact immutable scan correction

The first implemented object stream passed all nine identical offline cases and
reduced Git launches from 17,081 to 3,852, but took 440.066s instead of the
prototype's 188.349s. The completed profile records 213,533 calls and 250.015s
of self time in the newly introduced predecessor-marker scan. This is new
overhead, distinguished from inherited Git and traversal cost. The raw profile
and first sealed candidate remain preserved. No profile, scheduler, workflow,
assertion, acquisition or fixture change is justified by that observation.

This append-only successor moves marker detection inside a pure exact-byte
cache. Every cache input is capped at 8 MiB before admission; each cache holds
at most two entries. Keys include complete source bytes and reviewed source
and predecessor hashes. Outputs contain only reconstructed or unchanged bytes.
There is no repository, HEAD, Git-object, policy or validation-decision state.
The two new pure caches each retain at most 32 MiB of input/output; the first
performance layer's existing two-entry cache is preserved (at most another
32 MiB, conservatively 96 MiB across the three; shared byte objects may reduce
actual retention). Older frozen caches remain unchanged.

The exact previous guard is reconstructed/hash verified. All older schema and
policy bytes remain unchanged, including the first performance pair. Live
validation still sees changed HEAD, dirty/index/committed/resealed facts.
Changed source and cross-repository tests reject contamination. Unknown bytes
without this successor marker remain unchanged for their original validators;
that byte identity result is not an acceptance or validation decision.

The same bounded nine-case profile and fresh complete exact-head CI are
required. Earlier interrupted or superseded results cannot establish final
coverage. Measurements are not future runtime promises.
