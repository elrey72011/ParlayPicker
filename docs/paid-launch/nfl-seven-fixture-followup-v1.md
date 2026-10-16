# Local follow-up: exact historical input and bounded pure context

The single release cycle on f9902c98 completed seven application artifacts but
failed historical fixture and pure-context assertions. It is preserved and is
not retried. This locally prepared successor is not pushed and has no additional
release-cycle authorization.

The six-to-five historical adapter recognizes an exact six-way source. The live
seven-way source needs a separately hash-bound projection to that six-way input.
The old assertions remain unchanged and current seven-way tests read the live
source. Isolated historical copies cannot mutate the actual checkout.

The new shared reconstruction JSON reference added a513th context field to a
pure adapter with an existing512-field ceiling. Restore the existing inline-byte
representation; preserve both512 fields and the8MiB source ceiling. This changes
no cache algorithm, mutable validation, mathematical or acquisition behavior.
Peeling the new guard must restore the exact failed predecessor bytes, including
its original shared reference, so the failed revision and its bindings remain
honestly represented.
