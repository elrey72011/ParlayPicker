# First PostgreSQL driver failure - diagnostic successor v1

Owner authorization: retain the first genuine driver failure, qualify it offline,
publish a separate reviewable PR and use one conditional complete PR CI cycle.
Base is cleared main `6de0aa70413f96f42b174e9ad6973ad33a2bb913`, tree
`54405c65432c071ba1dd97261b743047e8a0bf20`. No merge is authorized.

The original hosted operation remains SPENT/UNKNOWN. The latest retained invocation (identified in the private evidence package)
remains SPENT/FAILED:
CONNECT / SECOND_CONNECTION_ATTEMPT_BLOCKED. No preceding transport cause
was retained. Settings and scoped running-build verification passed; successful
TLS, authentication, actual permissions and schema verification were not shown.
This successor does not reclassify either historical operation.

Psycopg 3.2.10 Connection.connect catches each propagated Error within its
candidate-address loop. The trace observes that error in the connect frame,
before the driver's handler. An error before the first generator is retained
with zero admitted driver attempts, rather than fabricating a connection. Genuine
driver and escaping OS errors become first_connect_failure and the primary failure.
StopIteration, GeneratorExit, interruption and normal suspension/resumption do
not become connection failures. A second generator entry remains blocked before
its body and appears separately as blocked_attempt and second_attempt_blocked.
One application connect call and one admitted generator are different counters;
neither establishes internal libpq socket/address counts.

The safe projection contains phase, allowlisted exception class, SQLSTATE and
allowlisted structured errno/winerror when supplied. Missing codes are null
(UNKNOWN). SQLSTATE and typed OS information take precedence over bounded
in-memory examination of standard libpq clauses. Only fixed categories leave
that examination: authentication rejection, timeout, address-resolution failure,
certificate verification failure, connection refused or network unreachable.
Unrecognized, oversized or hostile exception text remains unclassified. Raw
errors/args, SQL, credentials, URLs, hostnames/IPs, keys and secret-derived hashes
are never retained in the receipt. Receipt fields are projected again before
logging/display. Category evidence is not permission to change configuration.

No store, SQL, model, replay reader, transport, timeout, TLS setting or owner
approval behavior changes. Existing tests/assertions, shard count and CI deadlines
remain intact. The source binding adds a legitimate authenticated successor and
retains exact predecessor bytes and frozen historical expectations. The added
regression file receives a completed testcase-sum timing entry (setup, call and
teardown once), with the unchanged1.15 allowance and retained common job/collection
overhead. All released profile rows and sources remain exact. The earlier
unpublished whole-process estimate and its sub-120-second projection are retained.
The new receipt-specific renderer checks use the unchanged render function with a
controlled display sink; the complete original AppTest widget/reopen/authentication
suite remains unchanged and executes separately. No original assertion is removed. Final Ubuntu capacity must still meet 120 seconds reserve.

Offline qualification uses the real pinned connection loop with controlled
candidate addresses/errors and blocked external transports/driver entry. Real
hosted compatibility remains distinct: local Python 3.13.14/libpq16.9 and CI
Python3.12 differ from observed hosted Python3.13.18/libpq17.5. No test success
establishes actual Neon storage connectivity or predictive accuracy.

After owner merge, automatic main clearance and scoped running-build verification
must pass. One controlled restart is allowed only if automatic deployment is
insufficient. Both NFL switches stay the literal string false and Neon stays Free.
The owner authorized ONE new successor read-only storage check, linked to both
spent predecessors, with its allowance marked spent before action. It is held
through engineering/review; logout, process restart or another session cannot
replenish it. The UI guard is session-local, not a durable distributed allowance.
Failure/interruption prohibits retry. No database writes/migrations, provider
acquisition, predictions, secret/key/password/grant changes or model changes are
authorized by this correction. Persistence and authentic forecast acceptance are
separate subsequent decisions.

Timing-source provenance records the actual clean local checkout commit/tree and
measured source identities. The candidate only adds explicit safe pytest IDs to
the message-classification cases. The new regression additionally verifies
a genuine OS failure escaping the pinned driver loop. Every existing released
test and assertion remains unchanged. Synthetic markers and oversized error messages must not
become CI case names. This is not a completed Linux timing measurement.

Two directly affected timing entries use one additive phase basis. The added
regression file retains the slower1.852-second earlier heavier fixture upper
bound (current complete context1.771 seconds), times1.15. The changed receipt
helper workload uses the maximum complete14-case sum3.517 seconds, times1.15.
Its old7.43935-second two-file/process bound is preserved as provenance; it
includes other tests and collection/startup already budgeted elsewhere. Original
source records remain exact, every other released row is unchanged, and common
overhead121.10075 seconds and the120-second reserve remain unchanged. This is
phase-accounting normalization, not a lower observed runtime or a Linux claim.

A network-blocked real-loop regression demonstrated missing first-failure errno
for an escaping OSError. The connect-frame observer now retains driver.Error or
OSError; neither StopIteration nor interruption is included. A clean local
checkpoint passes74 Python3.12 tests and60 Python3.13 tests (one expected Windows
platform skip in each context). It is offline evidence, not a hosted connection
or a qualified deployment. The legitimate final binding/seal follows separately.
