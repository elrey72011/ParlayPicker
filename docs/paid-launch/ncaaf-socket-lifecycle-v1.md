# NCAAF HTTP response socket lifecycle correction

Verified main is `8f92728d7538afc1a9a6a842fbee5f172e241ec7`, tree
`3bd7dab87e663b131f18278e7a127211be47da55`. Original plan, custody,
authorization, reader and result versions remain unchanged. This is a transport
implementation correction, with an exact scope successor and policy-only seal.

## Reproduction and correction

Before editing transport, the new actual acquisition/socket regression recorded
**four failures and two passes**: content-length/chunked keep-alive completed;
closing responses and EOF-framed responses retained an incomplete bundle with
`NCAAF_PILOT_TRANSPORT_FAILURE` after one spent request. Real `HTTPSConnection`
and `HTTPResponse` ran against local socket pairs; only their DNS/TCP/TLS connect
boundary was synthetic. No external request or authentic input was used.

`HTTPConnection.getresponse()` clears `conn.sock` when the response will close.
The response's file still owns a readable reference. After its final framed read,
that file can close the socket too. The previous NCAAF stream dereferenced
`conn.sock` before each read, failing before reading a closing response.

`HttpsTransport` now retains the connected socket before response handling,
checks cancellation and the shared deadline before checking response completion,
and checks `HTTPResponse.fp is None` (the definition of `isclosed()`) before
touching a socket closed by the final framed read. Existing synthetic stream
protocols without an `fp` attribute still use their explicit EOF return.

Cleanup cancels, shuts down the retained socket, closes the response and closes
the connection. Only shutdown errors during cleanup are ignored; actual
settimeout/read errors still propagate and acquire retains its original public
failure codes. Windows may complete a pending select at its bounded read timeout;
the regression requires cancellation within the original shared budget, not an
unsupported instantaneous interrupt guarantee.

The earlier NFL document correction informed the socket/completion ordering.
NCAAF retains its own `read1`, JSON/gzip verification, credential handling,
10/10/60 capture and stricter discovery limits, and durable attempt accounting.
The NFL document transport/task is not changed.

## Validation and preservation

The focused offline socket, existing NCAAF capture/discovery and previous NFL
document socket suites pass: **174 passed**, no failures/skips. They include
original numerical/capture/export/display assertions; this correction adds no
authentic or historical inference. The first local test command stopped at an
inaccessible existing Windows pytest temp directory; its setup-error XML is
retained separately. The first candidate run had 173 passes and one new
cancellation-test failure because it assumed sub-second Windows shutdown; measured
shutdown/close evidence established completion at the configured two-second read
budget. Existing assertions were untouched. The revised regression tests the
unchanged shared deadline, cancellation rejection and absence of another request.

New tests cover closing/keep-alive length/chunked/EOF framing, multi-chunk bodies,
exact size/one-byte-over, truncated/malformed/short lengths, read errors, real
pending reads, completion/deadline ordering, cancellation, byte/hash and clock
preservation, actual capture/discovery read-back, immutable spent journals and no
retries. Synthetic failures are sanitized; no credentials enter receipts.

No existing test file/assertion, workflow, shard count, timing profile, scheduler,
model, admission, history/lag, freshness, alias, trust catalog or source permission
changes. All predecessor policies and the frozen version-10 baseline remain
byte-exact. The binding records exact original/source reconstruction hashes;
the subsequent policy-only seal binds implementation/tree and unchanged files
without a circular commit hash. Required final-head CI results and artifact /
exact-once identity reconciliation are reported in the PR receipt after execution.

## Thursday remains conditional

ECU AWAY at UAB HOME, October 15, 2026, 19:30 Eastern (23:30 UTC), remains a
proposal. The strict seven-day cutoff is October 8 at 23:30 UTC. Public schedules
and box scores support four ECU/five UAB scoring histories; weeks 2/3/4 can supply
the minimum three yardage histories each. Actual original CFBD bodies are absent
from inspected current custody, and season/weekly full-body fit remains UNKNOWN.

Separate discovery needs no future Odds event ID/hash. Capture/admission still
need attributable DraftKings feed/outcome-to-product/listing and effective
full-game/OT rule-edition applicability. The inspected general rules identify
West Virginia; other retail templates cannot silently establish this feed's
jurisdiction/product. Missing documentation establishes no extra licensing
obligation. No inquiry is sent and no listing facts or owner findings are invented.

After review/merge/release checks, resolve the exact mapping and target-bound
advance review, approve a fresh exact discovery plan and controlled configuration,
then separately approve capture. The proposed 11 a.m. NCAAF discovery / 5 p.m.
fresh capture and 7 p.m. NFL quote slots are unapproved and not scheduled.
No quote is acquired while these prerequisites remain unresolved. Owner review
and separately authorized inference must precede kickoff and stay within 900
seconds of the original provider observation; receipt/review cannot refresh it.

The approved October 15 10 a.m. NFL three-document task and October 16 Army–FAU
authorization remain unchanged, as do WNBA work, the saved Rams result, originals,
failed bundles, spent journals, storage, encryption and backups. No acquisition,
trust activation, inference, task/Drive change, deployment or wagering is included.
Unqualified results remain private research, PASS and zero stake.
