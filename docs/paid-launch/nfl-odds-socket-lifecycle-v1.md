# NFL odds response socket lifecycle

Read-only operational preparation reproduced a necessary defect on main
`a796056aca70ae79ec5c4f5fd4a644fb80f9cd51`: real HTTPResponse detaches
`conn.sock` for a `Connection: close` response. The old odds body iterator then
fails before reading its complete body. Keep-alive passes. This is a local
synthetic reproduction, not evidence about any authentic Odds response.

The separately approved patch retains the readable socket before getresponse,
tests framed completion after the existing deadline/cancellation checks, and
interrupts the detached read before closing the buffered response. Only cleanup
shutdown errors are tolerated; read errors and incomplete framing still reject.
Complete-body validation remains in read_body. Request/auth contracts, original
bytes, V1/V2 flags, TLS, bounds, public rejection codes and durable spent-attempt
accounting are unchanged. No new contract version is necessary.

Real local socketpairs replace only the DNS/TCP/TLS connect boundary. The actual
HTTPSConnection request, HTTPResponse framing, timed workers, caller, journal,
custody and read-back run against labelled synthetic JSON. Tests cover close and
keep-alive, length/chunked/EOF framing, multiple chunks, truncation, malformed
JSON, oversize, timeout, cancellation, exact deadline boundaries and no resume.
Synthetic CLI tests use the unchanged original harness for clocks/security/key
entry; the real runtime verifier separately rejects the blocked test environment.
No production verifier or external-network blocker is disabled.

This work is isolated from PR2427 (PDF normalization) and PR2426 (scanner
diagnostics), with no unmerged dependency. Main predecessor bindings, all tests,
six shards, workflow/profile/scheduler/timeouts, admission, models, operational
pins, original evidence and authorizations remain untouched. An exact additive
scope successor is followed by a policy-only seal. No acquisition is authorized.
