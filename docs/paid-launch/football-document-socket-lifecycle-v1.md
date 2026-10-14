# Original-document response completion

This correction is limited to the response-body iteration in the existing
original-document transport. Plan versions, authorization, TLS, requests,
limits, receipt meanings, readers and model behavior are unchanged.

## Reproduced defect

At sealed head `5f9643c3df65f38f3ac2d993aefa116539d602e6`, a real
`socket.socketpair()` and the standard-library `HTTPSConnection`,
`HTTPResponse` and timed worker yielded all 75 synthetic body bytes. A
keep-alive response completed. A `Connection: close` response then reported
`DOCUMENT_READ_FAILURE`, phase `BODY_READ`, OS error 10038 on Windows
(closed-socket errno 9 on POSIX). Chunked closing responses also reproduced
the failure. No external request was made.

The previous BytesIO-backed socket fixture retained an inert `settimeout`
method and therefore missed this lifecycle. Retaining `conn.sock` is necessary
for the body, but insufficient: the final framed read closes HTTPResponse's
file and the detached socket. The next loop iteration must not touch it.

## Correction and retained rejection behavior

Check cancellation and the existing monotonic deadline first. If the actual
HTTPResponse reports its body file closed, finish iteration without another
socket operation. Otherwise keep the existing socket timeout and timed read.
Do not catch or suppress socket errors. An unexpectedly damaged, unread socket
still produces the sanitized `DOCUMENT_READ_FAILURE` diagnostic.

The acquisition consumer still verifies exact Content-Length against all
collected bytes before retaining any successful object. A declared body that
ends seven bytes early remains `INCOMPLETE / DOCUMENT_INCOMPLETE`, consumes
one journaled attempt and cannot resume. Chunk framing failures still reject;
body/decoded/aggregate limits are unchanged. Cancellation or expiry after the
last yielded bytes still rejects, and a stalled body read still stops under
the unchanged shared deadline. Closing interrupts a pending detached read;
the fixture does not assume which of the OS timeout and caller deadline wins.

## Verification

Labelled local socket fixtures exercise the actual CLI, transport, timed worker,
journal, original-byte custody and immutable read-back for keep-alive and
closing Content-Length bodies below, at and above the 8,192-byte read boundary.
They also cover chunked and close-delimited bodies, actual truncation,
post-body cancellation/expiry, stalled reads and unexpected socket damage.
Synthetic DNS/TCP/TLS installation uses an already-connected local socketpair;
the HTTP implementation, socket ownership and buffered body reads are real.
External transports remain blocked. No authentic evidence, journal, source
review, catalog, probability or wagering authority is created or modified.

The original sealed head and its passing CI are retained as predecessor
evidence; they do not establish readiness of this corrected head. A fresh
implementation, exact successor binding, policy-only seal, all 15 checks,
artifact verification and five-way executed-identity reconciliation are required.
