# Standard closing responses and unchanged boundaries

Before operational handoff, a labelled synthetic HTTP/1.1 response with
`Connection: close` reproduced `DOCUMENT_READ_FAILURE` through the real
HTTPSConnection/timed worker. The standard library detaches `conn.sock` when
the response owns the body; accessing `conn.sock` afterwards loses a valid
complete response. This is an implementation defect, not an authentic provider
failure or a trust/TLS issue.

Keep the exact established read socket, apply the existing read deadline to
response headers and body reads, and close the detached body as well as the
connection. Interrupt pending reads before closing their buffered file. Clamp
the shared monotonic deadline to the exact collection authorization expiry.
Preserve original bytes, TLS verification, requests, redirects, body limits,
attempt accounting, receipts, old failure evidence and no-resume behavior.
Controlled configurations reject credential-bearing review text before journaling.

New synthetic fixtures use a realistic 15-minute collection window. Their
original quote/observation/review/inference clocks and stale/start assertions
are unchanged. No authentic clock or production freshness rule changes.

The document contract and four-document non-executable proposal are unchanged.
The document and NCAAF interpretation implementation/seal pairs remain frozen
predecessors. The final handoff must bind this subsequent exact software seal;
earlier completed checks do not establish final-head execution coverage.

The second sealed head completed CI with two frozen metadata assertions failing.
The source reconstruction returned the correct historical bytes, but inherited
in-memory metadata still named the parent after-images of the explicitly
versioned NCAAF changes. Compose exact current after-images and reverse patches
in the subsequent scope layer. Keep every historical before-hash, original
binding file and reconstructed predecessor function unchanged. This is metadata
preservation and grants no source acceptance or wagering authority. It supplies
no measured runtime saving. Unknown source edits still fail hash validation;
returned metadata is copied rather than cached as a mutable validation decision.
