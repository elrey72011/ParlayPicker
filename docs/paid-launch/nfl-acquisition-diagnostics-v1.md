# Private NFL acquisition failure diagnostics v1

Scope: diagnostic retention only for the existing bounded NFL schedules-release
acquisition. No new transport, trust setting, plan window, source, inference or
collection authority. NCAAF acquisition and all historical readers are unchanged.

## Reproduced loss and retained authentic limit

At base `0a19dae206d80b482a516e0b780275cfd8278a93`, a labelled synthetic
`PermissionError(EACCES)` at the actual HTTPS connect boundary passes through
`scripts/nfl_scoring_acquisition.py` into an immutable failure bundle, but the
bundle/report retain only `NFL_ACQUIRE_TRANSPORT_FAILURE`, `INCOMPLETE` and one
attempt. The added regression fails on that missing diagnostic before correction.
Initial local test setup also encountered the sandbox temporary-directory
restriction; rerunning with a fresh allowed `--basetemp` reproduced the code loss.
Raw local failure/probe receipts remain private, outside GitHub.

The authentic October 11 attempt at 16:46:20.981008 UTC failed before a stream
was returned. No HTTP status, response size or underlying OS/TLS error was
recorded. Its root cause remains **UNKNOWN**. No diagnostic network probe,
retry, resume or trust change is justified by that missing evidence.
Its existing failed bundle and journal are preserved byte-for-byte. This
extension cannot retroactively add facts to that record.

## Diagnostic contract and read-back

Only new `INCOMPLETE` bundles gain the optional `failure_diagnostics` record,
version `nfl-acquisition-failure-diagnostics-v1`, under the existing sealed private
`nfl-unaccepted-scoring-body-v2` wrapper. Successful bundles and reports do not
gain this extension; old bundles remain valid without it and are never rewritten.
The wrapper hash includes every diagnostic field. Acquisition retains exclusive
file creation, fsync and exact-byte read-back. The CLI verifies the sealed wrapper
and allowlisted fields before printing its **private execution report**.

| Field | Meaning / constraints |
|---|---|
| phase | Fixed failing boundary: connection setup, HTTPS connect, request send, response receive, headers, body read/validation, response validation, authorization, journal or custody. Unknown seam facts remain UNKNOWN. HTTPS_CONNECT includes TLS negotiation; category distinguishes a recognized SSL failure. |
| exception_category | Fixed class-based category: TLS certificate verification, TLS, DNS, timeout, connection, HTTP protocol, OS, local ValueError/policy rejection, or UNKNOWN. A category is not a root-cause diagnosis. |
| os_errno / os_winerror | Actual numeric exception fields only; no inferred platform code. |
| tls_error_code / tls_verify_code | Actual numeric SSL errno/certificate verify code only; no reason, certificate or verification-message text. |
| http_status | Valid numeric status only after a response was actually returned; UNKNOWN before that boundary. A 200 status does not prove a complete body. |
| elapsed_seconds | Monotonic seconds from the current transport attempt boundary to failure; for non-transport rejections, from that acquisition attempt boundary (or acquisition start before an attempt). Finite nonnegative measured value only; unavailable/invalid measurements remain UNKNOWN. It is not a wall clock, publisher clock or response-availability receipt. |

Never retain raw exception strings, arbitrary class names, filenames, response
headers, cookies, certificate descriptions, partial bodies or signed URLs in
diagnostics. Numeric fields and fixed enums fail validation if extended with
free text. Existing public rejection codes are retained: connect/request/response
failures remain `NFL_ACQUIRE_TRANSPORT_FAILURE`; unclassified body-read and
connection-constructor failures remain `NFL_ACQUIRE_STOPPED`; known header and
HTTP rejection codes stay distinct. No public-board consumer is changed.

## Verification and fresh-capture handoff after review

Labelled synthetic fixtures exercise the actual CLI → HTTPS transport → private
failure bundle → hash-verified read-back/report, with all external transports
blocked. They cover OS/DNS/TLS/timeouts, request/response/read errors, actual
HTTP rejection statuses, numeric-only fields, secret canaries, unchanged success
and historical reports, default TLS settings and spent-plan rejection.
No authentic inference or response acquisition occurs.

After this PR is reviewed and separately integrated, prepare **one new plan** and
exact authorization/configuration against that reviewed revision. Reuse unchanged
permission/storage receipts only if still applicable; recheck access and protections.
Robert must separately approve its future window and exact controlled configuration,
then explicitly instruct the single manual capture. Use a new plan identity and
exclusive journal, maximum two attempts (primary plus only the validated asset
hop), no retries/fallback, existing 10/10/60-second deadlines and size limits,
zero Odds/CFBD usage and $0 additional spending. Stop at CAPTURED_UNREVIEWED.
Do not run or renew the spent October 11 plan. No replacement window is proposed
while this fix is pending. A failure must preserve the new safe diagnostics and
stop; source review and inference remain separate later decisions.
