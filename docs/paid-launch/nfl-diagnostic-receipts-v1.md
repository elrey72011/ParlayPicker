# Owner-only diagnostic lifecycle successor

The owner authorized a bounded correction after the unchanged cleared main
`ad5662c99b6529fddc1f40dbaac6853563e01c8e`, tree
`de1ab55c0e4dce2a539ebc859df8f2acfd069ce3`. The prior browser action remains
spent with handler entry, connection attempts and outcome UNKNOWN. Its missing
receipt is not evidence of zero database traffic. Release evidence is preserved.

Offline reproduction demonstrated that the old UI computes disabled before
setting spent, creates no start receipt, reports zero on an unexpected exception,
and loses its action guard when private session state is revoked. These findings
do not establish the cause of the historical hosted event.

This successor moves dispatch into an authenticated callback with an independent
locked claim. It records a random invocation ID and actual UTC start before
imports/diagnostic work. Sanitized HANDLER_ENTERED, DIAGNOSTIC_STARTED and terminal
receipts go to the existing process stderr logs, outside the transient display.
Only known contract fields survive projection. No raw exception, setting value,
hostname, URL, credential, source body or secret-derived digest is emitted.
Unexpected exceptions and interruptions retain UNKNOWN counts. Native counts are
retained only when returned by the released instrumentation. STARTED without a
terminal record means interrupted/unknown, not failure before connection.

The session's nonsecret spent tombstone survives authorization loss; private
uploads, hostname and result display remain revocable. Reopening or rerendering
only reads retained session evidence. The tombstone and process locks are not a
distributed durable allowance across workers/restarts. External owner authority
must still be spent before any authorized hosted action. The existing process
logs provide session-independent evidence, subject to hosting retention; they
are not a backup or a database ledger. Loss of terminal evidence prohibits retry.

Unchanged: actual settings resolver, released diagnostic, Store.postgres/verify,
restricted login, verified TLS, five-second connection deadline, statement/lock
limits, driver-attempt guard, read-only queries, models and provider transports.
Both execution switches must remain the literal string `false`. This engineering
scope authorizes one conditional PR cycle from the existing unused allowance;
no additional hosted diagnostic job. Owner merge, automatic main clearance and
deployed loaded-code verification precede the separately authorized single
successor storage action. A controlled restart is conditional on needing it.

After release: Workspace -> Preview & Publish -> enter publishing token privately
-> Show private NFL storage diagnostic -> Read running NFL build information
(no connection). Verify current cleared commit/tree, seven loaded modules and
process evidence. Enter the independently verified direct production hostname
privately. Spend the successor allowance, then explicitly run the storage check
once. Retain its invocation ID, actual clocks and sanitized receipt. No retry,
second session or restart replenishes authority. Keep normal guided workflow,
Refresh, acquisition and inference inactive. No password/key/TLS/grant changes
are justified without observed evidence.

This does not authorize merge, immediate deployment, production writes or
prospective forecasts. Persistence, save/reopen/restart and authentic event-specific
workflow acceptance remain separate. Neon Free and all authentic NFL/NCAAF records
are preserved. Historical release guards and predecessor expectations remain
authenticated; successor seals approve only the exact bounded change set.

The added13-test file has one isolated estimated timing-profile entry. Its cost
is the entire completed26-test local process wall time times the existing1.15
allowance, rather than a narrowed favorable test-only timing. Every previous
profile field, source and cost remains exact. This includes collection overhead
again conservatively; the common retained job overhead remains unchanged. The
Windows estimate is not Ubuntu clearance. Final-head Linux jobs must still meet
the unchanged120-second reserve. The original fallback-only shortfall is retained.

Final interruption qualification: the terminal process receipt is emitted before
the interruptible session-state display update. A synthetic state-write interruption
proves this ordering; an abrupt process exit retains only pre-work evidence and
remains UNKNOWN. The final added file has14 cases. Its timing retains the slower
completed26-case process estimate after the complete27-case successor run; no
prior cost was reduced. Browser qualification verifies loaded fixture source.
An older owned local listener was found and cleaned up before final verification;
its preliminary browser observations do not establish final-source qualification.

Timing-source provenance preserves the established profile contract: each new
source records its observed local checkout HEAD, explicitly dirty worktree and
unverified checkout-tree status. Source-file and JUnit digests identify the
actual synthetic drafts. A checkout HEAD is not substituted for tested-tree
equality, and Windows timings are not relabelled as Linux measurements.
