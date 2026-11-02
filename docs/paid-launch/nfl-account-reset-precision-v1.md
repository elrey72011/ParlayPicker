# NFL account reset precision — local candidate

Authority: the owner's actual November 1, 2026 reply, “Prepare and locally qualify
the narrow correction.” This authorizes local preparation/qualification only.
It supplies no CI, publication, deployment, acquisition or authentic inference
authority. Base: `9fb69bf62c59f595a05478f2fc9cc2ed489840b9`, tree
`3edaa30fca0853d61d5da8879900b7724af93471`.

The released form required a reset date and exact UTC time even when the account
dashboard reported only a date. This candidate preserves that exact-time path
and adds a date-only choice. It does not infer midnight as the actual reset.

For exact-time input, the original `reset_at` string and authority fields remain
unchanged. For date-only input the signed acquisition claim instead retains:

```json
{
  "reset_observation": {
    "precision": "date",
    "date": "2026-12-01",
    "timezone": "UNKNOWN"
  },
  "reset_time": "UNKNOWN",
  "reset_not_before": "2026-11-30T00:00:00+00:00",
  "reset_bound_basis": "date_start_minus_24h_unknown_timezone_v1",
  "supported_utc_offset_hours": [-24, 24]
}
```

This example is illustrative, not an account observation or owner attestation.
`reset_not_before` is an admission bound, not an observed reset timestamp. For
UNKNOWN timezone the method deliberately covers the full offset envelope from
UTC−24 hours to UTC+24 hours. The earliest possible reset is bounded by the start
of the reported date minus 24 hours. For a reported UTC date the bound is the
start of that UTC date, while the actual reset time remains UNKNOWN. Other
timezone labels, missing dates, invalid dates and conflicting precision reject.

The complete capture window must end **strictly before** the bound. Equality or
overlap rejects before any operation/HTTP claim. Current entitlement, a verified
included-credit floor, private-use permission and $0/no-overage findings remain
required. The UI describes the credit value as a floor, never a fabricated exact
dashboard balance. Date/timezone changes clear the date-only account findings.

Changing precision or entering account data performs no acquisition. The owner
must submit the existing explicit capture action. Claims remain durable and
spent, transport deadlines and guards are unchanged, and there is no fallback.
Quote age still starts at the provider's original market update, with a maximum
900 seconds and pregame numerical calls. Source permissions, model mathematics,
native scenario review, authentication, database schema and static replay remain
unchanged. The original store key, database URL, CA selector and protected NFL
switches are not read or altered by local qualification.

Local tests use synthetic providers/SQLite and explicit synthetic approval data;
the repository external-network blocker stays active. AppTest evidence is not an
authentic hosted forecast or browser/restart acceptance. Existing authentic
records remain untouched. The local successor binding retains the exact cleared
parent and predecessor sources and seals this bounded change. Existing checks
still reject unapproved source edits. Three new timing rows are explicitly CI
estimates from completed Windows file phases, with a separate five-second
estimated shared-fixture allowance and the unchanged 1.15 multiplier. All prior
timing values and the 13-shard/120-second reserve contract remain unchanged.
The measured checkout identities refer to retained local Git objects, not the
later metadata/seal commit. Actual hosted capacity still requires a separately
authorized CI cycle. There is no authority to publish or deploy this candidate.


## Evidence-supported timing successor, November 1

The owner's subsequent authorization permits bounded timing-profile correction,
local qualification and, only after all local gates and every projected reserve
meet120 seconds, one draft PR and one CI cycle. No retry, merge, deployment,
acquisition, authentic inference or secret/configuration change is authorized.

Preserve sealed1d1742cb4e8ca7a7e5c3ef3e33a17874d3cf28ff, tree
7849626e66b79cd581ac7581d8ecce837fea444c and its117.264033-second forecast.
No existing timing row, source, assertion, file count, timeout, overhead or
variability multiplier is lowered. Two previously unprofiled complete files now
have budgets based on authenticated completed Linux main9fb69bf execution and a
complete pinned Windows context with identical test bytes. The larger measured
file phase is used, plus an explicitly estimated five-second shared-fixture
allowance, then unchanged1.15 variability. The rows remain estimates for future
CI; original measured observations and tested checkouts are identified separately.
The native scheduler reassigns all9358 identities over thirteen whole-file shards
without changing collection order. Final actual capacity is a separate gate.
The exact original profile is recoverable by removing only the two added rows
and their new source. Application/model/storage and all50 new test bodies remain
unchanged from the preserved candidate. Historical consumers still reconstruct
their independently authenticated predecessor bytes.

## Local standalone-entry qualification after the preserved failed cycle

Preserve PR2454 head b9a813824bec02d736cfb5fb5ca7131aa76b835e and its
single release cycle. The standalone scope entry loads authenticated files and
does not establish a scripts package search path. The new package-based account
dispatch therefore failed there; the local harness's inherited PYTHONPATH masked
that context. A fresh process without PYTHONPATH reproduces the exact failure.

This separate local preparation replaces only the three new account dispatches
with the established authenticated-file loader. It checks the helper's component
hash before executing its source, using the explicit root or authenticated
module location. Parent reconstruction, policy and validation assertions remain
unchanged. No application/model/storage/test-body/timing-row change is made.
The existing exact direct CLI, unchanged paid-scope case, package callers and
tampered-helper rejection must pass without an inherited repository path.

This local candidate and its necessary bindings/seal are prepared for owner
review only. The original single CI allowance is spent; no further publication,
CI cycle, retry, merge, deployment or operational activation is authorized.

## Bounded historical setup correction after failed capacity qualification

The owner's subsequent instruction authorizes measurement and correction only
of demonstrated repeated setup work, preserving fresh authentication, mutation
rejection, fixture isolation and every original assertion and identity. The
failed published head, qualified standalone correction, all measurements and
unqualified placement experiments remain preserved separately.

Ten native authentications compiled the same two complete helper byte strings
twenty times. The scoped correction retains immutable compiled code for those
two fixed helper paths. Each load still reads and authenticates physical source
and bindings and executes into a new module dictionary. Caller code, compiler,
complete source bytes, filename and interpreter/compile flags scope reuse. A
changed source or compiler cannot borrow an older entry. The helper is itself
pinned before execution. No approval, policy outcome, writable repository or
mutable module namespace is cached. The owning fixture suspends/restores the
compiler adapter with its other views and clears retained code on every exit.

The lower byte-transform memo prototype was measured and rejected because it
was slower. It is not part of this implementation. Existing reconstruction
memos and their original limits remain unchanged.

The next focused measurement found twelve exact source strings compiled by
unchanged native predecessor constructors on every fresh bootstrap. Compilation
used about 2.27 CPU seconds of a 3.56-second warm construction. A synthetic
compiled-code-only prototype reduced warm construction to 1.39 seconds, retaining
all native reader/output checks and fresh namespaces. The twelve source hashes
and byte lengths were independently retained from the unchanged measurement
seal d446ea65214ca6680dfd1f051e0e2829c057ec2e.

The fixture now admits code reuse only for that exact catalog, under a digest of
the complete authenticated bootstrap inputs, helper identity and interpreter.
The native byte readers still run on every construction. The compiler adapter
recognizes only code descended from the authenticated bootstrap/program code;
unknown programs keep the native path. Each execution creates and evaluates a
fresh namespace. The owning context restores every inserted compile/exec name
on exit and native-context suspension. Only immutable code and digest tuples
remain in a process-local pool, limited to sixteen entries, 32 MiB and 128,000
graph nodes, cleared by pytest configuration cleanup. No verdict, mutable
binding, writable repository or historical module is retained by that pool.
Each reuse verifies the identity of the original native-compiler code object
and its already-proven immutable constant graph. Replacing a cached code entry
is rejected even if the replacement's digest is recomputed. Source and binding
authentication and all program evaluations remain fresh.

The already-authorized ONE corrected-head publication/release CI cycle remains
available only after complete local and evidence-supported capacity gates pass.
The existing thirteen shards, timeouts, 120-second actual reserve, application,
model, storage and protected evidence remain unchanged. This source declaration
does not itself establish those gates. No merge, deployment, provider request,
authentic prediction, secret change or operational activation is authorized.


Bounded authenticated-reader successor (local qualification only)
The owner authorized one direct authenticated reader entry at the demonstrated
factory/reader double-authentication boundary. The real loader code remains the
compiler caller. The direct operation authenticates physical binding/helper
source before exec and the complete current component graph before decoding.
It passes the freshly returned binding directly to the existing pure decoder;
it retains no approval verdict. Standalone factory and later public module reads
still authenticate. Normalization, hashes, preimages and all substantive checks
remain. Exact inverse source transitions preserve historical guard assertions.
Frozen450672903bf5443457e87619767bf74cf5103ed6 and the179-pass/3-fail reuse
experiment remain unchanged. No cache variant is added. One conditional release
cycle remains held behind every local and capacity gate; no new cycle authority.
