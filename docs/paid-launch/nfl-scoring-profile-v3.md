# NFL original scoring profile v3

Base `cf85c803df4b34c2c20bb931128166b87369c5bd` is the actual owner merge
of #2414: ordered parents `909804595e5626dbc65cc9c79391ca0d1131aa3e` and
`dd6c9cf76b93fffd6defe81a27dfc3d46febd966`, tree
`771c5321bd4e5c2b1decd4f169f4fb15e4142188`. All 13 post-merge checks
completed successfully. Hosted revision is not established by those checks.
Original records, v1/v2 readers, old acquisition plans, source projections,
NCAAF's October 16 discovery and all existing workflows/tests remain unchanged.

## Original source and permission

The selected source is the **nflverse/nflverse-data schedules release games.csv**:
`https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv`.
Identity: `nflverse:nflverse-data:schedules:games.csv`, repository ID 452908115,
release tag `schedules`, exact asset name `games.csv`. It is one complete,
multi-season original CSV. Season/team/row selection happens in separately
hash-bound projections, never by filtering or reserializing the retained body.

The official [release](https://github.com/nflverse/nflverse-data/releases/tag/schedules)
describes replacement when nfldata changes. Metadata observed on October 11,
2026 identifies release 251386473, asset 630778441, **2,183,999 bytes**, update
2026-10-11T13:06:45Z and published digest
`7d27ecd3fee820b083e0473896f5facdec2ce7a00fb2412c89a9a4f78ee2c4e1`.
These are metadata observations, not a downloaded or verified body. Release
asset identity/digest may change before execution. An observed response gets
its own immutable body hash and subsequent review; the metadata digest is
not a future-response requirement or publisher first-publication clock.

The distribution repository's [LICENSE.md](https://github.com/nflverse/nflverse-data/blob/a78da18ca93f167bd0a2af274df48f539fa4153c/LICENSE.md),
[DESCRIPTION](https://github.com/nflverse/nflverse-data/blob/a78da18ca93f167bd0a2af274df48f539fa4153c/DESCRIPTION)
and [README](https://github.com/nflverse/nflverse-data/blob/a78da18ca93f167bd0a2af274df48f539fa4153c/README.md)
publish **CC BY 4.0** for its data releases. The
[license](https://creativecommons.org/licenses/by/4.0/) grants copying and
adaptation subject to its conditions; private retention and derivation need
no additional copyright permission merely because a body is downloaded.
Retain attribution to Lee Sharpe/nflverse, source and license links, license
notices, and a statement of projection changes. Do not imply endorsement.
Public availability alone is not a substitute for permission. The grant
does not establish operator rights, account restrictions, or settlement facts.
Owner advance review of this exact private scope is still required by the
repository acquisition contract; no such receipt is manufactured here.

`nflverse/nfldata` is a distinct upstream repository. Its inspected root tree
did not establish a corresponding data LICENSE. A software MIT license or the
downstream repository's CC BY license is not transferred to nfldata. This
successor selects the attributable licensed release, not the earlier raw
nfldata endpoint. The old plan and its rejection remain intact.

The official [update schedule](https://nflreadr.nflverse.com/articles/nflverse_data_schedule.html)
describes game/schedule updates every five minutes during the season. Assets
are mutable; every new body or correction requires a new subject and owner
review. Publisher first-publication time remains **UNKNOWN**. Verified complete
receipt/first observation supports only genuinely new prospective inference.

## Explicit versioning and bounds

Profile `nflverse-schedules-scoring-4m-v1`; custody
`nfl-original-scoring-custody-v3`; inputs `nfl-owner-scoring-private-inputs-v3`;
owner review `nfl-owner-scoring-verification-v3`; result
`nfl-owner-scoring-private-result-v3`. Routes dispatch explicitly. A v1/v2
packet or review cannot borrow this profile. v3 scoring custody is valid only
under the `scores` key with this exact source and transfer receipt.

| Boundary | v3 | Existing versions / NCAAF |
|---|---:|---:|
| Selected scoring CSV wire bytes | 4,194,304 | 524,288 |
| Selected scoring CSV decoded bytes | 4,194,304 | 524,288 |
| Every other original body | 524,288 | 524,288 |
| Aggregate decoded original objects | 6,291,456 | 2,097,152 |
| Serialized selected packet | 12,582,912 | 8,388,608 NFL v1/v2 |
| Combined selected packets | 16,777,216 | 16,777,216 |

The existing maximum four selected packets and 16 original objects remain.
Exact boundaries are inclusive; one byte over rejects. No truncation, source
splitting or reserialization can establish complete original custody. Full
CSV schema is verified; unchanged projections/native mathematics determine
the exact season, completed-before-current-UTC-day history, last-five inputs
and eleven ordered features. Neither model/artifacts nor formulas change.

The initial labelled maximum-size synthetic CSV (4 MiB, 63,717 rows) produced a
5,705,357-byte packet. Local measured parsing: 0.893 seconds, 63,340,173 peak
tracemalloc bytes; serialization: 0.0106 seconds, 14,133,064 peak bytes.
These are a synthetic shape and environment, not production upper bounds or
source-size guarantees. Parsing transient memory exceeds CSV bytes. The test
retains measured receipts; overhead still fits the packet/selection limits.
Near-metadata-size and maximum-size full-path tests preserve probabilities.
Final-head measurement receipts are retained separately. A malformed wide CSV
probe reproduced eager DictReader padding before rejection. The v3 parser now
checks each row's width before building its dictionary or consuming later rows;
well-formed row and empty-line semantics are unchanged. No source body is filtered
or rewritten, and original readers retain their original parser behavior.

## Isolated acquisition

Plan `nfl-scoring-original-plan-v2` permits one fixed primary HTTPS GET, with
one conditional asset GET. Maximum **two journaled attempts**, one original
object. The selected scoring snapshot is season 2026 research infrastructure;
it invents no target event, current offer or entitlement to infer.

The [GitHub asset contract](https://docs.github.com/en/rest/releases/assets#get-a-release-asset)
permits a 200 response or redirect; GitHub documents
[`release-assets.githubusercontent.com`](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
as its release download host. Only a primary **302** may lead to that HTTPS
host, exact repository path `/github-production-release-asset/452908115/`,
one UUID asset path and attachment filename `games.csv`. Fixed signed-query
keys only, no credentials/userinfo, port changes, fragments or additional hop.
The transient signed URL never enters receipts or logs: retain its SHA256,
sanitized path and query hash. It is not an accepted source fact by itself.

The plan also binds the exact permitted signed query keys and HTTPS-only
`spr` declaration. Microsoft's [SAS specification](https://learn.microsoft.com/en-us/rest/api/storageservices/create-service-sas)
documents that optional protocol field; unknown keys and HTTP-enabled declarations
still reject. A synthetic legitimate `spr=https` URL and repeated ignored
cookies reproduced rejection in the initial new transport. The correction
supports those forms, without recording signed URLs/cookies or changing any
framing/identity/limit gate. [RFC 6265](https://www.rfc-editor.org/rfc/rfc6265.html)
permits repeated Set-Cookie response fields; this client neither uses nor stores
them. No authentic scoring response or real asset-hop URL was retrieved.

No automatic redirects/retries, library importer, pagination, fallback provider,
Drive, model service, enrichment or normal live-analysis entrypoint. Each
attempt is fsynced **before** request dispatch; an exclusive plan journal
blocks resumption/repetition even after timeout or interruption. Status,
headers, TLS/host, content type/encoding, wire/decoded bounds, Content-Length,
gzip EOF/expansion and deadlines are checked. Duplicate framing, routing or pagination headers reject. Repeated unrelated
headers (including Set-Cookie) are ignored and never forwarded or retained.
Redirect headers are limited separately to 16 KiB; its body is not downloaded
as scoring evidence. The CSV transfer alone has the 4 MiB wire/decoded caps.
Stop on any unapproved response, auth/rate-limit error, incomplete body, deadline
or oversize. An interrupted/spent plan needs a separately reviewed new plan;
renaming its journal is prohibited.

Connect/read/total deadlines are **10/10/60 seconds** in the proposal. It requests
identity encoding; unexpected gzip is still independently bounded on wire and
decoded sizes. All attempts share the monotonic total deadline and execution
window. Original request/receipt/observation and transfer clocks survive.

Success produces an immutable, private-read-back **CAPTURED_UNREVIEWED** bundle
and stops. It creates no owner review, acceptance, prediction or selection.
Later owner review must address the exact bodies, permissions, projections,
feature derivation, event, offer and consumed model before fresh inference.

## Concrete proposal and commands

The supplied `nfl-scoring-acquisition-proposal-v2.json` is **non-executable**.
Proposal hash:
`ef5a72f1a6d5dd3f29cca5f5c6b39c9064a54def4b66e66eea9493d4df79c684`.
Window October 18, 2026 **11:30–11:31 a.m. America/New_York**
(15:30–15:31Z); authorization expiry 11:31:30 a.m. (15:31:30Z).
Revalidate that future window before finalization; no expired opportunity is
operationally ready. Source is public and requires **no credentials**.
Request count 1–2 public GitHub GETs, **zero Odds API credits, zero CFBD requests,
$0 additional spending**. No billable-service retry or upgrade is permitted.

Planning from a reviewed checkout (no credentials, network, storage or security
inspection):

```powershell
python scripts/nfl_scoring_acquisition.py docs/paid-launch/nfl-scoring-acquisition-proposal-v2.json
```

Execution command, **only after separate final-plan/collection/configuration
approval**, using real private files and their exact reviewed SHA256:

```powershell
python scripts/nfl_scoring_acquisition.py "$env:LOCALAPPDATA\ParlayPicker\ncaaf-pilot\custody\nfl-scoring-v2-final-plan.json" --execute --authorization "$env:LOCALAPPDATA\ParlayPicker\ncaaf-pilot\custody\nfl-scoring-v2-authorization.json" --custody-root "$env:LOCALAPPDATA\ParlayPicker\ncaaf-pilot\custody" --controlled-bindings "$env:LOCALAPPDATA\ParlayPicker\ncaaf-pilot\custody\nfl-scoring-v2-controlled-bindings.json" --bindings-sha256 THE_REVIEWED_CONFIGURATION_FILE_SHA256
```

The final executable plan must replace PROPOSAL/pending prerequisites with
recorded review facts; its changed hash requires exact owner authorization.
The command template is not executable authorization. Expected sanitized output:
`{"attempted": 2, "reason": null, "status": "CAPTURED_UNREVIEWED"}`
(one attempt on direct 200). Private files are
`PLAN_SHA256.nfl-scoring-attempts.jsonl` and
`PLAN_SHA256.nfl-scoring-unreviewed.json`. Do not delete/reset either or rerun
on failure. Inspect the sanitized reason and retain incomplete evidence.

## Controlled configuration and existing storage

`nfl-scoring-controlled-configuration-proposal-v1.json` prepares exactly three
process-local acquisition bindings: reviewed permission, exact owner collection
authorization, existing verified custody root. Its manifest payload hash is
`b2ab2bca416c6c1a3bc9017628a1b6010ce7df267b3037ba70faf36367c0eba4`;
whole-file SHA256 `07badc4fafdf721b816ce434c84a449317d5bc2ef7cdd53dcfa01668763386ea`.
It is unreviewed/non-executable: actual permission/authorization/configuration
review times are deliberately absent. No binding is applied by this PR.

Reuse STORAGE_PASS receipt SHA256
`1562cf1bc9f224c411af83ec0e1e51f743ff3639c4dd9e25a73eb29cee3e7ccd`;
existing encrypted/private custody root
`C:\Users\Robert\AppData\Local\ParlayPicker\ncaaf-pilot\custody`.
No setup, ACL changes, backup/Drive operation or database modification is needed
for preparation. Before execution, readonly existing runtime checks must still
verify private ACL, EFS encryption and no reparse points. A writable path is not
automatically verified custody.

Robert reviews the exact permission scope and final plan, separately authorizes
collection, then reviews the exact controlled-configuration subject and local
application method. A new exclusive private JSON file and exact byte read-back
are required after that approval. The CLI validates its supplied file SHA256,
payload hash, actual owner review and storage conditions, enables only its
three entries in an isolated process context, and clears them afterward. All
production acquisition/acceptance catalogs stay empty/unchanged. This supplies
no inference acceptance; private review mode remains OWNER_REVIEWED, never
independent acceptance.

## Full private path and unresolved authentic evidence

Explicit v3 selection routes through existing actual market caller, unchanged
native mathematics, capture/export, private replay and Streamlit private display.
The raw model, original blend and actual refresh stages remain separate;
UNKNOWN publisher publication time is explicit. No historical inference or
timestamp rewrite is performed. Old inputs/reviews remain rejected by old
readers. Public packages omit private bodies and owner-reviewed probabilities.
Novig FVS break-even/edge/EV restrictions, nonneutral half-point spread scope,
freshness/pregame checks and exact source/offer/rule bindings remain enforced.
Coverage keeps each supplied scheduled event and precise unavailable reasons.

| Remaining boundary | Owner / exact next action |
|---|---|
| Final scoring collection plan | Robert: advance review of this licensed private scope, revalidated future window, exact-plan collection approval, then exact controlled-configuration review/application |
| Authentic scoring body | Separately authorized bounded acquisition; full body must fit, then actual owner verification of immutable bytes/projections |
| Independent target event/schedule | Owner/source: original authoritative schedule, exact canonical/provider crosswalk, named sides/start/nonneutral facts |
| Fresh Odds quote and original clock meaning | Separate authorized capture with applicable account permission/budget; retain exact original response and documented quote-update meaning |
| Novig listing/product/period/OT/NFL-001/FVS | Authoritative applicable listing, product/jurisdiction, effective rules and selected-position/payoff binding; no generic or another operator's rules |
| Owner prospective input review | Robert: all six exact-subject findings after complete inputs; genuinely new inference only after review |
| Scientific qualification / wagering authority | Existing separate governing requirements; this private lane cannot establish either |

No authentic fresh result has been demonstrated or hosted revision verified.
Scoring-source readiness alone cannot remove production Unavailable states.
All unqualified selections remain **PASS at zero stake**. Provider inquiries
remain unsent; NCAAF's approved October 16 plan is untouched.
