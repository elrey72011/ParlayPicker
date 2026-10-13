# Bounded original-document retrieval and private linkage

The verified base is `4b89179e40f48cb555b00e3f9e440f20f96647be`.
#2421's ordered parents are `9aceef16d9f79fdb95c9a910b5411e3cd449e5ce`
and reviewed `641565ed3288fe94478ea38536795195c9a12c9e`; both reviewed
and merged trees are `6b5bb0e707ac743d34f482b6279a4b0c0dd41bac`.
All 15 completed post-merge checks and 7,306 exactly-once application identities
are retained locally. This change requires its own complete final-head CI.

## Reproduced boundaries

The NFL odds runner rejects the known Ludlow PDF endpoint with
`NFL_ACQUIRE_ENDPOINT_NOT_APPROVED` before transport. This is intentional
unsupported scope. It supplies neither original-document bytes nor receipts.

The existing NCAAF discovery contract accepts an unknown provider event ID.
The original capture plan still rejects an absent ID with `NCAAF_PILOT_TARGET`;
it also requires a product and settlement-review hash. A labelled synthetic
regression exercises advance acquisition permission, unaccepted ID discovery,
later applicable-terms review, capture-plan validation and the actual existing
owner-reviewed caller. No circular chronology was reproduced. No NCAAF reader,
history requirement, alias, permission or acceptance rule changes.

## Explicit acquisition contract

`football-original-document-plan-v1` is selected separately from sports plans.
The default CLI validates a supplied manifest without credentials, requests or
storage initialization. Planning cannot fill missing permissions or approvals.

Each manifest identifies exact HTTPS URLs, fixed public hostnames, source IDs,
purposes, formats and logical object destinations. Queries, credentials, ports,
fragments, traversal and unapproved hosts reject. There are at most four
documents and one explicitly named redirect destination per document, at most
eight attempts. Each primary/hop GET is durably journaled before DNS or dispatch.
A redirect requires the exact reviewed absolute destination. There is no retry,
fallback, pagination, crawl, embedded-resource fetch or returned-link dispatch.

DNS runs inside the connect deadline. Every resolved address must be public and
non-multicast. TLS uses the original hostname and default certificate validation;
the TCP connection is pinned to one validated address. A late worker cannot
reopen the connection or initiate another GET. Application test blockers are
never removed; the operational CLI requires the corrected isolated runtime.

Ceilings: 512 KiB wire and 512 KiB decoded per complete body; 2 MiB aggregate
wire and 2 MiB aggregate decoded; 6 MiB serialized bundle; 10-second connect,
10-second read and **60-second shared run deadline**. These document ceilings
do not alter any sports-provider profile. Known Novig PDFs are 403,625 and
293,631 bytes with 20 and 12 pages; other complete-body sizes remain UNKNOWN.
No probe, truncation or automatic enlargement is allowed.

Full wire bytes, full decoded entity bytes, their separate hashes, source/final
URL, each request start, complete receipt and actual observation clocks, HTTP
status, MIME, encoding and declared length are retained in the immutable private
bundle. Public response metadata is allowlisted; cookies, authorization headers,
arbitrary Location values and exception messages are excluded. An incomplete or
secret-bearing body is never labelled complete. Failures retain stable codes,
allowlisted diagnostic facts, completed earlier objects and the spent journal.
An interrupted journal, including a reserved empty journal, cannot be resumed.

The `destination` field names a logical original object **inside**
`<plan-sha>.documents-unreviewed.json`, not a second mutable file. The companion
`<plan-sha>.documents-attempts.jsonl` is append-only during that one execution.
Custody is the existing ACL/EFS/no-reparse-verified private folder. No store,
backup, journal or security configuration is initialized by this lane.

Execution needs: actual advance permission for these documents/private uses;
exact owner collection authorization with $0 additional spend; separately
reviewed controlled configuration binding that plan, permission, authorization
and STORAGE_PASS/root. Configuration is isolated to that process and retained
byte-for-byte with its file hash. Processing supplies no human review or trusted
source admission. Acquisition stops at `CAPTURED_UNREVIEWED`.

## Extraction, interpretation and private caller

The extraction input is the complete **decoded PDF**, byte-for-byte, not its
compressed wire wrapper or a rewritten PDF. Existing v5 extraction bounds,
pinned pypdf method, inert worker, page/excerpt locations, ambiguity checks and
active-content rejection remain unchanged. HTML/text remains original UTF-8;
scripts and links are not executed. Unreadable or unsupported originals remain
unavailable. Credential-looking bodies also remain fail-closed; successful
synthetic prose is not proof that an uncaptured website body will be admissible.

`nfl-owner-document-private-inputs-v6` wraps an unchanged v5 packet and exact
retrieval bundles/object IDs. It verifies equality of the projected original
body, receipt and PDF extraction before delegating **every** v5 event, offer,
permission, interpretation, edition, effective-interval, override, feature,
runtime and model gate. Robert's separate, substantive linkage receipt binds
the entire wrapper subject, supporting basis and actual review time. The v5
interpretation findings and six original owner conclusions are still required.
Neither route can borrow the other's review.

The original v5 packet and full v6 retrieval linkage survive caller, private
capture/export, static replay and private display. Results remain
OWNER_REVIEWED / PRIVATE_RESEARCH, uncalibrated, PASS, zero stake, excluded from
public packages. Source acceptance, calibration and qualification remain false.
The 900-second freshness clock remains the original provider-market observation;
receipt, extraction, interpretation and review cannot refresh it. Operator update
and publisher publication time remain UNKNOWN where not evidenced.

## NCAAF field and gate sequence

| Fact | Source / legitimate discovery time | Existing consuming gate |
|---|---|---|
| Private acquisition/retention permission, operator, exact request/custody/budget/window | Owner's substantive advance permission and separate exact authorization, **before discovery** | `ncaaf_pilot_discovery.authorize` |
| Canonical schedule event, named sides, kickoff, neutral-site | Attributable independent schedule and retained crosswalk, before discovery | discovery `validate_plan.target` |
| Provider event ID, original labels, observed line/price | Complete discovery response; IDs remain unaccepted | `identifiers`; later capture `validate_plan.target` |
| Applicable product/listing, period/OT, effective rule edition, permission uses | Exact authoritative product/listing bridge plus rules; may be verified **after discovery**, before fresh capture | capture `offer.product/settlement_review_sha256`; `check_chronology.terms_review` |
| Fresh quote hash, actual observation, exact offer verification | Fresh complete original Odds body; after capture, before inference | custody `_project_quote`, chronology observation/verification |
| Original CFBD scoring/yardage, seasons/weeks and availability, minimum three histories, seven-day lag | Complete original approved CFBD responses plus separate deterministic projection | custody, dependency verification and unchanged feature caller |
| Substantive owner findings, exact subjects and review clocks | Robert's actual answers, after supporting facts exist and before inference | `ncaaf_owner_research`, chronology owner verification |

No future response hash, line/price or acceptance receipt is required for
identifier discovery. An Odds response does **not** prove a DraftKings product,
jurisdiction or applicable rule edition. Public operator rules demonstrate OT
content, but a Louisiana retail edition cannot silently govern an unidentified
feed listing. This is an external applicability evidence gap, not a new licensing
or second-reviewer requirement.

## Operational handoff (non-executable)

The accompanying four-document proposal retains known exact URLs and no
redirects. Its window is a proposal, not scheduled or authorized. Complete
advance document-use review and exact collection/configuration review are
outstanding. Returned NFL listing links are untrusted; retrieve none until an
exact URL/host/purpose/redirect manifest has been separately verified and approved.

Network-free planning from the reviewed checkout:

```powershell
& $OperationalPython -I -B $VerifiedLocalLauncher docs/paid-launch/football-original-documents-proposal-v1.json
```

Future manual execution **only after** finalized approvals/configuration:

```powershell
& $OperationalPython -I -B $VerifiedLocalLauncher $ApprovedPlan --execute --controlled-bindings $ApprovedBindings --bindings-sha256 $ExactFileSha256 --custody-root $VerifiedCustody
```

These execution variables are not populated approvals. No credentials are needed
for public documents; do not pass API keys. The normal corrected Robert runtime,
unchanged private storage and native TLS checks must pass before dispatch.
The retained local launcher adds only the verified dependency/PDF library paths
to isolated Python, then dispatches this checkout's CLI without importing pytest,
scope instrumentation or startup hooks. Bare bundled Python has no application
dependencies under `-I`; it is not the supported execution command. Launcher and
code hashes, exact arguments and read-only preflight are retained locally.

NFL: retrieve/review schedule, timestamp guide and generic contracts first;
discover the exact listing using the existing NFL V2 odds lane; resolve that
listing's reference/position/threshold, period/overrides and effective edition
from attributable evidence. If the discovery quote expires, acquire a separately
authorized fresh quote and review its new subject before inference. Do not assume
document retrieval plus review fits the 900 seconds.

NCAAF: Delaware at Middle Tennessee is October 13, 19:00 Eastern / 23:00 UTC.
Five prior scoring games per team are calendar-eligible; complete current CFBD
bytes/availability and three qualifying yardage histories are not retained.
Minimum prepared scope: one CFBD season games request (`seasonType=both`,
`classification=fbs`), three week-3/4/5 team-stat requests, one fresh DraftKings
spread request. If a provider ID must first be discovered, add one separately
authorized Odds discovery: six total attempts, four CFBD and at most two Odds
credits. This is five scoring histories and three explicitly available yardage
histories; weeks 1/2 yardage are not represented as captured. Every full body must
fit the unchanged limits. The seven-request all-history alternative stays separate.

Tonight cannot be promised while this implementation is unreviewed and applicable
DraftKings listing/edition evidence is absent. The next calendar candidate is
WKU away at Sam Houston, October 14, 20:00 Eastern / October 15 00:00 UTC; it
offers more preparation time but has the same explicit evidence prerequisites.
October 16 Army–FAU discovery authorization remains unchanged and exclusive to
its existing target, hash and window.

Primary public research (parsed references are **not** original custody):
[Delaware schedule](https://bluehens.com/sports/football/schedule/2026),
[Middle Tennessee schedule](https://goblueraiders.com/sports/football/schedule/2026),
[WKU schedule](https://wkusports.com/sports/football/schedule/2026),
[Sam Houston schedule](https://gobearkats.com/sports/football/schedule/2026),
[DraftKings operator-hosted Louisiana retail rules, V15](https://lb.goldennugget.com/contentassets/9800d7ef44d6474dad6cd3e69d29dd99/draftkings-gnlc-retail-house-rules-v15.pdf),
[DraftKings football rules](https://sportsbook.draftkings.com/help/sport-rules/football),
[Odds V4 guide](https://the-odds-api.com/liveapi/guides/v4/),
[Novig Contracts](https://support.novig.com/en/articles/16083642-contracts).

Only unresolved owner facts: an existing authoritative DraftKings feed-to-product/
listing/jurisdiction/edition reference, if available, and later substantive reviews
and exact operational approvals. Identity, $0 ceiling, personal accounts, sole-owner
review role and STORAGE_PASS are already recorded. No inquiry is sent or required
merely to repeat established generic rule content.
