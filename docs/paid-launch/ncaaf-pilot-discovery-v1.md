# NCAAF identifier discovery v1

This adjacent route resolves unknown feed identifiers without weakening
`ncaaf-pilot-plan-v1`. The original capture reader still rejects a missing
provider event ID with `NCAAF_PILOT_TARGET`. Discovery is one target descriptor
scope, one full multi-game Odds API response and one spent attempt. It never
admits an offer, verifies a crosswalk, infers, displays a probability, creates
acceptance or assigns a stake. Original capture/chronology/custody readers,
mathematics, aliases and production acceptance catalogs remain unchanged.

## Explicit dispatch and advance authority

`python scripts/ncaaf_pilot_discovery.py <sealed-plan.json>` is planning only:
no credentials, requests, storage or catalog initialization. The distinct
`ncaaf-pilot-discovery-plan-v1` binds known canonical schedule descriptors,
named teams/IDs, original kickoff/neutral fact and schedule reference; selected
side, one book and spreads; exact HTTPS host/endpoint/query; clocks, custody,
limits and unresolved prerequisites. Unknown future provider IDs, response or
quote hashes, exact prices and offer acceptance are deliberately absent.

Explicit `ncaaf_pilot_discovery.acquire(plan, authorization, root=..., transport=...)`
requires all prerequisites resolved and a SYNTHETIC or PROSPECTIVE label.
PROPOSAL packets cannot execute. Advance discovery/retention permission binds
the exact request and target descriptor hashes, license-holder/evidence/reviewer
references, meaningful review/effectivity clocks and permitted use
`private_identifier_discovery_retention`. Review precedes request and permission
covers authorization expiry. This receipt does not know future bytes and is not
later exact-offer acceptance. Trusted separate owner authorization binds exact
plan hash, custody, owner, authorization/expiry, one attempt, one planned provider
credit, account-quota evidence and approved dollar ceiling. Subscription/error
billing still requires owner-specific evidence; a field is not a billing oracle.

`AUTHORIZED_DISCOVERIES`, `ACCEPTED_DISCOVERY_PERMISSIONS` and
`AUTHORIZED_CUSTODY_ROOTS` start empty. No upload, CLI, planning, setup or
execution registers them. A later controlled, independently reviewed deployment
of trust configuration is separate. Tests populate synthetic catalogs only.

## Transport and custody

Only `api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds`, US,
one selected non-Novig book, American spreads and an at-most-one-day target
commencement window are supported. A different endpoint/provider/market requires
a new review. The existing `HttpsTransport`, bounded decoding, strict JSON and
credential-echo checks are reused. There are no redirects, automatic pagination,
retries, fallback, enrichment, CFBD calls, Drive, model services or SQLite writes.
Maximum connect/read/total waits are 3/5/60 seconds. Wire and decoded bodies
remain at most 512 KiB; the inherited 16-object/2 MiB ceilings are not enlarged.
One response is acquired here. Oversize/incomplete/auth/rate-limit/timeout
failures stop; no filtering or truncation makes an incomplete body complete.

A credential-free, exclusive fsynced journal records authorization/plan identity
and the spent request before transport. Interrupted execution cannot resume or
repeat. Root identity must match controlled custody configuration. Losing a
directory or rewriting a plan is not authority to retry. Complete original
decoded bytes, body hash/count, request and actual local receipt clocks survive
private byte-exact read-back. Stable failures exclude exception text, secrets and
credential-bearing URLs. Credentials echoed in JSON/URL-encoded/escaped strings
are rejected rather than redacted into apparently authentic evidence.

Exact literal named descriptors and equivalent kickoff instants locate untrusted
candidate identifiers with original record indices/labels. No fuzzy matching or
alias expansion occurs. Duplicate/repeated candidates remain AMBIGUOUS; reversed
orientation, changed kickoff or conflicting ID descriptors remain CONFLICTING;
absent/unsupported labels remain unresolved. Their complete valid response body
is retained. No candidate is silently chosen. Neutral-site/listing/product are
not verified by matching labels. Listing/product remain unverified until a
genuinely applicable authoritative declaration is independently reviewed.

The distinct `ncaaf-pilot-unaccepted-identifiers-v1` bundle cannot pass
`project_capture` or `admitted_analysis`. Discovery receipts cannot substitute
for fresh quote/dependency custody, independent acceptance or source catalogs.
After discovery, resolve authoritative mapping/aliases and applicable operator
product/listing/period/overtime/rules/rights; review governing terms before a NEW
fresh capture, with a separate exact plan and authorization. Exact observed-offer
and dependency verification follow observation and independently trusted
acceptance precedes genuinely new inference. No clock renewal/backdating.

## Future proposals and honest accounting

The two companion Army HOME/Florida Atlantic AWAY plans are non-executable
PROPOSAL records, not observed offers or authorizations. Official Army/FAU public
schedules were rechecked during preparation; Army hosts FAU at Michie Stadium
October 17, 2026, noon Eastern (16:00Z). Native CFBD `401862802` comes from retained
September schedule evidence, not a newly acquired provider crosswalk. Exact feed
labels/ID, listing and accepted mapping remain unresolved. Schedule applicability
must be rechecked before any final execution plan; expiry never rolls forward.

Proposed discovery: October 16 15:30–15:31Z, expiry 15:31:30Z, one Odds request.
Proposed NEW capture: October 17 15:30–15:31Z, same expiry offset, six CFBD
requests (2026 season games and regular week 1, 2, 3, 4, 5 team statistics), then
one fresh Odds spreads request. Existing runner MAX_REQUESTS=16 already permits
seven structurally; this is not permission to exceed the old six-request plan.
Original six-request bytes/bindings stay untouched. Discovery plus capture is
EIGHT attempts, six CFBD and two Odds. Explicit combined-event budget and each
phase's final known hash require separate owner approval. No unknown future
capture hash is demanded before discovery.

Strict seven-day cutoff is October 10 16:00Z; a game exactly at cutoff is
excluded. Retained calendar support gives potentially four Army/five FAU prior
games. Week 3 includes FAU/FIU September 19; its original yardage response is not
retained. The old contingency omitted it. The unchanged model minimum is three
qualifying scoring/yardage histories per team, not an invented every-yardage-row
gate. Complete coverage and admissibility still need actual bytes, clocks,
correct weeks and independent review. Full-season 512 KiB feasibility, account
entitlement/quota, cash cost, source permission and backup readiness stay UNKNOWN.
Oversize stops; this PR adds no probes, endpoint switch or larger limits.

Sources for descriptor recheck: [Army schedule](https://goarmywestpoint.com/sports/football/schedule/text),
[FAU schedule](https://fausports.com/sports/football/schedule),
[Army official ticket listing](https://goarmywestpoint.evenue.net/event/F26/04?RDAT=gamecenter&RSRC=SIDEARM).
These establish public schedule facts, not feed IDs, listing contracts or rights.

Research, calibration, scientific qualification and funded authority remain
separate. Unqualified selections remain PASS, zero stake. Synthetic success
proves software mechanics only; no authentic hosted result is asserted.

Failure diagnostics use a fixed code whitelist. Arbitrary exception text,
including messages that imitate a diagnostic prefix, is replaced by
`NCAAF_DISCOVERY_STOPPED`; it cannot enter receipts. The retained synthetic
pre-fix canary reproduction failed this check. The successor regression proves
the canary is absent, while the spent attempt remains non-resumable.
