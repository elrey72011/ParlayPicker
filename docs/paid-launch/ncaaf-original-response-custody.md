# NCAAF original decoded-response custody, v1

This successor starts from main `00434dc8d14227648bb19795d8e05bb9304bca99`,
ordered parents `bb696fc7f8da436c9f7c9658756a254f8fdef29f` and
`e3bd64b4a837295e653a0c307150fb9d74fb6f6d`, tree
`4fb2f053e3cc38c75c73bf135dfb9c62f666847b`. Post-merge CI and candidate CI
have separate receipts. Owner preview acceptance does not prove a hosted revision.

## Reproduction and representation

Labelled synthetic bodies with different whitespace, key ordering and discarded
CFBD fields yield identical `ncaaf_prospective.fetch -> refresh ->
ncaaf_history._clean` batches. Whitespace-distinct Odds API bodies produce
identical `TheOddsAPIClient.get_odds` reconstructed debug JSON. Existing
`ncaaf_compatible_pipeline.objects` verifies those exact native bytes, not the
original responses. Private pre-change bodies/results are retained locally.

`ncaaf_response_custody.capture(bytes, metadata, native_batch=... | quote=...,
terms=...)` is transport-free. It receives **HTTP-client response-body bytes after
content decoding and before JSON/text parsing**. This is not wire-byte custody.
The caller must supply a genuine receipt and complete response, not reconstructed
JSON. Neither the adapter nor an upload can independently attest acquisition.
All historical packets retain their original status: original responses remain
unavailable unless actually retained. No old record is upgraded.

Only CFBD `games` and `games/teams`, and the primary Odds API
`v4/sports/americanfootball_ncaaf/odds` selected quote are supported. Credential-free
request scope is allowlisted. Arbitrary URLs, headers, credentials, historical
odds wrappers, fallback feeds and other providers are not supported. Metadata
contains status 200, complete=true, application/json, explicit representation
and actual local receipt time/meaning. Bodies are strict JSON lists: duplicate
keys, nonfinite numbers, malformed/truncated bodies and credential fields reject.
Errors expose stable codes only, never bodies or credential-bearing URLs.

Each immutable object records original base64 bytes, byte count and SHA256;
provider/endpoint/request scope; receipt; versioned projection and exact installed
source hashes; native SHA256 and original record-index/game/team locators, or
event/book/market/outcome indices and exact quote SHA256. CFBD projection calls
the unchanged cleaner on retained bytes and must equal the native batch; no
native byte or retrieval clock is rewritten. This initial serializer binds the
existing canonical native encoding (`ncaaf_model_compatibility.encode`); differently
encoded native JSON is unavailable, not silently reserialized as original.

The quote projection calls existing provider fact extraction on the retained
selected outcome. It verifies exact event, named orientation, kickoff, market,
side, signed line, American price, provider clock, period/rules and supported
quote fields. A repeated exact match rejects as ambiguous. Operator/listing/product
may have actual feed paths. When the feed does not declare them, the locator
explicitly names the **independently accepted exact terms review** as their source.
It never asserts they came from the feed. Missing independent listing applicability,
settlement, clock meaning or rights remain unavailable. The existing accepted
mapping and exact source review gates are still mandatory; generic rules do not
establish listing applicability. NFL/Novig source-intake authority is not extended.

Limits are 512 KiB per body, 16 objects, 2 MiB total original bodies, plus the
existing 8 MiB packet / four packets / 16 MiB selected-envelope limits. All
required native batches and exactly one selected quote response must resolve.
Oversize, missing and incomplete evidence rejects; there is no truncation or
claim of complete acquisition from matching counts. No transport is implemented.

Both the original CFBD body receipt and unchanged native projection must meet
the existing 24-hour input-age limit at inference. A reproduced synthetic probe
initially accepted a 93,604-second-old body with a four-second-old projection.
The successor rejects it before mathematics with `NCAAF_CUSTODY_INPUT_STALE`;
the exact 86,400-second boundary stays valid. Projection cannot renew source age.

## Explicit versions and immutable trust

`ncaaf-original-response-inputs-v1` wraps the complete unchanged
`ncaaf-compatible-normal-inputs-v2` packet (nested observation v3), response objects
and separate `ncaaf-response-custody-admission-v1`. Existing outer v1/v2, nested
v2/v3 and native exact-field schemas/readers remain unchanged and reject the
new envelope themselves. Results use `ncaaf-original-response-result-v1` and
contain the unchanged native computation receipt as a separately bound object.

| Clock | Meaning and required ordering |
|---|---|
| Advance terms/CFBD permissions | Independently accepted immutable scopes, rights and editions; no future response hashes. Strictly before each relevant response receipt and effective through inference. |
| Provider quote update | Original recorded provider market-last-update, never local receipt. No later than body receipt. |
| Body `received_at` | Actual local decoded-body receipt, known meaning; no later than projection availability or quote observation, and never future. |
| Native `retrieved_at` | Original native projection availability, not wire receipt; preserves #2407's clock and separate meaning. |
| Quote observation | Actual observation of exact quote, after/equal response receipt. |
| Exact native/offer verification | Existing separate #2407 hashes, fact-availability and chronology checks remain required. |
| Custody verification | Binds earlier native subject, all raw objects/projections, advance permissions/terms and exact native verification digest. Every included fact must exist by this time. |
| Independent custody acceptance | Trusted receipt binds exact subject/verification; reviewer differs from verifier. No earlier than verification, no later than checkpoint, strictly before inference. |
| Admission checkpoint and inference | Original checkpoint remains separate; actual new inference follows all required acceptance and keeps existing freshness/pregame rules. |

`ncaaf-response-custody-subject-v1` hashes the #2407 pre-admission native subject
and exact response objects. It excludes later acceptance, checkpoint and inference
hashes. The separate final-packet catalog binds the complete successor after
admission. Changed/rehashed bytes cannot reuse earlier trusted acceptance.
Processing, uploading, selection and tests cannot register independent acceptance.
All production catalogs, including `ACCEPTED_ADMISSIONS`, remain empty. Labelled
synthetic accepted catalogs exist only under test monkeypatch.

## Application and private custody

The existing owner-only private NCAAF input selector stages the explicitly
versioned envelope without storage initialization, collection or analysis. A
separately requested normal analysis uses the existing caller; custody checks
complete **before** centers/probabilities. Missing/corrupt/mismatched bodies retain
precise unavailable codes and complete UNVERIFIED slate decisions, with existing
wagering PASS and zero stake. Rejected candidates supply no fallback numbers.

The existing metadata carrier, capture, snapshot, per-game export and private
replay retain the original envelope, decoded bodies, clock receipts, native
computation, consumed reader/runtime and run identities. Private read-back is
byte-exact. Static diagnosis repeats integrity/applicability checks only, never
historical inference. Public package/HTML contain only permitted research fields,
not bodies, model bytes, admission objects or private canaries. No redacted body
is presented as original; any future redacted derivative needs its own label.

Synthetic spread home/away and total over/under follow unchanged distributions.
Raw native probability stays uncalibrated; original blend and absent UI-refresh
remain separate. No value-display restriction, source acceptance, scientific
qualification or wagering gate changes. Original model/predecessor/artifact,
recovery, formulas, feature order, minimum histories, seven-day lag, freshness,
half-point target, integer-push and Novig exclusions remain frozen.

## Remaining authentic boundaries

| Boundary | Status / responsible next action |
|---|---|
| Offline custody capability | Synthetic caller/capture/export/replay/display and negative regressions; exact-head CI required for review. |
| Authentic original response bodies | Not created by this PR. Owner collection needs separate authorization and actual original decoded bytes/receipts. Historical cleaned JSON cannot substitute. |
| Bounded pilot transport | Separate workstream: exclude Drive, unrelated multi-sport/enrichment/model-service and fallback requests before proposing execution. |
| Event/listing/source/rights | Independent reviewer must accept authentic exact mapping, full-game/overtime terms, operator/product/listing applicability, quote-clock meaning and account-specific CFBD/quote public-derived permissions. No alias expansion. |
| Scientific role and qualification | Independent scientific review and prospective protocol; capture is not automatic cohort admission, calibration or model qualification. |
| Wagering authority / hosted operation | None established. Separate deployment and authority decisions; all unqualified selections remain PASS at zero stake. |
