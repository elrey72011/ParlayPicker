# NCAAF bounded pilot v1

This adjacent lane acquires bounded responses for one target. It is not a
single-game provider endpoint: CFBD season games/week team statistics and the
primary Odds API NCAAF endpoint can return other games. Entire decoded original
bodies remain private. No response filtering, discovery, extra endpoint,
retry, pagination, fallback, Drive, enrichment, model service or fitting occurs.

`python scripts/ncaaf_pilot.py docs/paid-launch/ncaaf-pilot-army-proposal-v1.json`
is **planning only**. It reads a credential-free sealed plan, prints unresolved
prerequisites, and loads no credentials, initializes no store and makes no
requests. This release deliberately exposes no acquisition CLI/owner button.
Explicit Python execution requires a separately reviewed authorization and
advance permission bindings. Production catalogs remain empty. Selection,
upload, execution and fixtures cannot populate any trusted acceptance catalog.

## Immutable execution contract

`ncaaf-pilot-plan-v1` binds one canonical event, provider event, named orientation,
original start, neutral-site fact, selected market/side, book/product/full-game
terms digest; exact credential-free host/endpoint/query order; one attempt per
request; custody identity; execution and authorization clocks; advance permission
receipts and unresolved facts. Null line/price scope means an as-yet unobserved
half-point offer, not an invented price. A known line/price must match exactly.
Final quote verification always binds actual signed half-point line and American
price, original provider clocks, exact listing/product, source rights and rules.
Both target sides and totals are supported separately; Novig/integer lines remain
excluded. CFBD season and explicit week scopes cannot change during execution.

Only HTTPS `api.collegefootballdata.com/games`, `/games/teams`, and
`api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds` are allowed. One
CFBD games request comes first, fixed week requests follow, and one selected-book
spreads OR totals request is last. Requests consume attempts before transport.
Stdlib HTTPS has no redirects/retries; credential lookup occurs only after
authorization. Connect/read timeouts are bounded by one monotonic total deadline.
Caller waits are deadline-bounded even if OS DNS/connect/read ignores its socket
timeout. Each connect, request write, header and stream-read wait also enforces
its declared connect/read limit, clipped by the total deadline. A cancelled late
worker closes its connection and cannot progress to
another GET; the acquisition stops and records the spent attempt. DNS/TLS are
part of the fixed HTTPS transport, not extra sports-data requests.
Wire and decoded body limits are 512 KiB each; maximum 16 objects and 2 MiB
aggregate decoded bodies. Plan/packet/selection limits remain independently
enforced by existing readers. Gzip expansion, malformed JSON, incomplete streams,
unexpected pagination, auth/rate-limit responses, timeout or any budget failure
stop execution. Oversized evidence is INCOMPLETE, never truncated and called
complete. Diagnostic codes exclude exception text, credentials and URLs.
Literal, JSON-escaped and URL-encoded credential echoes are rejected before
private persistence. Original bodies are never redacted to obtain acceptance.

An exclusive, fsynced attempt journal precedes requests. A previously attempted
plan in the same owner-controlled durable custody directory cannot resume or
repeat, including after interruption. Execution also requires the custody ID's
trusted configured directory to match the resolved path; moving to another
directory cannot reuse the authorization. This production catalog is empty.
Changing/losing that directory is not a
resumption mechanism: any replacement requires owner reconciliation of spent
attempts and a separately authorized plan. Local durability across hosted
redeployments remains an operational prerequisite; this PR adds no remote sync.

## Capture, review and prospective inference

1. Independent advance permissions/terms review and separate owner collection
   authorization bind the exact plan before any future request. Effective terms
   and account permissions must cover capture, later inference and intended
   derived public output. This is not accepted-source or scientific admission.
2. Explicit `acquire(plan, authorization, root=existing_private_directory,
   transport=HttpsTransport(approved_credential_loader))` captures actual request
   starts, complete-body receipt clocks, original decoded bytes and hashes. It
   stops at a sealed `CAPTURED_UNACCEPTED` or `INCOMPLETE` bundle. No inference,
   SQLite initialization/restoration or acceptance follows acquisition.
3. `project_capture` passes exact decoded bodies into unchanged #2408 custody.
   Native scoring/yardage projections must match original bytes and availability
   clocks. Verification follows observation/feature availability; trusted,
   independent exact-byte and exact-offer acceptance precedes checkpoint/inference.
   Future acceptance hashes are not dependency subjects. Advance receipts remain
   unchanged. No backdating, refresh-clock renewal or historical reconstruction.
4. Only separately admitted #2408 inputs may enter
   `admitted_analysis(packet, source, inventory=retained_independent_inventory,
   bundle=original_capture_bundle)`. It verifies the bundle/plan/authorization,
   original body/projection and target bindings before the existing selected
   prospective caller. No general live-analysis entrypoint is called. Missing
   custody is unavailable; altered custody rejects before numerical computation.
   Rejection receipts retain the precise sanitized pilot cause inside the hashed
   private carrier; existing public fail-closed reasons remain unchanged.
5. `retain_analysis(analysis, path=existing_prediction_store)`
   uses existing finalization, explicit local capture, replay export, per-game
   readers and public package builder. An absent prediction store is rejected,
   not initialized. The nominal finalizer denominator is one; empty policies
   create no authority, and actual stakes must all remain zero. The artifact
   manifest binds the actual installed repository, separately from private
   storage; callers cannot substitute an empty custody directory. Exact private
   replay retains the original packet plus additional pilot request/authorization
   receipts inside the existing hashed NCAAF result carrier; original packet and
   frozen origin metadata remain unchanged. Public output excludes those receipts,
   raw bodies and credentials. Unavailable markets stay UNVERIFIED coverage rows;
   independent FBS/FCS inventory is preserved separately from Stage 1 policy.

Raw-native probability remains uncalibrated, original blend and absent UI-refresh
stay separate. Existing features, math, artifact/predecessor hashes, minimum three
histories per team, strict seven-day lag, input-age/freshness and pregame gates are
unchanged. All unqualified wagering remains PASS, zero stake. Acquisition or
display supplies no calibration, scientific qualification or wagering authority.

Feature order remains `home_ppg`, `away_ppg`, `home_oppg`, `away_oppg`,
`home_yards_per_game`, `away_yards_per_game`, `neutral_site`. Actual reader,
configuration/runtime and new inference receipts are retained by the existing
caller. Historical unrecorded Python/NumPy/alias details remain UNKNOWN.

## Concrete owner proposal and dependencies

The companion sealed PROPOSAL scopes Army HOME versus Tulane AWAY on October 10,
2026, noon EDT, Michie Stadium, nonneutral, canonical `cfbd:401862795`. These facts
come from the retained admission assessment/public schedule references; they do
not establish a current provider crosswalk or offer. Provider event ID, product,
actual signed line/price, listing applicability, accepted mapping (including the
Tulane label), source rights and terms/review hashes are unresolved. No alias is
added here. The proposal is BLOCKED and cannot execute.

Proposed response window: 15:30–15:31Z October 10; authorization expires 15:31:30Z.
Proposed maximum: six GET attempts, five CFBD (2026 games and regular weeks 1–4
team statistics) plus one US DraftKings NCAAF spreads Odds API request. Week
assignments require authoritative verification before approval. Three qualifying
histories per team and scoring AND yardage completeness must be established;
the strict cutoff is October 3, 16:00Z. Later games cannot fill history gaps.
The originally proposed Southern Miss +10 and Novig total cannot substitute.

No auxiliary enrichment/storage/model requests exist in this lane. CFBD quota:
five attempts maximum. Odds API: one attempt; one selected bookmaker/market's
advertised one-credit formula is a planning estimate, not an observed charge.
Account tier/quota, monetary subscription/overage, operator commercial permission,
private persistent storage, reviewer effort and operating cost are UNKNOWN. Owner
must verify these against the existing ceilings before authorization. No secrets
belong in the plan or review documents.

Whether a complete season-level CFBD body fits 512 KiB is UNKNOWN without an
original compatible body. Cleaned historical projections cannot prove that size.
If the body exceeds the existing ceiling the pilot stops unavailable; a different
endpoint/custody contract requires a separately reviewed successor, not filtering,
truncation or a limit increase in this lane.

Owner sequence: review implementation/CI; obtain applicable operator/feed/CFBD
permissions and exact event/listing/mapping evidence; have independent reviewers
accept advance governing terms and permitted uses; verify histories/weeks,
durable private custody and account budget; resolve the proposal and issue a
separate exact-plan collection authorization. After an authorized capture, stop
for exact-byte/offer verification and independent acceptance. Only then request
new prospective inference through the admitted route. Stop if incomplete bodies,
insufficient history, conflicting identities, unaccepted rights, quote expiry or
review latency prevent inference within existing 900-second freshness and
pregame limits. A later event needs a newly reviewed plan, never reused clocks.

The exact reviewed model remains
`881833a03169d8ff3cfdc7cfb6000a174ccb545b1d087657169fe06d5f8a1153`, predecessor
`6d7291fd464a9a367ba060ccf7bd526708bdba7b5809657930e1616f026173c0`.
Scientific role is proposed software/research development only; no automatic
validation/holdout or frozen Stage 1 cohort admission. Authentic successful
capture/display and the actual hosted revision remain unverified.
