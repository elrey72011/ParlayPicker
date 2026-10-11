# NFL owner-reviewed private research v1

This is a separate, explicitly selected prospective reader, not an upgrade to
`nfl-inference-inputs-v1`, independently accepted source intake, or historical
packets. Robert Velarde's task instruction authorizes owner verification only
for this personal private mode. `source_contract.ACCEPTED_LISTINGS` and
`source_evidence_intake` remain unchanged. Main's independent acceptance is a
repository admission policy (`app_core/source_contract.py:verify` and
`app_core/source_evidence_intake.py:assess`), not proof that a provider mandates
a second person. Retained Odds API terms/account review establishes no extra
DraftKings license requirement for the described private feed use. Personal
use does not establish unknown upstream rights or an offer's rules. Both
prepared provider inquiries remain unsent.

## Scope and provenance

Base main is d4b73dcc05b45abb1d57dc995571ce892d4cd5cf. #2411 and #2412 were open
and unmerged when this route was implemented. Neither is used as implementation
ancestry or assumed to be main. Shared presentation, UI and scope files require
reconciliation after an owner merge; fresh bindings and CI are then required.
Completed NCAAF storage/configuration and the October 16 discovery plan are
untouched. This route neither acquires data nor provides a bounded NFL transport.

Input schema: `nfl-owner-reviewed-private-inputs-v1`, outer `{payload,sha256}`.
Output: `nfl-owner-reviewed-private-result-v1` in `ml_estimate_metadata` as
`nfl_private_inputs`. No acceptance/catalog/store is initialized on selection.
UI staging permits RETAINED packets only; SYNTHETIC packets are used solely in
labelled offline fixtures. Context-local selection resets even on rejection.
At most four packets, 8 MiB each and 16 MiB selected. Original decoded response
bodies are limited to 512 KiB each, 16 objects and 2 MiB aggregate. Oversize,
duplicate object, corrupt hash, unknown clocks or credential echoes reject;
original bytes are never truncated, redacted or regenerated.

Five original objects are required: independent schedule, exact Odds API body,
native scoring-history CSV, applicable offer/listing document and documented
clock meanings. Credential-free source IDs/endpoints and actual request start,
complete-body receipt and first local observation accompany each byte hash.
The CSV is the original upstream body, not the returned pandas frame or a
projection serialized afterward. This v1 accepts scoring-history CSV
records with `game_id,season,gameday,home_team,away_team,home_score,away_score,
result,source_available_at`. Unscored rows remain in the original body with
explicit missing outcomes; unchanged native selection excludes them. Completed
records require documented publisher availability. A schedule-only source
cannot supply the required histories. Do not filter an authentic full body to
manufacture this contract. `gameday`
is a date, not an invented original kickoff or publisher timestamp.

Document projections point into retained original JSON; a hand-authored generic
rule template is not an authentic listing document. The exact event, provider
namespace/ID, named HOME/AWAY, independent canonical ID, season and explicit
non-neutral fact must agree. Ambiguous New York/Los Angeles labels and unknown
NFL teams reject. No new aliases are added. This initial route excludes neutral
sites because the consumed model has a home-field adjustment.

The original selected offer must have a supported American price, signed
half-point spread, provider market-update clock and an honestly present/absent
provider quote ID. The separate listing binds exact book/product/jurisdiction,
listing ID, offer, effective rule edition, full-game/overtime and payoff. Novig
FVS is labelled decided-game cover probability; binary EV, edge and break-even
remain unavailable. No void mass or settlement-compatible value is fabricated.
The initial operator scope is Novig's existing
`novig:nfl-001:half-point:full-game-ot:fvs:v1` only. Other operators or rule
variants remain unsupported. Totals and integer pushes are outside this new contract.
Explicit UNKNOWN/UNVERIFIED/UNAVAILABLE/NOT_RECORDED identifiers, listing facts
or clock citations remain missing under the existing intake sentinel rule. An
owner finding cannot promote a contradictory unknown declaration.

## Prospective chronology and owner verification

Advance permission review names Robert, the exact source/endpoint, effective
interval, separate private retention/private research use conclusions and credential
exclusion with a substantive basis. Retention permission alone cannot admit computation.
It precedes request start and covers observation through inference. It does
not require future body hashes. Original provider update precedes complete-body
observation; publisher per-record availability precedes score-body observation.
All observed objects precede feature availability. Exact owner verification
follows those facts and strictly precedes the actual new inference clock.

`owner_review` is `nfl-owner-verification-v1`: reviewer Robert Velarde, actual
`reviewed_at`, `subject_sha256`, OWNER_REVIEWED, PRIVATE_RESEARCH, human
attestation and six substantive VERIFIED findings: event mapping, offer and
settlement, permissions, clocks/original custody, feature provenance and model
binding. Its exact subject includes every original byte hash, original receipt,
permission, feature, target and model binding. A checkbox/upload alone creates
none of these. Independent receipts cannot be borrowed; this mode does not
populate their catalogs or masquerade as independent acceptance.

Every required publisher availability clock and its documented meaning must
exist. The historical New Orleans frame observation cannot supply one. Native
unchanged completed-before-current-UTC-day/stable-last-five aggregation derives
all eleven ordered features. Both teams require at least one completed game,
as in the existing NFL predictor. Source records, scores, season, distinct
teams, unique game IDs, original availability and supplied feature arithmetic
are checked before the numerical call. The fixed score parameters, prior
shrinkage, reliability/clipping and blend weights are unchanged.

Exact current code/artifact/callable/configuration identities are bound before
prospective inference; actual Python/NumPy/pandas/platform and consumed reader
hashes are retained separately. No historical runtime or artifact is rewritten.
Original quotes must be known, nonfuture, at most 900 seconds old and pregame
at inference. Reviews cannot backdate facts or renew a stale quote.

## Actual application path and separation

Publish board → Publishing token → Private NFL owner-reviewed research →
explicit mode checkbox → stage existing packet → select exact hashes. This
does not run analysis. A separately requested normal analysis uses the selected
contract at `market_probability_model.predict_market_probabilities`, which
checks original evidence before delegating to the unchanged native functions.
The general normal-analysis fetch path is not an isolated acquisition pilot;
real execution/acquisition remains separately authorized.

Capture and immutable private replay tables retain the same metadata/packet
through `prediction_evidence.capture_run`, `research_replay.retain_export`,
private CSV/download and static read-back. The private CSV uses the distinct
`nfl-owner-private-display-v1`; public research-display v1 remains unchanged.
No parallel archive or schema
migration is added. Replaying a captured observation is static; only labelled
synthetic fixtures numerically execute this route. Retention stores raw model,
original market-context blend and actual UI refresh receipts separately; the
legacy name `calibrated_probability` establishes no calibration. A UI refresh
retains its own clock/inputs and cannot change original clocks or raw output.
Static read-back checks chained previous/raw probabilities, exact consumed blend
identity and monotonic refresh clocks. It labels this a recorded UI stage, not
an arithmetic replay or calibration validation. Private display uses raw probability.

The owner-token-gated browser shows raw probability with OWNER_REVIEWED /
PRIVATE_RESEARCH. Private per-game readers check exact candidate/event/run,
original packet and retained outputs without inference. Public packages retain
the scheduled row but suppress this private probability, legacy numeric aliases,
EV and private evidence. Wager finalization adds a deny-only guard for these
packets; no independently accepted route is loosened. PASS and zero stake are
mandatory even if claimed approval fields were supplied. Coverage carries the
precise private rejection code and retains every scheduled event.

## Authentic readiness

| Boundary | Existing evidence | First required fact |
|---|---|---|
| Historical NFL observation | Retained original packet, static inspection only; exact private offer and raw/blended values remain local | Its saved rejection and absent UI refresh remain unchanged |
| Original scoring input | Retained frame projection, not original upstream CSV bytes | Authentic original body and documented per-record publisher availability |
| Independent target event | Not present in inspected chain | Original target schedule and deterministic named-side crosswalk |
| Exact offer | Original ID/side/line/price/clocks retained | Fresh prospective body and separately applicable product/listing/period/OT/FVS edition |
| Permissions and owner findings | No authentic new private subject reviewed | Advance private-use permission and later substantive Robert verification of exact bytes |
| Scientific/wager authority | Not established | Separate scientific qualification and current actual authority; this PR grants neither |

Hosted revision and successful authentic private display remain UNKNOWN. Synthetic
software acceptance and CI do not establish that either occurred. Missing source
documents require attributable provider evidence, not another historical numeric
rebuild or substitution of another game's dependencies.
