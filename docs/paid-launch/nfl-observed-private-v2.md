# NFL original-source prospective private research v2

Base: `909804595e5626dbc65cc9c79391ca0d1131aa3e`, owner merge of #2413
with ordered parents `f691c30f038889b8d91bea75c164b26932d4559f` and
`f7f0ab919948a6d839d5699fc5e8915fbf4032d3`, reviewed tree
`ee61a62220bdddf4993c34a11076eab196a6e075`. #2411–#2413 and their original
bindings remain immutable. Main's post-merge checks were incomplete at the
initial inspection; the final validation receipt records their actual outcomes.

## Reproduced boundary and upstream identity

`feature_processing.fetch_nfl_stats` calls `nfl.import_schedules([season])`.
The retained producer's installed `nfl_data_py` version is **0.3.2**;
`import_schedules` source SHA256 is
`3ad59513804a6580b4236549bd10d34b1f7f4f5eef80784d7b240209d4595069`.
The original private `post2406-nfl-original-bindings.json` contains that
callable and version in the native dependency's original source receipt.
This is a static verification of the recorded installed producer, not proof
of today's hosted environment. Neither the current local test venv nor its
bundled parent has `nfl_data_py` installed; offline tests use labelled fakes.
No dependency installation or authentic source request was performed.

The [published v0.3.2 implementation](https://github.com/nflverse/nfl_data_py/blob/v0.3.2/nfl_data_py/__init__.py)
reads `http://www.habitatring.com/games.csv` **before** filtering seasons.
The [schedule dictionary](https://nflreadr.nflverse.com/articles/dictionary_schedules.html)
defines game ID, season, game date, named teams, scores and result, but no
`source_available_at`. The old owner reader requires that field and rejects
other-season rows. Focused source-shaped synthetic probes record the two
actual old-reader failures: `NFL_PRIVATE_PUBLISHER_AVAILABILITY_MISSING` and
`NFL_PRIVATE_DEPENDENCY_IDENTITY_CONFLICT`. Its source bytes and behavior
remain unchanged; this successor is not an automatic packet migration.
Missing required native scoring columns reject separately as
`NFL_PRIVATE_DEPENDENCY_SCHEMA_MISSING`, rather than misidentifying that loss
as missing publisher availability. A consistently rehashed missing-column
synthetic subject reproduces the prior diagnostic and makes zero numerical calls.

The old document paths expect the provider to supply internal canonical event,
quote and clock declarations. The new `nfl-original-source-projection-v1`
instead binds original body and receipt hashes to separately attributable
JSON paths or exact text spans. Named teams use the existing deterministic
mapper, without new aliases. Crosswalk facts can reference both an independent
schedule and the exact Odds API event. Provider namespace is receipt attribution,
not a fabricated provider column. Offer/listing/rules facts must be supported
by original documents, with owner verification of their exact applicability.
Generic templates and a hash alone supply none of these facts.

## Explicit prospective successor

Input `nfl-owner-observed-private-inputs-v2`; custody
`nfl-original-response-custody-v2`; review
`nfl-owner-observed-verification-v2`; result
`nfl-owner-observed-private-result-v2`. Dispatch is explicit in
`nfl_private_routes`. v1 input/review cannot borrow v2 approval or vice versa.
Mixed-version duplicate offers reject before inference. Selection/staging
does not review, accept, collect, initialize storage or invoke inference.

Original decoded JSON, CSV or text bodies are retained byte for byte. Limits
stay 512 KiB/body, 16 objects, 2 MiB aggregate, four 8 MiB packets and 16 MiB
selected. CSV projection records exact selected-season row indices in source
order; filtering never changes the original body. Unique IDs and selected
season/team/scores/result are checked, then unchanged native stable-date,
completed-before-current-UTC-day and last-five mathematics derive the eleven
features. Both teams still need at least one completed game. Feature order,
supplied arithmetic, exact model/configuration/artifact and consumed reader
runtime are verified. No weights, formulas or scientific requirements change.

Advance permissions precede request start and cover retention and prospective
private use through inference. Actual request start <= complete-body receipt
<= first local observation <= feature availability <= substantive owner review
< **genuinely new** inference. Original provider quote update <= observation;
the unchanged 900-second, nonfuture, pregame checks remain. v2 scoring receipts
must declare `publisher_availability_unknown`. Output separately records
`publisher_available_at: null`, `publisher_availability: UNKNOWN`, and
`availability_basis: verified_first_observation_before_new_inference`.
Observation is never relabelled publication; it cannot establish historical
availability. The packet has `original_inference_time: null`. Changed bytes,
projections or clocks require a new exact subject and actual owner verification.

The original six substantive owner findings bind every original object/clock,
projection, permission, feature, event, offer and model. Reviewer remains
Robert Velarde, honestly OWNER_REVIEWED / PRIVATE_RESEARCH, with no mandatory
second person for this mode. Future observation/review, contradictory identities,
missing permission and unknown quote meaning remain unavailable before math.

Operator scope remains **Novig NFL-001 half-point full-game/overtime FVS**.
Listing/product/jurisdiction/effective rule/selected position must be evidenced.
Neutral-site, totals, integer push and other operators remain unsupported.
FVS binary EV/edge/break-even stay unavailable. Raw model, original blend and
actual UI refresh retain separate receipts; none is relabelled calibrated.

## Isolated acquisition and handoff

`python scripts/nfl_bounded_acquisition.py PLAN.json` is network/credential/store
free planning. `--execute --authorization AUTH.json --custody-root ROOT` is
separate and requires hash-bound trusted owner authorization, reviewed advance
permission and verified custody configuration. Production catalogs are empty;
the CLI cannot activate them. Controlled entries require separate review and
application. No NCAAF plan, catalogs, transport ceilings or storage receipt change.

The fixed public HTTPS request is
`https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`,
GET, no parameters or credentials. One target, one whole multi-season body,
one durable attempt; no implicit importer, retry, redirect, pagination or
fallback. The shared NCAAF deadline/journal primitives are reused unchanged.
Connect/read/total deadlines, streamed decoded/wire limits and complete-body
checks stop on failure; the fsynced exclusive-create journal prevents restart
from repeating an attempt. Failure does not truncate a body into completeness.
An unreviewed immutable bundle is private-read back and stops without inference,
source acceptance or review creation. Later supplied-input `admitted_analysis`
calls the existing market predictor with explicit packets, not general live
analysis, schedule recovery, enrichment, model services or Drive.

The supplied independent inventory determines coverage's Eastern date even
when a packet is rejected. A malformed rejected kickoff cannot raise during
coverage construction or remove its scheduled event. Without an inventory,
an original source kickoff is required; conflicting inventory dates fail
explicitly. This was reproduced with a labelled synthetic malformed packet
before correction and zero numerical calls, then retained as UNVERIFIED.

**Acquisition is currently blocked.** The repository's [games.csv metadata](https://github.com/nflverse/nfldata/blob/master/data/games.csv)
reports about **2.08 MB**, above the 512 KiB ceiling, independently of gzip.
That is metadata evidence, not an acquired scoring body or a measured receipt.
The legacy HTTP endpoint is not authorized as an HTTPS replacement; no bounded
same-contract season endpoint was established. The public repository's software
license alone is not treated as data/owner-specific permission. The included
one-request proposal is non-executable and carries size, source-permission,
target/window and authorization blockers. A separately reviewed compatible
bounded source is needed; this task does not enlarge limits or fetch a probe.

## Validation and authentic readiness

Labelled synthetic tests traverse source-shaped full bytes -> attributable
projections -> substantive owner review -> actual market caller -> original
private capture/export/replay -> per-game reader -> private Streamlit display.
They compare probabilities exactly and prove public exclusion/PASS/zero stake.
Rejection spies prove no numerical calls for missing/corrupt bodies, changed
subjects, chronology, feature arithmetic, identity and quote failures. Original
v1 tests and independent catalog behavior remain unchanged. Full required CI,
base checks, exact binding/implementation/seal/merge tree and raw failure history
are recorded separately in the PR/private validation receipts.
The full-suite review caught a successor wrapper returning an exception from
the legacy boolean guard comparator for tampered bytes. The comparator now
returns `False`; provenance reconstruction and the actual scope gate still
reject those same bytes. The original predecessor assertion is unchanged.

| Boundary | Authentic status | Smallest next evidence/decision | Owner |
|---|---|---|---|
| Original bounded scoring source | BLOCKED: documented full body oversized | Compatible complete original body source and applicable data permission under unchanged limits | Source/owner |
| Independent target schedule/crosswalk | Missing for retained New Orleans chain | Exact target mapping, named sides/start/non-neutral fact | Owner/source |
| Exact current quote | Historical only, expired | Separately authorized fresh bounded quote custody; its documented update meaning | Owner |
| Novig listing/product/period/OT/FVS | Missing exact applicability | Authoritative applicable listing and effective NFL-001/FVS bridge; substantive owner verification | Operator/source/owner |
| Owner review | No authentic v2 reviewed subject | Review actual bytes, projections and all six conclusions after capture | Robert |
| Custody/storage | Existing verified private setup reused | Separate NFL custody binding approval; no new setup/Drive work implied | Robert |
| Scientific qualification/wagering | Not established | Existing independent scientific route and separately authorized authority | Governing review/owner |

Historical New Orleans raw probability, distinct original blend, original
rejection and absent UI receipt remain unchanged; original packets and raw
validation receipts remain private and local.
No historical inference, collection, inquiries, publication or wagering occurred.
No authentic successful current-hosted display is claimed. Unsupported inputs
remain unavailable; unqualified selections remain PASS at zero stake.
