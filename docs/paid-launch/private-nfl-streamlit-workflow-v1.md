# Private NFL Streamlit workflow v1

## Authority and frozen boundary

The owner authorized bounded engineering in the October 26, 2026 conversation:
"Make routine private NFL predictions usable entirely through Streamlit."
Scope includes related UI/backend/storage/fixture corrections, bindings, local
validation, reviewable commits, a separate draft PR and one conditional complete
final-head CI cycle. This document records that instruction, not an operational
approval or an invented approval timestamp. No acquisition, credentials,
authentic inference, operational configuration application, merge or deployment
is authorized by engineering. Related files need no individual approval loop.

Base is `22f4dc219d8360001d54202418abcb78f5c72d98`, tree
`b9153afba757718cc48fef853da75a3f99f19049`. Its cleared 23 checks, 20 artifacts,
13 manifests and 9,061 reconciled identities (9,059 passed, two inherited skips)
are reused, not rerun. The exact successor scope seals this implementation and
the tested merge must have the same tree. Existing full-suite shards, job limits,
historical assertions and remote-export SQLite prohibition are unchanged.

The fixed `score-distribution-v1:nfl` formula, reliability **0.45**, eleven-feature
order, native scenario caller, live reader, replay readers and external ZIP trust
catalog stay unchanged. Neither fitted shrinkage nor the rejected ridge model is
used. Existing authentic results are neither recalculated nor rewritten. This
work never reads game outcomes to grade those results. Delaware–WKU's operation,
pinned checkout, configuration, approvals, custody and clocks are outside scope.

## Owner workflow

Open Workspace → Preview & Publish, enter the owner publishing token, then enable
**Guided private NFL forecasts**. This is a real default-off guard, before store
resolution or access. Unrelated publishing/token controls do not activate it.

1. Load retained NFL games, or prepare a bounded scoring capture. Read the source
   permission, exact request inventory and completion window, then explicitly
   authorize it in the form. Nothing is acquired by simply opening the panel.
2. Select up to four future non-neutral games from the original nflverse schedule.
   Prepare the selection; review home/away identity, kickoff, completed histories,
   all eleven feature values and their original observation times. This works
   for new schedule identities without a per-game catalog or source change.
3. Review the Novig spreads request for each selected event. Enter actual included
   entitlement/reset information and authorize only those requests when ready.
   CFBD/CBBD usage is separate and zero. No paid overage is permitted.
4. Review the authenticated opposing half-point lines, actual prices and clocks.
   Select both as explicit full-game, including-overtime mathematical assumptions.
   Check the six substantive findings and attestation, accept the limitations,
   then authorize one calculation per side and private capture. The actual answer,
   exact subjects and clock are durably recorded before numerical calls.
5. Both original outputs are saved before comparison. Use **View saved forecasts**
   to select and reopen them, including after kickoff or a fresh app session.
   Viewing is authenticated static replay, with the original calculation time
   shown separately from viewing time. Optional ZIP download is not routine input.

Supported target: selected final points plus signed handicap strictly exceeding
opponent final points, completed full game including OT. Integer lines, neutral
sites, conflicting identities, absent histories, expired permission, stale
quotes and kickoff fail explicitly. No line rounding, substitute moneyline,
fallback market, manual probability, silent adjustment or normal Refresh occurs.
A bounded kickoff-window listing may contain several NFL games. The released
mapper selects the unique exact named event by provider ID, preserving the entire
original and rejecting ambiguity. Quote request authority remains event-specific;
scoring originals can be reused across compatible games under their permission.
The original NFL reader has no additional invented scoring-age cutoff. Source
permission and actual original clocks still apply. Quote age is at most 15 minutes
from the market's actual update to each numerical call, and every call is pregame.

Labels remain **uncalibrated private research / PASS / zero stake**. Preferred side
means higher conditional model probability; tied estimates have no preferred side. Prices do not imply EV or verified
bookmaker settlement, and the model has no explicit quarterback/injury/inactive
adjustment. No calibration, validated accuracy, predictive advantage or wagering
admission is established. Billing, subscriptions, Gemini, customer publishing and
state-specific wagering admission are not requirements of this private route.

## Durable storage and hosted setup decision — not applied

The previous Windows EFS/SQLite operational store and fixed external ZIP replay
catalog cannot be presumed to work as a hosted execution authority. Streamlit
session state and `/mount/src/...` SQLite files are not durable custody. Shared
Drive evidence export is not an atomic dispatch lock. The new workflow uses a
dedicated durable PostgreSQL private schema for originals, approvals, operation
claims, results and receipts; no billing tables or existing SQLite migration.

Owner deployment approval must supply a private encrypted persistent PostgreSQL
database, encrypted backups, verified TLS hostname/certificate trust, a dedicated
least-privilege app login and separately protected random HMAC key (at least 32
bytes; retain for replay). The administrator-reviewed SQL preparation is
`docs/research/nfl-private-workflow-store-v1.sql`. It has not been applied by this
task. Schema/table owner is different from the application login; deny public
access, role membership, table modification/deletion/truncation and schema CREATE.
The app verifies protections on every operation and does not initialize/migrate
the database. Configuration is through existing private settings/secrets:

| Setting | Required deployment meaning |
| --- | --- |
| `PARLAYPICKER_PUBLISH_TOKEN` | Existing owner token, minimum 16 characters |
| `PARLAYPICKER_NFL_DATABASE_URL` | Dedicated login, host and `sslmode=verify-full`; secret |
| `PARLAYPICKER_NFL_STORE_KEY` | High-entropy private authentication key; secret |
| `PARLAYPICKER_NFL_ACQUISITION_ENABLED` | Default false; literal `true` only after owner operational setup approval |
| `PARLAYPICKER_NFL_INFERENCE_ENABLED` | Default false; literal `true` only after owner operational setup approval |
| `ODDS_API_KEY` | Existing private provider setting, resolved only at approved dispatch |

No secret is requested or read during engineering. Database connectivity is a
genuine deployment dependency. PostgreSQL network access is storage traffic,
distinct from provider requests. psycopg uses the existing service's 3.2.10 pin.

Unique durable claims commit before every HTTP hop and native call. Reruns,
double clicks, reconnects and multiple tabs adopt the same semantic operation.
Claims never expire/reset or authorize retry, even after interruption. A failed
or expired operation requires a visibly prepared successor and a new actual
request approval; its predecessor stays spent. Per-operation windows last at
most 20 minutes and end before kickoff. Launch requires a full 60-second budget
to remain; durable-claim time counts against the transport deadline. The original
quote is rechecked after the durable native claim at the recorded attempt clock.
Transport total is 60 seconds, with the
released connect/read, wire/body/header limits and validated redirect unchanged.
Scoring permits at most two HTTP requests; each Novig event request at most one
HTTP request/one included credit. No diagnostics request or fallback is sent.
Originals, receipts, source permission clocks, sanitized failure fields and spent
accounting remain append-only. Source originals can serve multiple compatible
games only under their original permission and clock contract.
The internal signed-record envelope is bounded at 32 MiB to hold a prepared
original and both native packets. Released CSV, quote, wire, header and individual
native-packet limits remain unchanged; this is not a larger acquisition allowance.

Native writer results are accepted using unchanged `nfl_scenario_research.read_result`
before store authentication. Static reads use freshly authenticated owner-scoped
HMAC records, producer source/model bindings and the existing ZIP static validator.
The trusted anchor comes from the private native writer, never a self-hashed
upload. New native results need no per-game source hash catalog. External ZIP
import remains optional and retains its fixed supported-binding security boundary;
new workflow exports are not automatically admitted by that external catalog.
Display reads 100 saved sides per page (up to eight selected for simultaneous
viewing). The owner can change pages to reopen older forecasts without a ZIP,
query or code change. Reads remain bounded and freshly authenticated. Store-key loss or unsupported
future frozen-reader/model bindings stops replay; it is not silently bypassed.

## Validation and release gate

Synthetic fixtures are explicit test-only constructors/entrypoints, never a
production secret, query parameter, upload or environment switch. External
network remains blocked. Real local browser tests exercised two uncatalogued
synthetic events, four native fixture outputs, durable reopen, switching, clearing,
fresh sessions and authorization loss. Saved viewing made predictor/provider/store
write attempts fail immediately; hashes and original clocks were unchanged.
AppTest regressions are additional evidence, not a replacement for browser use.
Actual hosted configuration, persistence after app replacement and authentic
prospective execution are not established by those tests.

Combined validation includes complete source-evidence, saved viewer/ZIP and remote
canonical files with the original owner-panel context, plus new backend/UI/scope
tests. The existing PostgreSQL service job adds synthetic native-store cases while
preserving its original three tests, check name and 15-minute limit. Thirteen
full-suite shards and 45-minute limits stay unchanged. Final collection, capacity,
one CI cycle and exact artifact/inventory reconciliation are owner-review gates.
CI failure is retained, without retry or an extra diagnostic job.

## Prepared deployment and authentic acceptance — future authority required

After owner merge approval, release clearance and deployment approval: configure
the dedicated store and secrets; verify serving revision and schema/login/TLS/key
protections, backup/restore and persistence after app process/filesystem replacement.
Keep acquisition/inference disabled until that decision passes. No migration of
existing authentic originals is part of this setup.

Then the owner selects **two different genuinely future eligible NFL events** in
the deployed app. Event names, identities, kickoff, histories, permission, model,
budgets and actual half-point lines must be reviewed at that time. A fresh scoring
capture, if needed, is at most two public HTTP requests and zero credits; two
event-specific Novig captures total at most two HTTP requests/two included Odds
credits, $0 additional spending and zero CFBD/CBBD usage. This is a prepared maximum
inventory, not an approval to spend it. Actual account entitlement and actual
event-specific request authorization must be entered in Streamlit before dispatch.
Finish avoidable preparation before quotes. If any line is unsupported or stale,
stop without automatic replacement. Record all six actual findings and authorize
exactly one native calculation per side per event (four calls total), before each
applicable cutoff. Preserve both outputs before comparison, reload via saved view
in a fresh session and after process replacement, verify original hashes/clocks,
then test static viewing with acquisition and inference disabled. No Codex,
PowerShell, prepared JSON, pasted hash or ZIP transfer is needed for this flow.

These future approvals do not exist yet. A two-game acceptance test validates
operation, not accuracy. Outcomes are outside this engineering task.

## Other sports: reuse map, outside this implementation

| Sport | Existing exact model/data contract | Missing guided-workflow adapter |
| --- | --- | --- |
| NCAAF | Existing immutable CFBD history/model/half-point spread pipeline, `ncaaf_pipeline_evidence`, `ncaaf_model_compatibility` and staged/private Refresh contracts; its operation-specific artifact/reader identities remain authoritative | Separate sport-specific UI authority and hosted custody adapter; do not substitute the NFL feature schema or disturb Delaware–WKU |
| NBA | `pregame-score-ridge-normal-v1`, `provider-final-score-form-v1`, sport/market-specific SPREAD or TOTAL artifact in `prospective_model`; authenticated events, paired quotes and available-at final-score lineage, 2–8 prior games within 240 days | A qualifying retained artifact/input inventory and NBA acquisition/approval/store adapter; model-building code alone proves no usable artifact |
| NCAAB | Same research model protocol, separately scoped NCAAB SPREAD/TOTAL artifact and basketball team/event/final-score identities; neutral sites/season and source availability must be represented correctly | CBBD-to-basketball evidence mapping and qualifying artifact/lineage review, then sport-specific adapter; football CFBD features do not transfer |
| NHL | Research protocol separately scoped PUCK_LINE/TOTAL; existing `nhl_puck_line_evidence` supplies the reviewed half-point puck-line contract, not generic NFL spreads | Qualifying authenticated goalie-omission/score/line evidence and exact sport adapter; do not infer supported totals from puck-line reader |
| WNBA | Fixed `score-distribution-v1` WNBA parameters (reliability 0.65), independent team scoring contract; not the NFL 0.45 model | Authentic compatible history/market/identity and private execution/replay adapter; no new acquisition here |

MLB remains paused. CFBD/CBBD shared usage and Odds credits remain separate. No
other sport's readiness or accuracy is inferred from this NFL software release.
