# Post-#2366 closure: execution and acceptance ledger

Base and audit anchor: `095b594460e94085051162a158b111dd8be31486`.
Actual GitHub main was checked again during this work and still matched that SHA.
Work branch: `codex/post2366-eligibility-capacity`. #2363–#2366 remain in ancestry.
The original checkout stays on `codex/post2362-external-verification` with its six
untracked owner files intact. This record is new; earlier audit history is unchanged.

## Real-entrypoint evidence

The initial two tests failed on untouched current main: the real `run_census`
returned one independent eligible game after an appended conflicting result while
the canonical manifest returned zero; the real `netlify_publishing.deploy` reached
its mocked upload after a separate $50 COMMITTED event consumed capacity. See
`reproduction-before.xml`. These are complete local entrypoint reproductions,
with writer-generated evidence and mocked publication transport, not remote
storage measurements, model qualification, hosted proof, or wagers.

The corrected census retains raw ready rows/games, canonical active-view rows,
active eligible rows/games after the manifest joins/exclusions, and independent
manifest games under `football-one-observation-v1`. It projects verified records
in memory and never opens or initializes an evidence database. Checkpoint v2
requires reparsing older v1 summaries that lack dependency fields. A changed source
revision still rejects resume. NFL/NCAAF have the supported canonical projection;
the other eight scopes return null, the precise missing remote manifest reader
reason, the existing exact-scope qualification reader and a bounded owner follow-up.
Cohort membership and model/calibration/plan/artifact binding remain explicitly
unevaluated; inventory cannot confer qualification. Nothing promotes legacy rows.

The publication adapter now calls the shared current allocation limits/headroom
policy against a newly read ledger and the cumulative unique package. It holds the
whole release when a saved amount no longer fits. It never resizes or reprices a
decision. Ledger reads use SQLite mode=ro. Private team identities are read from
hash-checked immutable prediction snapshots; absent/ambiguous identities use an
explicit conservative bound against the existing team cap. The strictest current
sport cap among package scopes applies cumulatively. Public straight releases
already count the full risk of overlapping committed parlays via their underlying
sport/game/team keys. Funded true-parlay allocation retains its separate existing
`core.true_parlay_engine._allocation_keys` limits; public proposed parlay products
remain non-funded. No separate straight overlap cap is invented.

An existing commitment is display-only only with exact contract digest or a
hash-checked immutable snapshot containing the identical contract, plus exact
stake, book, game, market, selection, line and odds. A lookalike, ambiguous or
corrupted binding cannot bypass capacity. Trials reread the existing exact
reservation under current consent and subtract their own already reserved risk
when checking; verification never creates/releases a reservation. Missing current
authority, expired/revoked authority, missing sources and insufficient capacity
retain separate diagnostics. Legacy history retains its existing preflight boundary.

Read-only checks establish capacity at their recorded snapshot; they do not
reserve funds or promise atomic exclusion of a later independent ledger write.
A later retry reads the ledger again. Capacity holds require the existing review
process, not silent changes to saved decisions or timestamps.

## Acceptance matrix

Implemented means repository behavior; executed means a real local entrypoint or
test; externally verified means independent CI or genuine authorized operational
execution. A CI fixture never establishes model or hosted subscriber proof.

| ID | Implemented | Executed | Externally verified / remaining evidence |
|---|---|---|---|
| D01 | PASS: current-main base and isolated checkout | Main/ancestry/owner status checked | Current GitHub main confirmed at anchor; final source identity is the PR head |
| D02 | PASS: separately named raw, active, independent counts | `test_d03`, `test_d04_d05`, `test_d02` | Remote counts NOT_RUN |
| D03 | PASS: conflicting score invalidates active/manifest eligibility | Real writer → encoder → run_census compared with canonical query | Remote revision inventory NOT_RUN |
| D04 | PASS: event revisions, missing quote/result/settlement/event, chronology | Nine canonical-reference cases | Remote records NOT_RUN |
| D05 | PASS: deterministic one-observation manifest | Duplicate/repriced writer records match exact manifest IDs | Remote manifest NOT_RUN |
| D06 | PASS: all twelve scopes known or precise null/reason/reader | Twelve-scope and partial/resume tests | Eight non-football remote training manifest calculations UNKNOWN |
| D07 | PASS: inventory binding remains NOT_VERIFIED and production_eligible=false | Existing synthetic activation rejection and census binding tests | Authentic bound qualification report NOT_RUN |
| E01 | PASS: late unrelated commitment holds actual publisher | Before/after real deploy entrypoint with mocked API | No production publication attempted |
| E02 | PASS: total/daily/weekly/game/team/sport and committed parlay underlying risk | Six limit cases, overlap and settlement turnover | Live ledger check NOT_RUN |
| E03 | PASS: cumulative new tickets, duplicate views, conflicting duplicates | Two tickets fit; package total exhausts; views do not count again | Production package NOT_RUN |
| E04 | PASS: retry and exact committed display; trial reservation reuse | Same bytes/history/package after retries; exact/corrupt/lookalike cases | No real funds reserved or commitments created |
| E05 | PASS: authority and capacity reasons remain separate | Prior closure expiry/revocation/source cases plus new holds | Current real authority NOT_RUN |
| E06 | PASS: release check never resizes/reprices/restamps | Immutable package equality and ledger/reservation byte checks | Any revised decision still requires existing review |
| O01 | Existing bounded read-only workflow retained | NOT_RUN: execution-surface authorization not supplied | Owner must authorize pinned revision/storage/resume |
| O02 | Raw inventory distinct from qualification | Local writer inventory only | Authentic census and exact-scope qualification UNKNOWN |
| O03 | Complete private trace producer preserved | No fresh provider run | Separate provider/quota authorization required |
| O04 | Existing staged API/worker/PostgreSQL/subscriber stack preserved | Local browser PASS; service tests 16 PASS, 10 PostgreSQL skips | Hosted journeys/attestations NOT_RUN; PostgreSQL CI reported separately |
| O05 | Existing timing/measurement implementation preserved | No new real cold/warm/provider/hosted measurements | NOT_RUN; no historical fake-provider percentage reused |
| Q01 | Workflow/CLI/compile/browser and combined CI required | Focused 112 PASS; local browser PASS; Windows full run has limitations below | Final GitHub application/safety/paid/PostgreSQL results must be read at final PR head |
| Q02 | Protected scope and no-production-effect ledger | Protected guard PASS, no existing test expectations changed | Draft PR only; no merge or production effects |

## Validation and artifact identities

- `focused.xml`: 112 local passes, including prior closure, allocation/trial and legacy history.
- `subscriber-local.xml`: 16 passes and 10 explicit PostgreSQL skips; Docker/PostgreSQL
  executables were unavailable locally. Existing PostgreSQL workflows supply the
  separate isolated database execution.
- Existing `tests/subscriber_journey_browser.cjs` passed against
  `web/subscriber/{index.html,subscriber.js,subscriber.css}`: checkout/account,
  cancellation, entitlement denial, all result states/corrections, late-response
  suppression, expiry clearing and mobile keyboard. Context: LOCAL_INTEGRATION.
- Nine workflow definitions validated. The actual census and paid verifier CLI
  help entrypoints and application/service compilation passed.
- Windows full suite before the final compatibility correction: 3,167 passes,
  22 failures and 38 passing subtests. Nineteen were SQLite cleanup/file-sharing
  errors. Two other failures (football stage2 expectation and runtime odds-secret
  environment) and a representative cleanup error also reproduce on an untouched
  export of `095b5944`: `anchor-comparison.xml`. The one introduced legacy-history
  interaction was fixed and its unchanged test passes in `focused.xml`.
  No existing expectation was weakened. This Windows result is not a passing
  final full-suite claim. CI uses the repository's Linux/pinned-runtime workflows.
- `local-entrypoint-reproductions.json` records exact source-file SHA256 values,
  writer/reader identities, manifest IDs, counts, real publisher blocker report
  and zero upload calls after the fix. It is explicitly fixture-marked.
- The machine-readable artifact register and protected-file report accompany
  this ledger. Final CI run/head/artifact identities are recorded separately.

## No-production-effect ledger

Only an isolated local branch/worktree, local fixture records, test/dependency
outputs and a draft GitHub PR are created. No remote evidence inventory, capture,
reconciliation, receipt repair, backup, provider/model run, production record,
publication/deployment, money reservation, wager, market/trial activation,
calibration replacement, threshold/cap/freshness change, billing/sales enablement
or automatic merge is performed. No secret values are requested or recorded.
Fixture reservations/commitments exist only inside isolated test directories.

The protected implementations, scope guard/baseline, benchmark history, candidate
identity/probability fixes and subscriber contracts are unchanged. Narrow
integration changes are limited to census projection/reporting, release adapter,
shared allocation key/headroom extraction and read-only exposure reads.

## Exact remaining owner actions

1. Review this draft PR and its final combined CI. Merge only by a separate owner
   action if desired; no auto-merge is enabled. Qualification and commercial
   approval are separate from merging a correctness fix.
2. Authorize `read-only-census.yml` on the reviewed fixed revision against the
   configured Drive storage, identify any resume run, and approve the object/byte/
   time budget. Use a tag/ref resolving to that one SHA for every slice. Retain
   `read-only-census-state` checkpoints/reports; confirm each run's head SHA, storage
   scope hash, reused/new objects, bytes/timings and completion before continuing.
   Old v1 checkpoints reparse; a different source SHA cannot resume silently.
   Supply/configure credentials through the approved environment, never chat.
3. Authorize a genuinely read-only exact-scope qualification report for unsupported
   manifest scopes and unevaluated cohorts/bindings. Inventory/model presence is
   insufficient to choose/activate the first market.
4. Separately approve the provider execution host, existing producer, scope,
   quota/spend budget and full private candidate trace retention. Retain original
   and current gate outcomes and quote → analysis → review → release intervals.
   Use `scripts/trace_current_wagers.py` on the complete saved candidate input,
   not only the selected board. A traceable explained zero-wager result is valid.
5. Name and authorize an isolated staging host for the existing
   `deploy/paid-launch/docker-compose.yml` API/worker/PostgreSQL/shell. Configure
   OIDC/MFA, Stripe sandbox/webhooks, TLS, SMTP, monitoring and recovery privately.
   Keep sales/live billing/markets/trials disabled. Run the existing hosted
   lifecycle/access/expiry/correction/notification/recovery/load/pilot scenarios.
6. Configure the out-of-band trusted attestation registry and signing secret on
   that approved surface. An authorized reviewer must inspect the real run and
   content-addressed artifacts before attesting. Never sign invented fixture
   reports as real model/hosted evidence. Record source/deployed/served/execution/
   review identities separately, then make any launch/commercial decision.

