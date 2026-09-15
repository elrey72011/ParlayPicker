# MLB prospective pregame receipt collector — 2026-09-15

HEAD BEFORE: `30acb57e2392b49bef2ce5e73c0400f221834e19` (main after merged #2284).
HEAD AFTER: commit containing this report on `codex/mlb-pregame-receipts`; exact SHA in the PR handoff.

TIED FINAL BUG: FIXED. `label()` rejects equal FINAL scores for all four targets. Valid non-tied finals, VOID, and betting-line PUSH outcomes remain supported.
RECEIPT COLLECTOR: IMPLEMENTED.

## Sources and point-in-time contract

- LIVE SOURCE FOR PROVIDER EVENT ID: MLB Stats API `/api/v1/schedule`, `gamePk`. The exact quote retains its separate Odds API event ID. The collector records both namespaces and never equates their IDs.
- LIVE SOURCE FOR HOME TEAM ID / AWAY TEAM ID: schedule `teams.home.team.id` / `teams.away.team.id`, represented as `mlb:<numeric ID>`. Names are never hashed into IDs. Missing or conflicting IDs reject capture.
- LIVE SOURCE FOR GAME START: schedule `gameDate`. Linking to an Odds API event requires the existing MLB team aliases, oriented home/away pair, same Eastern scheduled date, and exact UTC scheduled start. More than one event for the pair/date, including doubleheaders with an already completed game, remains ambiguous. No nearest-time guess.
- LIVE SOURCE FOR QUOTES: original live Odds API bookmaker objects, before candidate expansion. Existing `provider_quotes` supplies the exact signed spread or total, side, American price, book and provider ID. Prices are converted to decimal without changing the offered line. Fixed source preference is Novig, DraftKings, FanDuel, BetMGM; one first accepted receipt per event/target. Moneyline is not a target.
- LIVE SOURCE FOR QUOTE OBSERVED_AT: actual aware UTC clock at the live Odds API response boundary. Historical/date-requested calls do not acquire a live marker. Provider last-update time is retained separately and cannot be later than observation. Quote observation age must be at most 30 minutes, including after prior-game fetches.
- LIVE SOURCE FOR PRIOR GAMES: current-season regular-season schedule chooses the most recent ten completed games per team; raw `/api/v1.1/game/{gamePk}/feed/live` observations supply actual completion, scores and IDs. Existing `normalize_game` validates final status and actual play timestamps. Target games, duplicates, wrong namespaces, ties and invalid scores are rejected by the existing receipt contract.
- LIVE SOURCE FOR PRIOR GAME AVAILABLE_AT: actual UTC time the collector receives/parses that game feed. This is a conservative local availability observation, not a reconstructed historical publication time. Completion must precede observation; observation must precede capture. Cached observed scores/team IDs must agree with the newly observed schedule or be fetched again.

`captured_at` and `prediction_cutoff` are the actual receipt construction time. Both must precede scheduled start, and the storage write checks start again after obtaining its SQLite write lock. A feed batch crossing start cannot produce a receipt. The original schedule and quote observations, plus every referenced prior-game feed, are stored separately and linked by hash. Optional deterministic/configured-blend baselines are omitted because this raw provider boundary does not produce them; no baseline provenance is invented.

## Storage, reconciliation and integration

IMMUTABLE STORAGE: dedicated `mlb-pregame-receipts.sqlite3`, adjacent to the existing evidence database under `PARLAYPICKER_EVIDENCE_DIR` (default `data/prediction_evidence`). It does not write the approved evidence database. Separate observations, receipts and outcomes tables reject UPDATE, DELETE and duplicate SQL replacement. Exact duplicate application writes are idempotent; conflicting event/target payloads are rejected. Collection retains the first accepted event/target receipt and reports later attempts as duplicates.

PREGAME HASH / METHOD: the existing challenger `canonical()` and `digest()` functions, SHA-256 over sorted compact JSON with nonfinite numbers prohibited. The stored `snapshot.payload` and `snapshot.sha256` pass `receipt_features()` without adapter repair. Table payloads also have independent verified storage hashes.

POSTGAME RECONCILIATION: IMPLEMENTED as a separate explicit command. It fetches the exact MLB gamePk and requires matching provider, team IDs, season and scheduled start, verified non-tied FINAL scores, actual first play after capture, and completion no later than observation. It appends one outcome per event without editing any original snapshot. Rescheduled start conflicts remain unresolved; cancellation/postponement never automatically implies VOID. The storage API supports explicitly supplied, identity-matching VOID records, but the live reconciler emits only verified FINAL records.

STRICT CLOSE KEPT SEPARATE: PASS. Closing-line collectors, strict-close stores and existing immutable evidence are untouched. These receipts are not closing observations or CLV evidence.
CHALLENGER CONTRACT COMPATIBILITY: PASS. Collected fixtures pass `receipt_features()` and reconciled records pass `prepare_rows()` using the existing `snapshot` / `outcome` schema.

Normal live MLB analysis invokes capture on original provider objects. One season schedule plus up to 20 missing prior-game feeds is fetched per invocation (four concurrent feed requests, explicit connect/read timeouts). Cached verified observations allow later batches to make progress. Capture failure leaves ordinary research games available and reports skipped target counts/reasons. Provider IDs flow through candidate expansion and private authority projection. They resolve only the factual missing-team-ID condition; allocation and authority rules are unchanged.

The existing readiness expander displays capture counts and offers a receipt-health JSON download. There is no activation control. App restarts do not overwrite an existing database; however, an ephemeral host can lose local files. **Deployment must retain/back up this dedicated database on persistent storage. Existing remote approved-evidence synchronization is not automatically extended to this separate file in this PR.**

## Operation after deployment

Run normal MLB/all-sports analysis to collect pregame observations, or use:

```text
python scripts/capture_mlb_pregame_receipts.py capture
python scripts/capture_mlb_pregame_receipts.py status
python scripts/capture_mlb_pregame_receipts.py reconcile
python scripts/capture_mlb_pregame_receipts.py export --output receipts.json
```

The CLI uses the existing Odds API credential resolver; no new secret is committed. `--database` selects a dedicated store. `--max-feeds` bounds capture (default 20, maximum 100) or reconciliation batch size. Export defaults to settled records, requires a new output filename, and never trains. `--include-pending` explicitly includes records whose outcome is still null. Reconciliation is explicit; no scheduler or automatic training/promotion is added.

BLOCKED DATA PREREQUISITES: individual paths remain closed when the live quote marker, supported exact price, unambiguous matching event, matching scheduled start, stable IDs, ten observed prior finals per team, valid timing, or writable storage is missing. A first cold batch may not cover a whole slate; repeat capture before start to fill the cache. Outcomes with changed starts or incomplete final feeds remain unresolved. No live provider capture or real training was executed during development; source behavior was verified with representative provider fixtures, not claimed as a successful deployed capture. Persistent storage and deployment are operational prerequisites for durable accumulation.

AUTOMATIC TRAINING: NO.
AUTOMATIC PROMOTION: NO.
PRODUCTION ELIGIBILITY CHANGE: NONE. No model/calibration validation flags, active policy, thresholds, sizing, maturity, bankroll/exposure policy, Gemini behavior or execution code were modified. No real artifact was trained or promoted.

## Validation

- RECEIPT-01 through RECEIPT-16: covered by collector/model regression tests, including tied FINAL, missing IDs, all timing boundaries, prior leakage/duplicates/namespace, exact orientation, reproducible hashes, SQL replacement protection, conflict rejection, separate outcome append, trainer compatibility and unchanged authority.
- Additional checks: live-only observation stamping, doubleheader ambiguity, bounded batch resume, no unnecessary cached-feed refetch, game starting during collection, save-time start guard, source-failure isolation, stable IDs through private projection and zero allocation.
- Focused initial model/collector/training suite: 68 passed; final collector suite: 26 passed.
- FULL TESTS: 2,401 passed (125.81 seconds).
- PRODUCTION SAFETY: 545 passed using the current CI test list.
- PRODUCTION COMPILE: PASS, current CI command plus both MLB CLI scripts.
- GIT DIFF CHECK: PASS.
- Local validation on Windows; hosted Linux CI runs separately on the PR. Existing pandas deprecation/performance warnings remain.

FILES CHANGED:

- app_core/mlb_spread_total_model.py
- app_core/mlb_pregame_receipts.py
- app_core/odds_api.py
- core/streamlit_pipeline.py
- app/ui/readiness_dashboard.py
- scripts/capture_mlb_pregame_receipts.py
- tests/test_mlb_pregame_receipts.py
- tests/test_mlb_spread_total_model.py
- docs/audits/2026-09-15-mlb-prospective-receipt-collector.md

UNRELATED CHANGES: NONE in this change. Pre-existing untracked Medium article files are excluded. No captured live data or secrets are committed.
