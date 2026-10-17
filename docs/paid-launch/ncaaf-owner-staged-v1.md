# Owner-reviewed NCAAF staged originals v1

Implementation-only authorization: proposal `f8938f7b5140e0a41bd81ca71de157331ca394650c1f72b48c715621f1b375a8`.
Base main `f98015dbd969456990a01f9d45405697c7fc5251`, tree `bb55338dbaf1b1dea445e50ec4b3ac20a37fc852`.
No operational authorization, original body, journal, task, model artifact or acceptance catalog is changed.

The successor's exported historical-fixture metadata names the exact current after-images for four entries in the normal/chronology/custody fixture groups. Frozen preimage hashes and reconstruction edits remain unchanged; actual source readers still delegate through the exact original byte projections. Updating this metadata cannot accept altered source bytes, inputs or owner reviews.

## Explicit contracts

* `ncaaf-owner-evidence-crosswalk-v1`: only Memphis Tigers -> Memphis and Tulane Green Wave -> Tulane. Original discovery labels/bytes and qualified review stay separate. The original CFBD schedule must confirm the provisional 235/2655 team IDs and 401862801 game ID. Contradictions reject; no renumbering or mascot stripping.
* `ncaaf-cfbd-history-stage-plan-v1` -> `ncaaf-cfbd-history-unaccepted-v1`: four separately authorized CFBD GETs, no side, price, future quote hash or offer admission required.
* `ncaaf-prequote-history-features-v1`: network-free original-byte projection and frozen deterministic history selector. Binds actual preparation time, model artifact/runtime identities, ordered features and original receipt clocks. No inference or human review is created.
* `ncaaf-owner-prequote-review-v1`: actual substantive Robert review of confirmed event, original histories, deterministic features and model/source basis, with qualifications. This exact review must predate quote dispatch; a prepared feature object alone is insufficient. The quote plan binds the reviewed object's exact hash.
* `ncaaf-final-quote-stage-plan-v1` -> `ncaaf-final-quote-unaccepted-v1`: one separately authorized side-neutral DraftKings spreads GET, bound to the exact history bundle and preparation. It retains both outcomes unaccepted; it does not compare them or select a research side.
* `ncaaf-staged-evidence-binding-v1`: exact join, including both original plans, actual authorizations, durable journals, bodies, receipts, budgets and prepared features.
* `ncaaf-owner-reviewed-private-inputs-v3` / `ncaaf-owner-reviewed-private-result-v3`: explicit successor, with separate alias and stage-review conclusions in the final substantive owner review. Old v1/v2 packets/reviews/readers remain distinct.

## Request scope and accounting

History stage uses `https://api.collegefootballdata.com/games?year=2026&seasonType=both&classification=fbs`, then `games/teams` with year 2026, seasonType regular, and weeks 1, 2, 3, in that order. The exact target is Memphis AWAY at Tulane HOME, nonneutral, kickoff 2026-10-16T23:30:00Z. This version deliberately does not generalize to another target.

Final stage uses one GET `https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds` with regions=us, bookmakers=draftkings, markets=spreads, oddsFormat=american, dateFormat=iso and kickoff bounds 2026-10-16T23:29:59Z through 2026-10-16T23:30:01Z. No optional parameters or alternative endpoints. One included Odds credit maximum; four CFBD included requests; zero additional spending.

Each dispatch has its own exact owner authorization, interval, advance permission, entitlement evidence and controlled-configuration review. Empty production maps confer no authority. Configuration selection is context-local and cleared on exit. Native readonly private/encrypted/no-reparse custody checks remain mandatory.

Each stage retains the exact typed controlled-configuration file bytes/hash, including its actual review, authorization, custody and entitlement evidence. The journal's initial durable record binds that control receipt before dispatch. Static read-back verifies its plan/authorization/review relationship without refreshing any clock or scanning the control envelope as a provider response. The final joined subject retains both configuration receipts separately.

Configuration review cannot precede its authorization, permission review or account observation. Final-quote authorization cannot precede the actual prequote review it binds. Original review/receipt strings are retained; no stage rewrites them to establish chronology.

One attempt/request, no redirect, retry, fallback, pagination or automatic rescheduling. Exclusive journals are flushed/fsynced before dispatch. A final-quote claim keyed to the history bundle is exclusively created before GET. Changing a quote plan does not reset the claim. An incomplete/interrupted history cannot be used as a parent. Read-back verifies complete original bytes, hashes, metadata, journal order and cumulative budgets. Failures retain sanitized allowlisted codes, never exception text or rejected credential-bearing bodies.

A quote interruption after claiming the history leaves that claim spent, including interruption before the GET. Abrupt process exit can leave an unfinished journal with no final bundle; that is incomplete evidence, not permission to resume. The same plan and a replacement quote plan both reject. Copying history into a different custody root cannot reset the claim: both stages must retain the exact same reviewed custody identity, root and storage receipt.

512 KiB complete wire/decoded per body; 2 MiB combined decoded originals across stages; 6 MiB stage/join serialization; maximum 16 objects. HTTP client originals mean body after content decoding, before JSON, exactly as the preserved custody contract specifies; no claim that this retains compressed wire framing. Connect/read each 10 seconds. The 60-second monotonic active acquisition budget includes the history stage and the final stage. Human preparation/review waiting is not active acquisition, and does not replenish consumed time. The final deadline is 60 minus the durable history elapsed time. Original-body feasibility remains UNKNOWN before authentic complete capture.

## Admission remains separate

The frozen selector requires at least three eligible scoring and three yardage histories per team, all strictly earlier than kickoff minus seven days. CFBD originals must be no more than 24 hours old at actual inference. Publisher availability stays UNKNOWN, with the existing seven-day research proxy; no retrospective publication assertion.

The quote's original provider market observation remains the origin of the 900-second freshness clock. Receipt, preparation, review, export and replay never refresh it. Inference remains prospective/pregame. Exact DraftKings product/listing, period/OT, rule edition/effective interval and overrides still require attributable evidence and substantive owner review. Acquisition cannot supply missing applicability by default.

The final reviewed subject binds every original body and stage, confirmed mapping, features, source permissions, exact offer/terms and model/runtime. Review must predate inference and cover the exact subject with substantive findings. The preserved qualified name review is not acceptance of unseen CFBD bytes or a future offer.

`admitted_analysis` follows the existing pilot's explicit selected-market path into `market_probability_model.predict_market_probabilities`, followed by the normal custody/native predictor and private capture/export/replay/display. It does not request data or ask the multi-book slate normalizer to select an offer. Static replay verifies retained results; it does not rerun mathematics. Outputs remain OWNER_REVIEWED / PRIVATE_RESEARCH, uncalibrated, PASS/zero stake, excluded from public probabilities and qualification/catalog activation.

## Network-free commands and future operational steps

Use Robert's separately verified isolated Python runtime with `-I -B`. Planning:

```
python -I -B scripts/ncaaf_staged_acquisition.py plan reviewed-history-plan.json
python -I -B scripts/ncaaf_staged_acquisition.py prepare history.stage.json --model reviewed-model.json --output prequote-features.json
python -I -B scripts/ncaaf_staged_acquisition.py plan reviewed-final-quote-plan.json
python -I -B scripts/ncaaf_staged_acquisition.py join quote.stage.json --history history.stage.json --prequote prequote-features.json --output joined-originals.json
```

These names are templates, not executable approvals. A future execution additionally requires `execute`, exact `--authorization`, `--controlled-bindings`, `--bindings-sha256`, `--custody-root`; final quote also requires `--history` and `--prequote`. Only an interactive private terminal may supply the stage's masked CFBD/Odds key; no key in arguments, environment, files, chat or logs. The runtime gate rejects pytest/instrumented networking and never disables test blockers. This implementation does not create a scheduler/pre-arm helper.

After successful history capture: confirm original CFBD IDs and histories, prepare/review features, resolve exact applicable DraftKings terms, then authorize a separate final quote. After that unaccepted capture: select a genuinely observed supported half-point side, finish exact-subject review, separately authorize prospective inference, and verify input/result/export/replay/display agreement. No half-point offer is assumed. The retained 483-byte ±7 discovery is stale discovery evidence only.

## Deferred capabilities

Both-side comparison and integer spreads are separate implementation proposals, not enabled here. The existing native `ncaaf_research.probabilities` already computes discrete boundary push mass. Integer support still needs versioned cover/push/loss caller, receipts, settlement interpretation and replay/display validation; removing the zero-push guard alone is insufficient. A pair route must retain two attributable computations over the same event/market/terms/opposite signed lines, not call higher cover probability greater EV. Neither feature creates wagering qualification.

Massachusetts' active DraftKings rules candidate (implementation date October 7, 2026) does not connect this feed to that jurisdiction. A provider/operator-attributable mapping of this bookmaker feed's exact listing/product to governing edition/overrides is still missing. The owner supplied no such reference; that is not proof no reference exists or an additional license is required. No inquiry or document retrieval is authorized by this implementation.
