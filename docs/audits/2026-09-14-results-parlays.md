# Results reconciliation and parlay eligibility — September 14, 2026

## Baseline and scope

Current main before edits: `2dcc9bf2d5ac79e9e98def8bf0c47af6c1077b52` (PR #2270 merged). Branch: `codex/results-parlay-reconciliation`. Relevant baseline: **187 passed** in 6.68 seconds. Unrelated document files were left untouched.

No immutable lock, publication, loss, original odds or pick was rewritten. Code and test changes are prepared for PR review; production Drive/SFTP credentials are unavailable in this workspace. **Live backfill and publication have not been performed.**

## Root causes traced before editing

| Defect | Function | Before / failure | Fix |
|---|---|---|---|
| Delayed final rejected | `public_history.grade_leg` | Ordinary non-college picks required starts within 30 minutes | Provider-specific saved ID, then canonical teams/Eastern date; time only disambiguates multiple events |
| MLB provider gap | `public_history.fetch_scores` | ESPN-only retrieval | ESPN plus bounded official MLB schedule fallback, source IDs, retrieval provenance and unfinished-event identity protection |
| Opaque pending rows | `public_results.update_pending_results` | Fetch/rebuild with no per-row explanation; repeated revision timestamps created new blobs | Stable revision signature, before/after/source diagnostics, append-only persistence |
| Book inconsistency | `public_parlays.build_parlays` | Qualified filter hardcoded Novig | Shared `supported_quote`, same-book pairing, maturity restrictions and owner funnel |
| Service outage equals rejection | `gemini_bet_gate` and final gates | All unavailable reviews held at zero; transport errors discarded | Preserved timeout/5xx status; explicit reduced/capped deterministic straight-only policy; no Gemini approval |
| Lost review provenance | Portfolio allocator | Returned columns omitted outage/maturity status | Return the original review status and cap metadata with sized output |
| NFL missing stats | `fetch_nfl_stats` | Missing team rows looked like alias failures | Explicit no-completed-season-games diagnostic; existing DEN/KC mappings tested |

The start-time defect is demonstrated in regression tests. The exact original production score revisions and locked leg timestamps are not accessible here, so it is not proven that this explains every one of the four live rows. The NFL pending row may simply need a post-final grading run.

## Actual pending records and authoritative finals

The user approved proceeding with the actual four records after the initial discrepancy. Public record IDs were verified against the existing immutable lock identity hash `(locked-overall, sport, canonical away, canonical home, Eastern date)`; this verifies the public event identity, not a full Drive restore.

| Saved ID | Original published selection | ESPN event ID | Retrieved final (away–home) | Outcome |
|---|---|---|---|---|
| `164f1d536bf7ad64c8251de50d12d2e6d413c5c32ee336049317fe7e64dac74f` | Seattle at Athletics: Athletics +1.5 | 401816931 | 7–8 | WIN |
| `f01f0193b4b5e8db2603dc7d2e0dfadfbeeec9fde6683157f4a7b39f0467a84d` | Kansas City at Boston: Kansas City +1.5 | 401816922 | 1–4 | LOSS |
| `cac4e624e0c74d4a0744e62d49385e1fe3b29d0b7ec7a5188f22cd133cd6f0a1` | San Diego at San Francisco: Under 7.5 | 401816930 | 6–4 | LOSS |
| `6d0dd98c6a37f4518750b4daf4ae7090f9d2f12cde359e3d53cef513e19564a7` | Dallas at New York Giants: Over 47.5 | 401872930 | 20–28 | WIN |

Dallas/Giants totals **48**, so Over 47.5 wins. Seattle/Athletics Under 7.5 is a regression fixture from the initial request, not one of the four actual pending locks. It is not substituted into history.

Production retrieval returned 28 ESPN finals across MLB/NFL and 15 official MLB finals, with no provider errors. ESPN is preferred where providers agree; conflicting scores stay pending for review. Provider data was fetched through the production retrieval path, not hardcoded. Sources: ESPN scoreboard endpoints and `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=2026-09-13`.

**Live pending before: 4. Live pending after: still 4; no live write performed.** Two wins and two losses are supported by the retrieved finals and original displayed lines. The full immutable ledger must be restored before appending revisions and recomputing the public aggregate. No displayed win rate, ROI, units or category total was manually changed.

## Reconciliation behavior

`result_reconciliation.match_result` recognizes explicit ESPN/MLB IDs; generic sportsbook IDs are not misused as score-provider IDs. Without a mapped ID, it requires canonical sport/away/home/date. Multiple same-day events require a saved game number or uniquely close scheduled start. Ambiguity stays pending. Unfinished MLB events participate in disambiguation so one completed doubleheader game cannot stand in for its unfinished partner. Invalid or conflicting finals cannot grade a pick.

Score revisions retain source/provider IDs and retrieval time. Repeat payloads ignore retrieval-only timestamp differences for idempotency. Prior immutable picks and locks remain unchanged. `Update results and publish` still restores history first, persists revisions, rebuilds the report and requests publication through the existing publication action. Passive rendering makes no score-provider calls. Scheduler uses the same retrieval/matching and reports pending reasons.

Owner diagnostics distinguish missing finals, missing mapped IDs, team mismatch, doubleheader ambiguity, invalid scores and provider failure. Internal diagnostic objects are not added to public result rows.

## Parlay funnel — saved production snapshot

Measured at the original package build time `2026-09-14T16:43:19.065161+00:00`, not by refreshing its odds:

| Stage | Count |
|---|---:|
| Overall picks | 11 |
| Spread/total markets | 11 |
| Valid prices | 11 |
| Positive estimated price EV | 1 |
| Fresh positive-EV supported quotes | 1 |
| Individually approved | 0 |
| Qualified pairs before | 0 |
| Qualified pairs after | 0 |
| Research pairs | 0 |

Ten legs fail positive value and all eleven are unapproved. The corrected builder properly does not manufacture a second eligible leg. Saved public data lacks detailed review transport metadata, so the live snapshot alone cannot attribute all eleven failures to Gemini.

New qualified same-book pairs support Novig and, for NFL/NCAAF, DraftKings, FanDuel and BetMGM under the shared policy. Cross-book pairs are not qualified tickets. STANDARD/PREMIUM can qualify with approval/value/quote safeguards; PROVISIONAL is straight-only. Legacy APPROVED records retain their historical interpretation. Every generated product-of-leg-price parlay remains **RESEARCH ONLY with no funded ticket claim**; actual ticket pricing and joint validation are still required before staking a parlay.

New packages use `parlay_policy=supported-v2`; packages without it validate with the original builder, preventing retroactive rewriting or invalidation of saved ticket composition. Research combinations retain their existing separate policy.

## Gemini outage policy

Only recognized timeout/5xx transport errors qualify for outage handling. Empty/incomplete reviews, low confidence, abstention and factual vetoes remain holds. A deterministic ineligible or nonpositive-EV row cannot be promoted.

Default explicit settings:

- `PARLAYPICKER_GEMINI_OUTAGE_MODE=capped` (`hold` disables the exception).
- `PARLAYPICKER_GEMINI_OUTAGE_CAP=0.001`: at most 0.1% of known bankroll per affected straight.
- `PARLAYPICKER_GEMINI_OUTAGE_MULTIPLIER=0.5`: at most half the deterministic stake, subject to every existing tighter cap.

Invalid settings fail closed. Outage rows are labeled `OUTAGE_CAPPED`, `gemini_approved=False`, and PROVISIONAL. The final allocator enforces the cap after other sizing logic and preserves the status. Without known bankroll, already-sized rows receive zero. These conservative operating limits are not claimed to improve profitability or be empirically optimal. Existing MEDIUM review reduction remains unchanged.

## NFL statistics finding

The same NFL schedule source used for aggregation lists `2026_01_DEN_KC` on September 14 with no final scores/result. Neither Denver nor Kansas City has a completed 2026 game in that feed at inspection. `DEN`, `KC`, Denver Broncos and Kansas City Chiefs normalize correctly. Regression tests verify actual aggregation when scores exist and an empty result when none exist. No synthetic prior-season or league-average statistics were inserted to force eligibility.

## Files changed

- `app_core/result_reconciliation.py`, `result_providers.py`, `public_history.py`: matching, provenance, provider fallback and grading integration.
- `app/ui/public_results.py`, `app_core/public_grading_scheduler.py`: append-only reconciliation and owner diagnostics.
- `app_core/public_parlays.py`, `public_board.py`, `per_game_boards.py`, `app/ui/publish_panel.py`: supported same-book policy, metadata, package compatibility and funnel.
- `app_core/gemini_bet_gate.py`, `llm_assistant.py`, `integrations/gemini_client.py`, `core/streamlit_pipeline.py`, `streamlit_app.py`: explicit outage status and downstream caps.
- `app_core/feature_processing.py`: explain unavailable completed-season NFL statistics.
- `tests/test_results_parlay_reconciliation.py`, `test_college_public_grading.py`, `test_imported_recaps.py`: regression coverage and obsolete universal-time-cutoff expectations.

## Remaining production steps

1. Review/merge the PR and deploy the app.
2. Use **Update results and publish** with the configured production history store. Verify the four original stored IDs/legs and appended provider revisions; investigate any discrepancy instead of changing picks.
3. Verify the live pending count and ledger-derived summaries after publication. This is an outstanding acceptance item, not a completed backfill.

More parlays or a higher displayed hit rate is not evidence of profitability.

## Verification

Focused new reconciliation plus Gemini regressions: **43 passed**. Final complete suite: **2,031 passed**, 9,258 warnings, 93.38 seconds (`test-results/reconciliation-release.txt`). Production compilation and `git diff --check` passed. Existing warnings remain visible. Official MLB game IDs for the three reconciled MLB matchups are 824952 (Seattle/Athletics), 824708 (Kansas City/Boston), and 823171 (San Diego/San Francisco); their scores agree with ESPN.
