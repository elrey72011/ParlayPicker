# Reconciliation, total-input diagnostics, and Results clarity

HEAD BEFORE: ec7bafcf8ca1c0a2869278bb9f5644c497dddc5b (main)

HEAD AFTER: implementation commit recorded below after validation.

SEPTEMBER 15 USED FOR WEIGHT TUNING: NO

## September 15 source status

**Exact live reconciliation remains blocked by unavailable saved-history access in this checkout.** No Drive service-account, Drive folder, or public site history configuration was present in the process environment, and no local immutable public-history export was found. The candidate CSV, website aggregate, and previously quoted 11–4 record cannot establish the original Locked or first Published cohorts. No live numbers are invented and no generated live data is committed.

RECONCILIATION SOURCES: implemented read-only `History.publications()` (confirmed receipt/package hash verification), active `History.all('locks')` (honoring appended owner removals), and saved `History.all('scores')`. The UI's restore routine is deliberately not called by the utility because it can confirm deployments. No current analysis, new quote, or new score request is used. Imported recaps are excluded from these cohorts.

| Exact Sept 15 cohort | W | L | Push | Pending | Needs review | Win rate |
|---|---|---|---|---|---|---|
| Locked Overall | Unverified | Unverified | Unverified | Unverified | Unverified | Unverified |
| Published Overall | Unverified | Unverified | Unverified | Unverified | Unverified | Unverified |
| Published Sides | Unverified | Unverified | Unverified | Unverified | Unverified | Unverified |
| Published Totals | Unverified | Unverified | Unverified | Unverified | Unverified | Unverified |
| Approved | Unverified | Unverified | Unverified | Unverified | Unverified | Unverified |
| Research | Unverified | Unverified | Unverified | Unverified | Unverified | Unverified |

RECORDS REQUIRING REVIEW: live inventory unavailable. Synthetic tests verify identity collisions are NEEDS_REVIEW, exact repeats deduplicate by existing immutable ID, original lock prices/lines survive, and pushes/pending/review never enter W/(W+L).

Run with configured read-only source access: `python -m scripts.reconcile_public_results --site SITE --folder FOLDER --date 2026-09-15`. Alternatively pass `--source PATH` to an existing local immutable-history JSON with publications, revisions, and active locks. Output JSON and Markdown go to ignored `output/`. The owner UI can prepare reconciliation JSON/Markdown for its selected results date from already-restored history; this performs no grading or remote writes. A proposed full source download was excluded after automatic approval review; the downloads contain only requested reconciliation facts, not raw private history payloads.

## Results behavior

RESULTS FIRST-LOAD DEFAULT: Yesterday + Locked + Game picks & parlays if yesterday has Locked Overall rows, otherwise Yesterday + Published picks with explicit fallback. Valid saved filter values take precedence, including a deliberately empty slice. Filter changes no longer silently switch the selection group.

Dedicated Yesterday — Locked Overall uses only that date/group/category, independent of the main filters. No settled records displays explicit no-data text with counts; Published never substitutes. The existing selected-period locked summary remains separately labeled.

ACTIVE FILTER LABEL: period, group, kind, effective league and market, exact Eastern date bounds, and matching entry count. Main summary remains per category; no combined overlapping win-rate denominator.

CATEGORY OVERLAP PROTECTION: tested; Overall/Sides/Totals, Published/Locked/Approved/Research overlap. Approved and Research reconciliation aggregates describe category entries, not independent wagers.

## Instrumented total diagnostics

TOTAL INPUT FIELDS FOUND: league, market_type, theover_probability, market_probability, kalshi_probability, ml_probability, exact ml_target, recent_regime_penalty_reason enums, degraded_feature_subset_flag.

TOTAL INPUT FIELDS NOT AVAILABLE: per-candidate structured weather/pitcher/lineup completeness. Weather certificate log failures do not become candidate availability facts. Gemini prose is never parsed.

TOTAL STATUS RULE (`mlb-total-inputs-v1`): MLB total_over/total_under only. Missing exact target model or fewer than two instrumented probability signals => INCOMPLETE. Otherwise any explicit missing-TheOver/stale-empirical/degraded-feature reason => DEGRADED; otherwise COMPLETE. COMPLETE refers only to instrumented fields, not verified comprehensive context. Signal count measures presence, not independence or calibration. Version, status, completeness, sorted reason codes, signal count are carried in candidate exports, private immutable evidence, selected per-game rows, and allowlisted public payloads. Run counters report real candidate classifications.

PROBABILITIES MUTATED: NO

RANKING MUTATED: NO

SELECTION RULE MUTATED: NO

WAGER AUTHORITY MUTATED: NO

One research pick per game remains; diagnostics cannot approve it or create a stake. No suppression, direction flip, thresholds, Kelly, weights, active sport policy, or model promotion changes.

VALUE DISPLAY: canonical decimal-price conversion supplies no-push-equivalent break-even and a descriptive probability-minus-break-even comparison. Existing producer EV is copied, never recomputed from display probability. Public rows show break-even and positive/negative/zero/unavailable value. Missing price is unavailable. The public wager-contract conservative estimates remain authoritative where already applicable. Precision keeps its existing probability ranking; EV and edge aliases copy the existing candidate values, with a value label. No new wagering threshold.

## Prospective evaluation

PROSPECTIVE COHORT LABEL: frozen design in `2026-09-16-total-quality-prospective-plan.md`, version `mlb-total-inputs-v1`.

COHORT FROZEN AT PREDICTION TIME: append-only capture and post-grading regression verified. Existing unlabeled snapshots are excluded; reports never call live annotation.

`python -m scripts.report_total_quality PATH_TO_EVIDENCE_DB` opens SQLite with `mode=ro` and `query_only`, verifies prediction/score/closing hashes, and reports each quality cohort plus Over/Under. W/L excludes pushes; ROI includes priced pushes. Calibration metrics use only recorded calibrated_probability, never a fallback current model. Missing probabilities and closes retain their own denominators. Genuine closing observations are rechecked for namespace/event/book/time/line and original price consistency; no proxy CLV.

Units are candidate snapshots, with correlated repeats and opposite directions explicitly acknowledged. This is not an independent wager record or a superiority claim. No tiny-sample inference or September 15 causal claim is made.

AUTOMATIC GATE FROM COHORT: NO

MLB CHALLENGER STATUS: UNCHANGED / RESEARCH

## Validation

FULL TESTS: 2,452 passed; 9,975 existing warning emissions. Final focused reconciliation rerun: 4 passed.

PRODUCTION SAFETY: 545 passed (current CI command).

PRODUCTION COMPILE: passed (current CI command plus new scripts).

GIT DIFF CHECK: passed.

Initial full run found outdated lock-test fixtures: price mutation left derived display values inconsistent, and package build used wall-clock time with fixed September 15 starts. Fixtures now freeze build time and regenerate dependent display/funnel values after intentional edits, retaining all lock-immutability and freshness assertions. Production validation was not weakened.

FILES CHANGED: focused reconciliation and prospective-report helpers/CLIs; totals diagnostic and value display helpers; existing candidate projection/pipeline/per-game/public exports; Precision display/export; public Results template; owner reconciliation download; focused tests and audit documents.

UNRELATED CHANGES: NONE. Pre-existing untracked Medium article Markdown and Word document are excluded.

Remaining acceptance work: execute exact September 15 reconciliation against restored immutable production history and review the counts before deployment. This implementation does not certify the previously quoted live record.
