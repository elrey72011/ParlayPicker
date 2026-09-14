# Deterministic MLB event matching

Baseline HEAD: `e5cdb702e206baa28f0eea68a828eafd5cc2d1b3`.

## Scope and old failure mechanism

Inspected public_history, result_team_names, espn_results, public_results, result_reconciliation and result_providers. No selection, calibration, Gemini, parlay, or bankroll logic changed.

The baseline already accepted unique same-date delays. Its remaining MLB defects were using a 30-minute proximity filter for multiple same-provider events, using start-derived dates exclusively, accepting changed-date provider IDs for settlement, and supporting only a limited set of ID fields. This implementation replaces the MLB identity branch only.

## Files

- app_core/mlb_event_matcher.py: pure structured matcher, scheduled Eastern date, scoped IDs, normalized game numbers.
- app_core/result_reconciliation.py: delegates MLB and exposes structured owner diagnostics.
- app_core/public_history.py: derived report diagnostics; outcome arithmetic unchanged.
- app_core/public_board.py: optional diagnostics schema, restricted to known reason/method codes; compatibility only.
- app_core/result_providers.py: preserves official MLB scheduled date from existing fallback.
- tests/test_mlb_event_matcher.py: deterministic regression coverage.
- tests/test_results_parlay_reconciliation.py: replaces obsolete time-based doubleheader and generic reason expectations.

## Hierarchy and safeguards

Same-provider immutable IDs first, with team verification and changed-date settlement review. Otherwise canonical away/home and scheduled Eastern date. A unique event accepts arbitrary same-day time drift. Multiple events require a saved normalized game number; start proximity and API ordering never resolve ambiguity. Unfinished doubleheader schedule records remain protective evidence. Cross-provider final score conflicts remain unresolved.

IDs use mlb, espn, odds_api namespaces. Generic game_id is scoped only with an explicit provider. Result-source/event_id pairs remain supported for normalized legacy provider observations. Identical strings in different namespaces never match by ID; ordinary team/date fallback remains permitted.

Date priority: official_date, scheduled_date, game_date, event_date, scheduled_start, start, commence_time. Datetimes must be timezone-aware. Collection/completion times are not identity evidence.

Structured statuses include matched provider ID/unique event/game number, invalid saved/result event, no match, date mismatch, provider ID not found/conflict, doubleheader ambiguity, multiple conflicting events, and rescheduled review. Report diagnostics and owner diagnostics are derived, not written over saved selections.

## Validation and historical reconciliation

80 initial focused tests passed, including the four September 13 synthetic outcome fixtures (WIN, LOSS, LOSS, LOSS). Fixtures are not production overrides. Full suite result is recorded below.

No migration of immutable picks is required. No live records were rewritten or backfilled. After deploying, the normal result-refresh action must append refreshed provider evidence and recompute reports. Unique delayed September 13 MLB records can reconcile if their real saved team/date evidence matches; real immutable records were not accessed or graded in this task, so no historical repair is claimed. Rescheduled and ambiguous records remain pending for review.

Final full suite: 2,045 passed, 9,258 warnings, 94.33 seconds. Includes 14 dedicated MLB matcher cases and existing September 13 regression fixtures. git diff --check passed. Validation preceded the commit and PR. No deployment or live backfill was performed by this implementation task.
