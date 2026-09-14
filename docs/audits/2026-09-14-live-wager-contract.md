# Live wager contract and parlay promotion

HEAD before: 772d909735879db73105613565de0ca9c5e36ce9.
Baseline: 2,045 tests passed, 9,258 warnings, 96.60 seconds.

## Pre-change architecture

- Candidate creation and line integrity: ACTIVE in run_pipeline / build_best_picks_df.
- Ranking: ACTIVE probability-first family finalists; winner selected before final gates.
- Calibration: ACTIVE legacy blending, empirical tiers and absolute gate; not equivalent to independently validated conservative evidence.
- Gemini: ACTIVE selected-winner secondary review; legacy approval/outage adapter.
- Production gate: ACTIVE legacy status / empirical / export reconciliation; DUPLICATE authority across those layers.
- Portfolio: ACTIVE optimize_portfolio_allocation and empty-card recovery, before final export.
- core.wager_decisions and SportPolicy: NOT LIVE-WIRED; current default policies have no validation IDs and zero caps.
- Public export: ACTIVE per_game_boards and pick_record, legacy APPROVED + Bettable + positive stake.
- Public parlays: ACTIVE supported-book pairs, still gated by APPROVED; qualified tickets labeled RESEARCH ONLY.
- Smart parlay / best duos: separate legacy/research presentation generators; not actual combined ticket price authority.
- Public history: ACTIVE immutable publications; never retroactively promote saved research.

Implementation must retain zero policy caps and absent validation evidence as explicit blockers. No current slate can be promoted merely by connecting modules.

## Implemented live authority

The terminal live pipeline now calls `finalize_live_wagers` with all analysis candidates before its final funded selection. Each candidate must independently pass `candidate_decision`; the strongest eligible conservative-EV candidate may replace an ineligible family winner. If none qualifies, the original research lean remains with $0. Existing selected-ticket Gemini reviews are bound to that exact selection/market/price; they are not copied to other candidates. A candidate without its own review is explicitly unavailable.

`live-v1` snapshots carry exact selection, source, quote time, conservative metrics, maturity, policy/model/calibration/evidence versions, review state and allocated amount. Portfolio and export compatibility projections enforce this final authority. Legacy recovery cannot refund rejected rows. Moneyline remains excluded. No sportsbook execution, account action, or network request is introduced in rendering.

Gemini outage handling is opt-in. TIMEOUT, SERVICE_ERROR and UNAVAILABLE differ from veto/abstention. A configured outage cap must be positive and no more than 1% of bankroll, with a strictly reducing multiplier below 1. The resulting fraction is the minimum allowed by independent Kelly, maturity, sport and outage caps, followed by game/team/portfolio/daily/weekly exposure constraints. Missing or invalid configuration holds at $0. A factual hard veto remains blocked. Outages never change probabilities, EV, edge, evidence or maturity.

## Parlay behavior

New publications use `canonical-v3`; historical packages retain their prior validation semantics. Standard/Premium funded straight legs can form same-book two- and three-leg candidates only with positive conservative EV/edge, verified fresh exact quotes, unstarted games, resolved identity and completed acceptable secondary review. Provisional and outage-only straight recommendations do not enter this production parlay pool. Shared games/teams are excluded within tickets. Multiple totals in one league have unknown correlation and are excluded; LOW denotes the implemented structural screen, not an empirically validated correlation estimate.

The pool is bounded to 20 production legs and five displayed tickets, with at most two uses per game. At least a qualifying pair and triple are prioritized when available. Combined odds and probabilities remain illustrative estimates. No actual combined ticket price is verified here: every candidate says QUALIFIED - VERIFY TICKET PRICE and has $0 ticket stake. This does not authorize a production parlay wager.

Research generation now supports up to ten distinct same-book positive-EV pairs from a bounded 30-leg pool. Teams can recur across different tickets; never within a ticket. Research remains $0 and cannot authorize production. Saved original publications and decisions are not rewritten or promoted.

The public board and owner panel expose saved eligibility counts and exclusion reasons, including maturity, unavailable review, invalid/stale quotes and lack of compatible partners. Legacy APPROVED labels alone do not grant new authority.

## Configuration and activation limits

`PARLAYPICKER_WAGER_POLICY_PATH` names an owner-supplied JSON artifact. It requires timezone-aware `validated_at` and `expires_at`, plus `sports` entries matching the exact `SportPolicy` constructor and sport key. Every active sport needs its own validation ID and independently validated settings. No production policy artifact was fabricated or activated in this change; repository research defaults still have zero caps.

Optional `gemini_outage` keys: `mode` (capped), `cap` (bankroll fraction, at most .01), `multiplier` (strictly between 0 and 1). Absence means hold. `unit_value`, if supplied, converts dollars to displayed units.

`exposure` must include a timezone-aware `as_of` no older than 30 minutes, a `committed` mapping and explicit fraction limits `total_cap`, `daily_cap`, `weekly_cap`, `game_cap`, `team_cap`. Committed keys include total, daily, weekly, game:<sport>:<id>, sport:<sport>, team:<sport>:<id>. Parlay risk must count against each underlying game and team. Missing limits default to zero. This adapter consumes an explicit fresh exposure snapshot; it does not yet collect a live bankroll ledger automatically.

Candidate evidence must provide verified identity, stable game/team IDs, exact quote verification, validated model/calibration versions, a frozen evidence snapshot, effective sample size, conservative probability and explicit maturity. Legacy blended probabilities or a bucket win rate are not fabricated into these fields. Maturity remains explicitly supplied; automatic Research-to-Provisional assignment was not implemented after automatic approval review rejected that edit. The safe alternative retains RESEARCH until independently validated maturity is supplied.

Therefore this is a fail-closed authority integration, not proof of profitable predictions or an activated funded strategy. Completing upstream validated evidence/maturity production and exposure-ledger integration remains necessary before routine live funding. Do not create policy IDs or relax thresholds merely to produce recommendations.

## Saved-slate comparison

Read-only replay of the public package built at 2026-09-14T19:26:03.665239+00:00 used 11 overall selections. The legacy pool contained two positive-EV eligible research legs, zero approved production legs and one research pair. Under canonical replay, all 11 lack the new validated contract and remain Research: zero Provisional, Standard, Premium or production legs, zero qualified pairs/triples, and one research pair. The nine nonpositive-EV rows were not forced into tickets. These are replay diagnostics, not new historical results or an assertion that live execution was activated.

## Validation and delivery

Baseline: 2,045 passing tests. Final full-suite result is recorded below. Dedicated regression coverage includes default outage holds, all deterministic blockers during outages, hard vetoes, maturity preservation, caps, sport isolation, candidate-first selection, legacy stake override prevention, same-book/correlation restrictions, pair/triple generation, bounded research reuse, public schema round trips, immutable packages and rejection of a fabricated positive ticket stake. Browser-render tests verify the public qualification and $0 messaging.

Changes are local and uncommitted. No deployment, publication, backfill or history mutation was performed by this task. Unrelated article files were left untouched.

Final validation: **2,092 passed**, 9,261 warnings, 94.85 seconds. Full output: test-results/live-wager-final.txt (local ignored artifact). git diff --check passed. The warning count is reported without claiming all warnings were resolved.

Deployment clarification: see 2026-09-14-sport-deployment-assessment.md. Policies now have explicit independent deployment states and separate provisional evidence minima; default remains UNVALIDATED. This supersedes any implication that a validation ID alone grants authority.
