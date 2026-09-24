# Football V3 feasibility audit

This is a **read-only feasibility study**, not a V3 validation plan. V1 and V2 plans remain immutable. The result is `NO_DEFENSIBLE_SHORT_SEASON_V3` unless authenticated, replayable, exact-market evidence supports a different conclusion in a future independently reviewed change.

## Authenticated result, 24 September 2026

[GitHub Actions run #109](https://github.com/elrey72011/ParlayPicker/actions/runs/36033422917) completed successfully on commit `ae80baefdcad3f9f8afb3f9c7c365f2bafe8ba04`. Its [sanitized audit](audits/football-v3-feasibility-2026-09-24.json) has canonical SHA-256 `f8da82899c7c8b2c67fa4eaf704bd0076371d48d3a0b224884eac8975ac1a39c`. The downloaded artifact ZIP SHA-256 matches GitHub's `826726073b4fcb4b3a97d127ae5380736057db662fd7f86fe498ac2812c29bca`. The job verified 2,715 canonical objects, 17 NFL source objects, and 5 NCAAF source objects, with **zero remote writes**.

| Exact market | Source projected quote rows | Unique captured games | Verified pregame price games | V2 gate replay games | Legal V3 replay games | Exact canonical model/calibration pairs | Certified closes | Canonical 2026 predictions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| NFL Spread | 194 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| NFL Total | 196 | 9 | 0 | 0 | 0 | 0 | 0 | 0 |
| NCAAF Spread | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| NCAAF Total | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

The NFL source rows fail the unchanged V2 replay gate on identity, verified price, as-of feature, reproducible model/runtime, prediction chronology, and verified result availability. The five restored NCAAF source objects yield no projected exact-market quote rows. No historical season qualifies. The authenticated canonical store contains one V1 plan per football market and **no V2 plan record** for these markets; the V2 policy code remains present and unchanged. No canonical exact-scope model or calibration artifact exists. The missing binding is the primary blocker even before statistical design.

With zero legal replay games, Brier/log-loss variance, calibration uncertainty, push/void rate, dependence, effective sample, and cluster-aware sensitivity cannot be estimated. A bootstrap or sequential sample requirement would be invented here, so those fields stay null. A Bayesian design lacks an independently justified prior. V2's frozen distribution-free benchmark needs 9,451 independent decided games per market; an NFL regular season has at most 272 games. Neither 2026 full-slate coverage nor an eligible-games-per-week denominator is verified. Captured NFL games are only 9, and the NCAAF intended FBS population, conference coverage, and remaining capacity are unverified. No near-start snapshot is relabeled as a certified close.

**Conclusion: `NO_DEFENSIBLE_SHORT_SEASON_V3`.** The audit created no V3 plan, did not assign historical rows to prospective confirmation, and did not activate a market or stake. A later V3 proposal needs authenticated legal replay across multiple seasons, exact-scope reproducible model and calibration artifacts, a complete slate denominator, certified price/close coverage, and a prospectively frozen error-controlled method. No GitHub secret or variable change was needed for this audit; the existing service-account secret accessed the supplied Shared Drive.

## Authenticated evidence path

The `football_v3_audit` input on the existing research workflow runs a separate audit job. Normal research and activation jobs are skipped for that dispatch. The audit restores the canonical, NFL, and NCAAF stores from the configured Shared Drive through a wrapper that rejects every remote write. It verifies object hashes and produces a sanitized JSON artifact containing counts, plan hashes, model/calibration metadata, replay blockers, and a readiness row for each of the four markets. A local zero is labeled `LOCAL_ONLY_REMOTE_UNKNOWN`; only a completed authenticated restore may claim remote counts.

The job targets Shared Drive `0AAx101w9xyZcUk9PVA`, supplied by the owner. Its only required repository secret is `PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT`. The workflow does not log its value or call provider capture APIs. An interrupted or incomplete restore reports `AUTHENTICATED_RESTORE_FAILED`; no source row is promoted into canonical prospective evidence.

## Replay and statistical study

Each restored source row is evaluated with V2's as-of replay gate. Missing identity, verified timestamped prices, feature replay proof, model/runtime replay proof, training input availability, or result-availability proof blocks replay. Historical replay remains separate from prospective confirmation. Only passing rows can enter the exploratory season/week block-bootstrap study. Its deterministic seed and resample count are recorded, as are the number of independent events and clusters. The study does not infer a full-slate denominator from a captured subset and explicitly leaves team-repeat or season dependence unresolved.

Four methods are compared in each market's output: V2's 9,451-event distribution-free fixed checkpoint; a cluster-aware fixed bootstrap; a sequential confidence design; and Bayesian calibration. Without sufficient legal replay, exact-scope model/calibration binding, full-slate coverage, and independently justified Bayesian priors, the latter three have **no calculated prospective minimum**. Null means unavailable, never zero. Expected games per week, remaining 2026 capacity, and checkpoint dates also remain null until a complete authenticated schedule and price/settlement coverage audit supports them. The NFL's 272-game full-season regular-season count is only an upper bound, not an eligible-event estimate.

Paper ROI and EV are diagnostics and cannot override Brier, log loss, calibration, or coverage evidence. Near-start quote candidates are not certified closes. This audit creates zero V3 plans, activates no market, changes no parlay or straight-wager gate, and places no wager.
