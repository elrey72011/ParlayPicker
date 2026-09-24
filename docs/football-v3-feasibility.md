# Football V3 feasibility audit

This is a **read-only feasibility study**, not a V3 validation plan. V1 and V2 plans remain immutable. The result is `NO_DEFENSIBLE_SHORT_SEASON_V3` unless authenticated, replayable, exact-market evidence supports a different conclusion in a future independently reviewed change.

## Authenticated evidence path

The `football_v3_audit` input on the existing research workflow runs a separate audit job. Normal research and activation jobs are skipped for that dispatch. The audit restores the canonical, NFL, and NCAAF stores from the configured Shared Drive through a wrapper that rejects every remote write. It verifies object hashes and produces a sanitized JSON artifact containing counts, plan hashes, model/calibration metadata, replay blockers, and a readiness row for each of the four markets. A local zero is labeled `LOCAL_ONLY_REMOTE_UNKNOWN`; only a completed authenticated restore may claim remote counts.

The job targets Shared Drive `0AAx101w9xyZcUk9PVA`, supplied by the owner. Its only required repository secret is `PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT`. The workflow does not log its value or call provider capture APIs. An interrupted or incomplete restore reports `AUTHENTICATED_RESTORE_FAILED`; no source row is promoted into canonical prospective evidence.

## Replay and statistical study

Each restored source row is evaluated with V2's as-of replay gate. Missing identity, verified timestamped prices, feature replay proof, model/runtime replay proof, training input availability, or result-availability proof blocks replay. Historical replay remains separate from prospective confirmation. Only passing rows can enter the exploratory season/week block-bootstrap study. Its deterministic seed and resample count are recorded, as are the number of independent events and clusters. The study does not infer a full-slate denominator from a captured subset and explicitly leaves team-repeat or season dependence unresolved.

Four methods are compared in each market's output: V2's 9,451-event distribution-free fixed checkpoint; a cluster-aware fixed bootstrap; a sequential confidence design; and Bayesian calibration. Without sufficient legal replay, exact-scope model/calibration binding, full-slate coverage, and independently justified Bayesian priors, the latter three have **no calculated prospective minimum**. Null means unavailable, never zero. Expected games per week, remaining 2026 capacity, and checkpoint dates also remain null until a complete authenticated schedule and price/settlement coverage audit supports them. The NFL's 272-game full-season regular-season count is only an upper bound, not an eligible-event estimate.

Paper ROI and EV are diagnostics and cannot override Brier, log loss, calibration, or coverage evidence. Near-start quote candidates are not certified closes. This audit creates zero V3 plans, activates no market, changes no parlay or straight-wager gate, and places no wager.
