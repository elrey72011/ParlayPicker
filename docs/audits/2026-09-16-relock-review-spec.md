# Re-lock review implementation spec follow-up

Base: d0b10009fda9e6e3f69f0acac5fd88e76c04adde (main after merged PR #2288).
Branch: codex/relock-review-hardening.

This follow-up implements the stricter review spec without rebuilding the merged feature. The comparison is now a frozen mapping/dataclass with lock ID, archived hash, removal timestamp/reason, prior/current saved fields, explicit change flags, normalized spread/total/unknown families, severity, state and change code. Structured home/away orientation wins over display text. Legacy public legs use their saved matchup; only locks lacking team/matchup fields use the explicitly tested display fallback. Unknown fields remain Not recorded. Outcomes never influence classification.

Latest removed generation is chosen chronologically using timezone-aware times and verified against its archived hash. Saved paired moneyline or standard spread quotes can establish a favorite, but missing, conflicting or ambiguous pairs produce no favorite claim. This is informational; it does not alter quote binding or eligibility.

NORMAL line/price/book/time changes use one review button, no checkbox or modal. WARNING/HIGH require acknowledgment plus a final dialog; CRITICAL also requires exact RELOCK. Cancel is primary and clears acknowledgment on the next full rerun. Changing preview/history, restoring history, saving, cancelling or losing eligibility invalidates review state. Comparison always remains visible. Recorded differences enumerate before/after values and make no causal claims.

Immediately before storage, the UI reruns the existing lock candidate builder and compares exact reviewed legs. Reviewed storage calls rerun eligibility after reading authoritative history, reject concurrent active locks, and verify the reviewed removed-lock generation before writes. A conditional-write loser is reported as an existing concurrent lock rather than a successful replacement. First-write-wins remains unchanged for storage and compatibility callers. Batch races/failures may leave other successfully saved rows; messages do not promise an atomic rollback. No new lock schema fields or collections were introduced.

Logs contain lock identity, severity/change code, old/current market family and outcome only. No token, credential or quote payload is logged. No automated sportsbook execution is introduced.

Acceptance coverage: immutable prior source, latest generation lookup, normal/within-family/both family directions/team reversal, structured-team precedence, legacy missing data, paired favorite evidence, immutable comparison, metadata flags, side-by-side review, reset after Cancel/preview changes, single-click normal save, stale/started/other-date/invalid exclusions, candidate mutation with zero writes, fresh exact quote persistence, concurrent active lock before write, and conditional-write race loser. Existing lock and removal records remain unchanged.

Validation: full suite 2,483 passed; production safety 545 passed; focused lock/comparison/UI suite 61 passed. After final presentation/logging adjustments, all 31 comparison/UI tests passed again. Production compilation and git diff checks passed.

No selection/ranking, model/probability, blend, Gemini, Kelly, maturity, policy, lock identity, quote threshold or binding changes. No production history modified. No commit, push or deployment performed for this follow-up. Unrelated local files remain untouched.
