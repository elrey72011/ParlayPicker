# Probability integrity v1.1 review

Base: `origin/main` at `76b16c96e1498961e96a8fd1077f0c1e3c8251db`, the merge commit for PR #2351.

## Completion states

| Area | State | Evidence |
|---|---|---|
| Calibration math | PASS | Both counterexamples fixed; 10 randomized weighted datasets match sklearn within `1.12e-16`. |
| Probability path integrity | PASS (static and focused runtime) | Read-only call-site trace has zero parse errors; focused tests pass 81/81. |
| Real calibration refit | NOT RUN | No live artifact was rewritten. Candidate generation requires exact source predictor version and scope. |
| Current live calibration | REJECTED | Its train and holdout boundaries fall on the same date. Explicit loading is research-only. |
| Subscriber probability contract | PASS locally | Version 2 separates mean and conservative EV and rejects ambiguous semantics. PostgreSQL CI remains pending. |
| Qualification authority | UNCHANGED except stricter consumption | Existing binding remains; only exact upstream `APPROVED` is accepted. |
| Market activation | NOT PERFORMED | No market, sales, or billing activation is part of this change. |

## Implemented repair

The runtime PAV fitter now validates equal input lengths, finite in-range probabilities, binary labels, positive optional sample weights, and an explicit missing-data policy. It groups identical predictor values before fitting, then writes the fitted block value at every unique original x-coordinate. Calibration application rejects ambiguous knots, clamps within the fitted domain, and preserves missing values.

New calibration artifacts are schema v2 and carry a fitting implementation version, content digest, probability semantics, chronology, exact source predictor version, and exact training scope. The fit command writes only to a candidate path. `--force` can create research evidence but cannot write the live path or establish production authority.

Subscriber release v2 declares unconditional win/push/loss mass. It exposes mean EV and fixed-push conservative EV separately, enforces `p_win_conservative <= p_win`, and rejects unknown or conditional probability semantics. The subscriber service remains isolated from the research runtime; parity with `core.price_value.price_value` is established only in tests.

## Regression evidence

- Focused probability, promotion, refresh, semantics, and subscriber-contract suite: **81 passed**.
- Dedicated subscriber contract and isolation suite excluding the platform-specific scope hash test: **15 passed, 1 deselected**.
- PostgreSQL module: **7 collected and skipped locally** because neither Docker nor PostgreSQL is installed.
- Full application shard 1: **1,498 passed, 19 failed** locally. Eighteen failures are Windows SQLite cleanup locks; one reads a local Streamlit secret that the test expects absent.
- Full application shard 2: **1,512 passed, 2 failed** locally. One is a Windows SQLite cleanup lock; one is a pre-existing local historical-fixture row-count mismatch.
- The protected-file diff against current main is empty. The Windows checkout cannot reproduce the Linux scope guard's byte hashes because Git checked protected files out with CRLF line endings; the pull-request Linux job is authoritative.

None of the 21 local full-suite failures touches the probability-integrity diff. The PR must still pass both clean Linux application shards and the PostgreSQL subscriber job before merge readiness can be claimed.

## Exact remaining staging and launch blockers

1. Clean Linux full-suite CI and PostgreSQL subscriber CI have not yet run for this branch.
2. No staging database, OIDC, gateway, or hosted deployment credentials were supplied in this local environment.
3. No compatible, independently reviewed calibration v2 production candidate exists. The current live artifact remains rejected and unchanged.
4. A future candidate must record the immutable source predictor version and exact sport/market/time scope, pass a strictly future holdout, and receive the existing independent review and activation authority.
5. Commercial approvals, owner sales enablement, commercially enabled markets, and live billing remain disabled.

## Audit files

- `calibration-counterexamples.json`
- `calibration-reference-parity.json`
- `calibration-artifact-inventory.json`
- `probability-path-trace.json`
- `probability-semantic-tests.json`
- `subscriber-probability-compatibility.json`
- `protected-file-diff.txt`
