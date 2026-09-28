# Merge and reuse matrix

| Upstream item | Actual state at branch creation | Treatment in paid-launch work |
|---|---|---|
| PR #2349 | Merge commit `8dfeb4da` is an ancestor of current main | Reused as a protected dependency; no cherry-pick or rewrite |
| PR #2350 | `MERGED` at `2026-09-28T20:20:47Z`; merge commit `5fb8e13577c3092f1eda4ad9b787368d3f691c71`; checks passed | Used as exact branch base; status documentation not reproduced |
| Evidence Drive/remote storage | Present on main | Read-only upstream dependency; subscriber runtime cannot import it |
| Performance spans/stage timing | Present on main | Protected; inherited tests and benchmark rerun unchanged |
| Public history/lock UI/retry | Present on main | Protected; paid-release retry is a separate database job and writes no locks |
| Customer auth/billing/entitlement/database | No production-capable equivalent found | Added under `services/subscriber` with isolated dependencies |
| Existing public publisher | Present and public by design | Preserved; not reused for premium delivery; leak containment remains a launch blocker |

The baseline and exact protected Git blob IDs are in `launch-baseline-manifest.json`.
