# Go-live staking terms for the NCAAB Pinnacle plan (proposal, 2026-11-08)

Owner decision document. Nothing here is built yet. Part 1 must be decided **before the
NCAAB Pinnacle plan is frozen**; parts 2 and 3 can follow while the plan validates.

## Why this matters now

A plan's staking terms are frozen with it. `prospective_evidence.deployment_state` reads
`deployment_criteria` (its `maturity_rules` and `validated_policy`) from the **frozen plan's
payload**, and the live chain uses them:

| Check | Code | Needs from the frozen plan | Current template |
|---|---|---|---|
| Owner activation verifies | `core/sport_market_activation._policy_matches_frozen_plan` | `deployment_criteria.validated_policy`: the exact `SportPolicy` you will activate | **absent**, so activation can never verify |
| A pick reaches a stake tier | `core/candidate_maturity.assign` | `deployment_criteria.maturity_rules.PROVISIONAL = {max_uncertainty, min_prior_clv}` | **only yes/no promises** (`exact_market_model_and_calibration`, ...), so every pick stays RESEARCH at $0 |
| The tier's CLV check | `core/live_wager_contract` (line ~223) | the validation report's `metrics.price_clv_lower_95` | **not produced** by market-skill evaluation |

So a plan frozen from today's template would pass validation and still never stake. Fixing
that afterwards means a new plan version and restarting validation (weeks). Adding the terms
to the draft costs a day or two.

## Part 1. Staking terms frozen with the plan (decide before the freeze)

### 1a. `validated_policy` (the `SportPolicy` you will activate)

| Field | Meaning | Proposed | Notes |
|---|---|---|---|
| `sport`, `version` | | `NCAAB`, `ncaab-spread-pinnacle-v1` | |
| `deployment_state` | tier the plan targets | `PROVISIONAL_VALIDATED` | template's `target_state` |
| `provisional_allowed` | | `true` | only the provisional tier at first |
| `provisional_minimum_evidence` | effective sample a pick's evidence must reach | **400** | about validation + holdout games (2 x 200) |
| `minimum_evidence` | for STANDARD | 800 | not used while only PROVISIONAL is allowed |
| `kelly_fraction` | share of full Kelly | **0.10** | conservative while the edge is new |
| `provisional_stake_cap` | max stake per pick, share of bankroll | **0.005** (0.5%) | |
| `standard_stake_cap`, `premium_stake_cap` | | 0, 0 | later plan versions |
| `min_conservative_edge` | edge on the conservative probability | **0.01** | below the plan's 2% bet rule on the mean estimate |
| `sport_exposure_cap` | total open NCAAB stake, share of bankroll | **0.03** (3%) | |
| `uncertainty_quantile` | | 0.25 (default) | |
| others | | code defaults | |

### 1b. `maturity_rules.PROVISIONAL`

| Rule | Per pick | Proposed |
|---|---|---|
| `max_uncertainty` | `calibration_uncertainty` (2b) | **0.05**: the pick's conservative probability is at most 5 points under its mean |
| `min_prior_clv` | `prior_clv_lower` (1c) | **0.0** |

### 1c. `metrics.price_clv_lower_95` in the validation report (evaluation code)

Proposed: the **holdout bets' mean CLV minus 1.96 standard errors**. With `min_prior_clv = 0`,
that is a stricter test than the plan's own CLV gate (the mean CLV >= 0).

- **Option A (recommended):** the 95% lower bound, as above. Safer: with about 50 bets, a small positive mean CLV will often fail it.
- **Option B:** the holdout bets' mean CLV, the same test the plan already passes. Faster to stake; weaker.

This is evaluation code, not a frozen plan field, but it must be in place before the plan's
validation artifact is created, because the artifact's report must match a recomputation.

## Part 2. Per-pick fields the app bridge must fill (build while the plan validates)

`core/wager_decisions.candidate_decision` and `candidate_maturity.assign` require these on each
bridge row (`app_core/qualified_bridge.row`). Today none is set, so a bridge pick gets $0.
Each is **derived from stored evidence, never guessed**:

| Field | Required value | Proposed source |
|---|---|---|
| `model_validated` | `True` | `True` only when the market's `deployment_state` is VALIDATED and the pick's `model_id` / `model_version` equal the deployment's (the gate already checks equality) |
| `calibration_validated` | `True` | same, for `calibration_id` / `calibration_version` |
| `evidence_snapshot_id` | non-empty | the prediction's own `evidence_snapshot_id` (stored, not copied today) |
| `critical_feature_error` | `False` | `False` only when the prediction's feature snapshot holds Pinnacle's paired probability and its quote replays (`verify_provider_offer`); else `True` |
| `evidence_effective_sample_size` | ≥ `provisional_minimum_evidence` | the validation report's validation + holdout `effective_observations` |
| `calibration_uncertainty` | ≤ `max_uncertainty` | the prediction's `mean_probability - conservative_probability` |
| `current_regime_conflict` | `False` | **Decision:** (a) `True` when the market's last 50 settled live bets have mean CLV < 0 (a drift brake, recommended); or (b) always `False`, with no regime check for this market |

## Part 3. Go-live plumbing (build while the plan validates)

1. **The app reads a small file, not the full store.** Decision capture already holds the full store, runs every 15 minutes during slates, and has Drive write access. It would publish one file per qualified market: the deployment facts above plus the current predictions, quotes and events (including the raw provider bytes the gate replays for DraftKings, FanDuel and BetMGM). The bridge, qualification and the gate read it instead of downloading about 89,000 objects (which the app does not survive).
2. **Decision capture enters only at the plan's books** (`entry_books`): Novig, DraftKings, FanDuel, BetMGM.
3. **The exposure ledger gets a permanent home.** It is a local SQLite file that Streamlit wipes on reboot.

## Decisions needed

1. The policy values in 1a (bankroll shares, Kelly fraction, caps).
2. `max_uncertainty` in 1b.
3. The CLV bound in 1c: Option A (95% lower bound) or Option B (mean).
4. The regime brake in Part 2: (a) or (b).

**Then:** the Bound plan draft adds these terms to the plan, a new draft is reviewed, and you
freeze it. Expected delay: one to two days, against a guaranteed restart if frozen without them.
